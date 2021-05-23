import random
import time
from abc import ABC, abstractmethod
from typing import Type, Dict

import torch
import wandb
from nltk import WordPunctTokenizer
from nltk.translate.bleu_score import corpus_bleu
from torchtext.data import Field, TabularDataset, BucketIterator
from tqdm import tqdm

from utils import Task, get_text
from neural_network import NN_CATALOG


class Experiment(ABC):
    def __init__(self, config, device):
        self.config = config
        self.tokenizer = WordPunctTokenizer()
        self.device = device

        self.source = self.init_field()
        self.target = self.init_field()
        self.task = Task(*self.read_data())
        self.model = self.init_model()
        self.trainer = self.init_trainer()
        self.time = None

    def train(self):
        start_time = time.time()
        train_iterator, val_iterator, _ = self.task
        self.trainer.train(train_iterator, val_iterator)
        end_time = time.time()
        self.time = end_time - start_time

    def test(self):
        _, _, test_iterator = self.task
        vocab = self.target.vocab

        original_text = []
        generated_text = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iterator)):
                src = batch.src
                trg = batch.trg

                output = self.model(src, trg, 1.)  # turn on teacher forcing

                # trg = [trg sent len, batch size]
                # output = [trg sent len, batch size, output dim]
                output = output.argmax(dim=-1)

                original_text.extend([get_text(x, vocab) for x in trg.cpu().numpy().T])
                generated_text.extend([get_text(x, vocab) for x in output[1:].detach().cpu().numpy().T])

        bleu = corpus_bleu([[text] for text in original_text], generated_text) * 100
        print(f'Bleu: {bleu:.3f}')
        table = wandb.Table(data=[[self.time // 60, bleu]], columns=["time_spent (min)", "bleu"])
        if wandb.run:
            wandb.log({"bleu_score": wandb.plot.scatter(
                table,
                "time_spent (min)",
                "bleu", title="BLEU score")
            })

    def read_data(self):
        data_config = self.config['data']

        dataset = TabularDataset(
            path=data_config['path'],
            format='tsv',
            fields=[('trg', self.target), ('src', self.source)]
        )
        split_ratio = [data_config[f'{split}_size'] for split in ['train', 'val', 'test']]

        random.seed(42)
        random_state = random.getstate()
        train_data, valid_data, test_data = dataset.split(split_ratio=split_ratio, random_state=random_state)
        self.source.build_vocab(train_data, min_freq=data_config['word_min_freq'])
        self.target.build_vocab(train_data, min_freq=data_config['word_min_freq'])
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=data_config['batch_size'],
            device=self.device,
            sort_key=self._len_sort_key,
        )
        return train_iterator, valid_iterator, test_iterator

    def init_model(self):
        model_config = self.config['model']
        model_class = NN_CATALOG[model_config['name']]

        input_dim = len(self.source.vocab)
        output_dim = len(self.target.vocab)
        model = model_class(input_dim, output_dim, self.device, **model_config['params'])
        if 'model_path' in self.config:
            model.load(self.config['model_path'])
        model.to(self.device)
        return model

    @abstractmethod
    def init_trainer(self):
        raise NotImplementedError

    def init_field(self):
        field = Field(tokenize=self.tokenize,
                      init_token='<sos>',
                      eos_token='<eos>',
                      lower=True)
        return field

    def tokenize(self, x):
        token_collection = self.tokenizer.tokenize(x.lower())
        return token_collection

    @staticmethod
    def _len_sort_key(x):
        return len(x.src)


EXPERIMENT_CATALOG: Dict[str, Type[Experiment]] = {}
