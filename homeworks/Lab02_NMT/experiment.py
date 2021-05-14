import time
from abc import ABC, abstractmethod

import torch
import wandb
from nltk import WordPunctTokenizer
from nltk.translate.bleu_score import corpus_bleu
from torchtext.data import Field, TabularDataset
from tqdm import tqdm

from utils import Task, get_text
from my_network import CATALOG


class Experiment(ABC):
    def __init__(self, config):
        self.config = config
        self.tokenizer = WordPunctTokenizer()

        self.task = Task(*self.read_data())
        self.model = self.init_model()
        self.trainer = self.init_trainer()
        self.time = None

    def train(self):
        start_time = time.time()
        train_iterator, val_iterator, _ = self.task
        self.trainer.train(self.model, train_iterator, val_iterator)
        end_time = time.time()
        self.time = end_time - start_time

    def test(self):
        _, _, test_iterator = self.task
        vocab = test_iterator.dataset.fields['trg']

        original_text = []
        generated_text = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(test_iterator)):
                src = batch.src
                trg = batch.trg

                output = self.model(src, trg)  # turn off teacher forcing

                # trg = [trg sent len, batch size]
                # output = [trg sent len, batch size, output dim]
                output = output.argmax(dim=-1)

                original_text.extend([get_text(x, vocab) for x in trg.cpu().numpy().T])
                generated_text.extend([get_text(x, vocab) for x in output[1:].detach().cpu().numpy().T])

        bleu = corpus_bleu([[text] for text in original_text], generated_text) * 100
        table = wandb.Table(data=[[self.time, bleu]], columns=["time_spent", "bleu"])
        wandb.log({"my_custom_plot_id": wandb.plot.scatter(
            table,
            "time_spent",
            "bleu", title="BLEU score")
        })

    def read_data(self):
        data_config = self.config['data']
        source = Field(tokenize=self.tokenize,
                       init_token='<sos>',
                       eos_token='<eos>',
                       lower=True)

        target = Field(tokenize=self.tokenize,
                       init_token='<sos>',
                       eos_token='<eos>',
                       lower=True)

        dataset = TabularDataset(
            path=data_config['path'],
            format='tsv',
            fields=[('trg', target), ('src', source)]
        )
        split_ratio = [data_config[f'{split}_size'] for split in ['train', 'val', 'test']]
        train_data, valid_data, test_data = dataset.split(split_ratio=split_ratio)
        source.build_vocab(train_data, min_freq=data_config['word_min_freq'])
        target.build_vocab(train_data, min_freq=data_config['word_min_freq'])
        return train_data, valid_data, test_data

    def init_model(self):
        model_config = self.config['model']
        model_class = CATALOG[model_config['name']]
        model = model_class(**model_config['params'])
        return model

    @abstractmethod
    def init_trainer(self):
        raise NotImplementedError

    def tokenize(self, x):
        token_collection = self.tokenizer.tokenize(x.lower())
        return token_collection



