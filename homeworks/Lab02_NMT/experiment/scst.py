from collections import defaultdict

from experiment import EXPERIMENT_CATALOG
from experiment.base import Experiment
from train_stage import MainStage, ComposeStage, PretrainStage, SCSTStage
from utils import add_to_catalog


@add_to_catalog('scst', EXPERIMENT_CATALOG)
class SelfCriticalSeqTrain(Experiment):
    def init_trainer(self):
        pad_idx = self.target.vocab.stoi['<pad>']
        translator_dict = self.get_translator()

        pretrain_stage = PretrainStage(self.model, 'pretrain_stage', self.config['pretrain'], pad_idx)
        main_stage = MainStage(self.model, 'main_stage', self.config['train'], pad_idx)
        scst_stage = SCSTStage(self.model, 'scst_stage', self.config['scst'], pad_idx,
                               translator_dict, self.source.vocab, self.target.vocab)
        stage = ComposeStage([pretrain_stage,  main_stage, scst_stage])
        return stage

    def get_translator(self):
        train_dataset = self.task.train.dataset
        translator = defaultdict(set)
        train_dataset.filter_examples(['src', 'trg'])
        for exm in train_dataset.examples:
            translator[' '.join(exm.src)].add(' '.join(exm.trg))
        translator = {src: [t.split() for t in trg] for src, trg in translator.items()}
        return translator
