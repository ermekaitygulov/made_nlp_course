from collections import defaultdict

from experiment import EXPERIMENT_CATALOG
from experiment.base import Experiment
from train_stage import MainStage, ComposeStage, PretrainStage
from utils import add_to_catalog


@add_to_catalog('scst', EXPERIMENT_CATALOG)
class SelfCriticalSeqTrain(Experiment):
    def __init__(self, config, device):
        super(SelfCriticalSeqTrain, self).__init__(config, device)
        self.translator_dict = self.init_translator()

    def init_trainer(self):
        pad_idx = self.target.vocab.stoi['<pad>']
        pretrain_stage = PretrainStage(self.model, 'pretrain_stage', self.config['pretrain'], pad_idx)
        main_stage = MainStage(self.model, 'main_stage', self.config['train'], pad_idx)
        sgd_train_stage = MainStage(self.model, 'sgd_stage', self.config['sgd_train'], pad_idx)
        stage = ComposeStage([pretrain_stage,  main_stage, sgd_train_stage])
        return stage

    def init_translator(self):
        train_dataset = self.task.train.dataset
        translator = defaultdict(set)
        for exm in train_dataset.examples:
            translator[exm.src].add(exm.trg)
        return dict(translator)