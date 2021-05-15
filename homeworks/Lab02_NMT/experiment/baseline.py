from experiment import EXPERIMENT_CATALOG
from experiment.base import Experiment
from train_stage import MainStage, PretrainStage, ComposeStage
from utils import add_to_catalog


@add_to_catalog('baseline', EXPERIMENT_CATALOG)
class Baseline(Experiment):
    def init_trainer(self):
        pad_idx = self.target.vocab.stoi['<pad>']
        stage = MainStage(self.model, 'main_stage', self.config['train'], pad_idx)
        return stage


@add_to_catalog('pretrain_baseline', EXPERIMENT_CATALOG)
class Baseline(Experiment):
    def init_trainer(self):
        pad_idx = self.target.vocab.stoi['<pad>']
        pretrain_stage = PretrainStage(self.model, 'pretrain_stage', self.config['pretrain'], pad_idx)
        main_stage = MainStage(self.model, 'main_stage', self.config['train'], pad_idx)
        stage = ComposeStage([pretrain_stage,  main_stage])
        return stage
