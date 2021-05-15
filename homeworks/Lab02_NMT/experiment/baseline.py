from experiment import EXPERIMENT_CATALOG
from experiment.base import Experiment
from train_stage.main_stage import MainStage
from utils import add_to_catalog


@add_to_catalog('baseline', EXPERIMENT_CATALOG)
class Baseline(Experiment):
    def init_trainer(self):
        pad_idx = self.target.vocab.stoi['<pad>']
        stage = MainStage(self.model, 'main_stage', self.config['train'], pad_idx)
        return stage
