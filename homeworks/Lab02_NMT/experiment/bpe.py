import youtokentome as yttm
from experiment import EXPERIMENT_CATALOG
from experiment.base import Experiment
from train_stage import BPEStage, ComposeStage
from utils import add_to_catalog


@add_to_catalog('bpe', EXPERIMENT_CATALOG)
class Baseline(Experiment):
    def __init__(self, config, device):
        self.src_bpe = yttm.BPE(config['src_bpe'])
        self.trg_bpe = yttm.BPE(config['trg_bpe'])
        super().__init__(config, device)

    def init_trainer(self):
        pad_idx = self.target.vocab.stoi['<pad>']
        main_stage = BPEStage(self.model, 'main_stage', self.config['train'], pad_idx, self.trg_bpe)
        sgd_train_stage = BPEStage(self.model, 'sgd_stage', self.config['sgd_train'], pad_idx, self.trg_bpe)
        stage = ComposeStage([main_stage, sgd_train_stage])
        return stage

    def read_data(self):
        self.source.tokenize = self.src_tokenize
        self.source.lower = False
        self.target.tokenize = self.trg_tokenize
        self.target.lower = False
        return super(Baseline, self).read_data()

    def src_tokenize(self, x):
        return self.src_bpe.encode(x.lower())

    def trg_tokenize(self, x):
        return self.trg_bpe.encode(x.lower())

    def decode_translation(self, idx_text):
        bpe_text = super(Baseline, self).decode_translation(idx_text)
        text = self.trg_bpe.decode(bpe_text)[0]
        text = self.tokenize(text)
        return text