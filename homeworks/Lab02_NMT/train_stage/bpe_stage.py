from nltk import WordPunctTokenizer

from train_stage.main_stage import MainStage


class BPEStage(MainStage):
    def __init__(self, model, stage_name, stage_config, pad_idx, trg_bpe):
        super(BPEStage, self).__init__(model, stage_name, stage_config, pad_idx)
        self.trg_bpe = trg_bpe
        self.tokenizer = WordPunctTokenizer()

    def gen_translate(self, batch):
        bpe_org, bpe_gen = super(BPEStage, self).gen_translate(batch)
        org = self.trg_bpe.decode(bpe_org)
        gen = self.trg_bpe.decode(bpe_gen)
        org = [self.tokenizer.tokenize(x.lower()) for x in org]
        gen = [self.tokenizer.tokenize(x.lower()) for x in gen]
        return org, gen
