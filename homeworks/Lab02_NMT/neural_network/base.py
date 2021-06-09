from typing import Dict, Type

import torch
from torch.nn import Module


class BaseModel(Module):
    def __init__(self, input_dim, output_dim, device, trg_vocab, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kwargs = kwargs
        self.device = device
        self.trg_vocab = trg_vocab
        self.pad_idx = trg_vocab.stoi['<pad>']
        self.sos_idx = trg_vocab.stoi['<sos>']
        self.eos_idx = trg_vocab.stoi['<eos>']

    def load(self, path):
        with open(path, "rb") as fp:
            state_dict = torch.load(fp, map_location='cpu')
            self.load_state_dict(state_dict)

    def gen_translate(self, src, trg):
        output = self(src, trg, 1.)
        output = output.argmax(dim=-1)[1:].detach().cpu().numpy().T
        return output


NN_CATALOG: Dict[str, Type[BaseModel]] = {}
