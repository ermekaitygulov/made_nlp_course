from typing import Dict, Type

import torch
from torch.nn import Module


class BaseModel(Module):
    def __init__(self, input_dim, output_dim, device, pad_idx, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kwargs = kwargs
        self.device = device
        self.pad_idx = pad_idx

    def load(self, path):
        with open(path, "rb") as fp:
            state_dict = torch.load(fp, map_location='cpu')
            self.load_state_dict(state_dict)


NN_CATALOG: Dict[str, Type[BaseModel]] = {}
