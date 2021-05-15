from typing import Dict, Type
from torch.nn import Module


class BaseModel(Module):
    def __init__(self, input_dim, output_dim, device, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kwargs = kwargs
        self.device = device


NN_CATALOG: Dict[str, Type[BaseModel]] = {}
