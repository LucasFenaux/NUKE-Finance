from abc import abstractmethod

from torch import nn
from torch.utils import data


class Trainer:
    def __init__(self, model: nn.Module):
        self.model = model

    @abstractmethod
    def train(self, epochs: int, data_loader: data.DataLoader, **kwargs):
        raise NotImplementedError
