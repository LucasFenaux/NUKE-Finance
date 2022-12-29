import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm
from trainer.logger import SmoothedValue
from trainer.trainer import Trainer
import logging
logger = logging.getLogger("result_logger")


def update_target_model(online_net, target_net):
    # Target <- Net
    target_net.load_state_dict(online_net.state_dict())


class RlQTrainer(Trainer):

    def __init__(self, online_model: nn.Module, target_model: nn.Module):
        super().__init__(online_model)
        self.online_model = online_model
        self.target_model = target_model
        update_target_model(self.online_model, target_model)

    def train(self, epochs: int, data_loader: data.DataLoader, optimizer=None, scheduler=None, **kwargs):
        pass

    def test(self, data_loader: data.DataLoader):
        pass

