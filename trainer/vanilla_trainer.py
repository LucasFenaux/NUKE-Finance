from torch import nn
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm

from trainer.logger import SmoothedValue
from trainer.trainer import Trainer
import logging

logger = logging.getLogger("result_logger")


class VanillaTrainer(Trainer):

    def __init__(self, model: nn.Module):
        super().__init__(model)

    def train(self, epochs: int, data_loader: data.DataLoader, optimizer=None, scheduler=None, **kwargs):

        if optimizer is None:
            optimizer = Adam(lr=0.001, params=self.model.parameters())

        self.model.train()
        for e in range(epochs):
            train_loss = SmoothedValue()
            with tqdm(data_loader, desc=f"Epoch 1/{epochs}") as pbar:
                for x, in pbar:
                    x = x.to(self.model.device)
                    # output, _ = self.model(x[:, :-1])

                    output = self.model(x[:, :-1], x[:, :-1])

                    loss = self.model.loss(output, x)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    train_loss.update(loss.detach().cpu().numpy(), n=x.size()[0])
                    pbar.set_description(f"Epoch {e+1}/{epochs} Loss: {train_loss}")
            logger.info(f"Epoch {e + 1}/{epochs} Loss: {train_loss}")

    def test(self, data_loader: data.DataLoader):
        self.model.eval()
        test_loss = SmoothedValue()
        for x, in data_loader:
            x = x.to(self.model.device)

            output = self.model(x[:, :-1], x[:, 1:])

            loss = self.model.loss(output, x)

            test_loss.update(loss.detach().cpu().numpy(), n=x.size()[0])
        logger.info(f"Loss: {test_loss}")
        print(f"Loss: {test_loss}")
        return test_loss
