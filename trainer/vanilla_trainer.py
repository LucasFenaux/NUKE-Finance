import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm
from utils.accuracy import directional_accuracy
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
            train_loss = SmoothedValue(fmt='{global_avg:.3e}')
            train_penalty = SmoothedValue(fmt='{global_avg:.3e}')
            train_acc = SmoothedValue(fmt='{global_avg:.3f}')
            with tqdm(data_loader, desc=f"Epoch 1/{epochs}") as pbar:
                for x, in pbar:
                    x = x.to(self.model.device)
                    assert torch.count_nonzero(torch.isnan(x)).item() == 0
                    # output, _ = self.model(x[:, :-1])
                    # sequence_length = x.size(1) - 1
                    # tgt_mask = self.model.get_tgt_mask(x.size(0), sequence_length).to(self.model.device)

                    # output = self.model(x[:, :-1], x[:, 1:], tgt_mask)
                    output = self.model(x[:, :-1])

                    loss = self.model.loss(output, x)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()

                    train_loss.update(self.model.loss.loss_value.detach().cpu().numpy(), n=x.size()[0])
                    train_penalty.update(self.model.loss.penalty_value.detach().cpu().numpy(), n=x.size()[0])
                    train_acc.update(directional_accuracy(output, x), n=x.size()[0])
                    pbar.set_description(f"Epoch {e+1}/{epochs} L: {train_loss} G: {train_penalty} Acc: {train_acc}")
            logger.info(f"Epoch {e + 1}/{epochs} Loss: {train_loss} Guide: {train_penalty}")

    def test(self, data_loader: data.DataLoader):
        self.model.eval()
        test_loss = SmoothedValue()
        test_penalty = SmoothedValue()
        test_acc = SmoothedValue()
        feature_penalties = None
        feature_accuracies = None
        for x, in data_loader:
            x = x.to(self.model.device)
            # sequence_length = x.size(1) - 1
            # tgt_mask = self.model.get_tgt_mask(x.size(0), sequence_length).to(self.model.device)
            #
            # output = self.model(x[:, :-1], x[:, 1:], tgt_mask)
            output = self.model(x[:, :-1])

            loss = self.model.loss(output, x)

            num_features = self.model.loss.feature_penalties.size(0)
            if feature_penalties is None:
                feature_penalties = []
                feature_accuracies = []
                for features in range(num_features):
                    feature_penalties.append(SmoothedValue())
                    feature_accuracies.append(SmoothedValue())
            for i in range(num_features):
                feature_penalties[i].update(self.model.loss.feature_penalties[i].item(), n=x.size()[0])
                feature_accuracies[i].update(directional_accuracy(output[:, :, i], x[:, :, i]), n=x.size()[0])

            test_loss.update(self.model.loss.loss_value.item(), n=x.size()[0])
            test_penalty.update(self.model.loss.penalty_value.item(), n=x.size()[0])
            test_acc.update(directional_accuracy(output, x), n=x.size()[0])

        logger.info(f"Loss: {test_loss}")
        print(f"Loss: {test_loss}")
        logger.info(f"Guiding Loss: {test_penalty}")
        print(f"Guiding Loss: {test_penalty}")
        print(f"Test Accuracy: {test_acc}")
        print(f"Feature Penalties: {feature_penalties}")
        print(f"Feature Accuracies: {feature_accuracies}")
        return test_loss
