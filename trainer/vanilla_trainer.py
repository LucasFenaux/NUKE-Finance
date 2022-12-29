import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data
from tqdm import tqdm
from utils.accuracy import directional_accuracy, directional_decision_rate
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
                for x, _ in pbar:
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
        x_positives = None
        x_negatives = None
        output_positives = None
        output_negatives = None
        precision = None
        recall = None   # also called tpr
        false_positive_rate = None
        for x, _ in data_loader:
            x = x.to(self.model.device)
            # sequence_length = x.size(1) - 1
            # tgt_mask = self.model.get_tgt_mask(x.size(0), sequence_length).to(self.model.device)
            #
            # output = self.model(x[:, :-1], x[:, 1:], tgt_mask)
            print(x.size())
            output = self.model(x[:, :-1])

            loss = self.model.loss(output, x)

            num_features = self.model.loss.feature_penalties.size(0)
            if feature_penalties is None:
                feature_penalties = []
                feature_accuracies = []
                x_positives = []
                x_negatives = []
                output_positives = []
                output_negatives = []
                precision = []
                recall = []  # also called tpr
                false_positive_rate = []
                for features in range(num_features):
                    feature_penalties.append(SmoothedValue())
                    feature_accuracies.append(SmoothedValue())
                    x_positives.append(0)
                    x_negatives.append(0)
                    output_positives.append(0)
                    output_negatives.append(0)
                    precision.append(SmoothedValue())
                    recall.append(SmoothedValue())
                    false_positive_rate.append(SmoothedValue())
            for i in range(num_features):
                feature_penalties[i].update(self.model.loss.feature_penalties[i].item(), n=x.size()[0])
                feature_accuracies[i].update(directional_accuracy(output[:, :, i], x[:, :, i]), n=x.size()[0])
                x_p, x_n, o_p, o_n, pres, rec, fpr = directional_decision_rate(output[:, :, i], x[:, :, i])
                x_positives[i] += x_p
                x_negatives[i] += x_n
                output_positives[i] += o_p
                output_negatives[i] += o_n
                precision[i].update(pres, n=x.size()[0])
                recall[i].update(rec, n=x.size()[0])
                false_positive_rate[i].update(fpr, n=x.size()[0])

            test_loss.update(self.model.loss.loss_value.item(), n=x.size()[0])
            test_penalty.update(self.model.loss.penalty_value.item(), n=x.size()[0])
            test_acc.update(directional_accuracy(output, x), n=x.size()[0])

        logger.info(f"Loss: {test_loss}")
        print(f"Loss: {test_loss}")
        logger.info(f"Guiding Loss: {test_penalty}")
        print(f"Input Positive Diff Count: {x_positives}")
        print(f"Input Negative Diff Count: {x_negatives}")
        print(f"Output Positive Diff Count: {output_positives}")
        print(f"Output Negative Diff Count: {output_negatives}")
        print(f"Guiding Loss: {test_penalty}")
        print(f"Test Accuracy: {test_acc}")
        print(f"Feature Penalties: {feature_penalties}")
        print(f"Feature Accuracies: {feature_accuracies}")
        print(f"Precision: {precision}")
        print(f"Recall/True Positive Rate: {recall}")
        print(f"False Positive Rate: {false_positive_rate}")

        return test_loss
