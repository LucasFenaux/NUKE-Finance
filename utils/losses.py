import torch
import torch.nn as nn


class LSTMLoss(nn.Module):
    def __init__(self, alpha: float = 1., beta: float = 1.):
        super(LSTMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss = nn.MSELoss()
        self.relu = nn.ReLU()
        self.loss_value = 0.
        self.penalty_value = 0.
        self.feature_penalties = torch.tensor([])

    def compute_mask(self, x):
        mask = torch.where(x == -1.0, 0, 1)
        return mask

    def forward(self, output, x, *args, **kwargs):
        self.loss_value = self.loss(output, x[:, 1:]) * self.alpha

        x_diff = x[:, -2:-1] - x[:, -1:]
        output_diff = x[:, -2:-1] - output[:, -1:]
        self.feature_penalties = torch.mean(self.relu(-x_diff * output_diff), dim=0).squeeze() * self.beta
        self.penalty_value = torch.mean(self.feature_penalties)
        return self.loss_value + self.penalty_value
