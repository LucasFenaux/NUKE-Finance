import torch
import torch.nn as nn


class LSTMLoss(nn.Module):
    def __init__(self, alpha: float = 1., beta: float = 1.):
        super(LSTMLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.loss = nn.MSELoss()
        self.relu = nn.ReLU()

    def compute_mask(self, x):
        mask = torch.where(x == -1.0, 0, 1)
        return mask

    def forward(self, output, x, *args, **kwargs):
        mask = self.compute_mask(x[:, 1:])
        masked_output = mask * output
        masked_x = mask * x[:, 1:]
        loss = self.loss(masked_output, masked_x) * self.alpha

        x_diff = x[:, 1:] - x[:, :-1]
        output_diff = output - x[:, :-1]

        penalty = self.relu(-x_diff * output_diff * mask)

        loss += penalty * self.beta
        return loss
