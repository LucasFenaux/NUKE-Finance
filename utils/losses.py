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

    def compute_mask(self, x):
        mask = torch.where(x == -1.0, 0, 1)
        return mask

    def forward(self, output, x, *args, **kwargs):
        # mask = self.compute_mask(x[:, 1:])
        # masked_output = mask * output
        # masked_x = mask * x[:, 1:]
        # loss = self.loss(masked_output, masked_x) * self.alpha
        self.loss_value = self.loss(output, x[:, 1:]) * self.alpha
        # x_diff = x[:, 1:] - x[:, :-1]
        # output_diff = output - x[:, :-1]
        x_diff = x[:, -2:-1] - x[:, -1:]
        output_diff = x[:, -2:-1] - output[:, -1:]
        # print(x_diff.mean())
        # print(output_diff.mean())
        # print(x_diff)
        # print(output_diff)
        # penalty = self.relu(-x_diff * output_diff * mask)
        self.penalty_value = torch.mean(self.relu(-x_diff * output_diff)) * self.beta
        # self.penalty_value = torch.tensor(0.)
        return self.loss_value + self.penalty_value
