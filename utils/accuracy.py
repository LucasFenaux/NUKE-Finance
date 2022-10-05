import torch


def directional_accuracy(output, x):
    x_diff = torch.sign(x[:, -2:-1] - x[:, -1:])
    output_diff = torch.sign(x[:, -2:-1] - output[:, -1:])
    diff = x_diff == output_diff
    count = torch.sum(diff)
    total = 1.
    for size in diff.size():
        total *= size
    return count/total
