import torch


def directional_accuracy(output, x):
    x_diff = x[:, -2:-1] - x[:, -1:]
    output_diff = x[:, -2:-1] - output[:, -1:]
    count = torch.sum(x_diff == output_diff)
    total = 1
    for size in output.size():
        total *= size
    return count/total
