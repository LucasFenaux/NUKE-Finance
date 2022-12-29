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


def directional_decision_rate(output, x):
    x_diff = torch.sign(x[:, -2:-1] - x[:, -1:])
    x_positives = torch.where(x_diff > 0, 1, 0)
    x_negatives = torch.where(x_diff < 0, 1, 0)

    output_diff = torch.sign(x[:, -2:-1] - output[:, -1:])
    output_positives = torch.where(output_diff > 0, 1, 0)
    output_negatives = torch.where(output_diff < 0, 1, 0)

    true_positives = torch.sum(x_positives * output_positives)
    false_positives = torch.sum(output_positives * x_negatives)
    true_negatives = torch.sum(x_negatives * output_negatives)
    false_negatives = torch.sum(output_negatives * x_positives)

    precision = true_positives/(true_positives + false_positives)
    recall = true_positives/(true_positives + false_negatives)  # also tpr

    fpr = false_positives/(false_positives+true_negatives)

    return torch.sum(x_positives).item(), torch.sum(x_negatives).item(), torch.sum(output_positives).item(), \
           torch.sum(output_negatives).item(), precision.item(), recall.item(), fpr.item()
