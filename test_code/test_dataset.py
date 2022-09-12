import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def get_dataloaders(batch_size: int = 32, length: int = 10, end: int = 10000, train_proportion: float = 0.8):
    train_starting_points = np.random.choice(end, int(end*train_proportion), replace=False)
    test_starting_points = np.delete(np.arange(end), train_starting_points)

    train_data = []
    train_targets = []
    for sp in train_starting_points:
        train_data.append(np.arange(start=sp, stop=sp+length))
        train_targets.append(np.arange(start=sp+1, stop=sp+length+1))

    # train_data = torch.FloatTensor(normalize(np.array(train_data))).unsqueeze(1)
    # train_targets = torch.FloatTensor(normalize(np.array(train_targets))).unsqueeze(1)

    train_data = (torch.FloatTensor(np.array(train_data)).unsqueeze(1)/end).view(-1, length, 1)
    train_targets = (torch.FloatTensor(np.array(train_targets)).unsqueeze(1)/end).view(-1, length, 1)

    train_dataset = TensorDataset(train_data, train_targets)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=1)

    test_data = []
    test_targets = []
    for sp in test_starting_points:
        test_data.append(np.arange(start=sp, stop=sp + length))
        test_targets.append(np.arange(start=sp+1, stop=sp+length+1))

    # test_data = torch.FloatTensor(normalize(np.array(test_data))).unsqueeze(1)
    # test_targets = torch.FloatTensor(normalize(np.array(test_targets))).unsqueeze(1)

    test_data = (torch.FloatTensor(np.array(test_data)).unsqueeze(1)/end).view(-1, length, 1)
    test_targets = (torch.FloatTensor(np.array(test_targets)).unsqueeze(1)/end).view(-1, length, 1)

    test_dataset = TensorDataset(test_data, test_targets)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=1)

    return train_loader, test_loader