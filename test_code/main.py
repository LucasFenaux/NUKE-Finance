from test_dataset import get_dataloaders
from models.transformer import TransformerModel
from tqdm import tqdm
import torch
import torch.nn as nn
from collections import deque
from torch.nn import Transformer


def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def main():
    length = 10
    batch_size = 1000
    end = 100000
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = get_dataloaders(batch_size=batch_size, length=length, end=end)

    # model = TransformerModel(ntoken=length, ninp=length, nhead=1, nhid=64, nlayers=3).to(device)
    model = Transformer(d_model=1, nhead=1, batch_first=True).to(device)
    optimizer = torch.optim.Adam(lr=1e-3, params=model.parameters())

    loss = nn.MSELoss()
    average_loss = SmoothedValue(fmt='{global_avg:.3e}')

    for e in range(epochs):

        with tqdm(train_loader, desc=f"Epoch 1/{epochs} | ", disable=False) as pbar:
            for x, y in pbar:
        # for x, y in train_loader:
                x, y = x.to(device), y.to(device)

                p = model(src=x, tgt=y)

                l = loss(p, y)

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                average_loss.update(l.detach().cpu().item())

                pbar.set_description(f"E {e+1}/{epochs} | Loss: {average_loss}")

    print("Testing")

    test_loss = SmoothedValue(fmt='{global_avg:.3e}')

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)

        p = model(x, y)

        next = y[:, :, -1]
        predicted_next = p[:, :, -1]

        for i in range(next.size()[0]):
            print(x[i].squeeze(), y[i].squeeze(), p[i].squeeze())
        l = loss(p, y)

        test_loss.update(l.detach().cpu().item())

    print(f"Test Loss: {test_loss}")


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size: int = None, fmt: str = '{global_avg:.3f}'):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value: float, n: int = 1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def update_list(self, value_list):
        for value in value_list:
            self.deque.append(value)
            self.total += value
        self.count += len(value_list)

    def reset(self):
        self.deque = deque(maxlen=self.deque.maxlen)
        self.count = 0
        self.total = 0.0

    @property
    def median(self) -> float:
        try:
            d = torch.tensor(list(self.deque))
            return d.median().item()
        except Exception:
            return 0.0

    @property
    def avg(self) -> float:
        try:
            d = torch.tensor(list(self.deque), dtype=torch.float32)
            if len(d) == 0:
                return 0.0
            return d.mean().item()
        except Exception:
            return 0.0

    @property
    def global_avg(self) -> float:
        try:
            return self.total / self.count
        except Exception:
            return 0.0

    @property
    def max(self) -> float:
        try:
            return max(self.deque)
        except Exception:
            return 0.0

    @property
    def min(self) -> float:
        try:
            return min(self.deque)
        except Exception:
            return 0.0

    @property
    def value(self) -> float:
        try:
            return self.deque[-1]
        except Exception:
            return 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            min=self.min,
            max=self.max,
            value=self.value)

    def __format__(self, format_spec: str) -> str:
        return self.__str__()

    def __repr__(self) -> str:
        return self.__str__()



if __name__ == '__main__':
    main()