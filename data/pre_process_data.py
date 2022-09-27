from data.fetch_data import load_data
from datetime import datetime
import datetime as DT
from torch.utils.data import TensorDataset, DataLoader
import torch
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np

num_workers = 16

time_steps_ref = {
    ("1y", "1wk"): (52, DT.timedelta(weeks=52)),
    ("1mo", "1d"): (28, DT.timedelta(weeks=4)),
    ("1w", "1h"): (40, DT.timedelta(days=7))
}

current_day = datetime.today()

train_test_ratio = 0.8


def reshape_sliding_windows_into_batch(x):
    samples = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            samples.append(x[i, j].transpose(1, 0))

    return np.array(samples)


def get_week_data(num_weeks: int = 1, exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"), batch_size: int = 128,
                  keep_masked: bool = True):
    steps, delta = time_steps_ref[("1w", "1h")]
    start_day = (current_day - num_weeks * delta).strftime("%Y-%m-%d")

    data, tickers = process_horizon(period_start=start_day, period_end=None, exchanges=exchanges, interval="1h")

    num_steps = data.shape[1]
    index = int(num_steps * train_test_ratio)

    train_data = data[:, :index]
    windowed_train_data = reshape_sliding_windows_into_batch(sliding_window_view(train_data, window_shape=steps,
                                                                                 axis=1))
    print(windowed_train_data.shape)

    test_data = data[:, index:]
    windowed_test_data = reshape_sliding_windows_into_batch(sliding_window_view(test_data, window_shape=steps, axis=1))
    print(windowed_test_data.shape)
    if not keep_masked:
        print("Removing masked values")
        to_remove = []
        for i in range(windowed_train_data.shape[0]):
            if -1.0 in windowed_train_data[i]:
                to_remove.append(i)
        windowed_train_data = np.delete(windowed_train_data, to_remove, axis=0)
        print(windowed_train_data.shape)

        to_remove = []
        for j in range(windowed_test_data.shape[0]):
            if -1.0 in windowed_test_data[j]:
                to_remove.append(j)
        windowed_test_data = np.delete(windowed_test_data, to_remove, axis=0)
        print(windowed_test_data.shape)

    train_dataset = TensorDataset(torch.Tensor(windowed_train_data))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)

    test_dataset = TensorDataset(torch.Tensor(windowed_test_data))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


def process_horizon(period_start: str, period_end: str = None, exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"),
                    interval: str = "1h"):

    data, tickers = load_data(period=(period_start, period_end), exchanges=exchanges, interval=interval,
                              overwrite=False)

    print(data.shape)
    return data, tickers


if __name__ == '__main__':
    # process_horizon(period="1w", interval="1h")
    train_loader, test_loader = get_week_data(num_weeks=6)
