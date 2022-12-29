from data.fetch_data import load_data, load_clean_data
from datetime import datetime
import datetime as DT
from torch.utils.data import TensorDataset, DataLoader
import torch
from numpy.lib.stride_tricks import sliding_window_view
import numpy as np
from utils.normalization import normalization
import time
from datetime import timedelta

download_num_workers = 16
train_num_workers = 8

keys = {
    "week": ("1w", "1h"),
    "month": ("1mo", "1d"),
    "year": ("1y", "1wk")
}

time_steps_ref = {
    ("1y", "1wk"): (1000, DT.timedelta(weeks=52)),
    ("1mo", "1d"): (1000, DT.timedelta(weeks=4)),
    ("1w", "1h"): (1000, DT.timedelta(days=7))
}

max_increments = {
    "week": 100,  # can't get 1h data further back than 730 days with yfinance
    "month": 10000000,
    "year": 10000000
}

current_day = datetime.today()

train_test_ratio = 0.8


def reshape_sliding_windows_into_batch(x):
    samples = []
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            samples.append(x[i, j].transpose(1, 0))

    return np.array(samples)


def sliding_window_view_wrapper(arr: np.array, window_shape: int, axis: int = 0):
    if arr.shape[0] < window_shape:
        return
    return sliding_window_view(arr, window_shape=window_shape, axis=axis).transpose((0, 2, 1))


def get_un_indexed_data(num_increments: int = 1, sequence_length: int = 100, data_type: str = "week",
                        exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"), batch_size: int = 128,
                        overwrite: bool = True, input_dim: int = 5, use_cache: bool = False):
    start_time = time.monotonic()
    key = keys.get(data_type, None)

    if key is None:
        print(f"{data_type} data type is not supported; please use a key in {keys.keys()}")
        raise NotImplementedError

    max_seq_legnth, delta = time_steps_ref[key]

    assert sequence_length <= max_seq_legnth
    assert num_increments <= max_increments[data_type]

    start_day = (current_day - num_increments * delta).strftime("%Y-%m-%d")

    data, tickers = process_and_clean_horizon(period_start=start_day, period_end=None, exchanges=exchanges,
                                              interval=key[1], overwrite=overwrite, input_dim=input_dim,
                                              use_cache=use_cache)
    print(data[0].columns.values)
    train_dataset = []
    test_dataset = []
    # we first need to take the data array and generate the windows one by one
    for ticker_data in data:
        num_samples = ticker_data.shape[0]
        index = int(num_samples * train_test_ratio)
        train_ticker_windows = sliding_window_view_wrapper(ticker_data[:index].to_numpy(), window_shape=sequence_length,
                                                           axis=0)
        test_ticker_windows = sliding_window_view_wrapper(ticker_data[index:].to_numpy(), window_shape=sequence_length,
                                                          axis=0)
        if train_ticker_windows is not None:
            train_dataset.append(train_ticker_windows)
        if test_ticker_windows is not None:
            test_dataset.append(test_ticker_windows)

    train_dataset = np.concatenate(train_dataset, axis=0)
    test_dataset = np.concatenate(test_dataset, axis=0)

    print(train_dataset.shape)
    print(test_dataset.shape)

    assert np.count_nonzero(np.isnan(train_dataset)) == 0 and np.count_nonzero(np.isnan(test_dataset)) == 0

    train_tensor = torch.Tensor(train_dataset)
    # we take the pre-normalized value
    train_values = train_tensor[:, -2, :]
    train_dataset = TensorDataset(normalization(train_tensor, use_multiprocessing=True,
                                                num_workers=download_num_workers), train_values)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=train_num_workers)

    test_tensor = torch.Tensor(test_dataset)
    test_values = test_tensor[:, -2, :]
    test_dataset = TensorDataset(normalization(test_tensor, use_multiprocessing=True,
                                               num_workers=download_num_workers), test_values)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=train_num_workers)
    end_time = time.monotonic()
    print(f"Preprocessing took: {timedelta(seconds=end_time - start_time)}")
    return train_dataloader, test_dataloader


def get_week_data(num_weeks: int = 1, exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"), batch_size: int = 128,
                  keep_masked: bool = True, overwrite: bool = True):
    steps, delta = time_steps_ref[("1w", "1h")]
    start_day = (current_day - num_weeks * delta).strftime("%Y-%m-%d")

    data, tickers = process_horizon(period_start=start_day, period_end=None, exchanges=exchanges, interval="1h",
                                    overwrite=overwrite)

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
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=train_num_workers)

    test_dataset = TensorDataset(torch.Tensor(windowed_test_data))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=train_num_workers)

    return train_dataloader, test_dataloader


def process_and_clean_horizon(period_start: str, period_end: str = None, exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"),
                    interval: str = "1h", overwrite: bool = True, input_dim: int = 5, use_cache: bool = False):
    data, tickers = load_clean_data(period=(period_start, period_end), exchanges=exchanges, interval=interval,
                                    overwrite=overwrite, num_workers=download_num_workers, input_dim=input_dim,
                                    use_cache=use_cache)

    return data, tickers


def process_horizon(period_start: str, period_end: str = None, exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"),
                    interval: str = "1h", overwrite: bool = True):

    data, tickers = load_data(period=(period_start, period_end), exchanges=exchanges, interval=interval,
                              overwrite=overwrite)

    print(data.shape)
    return data, tickers


if __name__ == '__main__':
    # process_horizon(period="1w", interval="1h")
    train_loader, test_loader = get_week_data(num_weeks=6)
