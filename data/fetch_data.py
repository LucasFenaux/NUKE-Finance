import csv
import queue

import numpy as np
import pandas as pd
import yfinance as yf
import multiprocessing as mp
import os
import pickle
import yfinance.shared as shared
from typing import Union

save_dir = "data/values/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# when downloading a yfinance ticker using a start-end, one can make the end time a day later to capture the current day


def download_data_worker(tickers: list, period: Union[str, tuple], interval: str, q: mp.Queue):

    available_tickers = []
    multi_tickers = ""

    for ticker in tickers:
        available_tickers.append(ticker)
        multi_tickers += ticker + " "

    try:
        if type(period) == str:
            data = yf.download(multi_tickers, period=period, interval=interval, group_by='ticker')

        elif type(period) == tuple:
            assert len(period) == 2
            data = yf.download(multi_tickers, start=period[0], end=period[1], interval=interval, group_by='ticker')

        else:
            print(f"Period variable is neither a string nor a tuple but a type {type(period)}")
            raise TypeError

        errors = list(shared._ERRORS.keys())
        available_tickers = list(set(available_tickers) - set(errors))
        print("Errors " + str(errors))
        print(f"Available tickers: {available_tickers}")
        q.put((data, available_tickers))
        return

    except ConnectionError:
        print(f"Something went wrong when fetching all the data")
        # so the process doesn't get stuck in an infinite loop
        q.put(None)
        return


def download_data(period: Union[str, tuple], exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"), interval: str = "1h",
                  num_workers: int = 16):
    """ Reads all the existing tickers on a stock exchange from a pre-downloaded list stored in data/tickers"""
    ticker_name_list = []
    for exchange in exchanges:
        if exchange in ("nyse", "nasdaq", "amex"):

            file_name = f"data/tickers/nasdaq_screener_{exchange}.csv"

            with open(file_name) as ticker_csv:
                ticker_reader = csv.reader(ticker_csv)
                next(ticker_reader, None)  # skip the headers

                for row in ticker_reader:
                    if row[0].strip() not in ticker_name_list:  # we only take one listing a given ticker
                        ticker_name_list.append(row[0].strip())

        elif exchange == "tsx":
            file_name = f"data/tickers/tsx_tickers.csv"

            with open(file_name) as ticker_csv:
                ticker_reader = csv.reader(ticker_csv)
                next(ticker_reader, None)  # skip the headers

                for row in ticker_reader:
                    if row[3].strip() not in ticker_name_list:  # we only take one listing a given ticker
                        ticker_name_list.append(row[3].strip())
                    if row[3].strip() + ".TO" not in ticker_name_list:
                        ticker_name_list.append(row[3].strip() + ".TO")

    # we first get them to see which tickers don't work anymore
    print(period)
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_workers)

    p = int(len(ticker_name_list)/num_workers)
    c = 0
    jobs = []

    for i in range(num_workers - 1):
        job = pool.apply_async(download_data_worker, (ticker_name_list[c:c+p], period, interval, q))
        jobs.append(job)
        c += p

    job = pool.apply_async(download_data_worker, (ticker_name_list[c:], period, interval, q))
    jobs.append(job)

    for job in jobs:
        job.get()

    tickers = None
    for i in range(num_workers):
        if tickers is None:
            _, tickers = q.get()
        else:
            _, t = q.get()
            tickers += t

    pool.close()
    pool.join()

    # we now get the data for real

    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_workers)

    p = int(len(tickers)/num_workers)
    c = 0
    jobs = []

    for i in range(num_workers - 1):
        job = pool.apply_async(download_data_worker, (tickers[c:c+p], period, interval, q))
        jobs.append(job)
        c += p

    job = pool.apply_async(download_data_worker, (tickers[c:], period, interval, q))
    jobs.append(job)

    for job in jobs:
        job.get()

    pool.close()
    pool.join()

    data = None

    for i in range(num_workers):
        try:
            d, _ = q.get(timeout=10)
        except queue.Empty:
            print("Getting from queue timed-out")
            return None
        ## We don't care about localizing anymore since treat every window as its own object
        # d = pd.to_datetime(d, utc=True)
        # d.index = pd.to_datetime(d.index, infer_datetime_format=True, utc=True)
        # d.index.tz_convert("UTC")
        # try:
        #     d.index = d.index.tz_localize('UTC')
        #     d.index = d.index.tz_convert('UTC')
        # except:
        #     # d.index = d.index.tz_convert('UTC')
        #     pass

        # d.index = d.index.tz_convert('UTC')
        # d = d.asfreq(interval)

        # we now do the cleanup and generate the windows here
        # print(d)
        # d.dropna(axis=1, how='any', inplace=True)
        # print(d)
        if data is None:
            data = d
        else:
            data = pd.concat([data, d], axis=1)

    # we save to pickle instead of csv to save the indexing structure properly
    # data.to_pickle(save_dir + f"{exchanges}_{period}_{interval}.pkl")
    with open(save_dir + f"{exchanges}_{period}_{interval}.pkl", "wb") as f:
        pickle.dump(data, f)

    return data


def cleanup_ticker_data(data: pd.DataFrame, input_dim: int = 5):
    # in this case, we do not care about the adjusted close
    if input_dim < 6:
        data = data.drop(columns='Adj Close')
    if input_dim < 5:
        data = data.drop(columns='Volume')

    return data.dropna(axis=0, how='any')


def download_and_clean_data_worker(tickers: list, period: Union[str, tuple], interval: str, input_dim: int,
                                   q: mp.Queue):
    for ticker in tickers:

        try:
            if type(period) == str:
                data = yf.download(ticker, period=period, interval=interval, group_by='ticker')

            elif type(period) == tuple:
                assert len(period) == 2
                data = yf.download(ticker, start=period[0], end=period[1], interval=interval, group_by='ticker')

            else:
                print(f"Period variable is neither a string nor a tuple but a type {type(period)}")
                raise TypeError

            errors = list(shared._ERRORS.keys())
            if ticker not in errors:
                # we now cleanup the data
                data = cleanup_ticker_data(data, input_dim=input_dim)
                q.put((data, ticker))
            else:
                print("Errors " + str(errors))
        except ConnectionError:
            print(f"Something went wrong when fetching {ticker}")
            # so the process doesn't get stuck in an infinite loop
            # q.put(None)
            # return


def download_and_clean_data(period: Union[str, tuple], exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"),
                            interval: str = "1h", num_workers: int = 16, input_dim: int = 5):
    """ Reads all the existing tickers on a stock exchange from a pre-downloaded list stored in data/tickers"""
    ticker_name_list = []
    for exchange in exchanges:
        if exchange in ("nyse", "nasdaq", "amex"):

            file_name = f"data/tickers/nasdaq_screener_{exchange}.csv"

            with open(file_name) as ticker_csv:
                ticker_reader = csv.reader(ticker_csv)
                next(ticker_reader, None)  # skip the headers

                for row in ticker_reader:
                    if row[0].strip() not in ticker_name_list:  # we only take one listing a given ticker
                        ticker_name_list.append(row[0].strip())

        elif exchange == "tsx":
            file_name = f"data/tickers/tsx_tickers.csv"

            with open(file_name) as ticker_csv:
                ticker_reader = csv.reader(ticker_csv)
                next(ticker_reader, None)  # skip the headers

                for row in ticker_reader:
                    if row[3].strip() not in ticker_name_list:  # we only take one listing a given ticker
                        ticker_name_list.append(row[3].strip())
                    if row[3].strip() + ".TO" not in ticker_name_list:
                        ticker_name_list.append(row[3].strip() + ".TO")

    # we first get them to see which tickers don't work anymore
    print(period)

    # we now get the data for real
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_workers)

    p = int(len(ticker_name_list)/num_workers)
    c = 0
    jobs = []

    for i in range(num_workers - 1):
        job = pool.apply_async(download_and_clean_data_worker, (ticker_name_list[c:c+p], period, interval, input_dim,
                                                                q))
        jobs.append(job)
        c += p

    job = pool.apply_async(download_and_clean_data_worker, (ticker_name_list[c:], period, interval, input_dim, q))
    jobs.append(job)

    for job in jobs:
        job.get()

    pool.close()
    pool.join()

    data = []
    tickers = []

    for i in range(len(ticker_name_list)):
        try:
            d, ticker = q.get(timeout=10)
        except queue.Empty:
            print("Getting from queue timed-out")
            print("We can now stop fetching")
            break

        data.append(d)
        tickers.append(ticker)
    print(len(data))
    # we save to pickle instead of csv to save the indexing structure properly
    # data.to_pickle(save_dir + f"{exchanges}_{period}_{interval}.pkl")
    with open(save_dir + f"{exchanges}_{period}_{interval}_{input_dim}.pkl", "wb") as f:
        pickle.dump((data, tickers), f)

    return data, tickers


def load_clean_data(period: Union[str, tuple], exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"), interval: str = "1h",
              num_workers: int = 16, overwrite: bool = True, input_dim: int = 5):
    """ Loads saved stock market data, if it isn't saved, it downloads it
         Period can either be a string (representing the last amount of hours/days/weeks/months/years etc...
         or it can be a tuple of the form (starting_date, end_date) where if end_date is None then it defaults to
         (starting_date, now)
     """
    try:
        # data = pd.read_pickle(save_dir + f"{exchanges}_{period}_{interval}.pkl")
        if type(period) == tuple:

            if len(period) == 1:
                period = (period[0], None)

            if period[1] is None and overwrite:
                # we only have a starting date and the current date is now so we re-download the data
                print(f"overwrite is set to True with the second period None, hence we re-download the data")
                print(f"It will now be downloaded using {num_workers} workers")
                data, tickers = download_and_clean_data(period=period, exchanges=exchanges, interval=interval,
                                                        num_workers=num_workers, input_dim=input_dim)

            else:

                with open(save_dir + f"{exchanges}_{period}_{interval}_{input_dim}.pkl", "rb") as f:
                    data, tickers = pickle.load(f)

        else:

            with open(save_dir + f"{exchanges}_{period}_{interval}_{input_dim}.pkl", "rb") as f:
                data, tickers = pickle.load(f)

    except FileNotFoundError:

        print(f"Data for exchanges {exchanges} with period {period} and interval {interval} was not found")
        print(f"It will now be downloaded using {num_workers} workers")
        data, tickers = download_and_clean_data(period=period, exchanges=exchanges, interval=interval,
                                                num_workers=num_workers, input_dim=input_dim)

    return data, tickers


def load_data(period: Union[str, tuple], exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"), interval: str = "1h",
              num_workers: int = 16, overwrite: bool = True):
    """ Loads saved stock market data, if it isn't saved, it downloads it
        Period can either be a string (representing the last amount of hours/days/weeks/months/years etc...
        or it can be a tuple of the form (starting_date, end_date) where if end_date is None then it defaults to
        (starting_date, now)
    """
    try:
        # data = pd.read_pickle(save_dir + f"{exchanges}_{period}_{interval}.pkl")
        if type(period) == tuple:

            if len(period) == 1:
                period = (period[0], None)

            if period[1] is None and overwrite:
                # we only have a starting date and the current date is now so we re-download the data
                print(f"overwrite is set to True with the second period None, hence we re-download the data")
                print(f"It will now be downloaded using {num_workers} workers")
                data = download_data(period=period, exchanges=exchanges, interval=interval, num_workers=num_workers)

            else:

                with open(save_dir + f"{exchanges}_{period}_{interval}.pkl", "rb") as f:
                    data = pickle.load(f)

        else:

            with open(save_dir + f"{exchanges}_{period}_{interval}.pkl", "rb") as f:
                data = pickle.load(f)

    except FileNotFoundError:

        print(f"Data for exchanges {exchanges} with period {period} and interval {interval} was not found")
        print(f"It will now be downloaded using {num_workers} workers")
        data = download_data(period=period, exchanges=exchanges, interval=interval, num_workers=num_workers)

    number_of_features = 6
    tickers = data.keys().get_level_values(0).unique().values
    s = data.shape
    # for now we drop the volume and adjusted close price
    for l, ticker in enumerate(tickers):

        # sometimes some rows have duplicate indices, I do not know how to fix it without it taking forever and it's
        # only been like 5 tickers out of 7500 so we just drop them

        try:
            assert data[ticker].shape == (s[0], number_of_features)

        except AssertionError:

            print("Dropping ticker:")
            print(l, ticker, data[ticker].shape)
            data.drop(ticker, axis=1, inplace=True)

    tickers = data.keys().get_level_values(0).unique().values
    print(f"There are {len(tickers)} tickers in the available training data")

    # only keep the times within market open
    # we're working in UTC, we keep into account the tsx stocks closing time
    if interval != "1d" and interval != "1wk":
        data = data.between_time(start_time="13:29", end_time="21:01")

    s = data.shape

    # reshape it into a numpy array of the form (ticker, timestamp, column attribute)
    # where the column attributes are the features
    data = data.fillna(-1.0).values.reshape((s[0], len(tickers), number_of_features))
    data = data.transpose(1, 0, 2)
    # we drop the last two features (Adj Close and Volume)
    data = data[:, :, :-2]
    return data, tickers


# if __name__ == '__main__':
#     # load_data(period=("2022-04-18", None), interval="1h", num_workers=16)
#     data = load_data(period=("2022-04-13", "2022-04-21"), interval="1h", num_workers=16)
#     print(data[0].shape, data[1])
