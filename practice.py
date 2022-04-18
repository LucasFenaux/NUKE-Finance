import csv

import numpy as np
import pandas as pd
import yfinance as yf
import multiprocessing as mp
import os
import pickle
import time
import yfinance.shared as shared

save_dir = "data/values/"
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def download_data_worker(tickers: list, period, interval, q):
    print(tickers)
    available_tickers = []
    multi_tickers = ""
    for ticker in tickers:
        available_tickers.append(ticker)
        multi_tickers += ticker + " "
    # i = 0
    # for ticker_name in tickers:
    #     try:
    #         data = yf.download(ticker_name, period=period, interval=interval)
    #         # if ticker_data is None:
    #         #     ticker_data = data
    #         # else:
    #         #     ticker_data = pd.concat([ticker_data, data])
    #         available_tickers.append(ticker_name)
    #         multi_tickers += ticker_name + " "
    #         i += 1
    #         print(str(i) + "/" + str(len(tickers)))
    #     except:
    #         print(f"Something went wrong when fetching {ticker_name}")
    #         i += 1
    #         print(str(i) + "/" + str(len(tickers)))

    try:
        data = yf.download(multi_tickers, period=period, interval=interval, group_by='ticker')
        import yfinance.shared as shared
        errors = list(shared._ERRORS.keys())
        available_tickers = list(set(available_tickers) - set(errors))
        print("Errors " + str(errors))
        print(f"Available tickers: {available_tickers}")
        q.put((data, available_tickers))
        return
    except:
        print(f"Something went wrong when fetching all the data")
        # so the process doesn't get stuck in an infinite loop
        q.put(None)
        return


def download_data(exchanges: tuple = ("nyse", "nasdaq", "amex"), period: str = "1d", interval: str = "1h",
                  num_workers: int = 20):
    """ Reads all the existing tickers on a stock exchange from a pre-downloaded list stored in data/tickers"""
    ticker_name_list = []
    for exchange in exchanges:
        file_name = f"data/tickers/nasdaq_screener_{exchange}.csv"
        with open(file_name) as ticker_csv:
            ticker_reader = csv.reader(ticker_csv)
            next(ticker_reader, None)  # skip the headers
            for row in ticker_reader:
                if row[0] not in ticker_name_list:  # we only take one listing a given ticker
                    ticker_name_list.append(row[0].strip())

    # we first get them to see which tickers don't work anymore

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
            d, t = q.get()
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

    data = None
    for i in range(num_workers):
        d, _ = q.get()
        if data is None:
            data = d
        else:
            data = pd.concat([data, d], axis=1)

    pool.close()
    pool.join()

    # we save to pickle instead of csv to save the indexing structure properly
    # data.to_pickle(save_dir + f"{exchanges}_{period}_{interval}.pkl")
    with open(save_dir + f"{exchanges}_{period}_{interval}.pkl", "wb") as f:
        pickle.dump((data, tickers), f)

    return data, tickers


def load_data(exchanges: tuple = ("nyse", "nasdaq", "amex"), period: str = "1d", interval: str = "1h",
              num_workers: int = 20):
    """ Loads saved stock market data, if it isn't saved, it downloads it"""
    try:
        # data = pd.read_pickle(save_dir + f"{exchanges}_{period}_{interval}.pkl")
        with open(save_dir + f"{exchanges}_{period}_{interval}.pkl", "rb") as f:
            data, tickers = pickle.load(f)
    except FileNotFoundError:
        print(f"Data for exchanges {exchanges} with period {period} and interval {interval} was not found")
        print(f"It will now be downloaded using {num_workers} workers")
        data, tickers = download_data(exchanges=exchanges, period=period, interval=interval, num_workers=num_workers)

    return data, tickers


if __name__ == '__main__':
    load_data(interval="90m", num_workers=20)
