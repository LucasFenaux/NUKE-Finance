import csv
from data.pre_process_data import current_day, keys, time_steps_ref, max_increments, train_num_workers, \
    train_test_ratio, download_num_workers
from data.fetch_data import save_dir, download_and_clean_data_worker
from typing import Union
import multiprocessing as mp
import queue
import pickle
import time
from datetime import timedelta, datetime
import os
import pandas as pd
import numpy as np
from collections import namedtuple, deque
import torch
from configs.trading_config import sequence_length
import datetime as dt
from utils.normalization import parallelized_normalization_trader
import torch.nn.functional as F
import pytz
import random

utc=pytz.UTC

year_converter = {'week': 50,
                  'month': 12,
                  'year': 1}

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask', 'liquidity', 'new_liquidity',
                                       'portfolio', 'new_portfolio'))

sell_lot_threshold = 500
sell_little_threshold = 100
buy_little_threshold = 100
buy_lot_threshold = 500
# actions: 0 sell a lot: 500$s worth,
# 1 sell a little: 100$s worth,
# 2 do nothing
# 3 buy a little: 100$s worth
# 4 buy a lot: 500$ worth


class TradingEnv:
    def __init__(self, starting_liquidity: float, num_actions: int, data, pred_data, indices, tickers, device):
        self.num_actions = num_actions

        self.starting_liquidity = starting_liquidity
        self.stock_portfolio = torch.zeros(len(tickers)).to(device)
        self.model_portfolio = torch.zeros(len(tickers)).to(device)  # percentage version
        # we store all of this to cpu
        self.device = device
        self.week_data, self.month_data, self.year_data = data
        self.pred_week, self.pred_month, self.pred_year = pred_data
        self.week_index, self.month_index, self.year_index = indices[0].to_pydatetime(),\
                                                             indices[1].to_pydatetime(), \
                                                             indices[2].to_pydatetime()
        self.tickers = tickers

        # we now find the earliest and latest we can go, we assume the indices are sorted
        # since we want our model to play for at least double the sequence_length
        self.last_day = self.week_index[-2*sequence_length]
        self.last_index = len(self.week_index)-2*sequence_length
        # we need at least a history of 100 days
        self.first_day = self.week_index[sequence_length]
        self.first_index = sequence_length

        self.current_week_index = None
        self.current_month_index = None
        self.current_year_index = None
        self.current_week_datetime = None
        self.current_valuations = None
        self.episode_length = None
        self.final_episode_index = None
        self.current_liquidity = starting_liquidity
        self.current_worth = starting_liquidity

    @torch.no_grad()
    def update_month_and_year_indices(self):
        self.current_week_datetime = self.week_index[self.current_week_index].replace(tzinfo=utc)

        # we find the corresponding month index
        i = 0
        while i < len(self.month_index) and self.month_index[i].replace(tzinfo=utc) <= self.current_week_datetime:
            i += 1
        self.current_month_index = i - 1

        # we find the corresponding year index
        j = 0
        while j < len(self.year_index) and self.year_index[j].replace(tzinfo=utc) <= self.current_week_datetime:
            j += 1
        self.current_year_index = j - 1

    @torch.no_grad()
    def reset(self):
        # we re-initialize the portfolio and the liquidity
        self.current_liquidity = self.starting_liquidity
        self.current_worth = self.starting_liquidity

        self.stock_portfolio = torch.zeros(len(self.tickers)).to(self.device)
        self.model_portfolio = torch.zeros(len(self.tickers)).to(self.device)  # percentage version

        # we then re-select a starting index and time
        self.current_week_index = torch.randint(low=self.first_index, high=self.last_index, size=(1,)).item()
        self.update_month_and_year_indices()

        self.current_valuations = torch.Tensor(self.week_data[self.current_week_index, :, 0]).to(self.device)
        zero = torch.zeros((1,), device=self.device)
        self.current_valuations = torch.where(self.current_valuations >= 0., self.current_valuations, zero)

        week_data = parallelized_normalization_trader(
            torch.cat([torch.Tensor(self.week_data[self.current_week_index-99:self.current_week_index]).to(self.device),
                      self.pred_week[self.current_week_index-sequence_length].unsqueeze(0).to(self.device)]).
                transpose(0, 1))[0]
        month_data = parallelized_normalization_trader(
            torch.cat([torch.Tensor(self.month_data[self.current_month_index-99:self.current_month_index]).to(self.device),
                      self.pred_month[self.current_month_index-sequence_length].unsqueeze(0).to(self.device)]).
                transpose(0, 1))[0]
        year_data = parallelized_normalization_trader(
            torch.cat([torch.Tensor(self.year_data[self.current_year_index-99:self.current_year_index]).to(self.device),
                      self.pred_year[self.current_year_index-sequence_length].unsqueeze(0).to(self.device)]).
                transpose(0, 1))[0]

        self.episode_length = torch.randint(low=2 * sequence_length - 1, high=len(self.week_index) - self.current_week_index - 1,
                                            size=(1,)).item()
        # print(f"New Episode Length: {self.episode_length}")
        self.final_episode_index = self.current_week_index + self.episode_length
        state = torch.cat([week_data, month_data, year_data], dim=2).to(self.device)
        return self.model_portfolio, self.current_liquidity, state, self.episode_length

    @torch.no_grad()
    def sell_stocks(self, actions: torch.Tensor):
            # we first sell, then buy
            # we first find big sell orders
            big_sell = (actions == 0).nonzero(as_tuple=True)[0]
            small_sell = (actions == 1).nonzero(as_tuple=True)[0]

            # we first check that there are stocks that can be sold
            to_sell_lot = (self.stock_portfolio[big_sell] > 0).nonzero(as_tuple=True)[0]
            to_sell_little = (self.stock_portfolio[small_sell] > 0).nonzero(as_tuple=True)[0]

            # we remove the ones that have no current saved valuations
            to_sell_lot = (self.current_valuations[to_sell_lot] > 0).nonzero(as_tuple=True)[0]
            to_sell_little = (self.current_valuations[to_sell_little] > 0).nonzero(as_tuple=True)[0]

            # we then sell and compute the gained value based on current valuations
            # current_values = self.current_valuations[to_sell_big]
            current_value_of_holdings = self.current_valuations * self.stock_portfolio

            to_sell_lot_less_than_threshold = (current_value_of_holdings[to_sell_lot] < sell_lot_threshold).nonzero(
                as_tuple=True)[0]
            # we first empty those positions and add up the profits into our current liquidity
            sell_lot_less_than_threshold_value = torch.sum(current_value_of_holdings[to_sell_lot_less_than_threshold])
            self.current_liquidity += sell_lot_less_than_threshold_value.item()
            # we then set the ownership to 0
            self.stock_portfolio[to_sell_lot_less_than_threshold] = 0

            # we do the whole process again for selling little
            to_sell_little_less_than_threshold = (
                        current_value_of_holdings[to_sell_little] < sell_little_threshold).nonzero(as_tuple=True)[0]
            to_sell_little_less_than_threshold_value = torch.sum(
                current_value_of_holdings[to_sell_little_less_than_threshold])
            self.current_liquidity += to_sell_little_less_than_threshold_value.item()
            # we then set the ownership to 0
            self.stock_portfolio[to_sell_little_less_than_threshold] = 0

            # we now sell what we have that is above the threshold and deduct the sold shares from the portfolio

            # first lot
            to_sell_lot_more_than_threshold = (current_value_of_holdings[to_sell_lot] >= sell_lot_threshold).nonzero(
                as_tuple=True)[0]
            # we now get how much sell_lot_threshold is w.r.t. to a single share
            to_sell_lot_share_ratio = sell_lot_threshold / self.current_valuations[to_sell_lot_more_than_threshold]
            # we then remove that many shares and add the gained value to the current liquidity
            self.stock_portfolio[to_sell_lot_more_than_threshold] -= to_sell_lot_share_ratio
            self.current_liquidity += sell_lot_threshold * len(to_sell_lot_more_than_threshold)

            # then little
            to_sell_little_more_than_threshold = (
                        current_value_of_holdings[to_sell_little] >= sell_little_threshold).nonzero(as_tuple=True)[0]
            # we now get how much sell_lot_threshold is w.r.t. to a single share
            to_sell_little_share_ratio = sell_little_threshold / self.current_valuations[to_sell_little_more_than_threshold]
            # we then remove that many shares and add the gained value to the current liquidity
            self.stock_portfolio[to_sell_little_more_than_threshold] -= to_sell_little_share_ratio
            self.current_liquidity += sell_little_threshold * len(to_sell_little_more_than_threshold)

    @torch.no_grad()
    def buy_stocks(self, actions: torch.Tensor):
        # we buy the stocks one by one until we run out of stocks to buy or we run out of liquidity
        # we first buy small
        with torch.no_grad():
            small_buy = (actions == 3).nonzero(as_tuple=True)[0]
            buy_little_ratio = buy_little_threshold / self.current_valuations[small_buy]
            # we buy at random
            idx = torch.randperm(len(small_buy))
            i = 0
            while i < len(small_buy) and self.current_liquidity >= buy_little_threshold:
                self.stock_portfolio[small_buy[idx[i]]] += buy_little_ratio[idx[i]]
                self.current_liquidity -= buy_little_threshold

            # we check if we still have enough money for the big buys
            if self.current_liquidity < buy_lot_threshold:
                return

            # otherwise we keep buying
            big_buy = (actions == 4).nonzero(as_tuple=True)[0]
            buy_lot_ratio = buy_lot_threshold / self.current_valuations[big_buy]
            # we buy at random
            idx = torch.randperm(len(big_buy))
            i = 0
            while i < len(big_buy) and self.current_liquidity >= buy_lot_threshold:
                self.stock_portfolio[big_buy[idx[i]]] += buy_lot_ratio[big_buy[idx[i]]]
                self.current_liquidity -= buy_lot_threshold

    @torch.no_grad()
    def step(self, actions: torch.Tensor):
        actions = actions.squeeze()
        current_value_of_holdings = torch.sum(self.current_valuations * self.stock_portfolio).item() + \
                                    self.current_liquidity
        self.sell_stocks(actions)
        self.buy_stocks(actions)
        self.update_model_portfolio()
        # we now take a step in the timeline
        self.current_week_index += 1
        if self.current_week_index > self.final_episode_index:
            done = True
        else:
            done = False
        # we then check if the month/year index needs to move depending on if the day changed
        self.update_month_and_year_indices()

        new_valuations = torch.Tensor(self.week_data[self.current_week_index, :, 0]).to(self.device)
        positive_new_valuations = (new_valuations >= 0.).nonzero(as_tuple=True)[0]
        self.current_valuations[positive_new_valuations] = new_valuations[positive_new_valuations]

        # self.current_valuations = torch.Tensor(self.week_data[self.current_week_index, :, 0]).to(self.device)

        week_data = parallelized_normalization_trader(
            torch.cat([torch.Tensor(self.week_data[self.current_week_index-99:self.current_week_index]).to(self.device),
                      self.pred_week[self.current_week_index-sequence_length].unsqueeze(0).to(self.device)]).
                transpose(0, 1))[0]
        month_data = parallelized_normalization_trader(
            torch.cat([torch.Tensor(self.month_data[self.current_month_index-99:self.current_month_index]).to(self.device),
                      self.pred_month[self.current_month_index-sequence_length].unsqueeze(0).to(self.device)]).
                transpose(0, 1))[0]
        year_data = parallelized_normalization_trader(
            torch.cat([torch.Tensor(self.year_data[self.current_year_index-99:self.current_year_index]).to(self.device),
                      self.pred_year[self.current_year_index-sequence_length].unsqueeze(0).to(self.device)]).
                transpose(0, 1))[0]

        next_state = torch.cat([week_data, month_data, year_data], dim=2).to(self.device)

        new_value_of_holdings = torch.sum(self.current_valuations * self.stock_portfolio).item() + \
                                    self.current_liquidity

        reward = new_value_of_holdings - current_value_of_holdings

        # torch.cuda.empty_cache()

        return self.model_portfolio, self.current_liquidity, next_state, reward, done

    def update_model_portfolio(self):
        # we update the model portfolio to reflect the stock portfolio with the relative percentage worth of each stock
        current_value_of_holdings = torch.sum(self.current_valuations * self.stock_portfolio).item() + \
                                    self.current_liquidity
        self.model_portfolio = self.stock_portfolio / current_value_of_holdings
        self.model_portfolio = F.normalize(self.model_portfolio, dim=0, p=1)

    def get_random_action(self):
        # we weight the action [0.1, 0.1, 0.6, 0.1, 0.1] so tha
        return torch.Tensor(np.random.choice(a=[0, 1, 2, 3, 4], size=len(self.tickers), p=[0.01, 0.05, 0.88, 0.05, 0.01]
                                             , replace=True)).to(self.device)


class ExperienceReplay:
    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.device = device

    def push(self, state, next_state, action, reward, mask, liquidity, new_liquidity, portfolio, new_portfolio):
        self.memory.append(Transition(state.to(self.device), next_state.to(self.device),
                                      action.to(torch.float).to(self.device), torch.Tensor([reward]).to(self.device),
                                      torch.Tensor([mask]).to(self.device), torch.Tensor([liquidity]).to(self.device),
                                      torch.Tensor([new_liquidity]).to(self.device), portfolio.to(self.device),
                                      new_portfolio.to(self.device)))

    def sample(self, batch_size):
        # batch_state, batch_next_state, batch_action, batch_reward, batch_mask, batch_liquidity, batch_new_liquidity, \
        # batch_portfolio, batch_new_portfolio = [], [], [], [], [], [], [], [], []
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def move_to_device(self):
        new_memory = deque(maxlen=self.capacity)
        for transition in self.memory:
            state = transition.state.to(self.device)
            next_state = transition.next_state.to(self.device)
            action = transition.action.to(self.device)
            reward = transition.reward.to(self.device)
            mask = transition.mask.to(self.device)
            liquidity = transition.liquidity.to(self.device)
            new_liquidity = transition.new_liquidity.to(self.device)
            portfolio = transition.portfolio.to(self.device)
            new_portfolio = transition.new_portfolio.to(self.device)

            new_memory.append(Transition(state, next_state, action, reward, mask, liquidity, new_liquidity, portfolio,
                                         new_portfolio))

        self.memory = new_memory

    def __len__(self):
        return len(self.memory)

    def reset(self):
        self.memory = deque(maxlen=self.capacity)


def read_tickers(index: str = 'S&P500'):
    ticker_name_list = []

    if index in ('S&P500'):
        file_name = f"data/tickers/{index}_tickers.csv"

        with open(file_name) as ticker_csv:
            ticker_reader = csv.reader(ticker_csv)
            next(ticker_reader, None)  # skip the headers

            for row in ticker_reader:
                if row[0].strip() not in ticker_name_list:  # we only take one listing a given ticker
                    ticker_name_list.append(row[0].strip())

    return ticker_name_list


def download_and_clean_data(period: Union[str, tuple], index: str = 'S&P500',
                            interval: str = "1h", num_workers: int = 16, input_dim: int = 5):
    ticker_name_list = read_tickers(index=index)
    # we first get them to see which tickers don't work anymore
    print(period)

    # we now get the data for real
    manager = mp.Manager()
    q = manager.Queue()
    pool = mp.Pool(num_workers)

    p = int(len(ticker_name_list) / num_workers)
    c = 0
    jobs = []

    for i in range(num_workers - 1):
        job = pool.apply_async(download_and_clean_data_worker, (ticker_name_list[c:c + p], period, interval, input_dim,
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
    with open(save_dir + f"{index}_{period}_{interval}_{input_dim}.pkl", "wb") as f:
        pickle.dump((data, tickers), f)

    return data, tickers


def load_data(period: Union[str, tuple], index: str = 'S&P500', interval: str = "1h",
              num_workers: int = 16, overwrite: bool = True, input_dim: int = 5, use_cache: bool = False):
    if use_cache:
        if os.path.exists(os.path.join(save_dir, f"{index}_{interval}_{input_dim}_cache.pkl")):
            with open(save_dir + f"{index}_{interval}_{input_dim}_cache.pkl", "rb") as f:
                data, tickers = pickle.load(f)
                return data, tickers
    try:
        # data = pd.read_pickle(save_dir + f"{exchanges}_{period}_{interval}.pkl")
        if type(period) == tuple:

            if len(period) == 1:
                period = (period[0], None)

            if period[1] is None and overwrite:
                # we only have a starting date and the current date is now so we re-download the data
                print(f"overwrite is set to True with the second period None, hence we re-download the data")
                print(f"It will now be downloaded using {num_workers} workers")
                data, tickers = download_and_clean_data(period=period, index=index, interval=interval,
                                                        num_workers=num_workers, input_dim=input_dim)

            else:

                with open(save_dir + f"{index}_{period}_{interval}_{input_dim}.pkl", "rb") as f:
                    data, tickers = pickle.load(f)

        else:

            with open(save_dir + f"{index}_{period}_{interval}_{input_dim}.pkl", "rb") as f:
                data, tickers = pickle.load(f)

    except FileNotFoundError:

        print(f"Data for exchanges {index} with period {period} and interval {interval} was not found")
        print(f"It will now be downloaded using {num_workers} workers")
        data, tickers = download_and_clean_data(period=period, index=index, interval=interval,
                                                num_workers=num_workers, input_dim=input_dim)

    # we refresh the cache
    with open(save_dir + f"{index}_{interval}_{input_dim}_cache.pkl", "wb") as f:
        pickle.dump((data, tickers), f)

    return data, tickers


def get_training_trade_data(index: str = 'S&P500', input_dim: int = 5,
                            overwrite: bool = False, episode_length: int = 1000, use_cache: bool = False):
    # if you use more than 2 years, some data will not have a hour by hour granularity
    # episode length is the length by the smallest granularity for the episode
    start_time = time.monotonic()
    if use_cache:
        with open(os.path.join(save_dir, "weeks_cache.pkl"), "rb") as f:
            weeks = pickle.load(f)
        with open(os.path.join(save_dir, "months_cache.pkl"), "rb") as f:
            months = pickle.load(f)
        with open(os.path.join(save_dir, "years_cache.pkl"), "rb") as f:
            years = pickle.load(f)
    else:
        datas = []
        tickerss = []
        # for now we only look at the last two years
        num_years = 2
        column_names = None
        data_types = ('week', 'month', 'year')
        for data_type in data_types:

            key = keys.get(data_type, None)

            if key is None:
                print(f"{data_type} data type is not supported; please use a key in {keys.keys()}")
                raise NotImplementedError

            num_increments = year_converter[data_type] * num_years
            max_seq_legnth, delta = time_steps_ref[key]

            if data_type == "week":
                assert sequence_length <= max_seq_legnth
                assert num_increments <= max_increments[data_type]
                start_day = (current_day - num_increments * delta).strftime("%Y-%m-%d")
            elif data_type == "month":
                assert sequence_length <= max_seq_legnth
                assert num_increments + 10 <= max_increments[data_type]
                # we pad by an extra 100 to always have at least 100 past timesteps for the start of the week data
                start_day = (current_day - ((num_increments + 10) * delta)).strftime("%Y-%m-%d")
            elif data_type == "year":
                assert sequence_length <= max_seq_legnth
                assert num_increments + 2 <= max_increments[data_type]
                # we pad by an extra 100 to always have at least 100 past timesteps for the start of the week data
                start_day = (current_day - ((num_increments + 2) * delta)).strftime("%Y-%m-%d")
            else:
                raise NotImplementedError

            data, tickers = load_data(period=(start_day, None), index=index, interval=key[1],
                                      num_workers=download_num_workers, overwrite=overwrite, input_dim=input_dim,
                                      use_cache=use_cache)
            column_names = data[0].columns.values

            datas.append(data)
            tickerss.append(tickers)
            print(len(tickers))
            print(len(data))

        tickers = tickerss[0]
        ticker_dict = {}
        for ticker in tickers:
            missing_ticker = False
            # first we check if the ticker is available at all horizons
            indices = []
            for i in range(len(data_types)):
                try:
                    indices.append(tickerss[i].index(ticker))
                except ValueError:
                    missing_ticker = True

            if not missing_ticker:
                # we then get the data for that ticker for each horizon
                ticker_data_list = []
                for j, data_type in enumerate(data_types):
                    ticker_data_list.append(datas[j][indices[j]])
                ticker_dict[ticker] = ticker_data_list

        # now that we have the data lists sorted by ticker in a dictionary, we can start indexing them by our smallest
        # timestep and generating our data windows
        constructed_data = {}
        tickers = []
        weeks = []
        months = []
        years = []
        for ticker, data in ticker_dict.items():
            constructed_data[ticker] = []
            # we now move by our smallest time increment
            week_data, month_data, year_data = data
            weeks.append(week_data)
            months.append(month_data)
            years.append(year_data)
            tickers.append(ticker)
        weeks = pd.concat(weeks, axis=1)
        weeks.columns = pd.MultiIndex.from_product([tickers, column_names], names=['ticker', 'price'])
        months = pd.concat(months, axis=1)
        months.columns = pd.MultiIndex.from_product([tickers, column_names], names=['ticker', 'price'])
        years = pd.concat(years, axis=1)
        years.columns = pd.MultiIndex.from_product([tickers, column_names], names=['ticker', 'price'])

        # we refresh the cache
        weeks.to_pickle(os.path.join(save_dir, "weeks_cache.pkl"))
        months.to_pickle(os.path.join(save_dir, "months_cache.pkl"))
        years.to_pickle(os.path.join(save_dir, "years_cache.pkl"))

    # we leave perform the sliding window at train time to save memory
    # we save the index to perform the sliding properly
    tickers = weeks.columns.levels[0]
    week_index = weeks.index
    month_index = months.index
    year_index = years.index

    weeks = weeks.to_numpy(na_value=-1.0).reshape((len(week_index), len(tickers), input_dim))
    months = months.to_numpy(na_value=-1.0).reshape((len(month_index), len(tickers), input_dim))
    years = years.to_numpy(na_value=-1.0).reshape((len(year_index), len(tickers), input_dim))

    end_time = time.monotonic()
    print(f"Preprocessing took: {timedelta(seconds=end_time - start_time)}")
    return ((weeks, week_index), (months, month_index), (years, year_index)), tickers


if __name__ == '__main__':
    ((weeks, week_index), (months, month_index), (years, year_index)), tickers = get_training_trade_data(use_cache=True)
    print(type(week_index[-2*sequence_length]))
