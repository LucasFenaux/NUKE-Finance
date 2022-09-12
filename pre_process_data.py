from fetch_data import load_data
from typing import Union

num_workers = 16

time_steps_ref = {
    ("1y", "1w"): 53,
    ("1mo", "1d"): 32,
    ("1w", "1h"): 56
}


def process_horizon(period: Union[str, tuple], exchanges: tuple = ("nyse", "nasdaq", "amex", "tsx"),
                    interval: str = "1h", time_steps: int = 0):
    data, tickers = load_data(period=period, exchanges=exchanges, interval=interval)

    print(data.shape)


if __name__ == '__main__':
    process_horizon(period="1w", interval="1h")
