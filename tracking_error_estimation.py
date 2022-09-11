import yfinance as yf
import copy
from currency_converter import CurrencyConverter, SINGLE_DAY_ECB_URL
import math

current_cash = 0


portfolio = {'VOO': (2, 9.0, 'USD'),
             'VXUS': (19, 13.0, 'USD'),
             'AVUV': (15, 11.0, 'USD'),
             'VUN.TO': (28, 15.0, 'CAD'),
             'AVDV': (18, 12.0, 'USD'),
             'VCN.TO': (80, 30.0, 'CAD'),
             'VLUE': (4, 0.0, 'USD'),
             'BND': (0, 10.0, 'USD')}  # layout is portfolio[ticker] = (# shares owned, ideal % weight in portfolio)


def get_current_exchange_rate(origin: str):
    c = CurrencyConverter(SINGLE_DAY_ECB_URL)
    rate = c.convert(1, origin, 'CAD')
    print(f"Current USD to CAD rate: {rate}")
    return rate


def get_current_price(ticker: str, rate: float):
    ticker_yahoo = yf.Ticker(ticker)
    data = ticker_yahoo.history()
    last_quote = data['Close'].iloc[-1]
    if portfolio[ticker][2] != 'CAD':
        last_quote = last_quote * rate
    print(f"{ticker} {last_quote:.3f}")
    return last_quote


def get_individual_ticker_error(ticker: str, shares: int, portfolio_value: float, prices: dict):
    current_percentage = 100 * (shares * prices[ticker]) / portfolio_value
    return current_percentage - portfolio[ticker][1]


def get_portfolio_error(current_portfolio: dict, portfolio_value: float, prices: dict):
    total_error = 0.
    for ticker, shares in current_portfolio.items():
        total_error += math.pow(get_individual_ticker_error(ticker, shares, portfolio_value, prices), 2)

    return math.sqrt(total_error)


def greedy_step(current_portfolio: dict, value: float, prices: dict, max_value: float):
    errors = {}
    done = True
    for ticker in current_portfolio.keys():
        # add one share
        new_value = value + prices[ticker]
        if new_value < max_value:
            done = False
            # add one share to the portfolio
            new_portfolio = copy.deepcopy(current_portfolio)
            new_portfolio[ticker] += 1
            # compute current error
            errors[ticker] = get_portfolio_error(new_portfolio, new_value, prices)
    if not done:
        best_ticker = min(errors, key=errors.get)
        current_portfolio[best_ticker] += 1
        best_value = value + prices[best_ticker]
        return current_portfolio, done, best_value
    else:
        return current_portfolio, done, value


def greedy_solve(initial_portfolio, initial_portfolio_value: float, prices: dict, tickers: list):
    current_portfolio = copy.deepcopy(initial_portfolio)
    current_value = initial_portfolio_value
    max_value = current_value + current_cash
    done = False
    while not done:
        current_portfolio, done, current_value = greedy_step(current_portfolio, current_value, prices, max_value)

    return current_portfolio, current_value


def main():
    # we add the variables for the solver
    prices = {}
    portfolio_value = 0
    usd_to_cad_rate = get_current_exchange_rate('USD')
    initial_portfolio = {}
    tickers = []
    for ticker in portfolio.keys():
        initial_portfolio[ticker] = portfolio[ticker][0]
        prices[ticker] = get_current_price(ticker, usd_to_cad_rate)
        portfolio_value += portfolio[ticker][0] * prices[ticker]
        tickers.append(ticker)

    best_porfolio, new_value = greedy_solve(initial_portfolio, portfolio_value, prices, tickers)

    print(f"\nBest Standard Deviation: {get_portfolio_error(best_porfolio, new_value, prices):.3f}")

    print(f"New Portfolio Value: {new_value:.3f}")
    print(f"Cash Left: {portfolio_value + current_cash - new_value:.3f}\n")

    print(f"Minimum Tracking Error Combination")
    for ticker, shares in best_porfolio.items():
        print(f"{ticker}: {portfolio[ticker][0]} + {shares - portfolio[ticker][0]} -> {shares}")


if __name__ == '__main__':
    main()
    # get_current_exchange_rate()
