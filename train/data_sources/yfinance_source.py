import yfinance as yf

from .base import MarketDataSource


class YFinanceDataSource:
    def __init__(self, interval="1d"):
        self.interval = interval

    def load(self, ticker, start, end):
        return yf.download(
            ticker,
            start=start,
            end=end,
            interval=self.interval,
            progress=False,
        )
