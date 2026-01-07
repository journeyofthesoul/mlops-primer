import yfinance as yf

from .base import MarketDataSource


class YFinanceDataSource(MarketDataSource):
    def __init__(self, period: str = "5y", interval: str = "1d"):
        self.period = period
        self.interval = interval

    def load(self, ticker: str):
        return yf.download(
            ticker,
            period=self.period,
            interval=self.interval,
            progress=False,
        )
