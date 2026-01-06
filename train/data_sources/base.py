from abc import ABC, abstractmethod

import pandas as pd


class MarketDataSource(ABC):
    """Abstract interface for market data sources."""

    @abstractmethod
    def load(self, ticker: str) -> pd.DataFrame:
        """Return historical OHLCV data for a ticker."""
        raise NotImplementedError
