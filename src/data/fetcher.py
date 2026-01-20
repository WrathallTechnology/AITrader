"""
Data fetching module for retrieving market data from Alpaca.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.client import AlpacaClient

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Fetches historical and real-time market data from Alpaca.
    """

    def __init__(self, client: AlpacaClient):
        self.client = client
        self._cache: dict[str, pd.DataFrame] = {}

    def get_historical_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        days: int = 365,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Get historical price bars for a symbol.

        Args:
            symbol: Ticker symbol
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
            days: Number of days of history (used if start not specified)
            start: Start datetime
            end: End datetime
            use_cache: Whether to use cached data

        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{symbol}_{timeframe}_{days}"

        if use_cache and cache_key in self._cache:
            logger.debug(f"Using cached data for {symbol}")
            return self._cache[cache_key]

        # Calculate date range
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=days)

        logger.info(f"Fetching {timeframe} bars for {symbol} from {start} to {end}")

        try:
            is_crypto = self.client.is_crypto(symbol)

            if is_crypto:
                bars = self.client.get_crypto_bars(
                    symbols=[symbol],
                    timeframe=timeframe,
                    start=start,
                    end=end,
                )
            else:
                # Use IEX feed (free tier) instead of SIP (requires paid subscription)
                bars = self.client.get_stock_bars(
                    symbols=[symbol],
                    timeframe=timeframe,
                    start=start,
                    end=end,
                    feed="iex",
                )

            # Convert to DataFrame
            df = bars.df

            # Handle multi-index if present
            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level="symbol")

            # Reset index to make timestamp a column, then set it back
            df = df.reset_index()
            df = df.rename(columns={"timestamp": "datetime"})
            df = df.set_index("datetime")

            # Ensure we have standard OHLCV columns
            df = df[["open", "high", "low", "close", "volume"]]

            # Sort by date
            df = df.sort_index()

            # Cache the data
            if use_cache:
                self._cache[cache_key] = df

            logger.info(f"Fetched {len(df)} bars for {symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            raise

    def get_multiple_symbols(
        self,
        symbols: list[str],
        timeframe: str = "1Day",
        days: int = 365,
    ) -> dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.

        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}

        for symbol in symbols:
            try:
                result[symbol] = self.get_historical_bars(
                    symbol=symbol,
                    timeframe=timeframe,
                    days=days,
                )
            except Exception as e:
                logger.warning(f"Failed to fetch data for {symbol}: {e}")

        return result

    def get_latest_bar(self, symbol: str) -> Optional[pd.Series]:
        """Get the most recent bar for a symbol."""
        df = self.get_historical_bars(
            symbol=symbol,
            timeframe="1Day",
            days=5,
            use_cache=False,
        )

        if df.empty:
            return None

        return df.iloc[-1]

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the most recent closing price for a symbol."""
        bar = self.get_latest_bar(symbol)
        if bar is not None:
            return float(bar["close"])
        return None

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached data."""
        if symbol:
            keys_to_remove = [k for k in self._cache if k.startswith(symbol)]
            for key in keys_to_remove:
                del self._cache[key]
        else:
            self._cache.clear()
