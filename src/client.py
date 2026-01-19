"""
Alpaca API client wrapper for trading operations.
Supports both stocks and crypto trading.
"""

import logging
from datetime import datetime
from typing import Optional
from decimal import Decimal

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    StopLossRequest,
    TakeProfitRequest,
    GetOrdersRequest,
)
from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, AssetClass
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame

from config import AlpacaConfig

logger = logging.getLogger(__name__)


class AlpacaClient:
    """
    Unified client for Alpaca trading and data operations.
    Handles both stocks and cryptocurrency.
    """

    def __init__(self, config: AlpacaConfig):
        self.config = config
        self._trading_client: Optional[TradingClient] = None
        self._stock_data_client: Optional[StockHistoricalDataClient] = None
        self._crypto_data_client: Optional[CryptoHistoricalDataClient] = None

    @property
    def trading(self) -> TradingClient:
        """Lazy-loaded trading client."""
        if self._trading_client is None:
            self._trading_client = TradingClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
                paper=self.config.is_paper,
            )
        return self._trading_client

    @property
    def stock_data(self) -> StockHistoricalDataClient:
        """Lazy-loaded stock data client."""
        if self._stock_data_client is None:
            self._stock_data_client = StockHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
            )
        return self._stock_data_client

    @property
    def crypto_data(self) -> CryptoHistoricalDataClient:
        """Lazy-loaded crypto data client."""
        if self._crypto_data_client is None:
            self._crypto_data_client = CryptoHistoricalDataClient(
                api_key=self.config.api_key,
                secret_key=self.config.secret_key,
            )
        return self._crypto_data_client

    # ==================== Account Methods ====================

    def get_account(self):
        """Get account information."""
        return self.trading.get_account()

    def get_buying_power(self) -> Decimal:
        """Get available buying power."""
        account = self.get_account()
        return Decimal(account.buying_power)

    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value."""
        account = self.get_account()
        return Decimal(account.portfolio_value)

    def get_cash(self) -> Decimal:
        """Get available cash."""
        account = self.get_account()
        return Decimal(account.cash)

    # ==================== Position Methods ====================

    def get_all_positions(self):
        """Get all open positions."""
        return self.trading.get_all_positions()

    def get_position(self, symbol: str):
        """Get position for a specific symbol."""
        try:
            return self.trading.get_open_position(symbol)
        except Exception:
            return None

    def close_position(self, symbol: str):
        """Close a position entirely."""
        logger.info(f"Closing position: {symbol}")
        return self.trading.close_position(symbol)

    def close_all_positions(self, cancel_orders: bool = True):
        """Close all positions."""
        logger.warning("Closing all positions!")
        return self.trading.close_all_positions(cancel_orders=cancel_orders)

    # ==================== Order Methods ====================

    def submit_market_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        time_in_force: str = "day",
    ):
        """
        Submit a market order.

        Args:
            symbol: Ticker symbol (e.g., 'AAPL' or 'BTC/USD')
            qty: Quantity to trade
            side: 'buy' or 'sell'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        # Crypto only supports gtc or ioc, not day
        if "/" in symbol:  # Crypto symbols have '/' (e.g., BTC/USD)
            time_in_force = "gtc"

        tif = self._parse_time_in_force(time_in_force)

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
        )

        logger.info(f"Submitting market order: {side} {qty} {symbol}")
        return self.trading.submit_order(request)

    def submit_limit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        limit_price: float,
        time_in_force: str = "day",
    ):
        """Submit a limit order."""
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

        # Crypto only supports gtc or ioc, not day
        if "/" in symbol:  # Crypto symbols have '/' (e.g., BTC/USD)
            time_in_force = "gtc"

        tif = self._parse_time_in_force(time_in_force)

        request = LimitOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            limit_price=limit_price,
            time_in_force=tif,
        )

        logger.info(f"Submitting limit order: {side} {qty} {symbol} @ {limit_price}")
        return self.trading.submit_order(request)

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        take_profit_price: float,
        stop_loss_price: float,
        time_in_force: str = "day",
    ):
        """
        Submit a bracket order with take profit and stop loss.
        """
        order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
        tif = self._parse_time_in_force(time_in_force)

        request = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=order_side,
            time_in_force=tif,
            order_class="bracket",
            take_profit=TakeProfitRequest(limit_price=take_profit_price),
            stop_loss=StopLossRequest(stop_price=stop_loss_price),
        )

        logger.info(
            f"Submitting bracket order: {side} {qty} {symbol} "
            f"TP: {take_profit_price}, SL: {stop_loss_price}"
        )
        return self.trading.submit_order(request)

    def get_orders(
        self,
        status: str = "open",
        limit: int = 100,
        symbols: Optional[list[str]] = None,
    ):
        """Get orders with optional filtering."""
        order_status = OrderStatus(status) if status != "all" else None

        request = GetOrdersRequest(
            status=order_status,
            limit=limit,
            symbols=symbols,
        )

        return self.trading.get_orders(request)

    def cancel_order(self, order_id: str):
        """Cancel a specific order."""
        logger.info(f"Canceling order: {order_id}")
        return self.trading.cancel_order_by_id(order_id)

    def cancel_all_orders(self):
        """Cancel all open orders."""
        logger.warning("Canceling all orders!")
        return self.trading.cancel_orders()

    # ==================== Data Methods ====================

    def get_stock_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
    ):
        """
        Get historical stock bars.

        Args:
            symbols: List of stock symbols
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            start: Start datetime
            end: End datetime (defaults to now)
        """
        tf = self._parse_timeframe(timeframe)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
        )

        return self.stock_data.get_stock_bars(request)

    def get_crypto_bars(
        self,
        symbols: list[str],
        timeframe: str,
        start: datetime,
        end: Optional[datetime] = None,
    ):
        """
        Get historical crypto bars.

        Args:
            symbols: List of crypto symbols (e.g., ['BTC/USD', 'ETH/USD'])
            timeframe: '1Min', '5Min', '15Min', '1Hour', '1Day'
            start: Start datetime
            end: End datetime (defaults to now)
        """
        tf = self._parse_timeframe(timeframe)

        request = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
        )

        return self.crypto_data.get_crypto_bars(request)

    # ==================== Asset Methods ====================

    def get_asset(self, symbol: str):
        """Get asset information."""
        return self.trading.get_asset(symbol)

    def is_tradable(self, symbol: str) -> bool:
        """Check if an asset is tradable."""
        try:
            asset = self.get_asset(symbol)
            return asset.tradable
        except Exception:
            return False

    def is_crypto(self, symbol: str) -> bool:
        """Check if a symbol is a crypto asset."""
        return "/" in symbol  # Crypto symbols are like 'BTC/USD'

    # ==================== Market Status ====================

    def get_clock(self):
        """Get market clock."""
        return self.trading.get_clock()

    def is_market_open(self) -> bool:
        """Check if the market is currently open."""
        clock = self.get_clock()
        return clock.is_open

    # ==================== Helper Methods ====================

    def _parse_time_in_force(self, tif: str) -> TimeInForce:
        """Parse time in force string to enum."""
        mapping = {
            "day": TimeInForce.DAY,
            "gtc": TimeInForce.GTC,
            "ioc": TimeInForce.IOC,
            "fok": TimeInForce.FOK,
        }
        return mapping.get(tif.lower(), TimeInForce.DAY)

    def _parse_timeframe(self, tf: str) -> TimeFrame:
        """Parse timeframe string to TimeFrame object."""
        mapping = {
            "1min": TimeFrame.Minute,
            "5min": TimeFrame(5, "Min"),
            "15min": TimeFrame(15, "Min"),
            "1hour": TimeFrame.Hour,
            "1day": TimeFrame.Day,
        }
        return mapping.get(tf.lower(), TimeFrame.Day)
