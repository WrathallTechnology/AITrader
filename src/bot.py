"""
Main trading bot orchestrator.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.client import AlpacaClient
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.portfolio import PortfolioManager
from src.strategies.base import Signal, SignalType
from src.strategies.hybrid import HybridStrategy
from src.strategies.technical import TechnicalStrategy
from src.strategies.ml_strategy import MLStrategy
from src.utils.logger import setup_logger, get_trade_logger
from config import Config, load_config

logger = logging.getLogger(__name__)


class TradingBot:
    """
    Main trading bot that orchestrates data fetching, strategy execution,
    and order management.
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        watchlist: Optional[list[str]] = None,
    ):
        """
        Initialize the trading bot.

        Args:
            config: Configuration object
            watchlist: List of symbols to trade
        """
        self.config = config or load_config()

        # Default watchlist: popular stocks and crypto
        self.watchlist = watchlist or [
            # Stocks
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
            "META", "TSLA", "JPM", "V", "SPY",
            # Crypto
            "BTC/USD", "ETH/USD",
        ]

        # Filter by asset class
        if not self.config.trading.trade_stocks:
            self.watchlist = [s for s in self.watchlist if "/" in s]
        if not self.config.trading.trade_crypto:
            self.watchlist = [s for s in self.watchlist if "/" not in s]

        # Initialize components
        self.client = AlpacaClient(self.config.alpaca)
        self.data_fetcher = DataFetcher(self.client)
        self.portfolio = PortfolioManager(self.client, self.config.trading)

        # Initialize strategy
        self.strategy = self._create_hybrid_strategy()

        # Trade logger
        self.trade_logger = get_trade_logger()

        # State
        self._is_running = False
        self._last_run = None

    def _create_hybrid_strategy(self) -> HybridStrategy:
        """Create the hybrid strategy with configured weights."""
        strategy = HybridStrategy(
            name="main",
            min_confidence=0.6,
            consensus_required=0.5,
        )

        # Add technical strategy
        technical = TechnicalStrategy(
            name="technical",
            weight=self.config.strategy.technical_weight,
            config=self.config.strategy,
        )
        strategy.add_strategy(technical)

        # Add ML strategy
        ml = MLStrategy(
            name="ml",
            weight=self.config.strategy.ml_weight,
            config=self.config.strategy,
            min_confidence=0.55,
        )
        strategy.add_strategy(ml)

        return strategy

    # ==================== Training ====================

    def train_strategies(self, days: int = 365) -> None:
        """
        Train ML-based strategies on historical data.

        Args:
            days: Number of days of historical data to use
        """
        logger.info(f"Training strategies on {days} days of data...")

        # Use a representative symbol for training
        training_symbol = "SPY"  # S&P 500 ETF

        try:
            # Fetch historical data
            data = self.data_fetcher.get_historical_bars(
                symbol=training_symbol,
                timeframe="1Day",
                days=days,
                use_cache=False,
            )

            # Add indicators
            data = DataProcessor.add_technical_indicators(data)

            # Train the hybrid strategy (which trains all sub-strategies)
            self.strategy.train(data)

            logger.info("Strategy training complete!")

        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    # ==================== Signal Generation ====================

    def generate_signals(self) -> dict[str, Signal]:
        """
        Generate trading signals for all watchlist symbols.

        Returns:
            Dictionary mapping symbol to Signal
        """
        signals = {}

        for symbol in self.watchlist:
            try:
                signal = self._generate_signal(symbol)
                signals[symbol] = signal

                if signal.is_actionable:
                    logger.info(f"Signal: {signal}")

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")

        return signals

    def _generate_signal(self, symbol: str) -> Signal:
        """Generate signal for a single symbol."""
        # Fetch recent data
        data = self.data_fetcher.get_historical_bars(
            symbol=symbol,
            timeframe="1Day",
            days=100,
            use_cache=True,
        )

        # Add indicators
        data = DataProcessor.add_technical_indicators(data)

        # Generate signal
        return self.strategy.generate_signal(symbol, data)

    # ==================== Trade Execution ====================

    def execute_signals(self, signals: dict[str, Signal]) -> list[str]:
        """
        Execute actionable signals.

        Returns:
            List of order IDs for executed trades
        """
        executed_orders = []

        for symbol, signal in signals.items():
            if not signal.is_actionable:
                continue

            # Handle existing positions
            existing_position = self.portfolio.get_position(symbol)

            if signal.signal_type == SignalType.SELL and existing_position:
                # Close existing long position
                if self.portfolio.close_position(symbol):
                    self._log_trade("CLOSE", symbol, existing_position.quantity, signal)
                continue

            if signal.signal_type == SignalType.BUY and existing_position:
                # Skip if already have position
                logger.debug(f"Already have position in {symbol}, skipping")
                continue

            # Generate order
            order = self.portfolio.signal_to_order(signal, use_bracket=True)
            if order is None:
                continue

            # Execute order
            order_id = self.portfolio.execute_order(order)
            if order_id:
                executed_orders.append(order_id)
                self._log_trade(order.side.upper(), symbol, order.quantity, signal)

        return executed_orders

    def _log_trade(
        self,
        action: str,
        symbol: str,
        quantity: float,
        signal: Signal,
    ) -> None:
        """Log trade to trade logger."""
        self.trade_logger.info(
            f"{action} | {symbol} | qty={quantity} | "
            f"confidence={signal.confidence:.2%} | {signal.reason}"
        )

    # ==================== Main Loop ====================

    def run_once(self) -> dict[str, Signal]:
        """
        Run a single iteration of the trading bot.

        Returns:
            Generated signals
        """
        logger.info("=" * 50)
        logger.info(f"Running trading cycle at {datetime.now()}")

        # Check market status for stocks
        if self.config.trading.trade_stocks:
            if not self.client.is_market_open():
                logger.info("Stock market is closed")
                # Still process crypto if enabled
                if not self.config.trading.trade_crypto:
                    return {}
                # Filter to only crypto
                original_watchlist = self.watchlist
                self.watchlist = [s for s in self.watchlist if "/" in s]

        # Log portfolio status
        self._log_portfolio_status()

        # Generate signals
        signals = self.generate_signals()

        # Execute signals
        if signals:
            executed = self.execute_signals(signals)
            if executed:
                logger.info(f"Executed {len(executed)} orders")

        # Restore watchlist if modified
        if "original_watchlist" in locals():
            self.watchlist = original_watchlist

        self._last_run = datetime.now()
        return signals

    def run(
        self,
        interval_minutes: int = 60,
        max_iterations: Optional[int] = None,
    ) -> None:
        """
        Run the trading bot continuously.

        Args:
            interval_minutes: Minutes between trading cycles
            max_iterations: Maximum number of iterations (None for infinite)
        """
        logger.info(f"Starting trading bot (interval: {interval_minutes} min)")
        logger.info(f"Trading mode: {self.config.trading.mode}")
        logger.info(f"Watchlist: {self.watchlist}")

        self._is_running = True
        iteration = 0

        try:
            while self._is_running:
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached max iterations ({max_iterations})")
                    break

                self.run_once()
                iteration += 1

                # Wait for next cycle
                if self._is_running:
                    logger.info(f"Sleeping for {interval_minutes} minutes...")
                    time.sleep(interval_minutes * 60)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the trading bot."""
        logger.info("Stopping trading bot...")
        self._is_running = False

    # ==================== Status & Reporting ====================

    def _log_portfolio_status(self) -> None:
        """Log current portfolio status."""
        try:
            portfolio_value = self.portfolio.get_portfolio_value()
            buying_power = self.portfolio.get_buying_power()
            positions = self.portfolio.get_positions()

            logger.info(f"Portfolio value: ${portfolio_value:,.2f}")
            logger.info(f"Buying power: ${buying_power:,.2f}")
            logger.info(f"Open positions: {len(positions)}")

            for pos in positions:
                logger.info(
                    f"  {pos.symbol}: {pos.quantity} shares, "
                    f"P/L: ${pos.unrealized_pl:,.2f} ({pos.unrealized_pl_pct:+.2f}%)"
                )

        except Exception as e:
            logger.error(f"Error getting portfolio status: {e}")

    def get_status(self) -> dict:
        """Get current bot status."""
        return {
            "is_running": self._is_running,
            "last_run": self._last_run,
            "trading_mode": self.config.trading.mode,
            "watchlist": self.watchlist,
            "strategy_weights": self.strategy.get_strategy_weights(),
            "portfolio_value": float(self.portfolio.get_portfolio_value()),
            "position_count": self.portfolio.get_position_count(),
        }


def main():
    """Main entry point."""
    # Setup logging
    logger = setup_logger("aitrader", level="INFO", log_file="logs/trading.log")

    # Create bot
    bot = TradingBot()

    # Train strategies (do this once, or periodically)
    # bot.train_strategies(days=365)

    # Run bot
    bot.run(interval_minutes=60)


if __name__ == "__main__":
    main()
