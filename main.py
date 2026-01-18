"""
AITrader - Main Entry Point

This is the main orchestrator that runs the trading bot.
It initializes all components and runs the trading loop.

Usage:
    python main.py                    # Run with default settings
    python main.py --mode stocks      # Trade only stocks
    python main.py --mode crypto      # Trade only crypto
    python main.py --mode options     # Trade only options
    python main.py --mode all         # Trade everything (default)
    python main.py --dry-run          # Analyze but don't execute trades
"""

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from typing import Optional

from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from config import config, AlpacaConfig
from src.client import AlpacaClient
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.strategies import HybridStrategy, TechnicalStrategy, MLStrategy, SignalType
from src.risk import (
    PositionSizer,
    DrawdownProtection,
    CircuitBreaker,
    RiskState,
    PortfolioCorrelationManager,
)
from src.options import (
    OptionsClient,
    OptionsStrategyManager,
    OptionsRiskManager,
    OptionsScanner,
    create_risk_limits_conservative,
)

# Setup logging
def setup_logging():
    """Configure logging for the bot."""
    log_dir = os.path.dirname(config.log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(config.log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("AITrader")


logger = setup_logging()


class AITrader:
    """
    Main trading bot orchestrator.

    Coordinates all components:
    - Data fetching
    - Signal generation
    - Risk management
    - Order execution
    """

    def __init__(
        self,
        mode: str = "all",
        dry_run: bool = False,
    ):
        """
        Initialize the trading bot.

        Args:
            mode: Trading mode - "stocks", "crypto", "options", or "all"
            dry_run: If True, analyze but don't execute trades
        """
        self.mode = mode
        self.dry_run = dry_run
        self.running = False
        self.starting_account_value: Optional[float] = None

        # Validate API keys
        if not config.alpaca.api_key or not config.alpaca.secret_key:
            raise ValueError(
                "Alpaca API keys not configured. "
                "Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file."
            )

        logger.info("=" * 60)
        logger.info("AITrader Initializing")
        logger.info("=" * 60)
        logger.info(f"Mode: {mode}")
        logger.info(f"Dry Run: {dry_run}")
        logger.info(f"Paper Trading: {config.alpaca.is_paper}")
        logger.info(f"Initial Capital: ${config.trading.initial_capital:,.2f}"
                   if config.trading.initial_capital else "Using full account")

        # Initialize components
        self._init_data_components()
        self._init_strategy_components()
        self._init_risk_components()
        self._init_execution_components()

        if mode in ("options", "all"):
            self._init_options_components()

        # Watchlist
        self.stock_watchlist = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "AMD", "SPY", "QQQ",
        ]
        self.crypto_watchlist = ["BTC/USD", "ETH/USD"]

        logger.info("Initialization complete")

    def _init_data_components(self):
        """Initialize data fetching components."""
        logger.info("Initializing data components...")
        self.client = AlpacaClient(config.alpaca)
        self.data_fetcher = DataFetcher(self.client)
        self.data_processor = DataProcessor()

    def _init_strategy_components(self):
        """Initialize strategy components."""
        logger.info("Initializing strategy components...")

        # Create sub-strategies with weights from config
        technical = TechnicalStrategy(
            name="technical",
            weight=config.strategy.technical_weight,
            config=config.strategy,
        )

        ml = MLStrategy(
            name="ml",
            weight=config.strategy.ml_weight,
            config=config.strategy,
            min_confidence=0.55,
        )

        # Create hybrid strategy that combines sub-strategies
        self.strategy = HybridStrategy(
            name="hybrid",
            strategies=[technical, ml],
            min_confidence=0.5,
            consensus_required=0.4,
        )

    def _init_risk_components(self):
        """Initialize risk management components."""
        logger.info("Initializing risk components...")
        # Note: DrawdownProtection and PositionSizer will be initialized
        # in run() once we have account value
        self.drawdown_protection: Optional[DrawdownProtection] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.position_sizer: Optional[PositionSizer] = None

        self.correlation_manager = PortfolioCorrelationManager(
            max_sector_weight=0.30,
            max_single_position=0.10,
        )

    def _init_execution_components(self):
        """Initialize order execution components."""
        logger.info("Initializing execution components...")
        # Using self.client initialized in _init_data_components

    def _init_options_components(self):
        """Initialize options trading components."""
        logger.info("Initializing options components...")
        self.options_client = OptionsClient(
            api_key=config.alpaca.api_key,
            secret_key=config.alpaca.secret_key,
            paper=config.alpaca.is_paper,
        )
        self.options_strategy = OptionsStrategyManager(self.options_client)
        self.options_risk = OptionsRiskManager(
            limits=create_risk_limits_conservative()
        )
        self.options_scanner = OptionsScanner(
            client=self.options_client,
            watchlist=self.stock_watchlist,
        )

    def get_effective_capital(self) -> float:
        """Get the capital amount to use for trading."""
        account = self.client.get_account()
        account_value = float(account.portfolio_value)

        return config.trading.get_effective_capital(
            account_value=account_value,
            starting_account_value=self.starting_account_value,
        )

    def run(self):
        """Main trading loop."""
        self.running = True
        logger.info("Starting trading loop...")

        # Get initial account value
        account = self.client.get_account()
        self.starting_account_value = float(account.portfolio_value)
        logger.info(f"Starting Account Value: ${self.starting_account_value:,.2f}")

        effective_capital = self.get_effective_capital()
        logger.info(f"Effective Trading Capital: ${effective_capital:,.2f}")

        # Initialize risk management with actual capital
        self.drawdown_protection = DrawdownProtection(initial_value=effective_capital)
        self.circuit_breaker = CircuitBreaker(self.drawdown_protection)
        self.position_sizer = PositionSizer(
            circuit_breaker=self.circuit_breaker,
            base_risk_pct=config.trading.risk_per_trade,
            max_position_pct=config.trading.default_position_size_pct,
        )

        while self.running:
            try:
                loop_start = datetime.now()

                # Check market hours for stocks
                if self.mode in ("stocks", "all"):
                    if self._is_market_open():
                        self._run_stock_cycle()
                    else:
                        logger.debug("Stock market closed")

                # Crypto trades 24/7
                if self.mode in ("crypto", "all"):
                    self._run_crypto_cycle()

                # Options during market hours
                if self.mode in ("options", "all"):
                    if self._is_market_open():
                        self._run_options_cycle()

                # Check drawdown protection
                self._check_risk_state()

                # Sleep until next cycle
                elapsed = (datetime.now() - loop_start).total_seconds()
                sleep_time = max(0, 60 - elapsed)  # Run every minute

                if self.running:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                logger.info("Shutdown requested...")
                self.stop()
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                time.sleep(60)  # Wait before retrying

        logger.info("Trading loop stopped")

    def _is_market_open(self) -> bool:
        """Check if stock market is open."""
        try:
            return self.client.is_market_open()
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False

    def _run_stock_cycle(self):
        """Run one cycle of stock trading."""
        logger.debug("Running stock trading cycle...")

        for symbol in self.stock_watchlist:
            try:
                self._analyze_and_trade(symbol, "stock")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    def _run_crypto_cycle(self):
        """Run one cycle of crypto trading."""
        logger.debug("Running crypto trading cycle...")

        for symbol in self.crypto_watchlist:
            try:
                self._analyze_and_trade(symbol, "crypto")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    def _run_options_cycle(self):
        """Run one cycle of options trading."""
        logger.debug("Running options trading cycle...")

        try:
            # Scan for opportunities
            opportunities = self.options_scanner.get_top_opportunities(
                count=5,
                market_trend="neutral",  # TODO: integrate with market regime detection
            )

            for opp in opportunities:
                if opp.signal and opp.score >= 60:
                    self._evaluate_options_trade(opp)

        except Exception as e:
            logger.error(f"Error in options cycle: {e}")

    def _analyze_and_trade(self, symbol: str, asset_type: str):
        """Analyze a symbol and execute trade if signal generated."""
        # Fetch data
        try:
            df = self.data_fetcher.get_historical_bars(
                symbol=symbol,
                timeframe="1Hour",
                days=30,
            )
        except Exception as e:
            logger.debug(f"Failed to fetch data for {symbol}: {e}")
            return

        if df is None or len(df) < 50:
            return

        # Process data - add technical indicators
        df = DataProcessor.add_technical_indicators(df)

        # Generate signal
        signal = self.strategy.generate_signal(symbol, df)

        if signal.signal_type == SignalType.HOLD:
            return

        logger.info(f"Signal generated: {signal}")

        # Check risk limits
        if not self._check_risk_limits(symbol, signal):
            logger.info(f"Trade blocked by risk limits: {symbol}")
            return

        # Calculate position size
        effective_capital = self.get_effective_capital()
        current_price = float(df["close"].iloc[-1])

        shares = self.position_sizer.calculate_position_size(
            portfolio_value=effective_capital,
            entry_price=current_price,
            stop_loss=signal.stop_loss or current_price * 0.95,
            signal_confidence=signal.confidence,
        )

        if shares == 0:
            logger.debug(f"Position size too small for {symbol}")
            return

        # Execute trade
        if not self.dry_run:
            self._execute_trade(symbol, signal, shares, current_price)
        else:
            logger.info(
                f"[DRY RUN] Would {signal.signal_type.value} {shares:.2f} "
                f"shares of {symbol} at ${current_price:.2f}"
            )

    def _check_risk_limits(self, symbol: str, signal) -> bool:
        """Check if trade passes all risk limits."""
        # Check drawdown protection state
        state = self.circuit_breaker.check_state()
        if state == RiskState.HALTED:
            logger.warning("Trading halted due to drawdown protection")
            return False
        if state == RiskState.RESTRICTED:
            if signal.signal_type == SignalType.BUY:
                logger.warning("New buys restricted due to drawdown")
                return False

        # Check position limits
        positions = self.client.get_all_positions()
        if len(positions) >= config.trading.max_positions:
            if signal.signal_type == SignalType.BUY:
                logger.info("Max positions reached")
                return False

        # Check correlation/sector limits
        current_positions = {
            p.symbol: float(p.market_value) for p in positions
        }
        effective_capital = self.get_effective_capital()

        can_add, violations = self.correlation_manager.can_add_position(
            symbol=symbol,
            proposed_value=effective_capital * config.trading.default_position_size_pct,
            current_positions=current_positions,
        )

        if not can_add:
            for v in violations:
                logger.info(f"Risk violation: {v}")
            return False

        return True

    def _execute_trade(self, symbol: str, signal, shares: float, price: float):
        """Execute a trade."""
        try:
            if signal.signal_type == SignalType.BUY:
                order = self.client.submit_limit_order(
                    symbol=symbol,
                    qty=shares,
                    side="buy",
                    limit_price=price,
                    time_in_force="day",
                )
                logger.info(f"BUY order submitted: {order.id if order else 'FAILED'}")

            elif signal.signal_type == SignalType.SELL:
                # Check if we have a position to sell
                position = self.client.get_position(symbol)
                if position:
                    order = self.client.submit_limit_order(
                        symbol=symbol,
                        qty=abs(float(position.qty)),
                        side="sell",
                        limit_price=price,
                        time_in_force="day",
                    )
                    logger.info(f"SELL order submitted: {order.id if order else 'FAILED'}")

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")

    def _evaluate_options_trade(self, opportunity):
        """Evaluate and potentially execute an options trade."""
        signal = opportunity.signal
        if not signal:
            return

        # Check risk limits
        account = self.client.get_account()
        account_value = float(account.portfolio_value)

        for contract in signal.contracts:
            from src.options import OptionOrder, OrderAction

            order = OptionOrder(
                contract=contract,
                action=OrderAction.BUY_TO_OPEN,
                quantity=1,
                limit_price=contract.mid_price,
            )

            passes, violations = self.options_risk.check_order(order, account_value)

            if not passes:
                for v in violations:
                    logger.info(f"Options risk violation: {v}")
                return

        # Log opportunity (actual execution would go here)
        logger.info(
            f"Options opportunity: {opportunity.symbol} - {signal.spread_type.value} "
            f"(score: {opportunity.score:.1f})"
        )

        if not self.dry_run:
            # TODO: Implement actual options order submission
            logger.info("[OPTIONS] Order execution not yet implemented")

    def _check_risk_state(self):
        """Check and log current risk state."""
        account = self.client.get_account()
        current_value = float(account.portfolio_value)

        # Update drawdown protection
        self.drawdown_protection.update_value(current_value)
        state = self.circuit_breaker.check_state()

        if state != RiskState.NORMAL:
            logger.warning(f"Risk state: {state.value}")

            if state == RiskState.HALTED:
                logger.error("TRADING HALTED - Manual intervention required")

    def stop(self):
        """Stop the trading bot gracefully."""
        logger.info("Stopping AITrader...")
        self.running = False

        # Cancel any open orders
        try:
            self.client.cancel_all_orders()
            logger.info("Cancelled all open orders")
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

        # Log final state
        try:
            account = self.client.get_account()
            final_value = float(account.portfolio_value)
            pnl = final_value - self.starting_account_value
            pnl_pct = pnl / self.starting_account_value * 100

            logger.info("=" * 60)
            logger.info("SESSION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Starting Value: ${self.starting_account_value:,.2f}")
            logger.info(f"Final Value: ${final_value:,.2f}")
            logger.info(f"P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            logger.info("=" * 60)
        except Exception as e:
            logger.error(f"Error getting final state: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AITrader - AI-Powered Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["stocks", "crypto", "options", "all"],
        default="all",
        help="Trading mode (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze but don't execute trades",
    )
    args = parser.parse_args()

    # Handle shutdown signals
    def signal_handler(sig, frame):
        logger.info("Shutdown signal received")
        if trader:
            trader.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    trader = None
    try:
        trader = AITrader(
            mode=args.mode,
            dry_run=args.dry_run,
        )
        trader.run()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
