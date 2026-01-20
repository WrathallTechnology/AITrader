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
from src.data.crypto_scanner import CryptoScanner
from src.strategies import AdvancedHybridStrategy, TechnicalStrategy, MLStrategy, SignalType
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
import json
from pathlib import Path

# Transaction log file
TRANSACTIONS_FILE = Path("logs/transactions.json")


def load_transactions() -> list:
    """Load transaction history from file."""
    if TRANSACTIONS_FILE.exists():
        try:
            with open(TRANSACTIONS_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_transaction(transaction: dict):
    """Save a transaction to the log file."""
    transactions = load_transactions()
    transactions.append(transaction)
    # Keep only last 500 transactions
    transactions = transactions[-500:]

    TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TRANSACTIONS_FILE, "w") as f:
        json.dump(transactions, f, indent=2, default=str)


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

        # Watchlist - must be defined before _init_options_components
        self.stock_watchlist = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "AMD", "SPY", "QQQ",
        ]
        self.crypto_watchlist = ["BTC/USD", "ETH/USD"]

        if mode in ("options", "all"):
            self._init_options_components()

        logger.info("Initialization complete")

    def _init_data_components(self):
        """Initialize data fetching components."""
        logger.info("Initializing data components...")
        self.client = AlpacaClient(config.alpaca)
        self.data_fetcher = DataFetcher(self.client)
        self.data_processor = DataProcessor()
        self.crypto_scanner = CryptoScanner(
            client=self.client,
            min_volume_usd=1_000,  # Very low - small account doesn't need high liquidity
            max_pairs=10,
        )

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
            min_confidence=0.51,  # Lower threshold - act on weaker ML signals
        )

        # Load pre-trained ML model if it exists
        from pathlib import Path
        model_path = Path("models/price_predictor.pkl")
        if model_path.exists():
            try:
                ml.load_model(model_path)
                logger.info("Loaded pre-trained ML model")
            except Exception as e:
                logger.warning(f"Could not load ML model: {e}")

        # Create hybrid strategy that combines sub-strategies
        # AGGRESSIVE SETTINGS - trades more frequently
        self.strategy = AdvancedHybridStrategy(
            name="hybrid",
            strategies=[technical, ml],
            min_confidence=0.25,  # Very low threshold - act on weak signals
            consensus_required=0.0,  # No consensus needed - single strategy can trigger
            use_adaptive_weights=False,  # Don't reduce weights based on performance
            use_regime_detection=False,  # Don't avoid ranging markets
            use_correlation_adjustment=False,  # Don't reduce correlated signals
            use_time_filter=False,  # Trade any time (24/7 for crypto)
            use_signal_scoring=False,  # Disable - use simple weighted voting instead
        )

    def _init_risk_components(self):
        """Initialize risk management components."""
        logger.info("Initializing risk components...")
        # Note: DrawdownProtection and PositionSizer will be initialized
        # in run() once we have account value
        self.drawdown_protection: Optional[DrawdownProtection] = None
        self.circuit_breaker: Optional[CircuitBreaker] = None
        self.position_sizer: Optional[PositionSizer] = None

        # For crypto-only mode with small capital, use relaxed limits
        # For mixed portfolios, you'd want stricter limits
        self.correlation_manager = PortfolioCorrelationManager(
            max_sector_weight=1.0,  # Allow 100% crypto since we're crypto-only
            max_single_position=1.0,  # Allow 100% in single position for small accounts
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

        # SAFETY: If initial_capital is set, always use it as the cap
        # This prevents trading with full account even if get_effective_capital has issues
        if config.trading.initial_capital is not None:
            effective_capital = min(effective_capital, config.trading.initial_capital * 1.5)  # Allow 50% profit growth
            logger.info(f"Using capped effective capital: ${effective_capital:,.2f} (initial: ${config.trading.initial_capital:,.2f})")

        # Initialize risk management with actual capital
        self.drawdown_protection = DrawdownProtection(initial_value=effective_capital)
        self.circuit_breaker = CircuitBreaker(self.drawdown_protection)
        # Position sizing for small accounts:
        # - Use 20-30% per position to allow 3-4 positions
        # - For crypto with small capital, meaningful trades require larger %
        if effective_capital < 5_000:
            max_pos_pct = 0.30  # 30% max per position for small accounts
        elif effective_capital < 10_000:
            max_pos_pct = 0.20  # 20% for medium accounts
        else:
            max_pos_pct = config.trading.default_position_size_pct  # Use config for large accounts

        logger.info(f"Position sizing: {max_pos_pct:.0%} max per trade (${effective_capital * max_pos_pct:,.0f})")

        self.position_sizer = PositionSizer(
            circuit_breaker=self.circuit_breaker,
            base_risk_pct=config.trading.risk_per_trade,
            max_position_pct=max_pos_pct,
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

                logger.info(f"Cycle complete in {elapsed:.1f}s. Sleeping {sleep_time:.1f}s...")

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
        logger.info("Running crypto trading cycle...")

        # Run scanner on first cycle (no _last_crypto_scan) or every 10 minutes
        should_scan = not hasattr(self, "_last_crypto_scan") or \
                      (datetime.now() - self._last_crypto_scan).total_seconds() > 600

        if should_scan:
            logger.info("Running crypto scanner to find opportunities...")
            try:
                opportunities = self.crypto_scanner.get_top_opportunities(
                    count=8,
                    min_score=30,  # Lower threshold to find more pairs
                )
                if opportunities:
                    self.crypto_watchlist = [o.symbol for o in opportunities]
                    logger.info(f"Scanner found {len(opportunities)} opportunities:")
                    for opp in opportunities[:5]:
                        logger.info(f"  {opp.symbol}: score={opp.score:.0f}, RSI={opp.rsi:.0f}, {opp.reason}")
                else:
                    logger.warning("Scanner found no opportunities, using default watchlist")
                self._last_crypto_scan = datetime.now()
            except Exception as e:
                logger.warning(f"Crypto scan failed, using default watchlist: {e}")

        for symbol in self.crypto_watchlist:
            try:
                self._analyze_and_trade(symbol, "crypto")
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

    def _run_options_cycle(self):
        """Run one cycle of options trading."""
        logger.info("Running options trading cycle...")

        # Run scanner on first cycle or every 10 minutes
        should_scan = not hasattr(self, "_last_options_scan") or \
                      (datetime.now() - self._last_options_scan).total_seconds() > 600

        if should_scan:
            logger.info("Running options scanner to find opportunities...")
            try:
                # Determine market trend based on SPY
                market_trend = self._detect_market_trend()

                opportunities = self.options_scanner.get_top_opportunities(
                    count=10,
                    market_trend=market_trend,
                )

                if opportunities:
                    logger.info(f"Options scanner found {len(opportunities)} opportunities:")
                    for opp in opportunities[:5]:
                        signal_info = ""
                        if opp.signal:
                            signal_info = f" - {opp.signal.spread_type.value} ({opp.signal.confidence:.0%})"
                        logger.info(f"  {opp.symbol}: {opp.opportunity_type} score={opp.score:.0f}{signal_info}")

                    # Evaluate top opportunities for trading
                    for opp in opportunities:
                        if opp.signal and opp.score >= 60:
                            self._evaluate_options_trade(opp)
                else:
                    logger.info("No options opportunities found")

                self._last_options_scan = datetime.now()
            except Exception as e:
                logger.warning(f"Options scan failed: {e}")
        else:
            logger.debug("Skipping options scan (within 10 min window)")

    def _detect_market_trend(self) -> str:
        """Detect overall market trend using SPY."""
        try:
            df = self.data_fetcher.get_historical_bars(
                symbol="SPY",
                timeframe="1Hour",
                days=5,
                use_cache=True,
            )
            if df is None or len(df) < 20:
                return "neutral"

            df = DataProcessor.add_technical_indicators(df)
            latest = df.iloc[-1]

            # Simple trend detection
            rsi = latest.get("rsi", 50)
            sma_20 = latest.get("sma_20", latest["close"])
            current_price = latest["close"]

            if rsi > 60 and current_price > sma_20:
                return "bullish"
            elif rsi < 40 and current_price < sma_20:
                return "bearish"
            return "neutral"
        except Exception:
            return "neutral"

    def _analyze_and_trade(self, symbol: str, asset_type: str):
        """Analyze a symbol and execute trade if signal generated."""
        # Fetch data - disable cache to get fresh data each cycle
        try:
            df = self.data_fetcher.get_historical_bars(
                symbol=symbol,
                timeframe="1Hour",
                days=30,
                use_cache=False,  # Always fetch fresh data
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

        logger.info(f"{symbol}: {signal.signal_type.value.upper()} (conf={signal.confidence:.1%}) - {signal.reason}")

        if signal.signal_type == SignalType.HOLD:
            return

        # Check risk limits
        if not self._check_risk_limits(symbol, signal):
            logger.info(f"Trade blocked by risk limits: {symbol}")
            return

        # Calculate position size using INITIAL capital, not current account value
        # This prevents position size from growing as positions are added
        if config.trading.initial_capital is not None:
            effective_capital = config.trading.initial_capital
        else:
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

        # For small accounts (<$10k), skip complex correlation/sector checks
        # These are designed for larger diversified portfolios
        effective_capital = self.get_effective_capital()
        if effective_capital >= 10_000:
            # Check correlation/sector limits for larger accounts
            current_positions = {
                p.symbol: float(p.market_value) for p in positions
            }
            proposed_pct = max(config.trading.default_position_size_pct, 0.50)
            can_add, violations = self.correlation_manager.can_add_position(
                symbol=symbol,
                proposed_value=effective_capital * proposed_pct,
                current_positions=current_positions,
            )

            if not can_add:
                for v in violations:
                    logger.info(f"Risk violation: {v}")
                return False

        return True

    def _execute_trade(self, symbol: str, signal, shares: float, price: float):
        """Execute a trade and log the transaction."""
        try:
            order = None
            trade_value = shares * price

            if signal.signal_type == SignalType.BUY:
                # CHECK IF WE ALREADY HAVE A POSITION - prevents duplicate buys
                existing_position = self.client.get_position(symbol)
                if existing_position:
                    logger.debug(f"Already have position in {symbol} ({float(existing_position.qty):.4f} shares), skipping buy")
                    return

                # Also check for pending orders to avoid duplicate orders
                open_orders = self.client.get_orders(status="open")
                pending_for_symbol = [o for o in open_orders if o.symbol == symbol.replace("/", "")]
                if pending_for_symbol:
                    logger.debug(f"Already have pending order(s) for {symbol}, skipping buy")
                    return

                order = self.client.submit_limit_order(
                    symbol=symbol,
                    qty=shares,
                    side="buy",
                    limit_price=price,
                    time_in_force="day",
                )
                logger.info(f"BUY order submitted: {order.id if order else 'FAILED'}")

                # Log the transaction
                save_transaction({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "action": "BUY",
                    "quantity": shares,
                    "price": price,
                    "value": round(trade_value, 2),
                    "confidence": round(signal.confidence * 100, 1),
                    "reason": signal.reason,
                    "order_id": str(order.id) if order else None,
                    "status": "submitted",
                })

            elif signal.signal_type == SignalType.SELL:
                # Check if we have a position to sell
                position = self.client.get_position(symbol)
                if position:
                    sell_qty = abs(float(position.qty))
                    sell_value = sell_qty * price
                    entry_price = float(position.avg_entry_price)
                    pnl = (price - entry_price) * sell_qty
                    pnl_pct = ((price / entry_price) - 1) * 100

                    order = self.client.submit_limit_order(
                        symbol=symbol,
                        qty=sell_qty,
                        side="sell",
                        limit_price=price,
                        time_in_force="day",
                    )
                    logger.info(f"SELL order submitted: {order.id if order else 'FAILED'}")

                    # Log the transaction
                    save_transaction({
                        "timestamp": datetime.now().isoformat(),
                        "symbol": symbol,
                        "action": "SELL",
                        "quantity": sell_qty,
                        "price": price,
                        "value": round(sell_value, 2),
                        "entry_price": entry_price,
                        "pnl": round(pnl, 2),
                        "pnl_pct": round(pnl_pct, 2),
                        "confidence": round(signal.confidence * 100, 1),
                        "reason": signal.reason,
                        "order_id": str(order.id) if order else None,
                        "status": "submitted",
                    })

        except Exception as e:
            logger.error(f"Error executing trade for {symbol}: {e}")
            # Log failed transaction
            save_transaction({
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": signal.signal_type.value.upper(),
                "quantity": shares,
                "price": price,
                "reason": signal.reason,
                "status": "failed",
                "error": str(e),
            })

    def _evaluate_options_trade(self, opportunity):
        """Evaluate and potentially execute an options trade."""
        signal = opportunity.signal
        if not signal:
            return

        from src.options import OptionOrder, OrderAction

        # Check risk limits
        account = self.client.get_account()
        account_value = float(account.portfolio_value)

        # Use initial_capital if set
        if config.trading.initial_capital is not None:
            account_value = config.trading.initial_capital

        # Build orders for all contracts in the signal
        orders_to_submit = []
        for contract in signal.contracts:
            order = OptionOrder(
                contract=contract,
                action=OrderAction.BUY_TO_OPEN,
                quantity=1,
                limit_price=contract.mid_price,
            )

            passes, violations = self.options_risk.check_order(order, account_value)

            if not passes:
                for v in violations:
                    logger.info(f"Options risk violation for {contract.symbol}: {v}")
                return

            orders_to_submit.append(order)

        # Log opportunity
        logger.info(
            f"Options opportunity: {opportunity.symbol} - {signal.spread_type.value} "
            f"(score: {opportunity.score:.1f}, confidence: {signal.confidence:.0%})"
        )

        if self.dry_run:
            logger.info(f"[DRY RUN] Would execute {signal.spread_type.value} on {opportunity.symbol}")
            for order in orders_to_submit:
                logger.info(f"  {order.action.value}: {order.contract.symbol} @ ${order.limit_price:.2f}")
            return

        # Execute the options trade
        try:
            for order in orders_to_submit:
                order_id = self.options_client.submit_order(order)

                if order_id:
                    logger.info(f"Options order submitted: {order_id} - {order.contract.symbol}")

                    # Log the transaction
                    save_transaction({
                        "timestamp": datetime.now().isoformat(),
                        "symbol": order.contract.symbol,
                        "underlying": order.contract.underlying,
                        "action": f"OPTIONS_{order.action.value}",
                        "option_type": order.contract.option_type.value if hasattr(order.contract.option_type, 'value') else str(order.contract.option_type),
                        "strike": order.contract.strike,
                        "expiration": str(order.contract.expiration),
                        "quantity": order.quantity,
                        "price": order.limit_price,
                        "value": round(order.limit_price * order.quantity * 100, 2),
                        "confidence": round(signal.confidence * 100, 1),
                        "reason": signal.rationale,
                        "spread_type": signal.spread_type.value if hasattr(signal.spread_type, 'value') else str(signal.spread_type),
                        "expected_profit": signal.expected_profit,
                        "max_loss": signal.max_loss,
                        "order_id": order_id,
                        "status": "submitted",
                    })
                else:
                    logger.error(f"Failed to submit options order for {order.contract.symbol}")
                    save_transaction({
                        "timestamp": datetime.now().isoformat(),
                        "symbol": order.contract.symbol,
                        "underlying": order.contract.underlying,
                        "action": f"OPTIONS_{order.action.value}",
                        "reason": signal.rationale,
                        "status": "failed",
                        "error": "Order submission returned None",
                    })

        except Exception as e:
            logger.error(f"Error executing options trade: {e}")
            save_transaction({
                "timestamp": datetime.now().isoformat(),
                "symbol": opportunity.symbol,
                "action": "OPTIONS_TRADE",
                "reason": signal.rationale,
                "status": "failed",
                "error": str(e),
            })

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
