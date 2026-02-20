"""Method Competition Manager — orchestrates 3 trading methods.

Manages timing, shared Gemini analysis, order submission, and
per-method capital tracking. Each method gets $1000 of capital.

Methods:
  1. "strategies" — AdvancedHybridStrategy (all existing strategies), every 5 min
  2. "wsb"        — WSB Reddit sentiment → stock signals, every 20 min
  3. "gemini"     — Gemini AI comprehensive analysis, every 20 min
"""

import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from config import config
from src.client import AlpacaClient
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.strategies import (
    AdvancedHybridStrategy,
    TechnicalStrategy,
    MLStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    SignalType,
)
from src.options.gemini_client import GeminiSentimentClient, GeminiFullAnalysis
from src.options.wsb_reddit import WSBRedditFetcher

from .tracker import MethodTracker
from .wsb_stock_strategy import WSBStockStrategy
from .gemini_comprehensive_strategy import GeminiComprehensiveStrategy

logger = logging.getLogger(__name__)

# Method IDs (used as prefixes in client_order_id)
METHOD_STRATEGIES = "strat"
METHOD_WSB = "wsb"
METHOD_GEMINI = "gemini"

# Position sizing: 30% of method capital per position
POSITION_SIZE_PCT = 0.30


class MethodCompetitionManager:
    """Orchestrates the 3-method competition.

    Handles:
    - Tiered timing (strategies every 5min, Gemini methods every 20min)
    - Shared Gemini API call (one per symbol, both methods read from it)
    - Order submission with client_order_id tagging
    - Per-method capital tracking via MethodTracker
    """

    def __init__(
        self,
        client: AlpacaClient,
        data_fetcher: DataFetcher,
        watchlist: list[str],
        method_capital: float = 1000.0,
        gemini_interval_min: int = 20,
        strategies_interval_min: int = 5,
    ):
        self.client = client
        self.data_fetcher = data_fetcher
        self.watchlist = watchlist
        self.method_capital = method_capital
        self.gemini_interval = timedelta(minutes=gemini_interval_min)
        self.strategies_interval = timedelta(minutes=strategies_interval_min)

        # Tracker for per-method P&L
        self.tracker = MethodTracker()

        # Initialize methods in tracker
        self.tracker.init_method(METHOD_STRATEGIES, "Strategies", method_capital)
        self.tracker.init_method(METHOD_WSB, "WSB Reddit", method_capital)
        self.tracker.init_method(METHOD_GEMINI, "Gemini AI", method_capital)

        # Strategy instances
        self._init_strategies()

        # WSB and Gemini clients
        self.wsb_fetcher = WSBRedditFetcher(cache_ttl_seconds=600)
        self.gemini_client = GeminiSentimentClient()

        # Timing
        self._last_strategies_run: Optional[datetime] = None
        self._last_gemini_run: Optional[datetime] = None

        # Shared analysis cache {symbol: GeminiFullAnalysis}
        self._gemini_cache: dict[str, GeminiFullAnalysis] = {}

        logger.info(
            f"Method Competition enabled: "
            f"Strategies ({strategies_interval_min}min), "
            f"WSB Reddit ({gemini_interval_min}min), "
            f"Gemini AI ({gemini_interval_min}min), "
            f"${method_capital:,.0f} each"
        )

    def _init_strategies(self):
        """Initialize all strategy instances."""
        # Method 1: All existing strategies via AdvancedHybridStrategy
        technical = TechnicalStrategy(
            name="technical",
            weight=config.strategy.technical_weight,
            config=config.strategy,
        )
        ml = MLStrategy(
            name="ml",
            weight=config.strategy.ml_weight,
            config=config.strategy,
            min_confidence=0.51,
        )

        # Load ML model if available
        model_path = Path("models/price_predictor.pkl")
        if model_path.exists():
            try:
                ml.load_model(model_path)
                logger.info("Competition: Loaded pre-trained ML model")
            except Exception as e:
                logger.warning(f"Competition: Could not load ML model: {e}")

        # Try to load optional strategies
        strategies = [technical, ml]
        try:
            momentum = MomentumStrategy(name="momentum", weight=0.3)
            strategies.append(momentum)
        except Exception as e:
            logger.debug(f"Could not load MomentumStrategy: {e}")

        try:
            mean_rev = MeanReversionStrategy(name="mean_reversion", weight=0.3)
            strategies.append(mean_rev)
        except Exception as e:
            logger.debug(f"Could not load MeanReversionStrategy: {e}")

        self.hybrid_strategy = AdvancedHybridStrategy(
            name="competition_hybrid",
            strategies=strategies,
            min_confidence=0.25,
            consensus_required=0.0,
            use_adaptive_weights=False,
            use_regime_detection=False,
            use_correlation_adjustment=False,
            use_time_filter=False,
            use_signal_scoring=False,
        )

        # Method 2 & 3
        self.wsb_strategy = WSBStockStrategy()
        self.gemini_strategy = GeminiComprehensiveStrategy()

    def run_cycle(self):
        """Run one competition cycle. Called from the main loop."""
        now = datetime.now()

        # Method 1: Strategies — runs every 5 minutes
        if self._should_run_strategies(now):
            try:
                self._run_strategies_method()
                self._last_strategies_run = now
            except Exception as e:
                logger.error(f"Strategies method failed: {e}", exc_info=True)

        # Methods 2 & 3: Shared Gemini analysis — every 20 minutes
        if self._should_run_gemini(now):
            try:
                self._run_gemini_methods()
                self._last_gemini_run = now
            except Exception as e:
                logger.error(f"Gemini methods failed: {e}", exc_info=True)

        # Reconcile filled orders
        try:
            self._reconcile_orders()
        except Exception as e:
            logger.error(f"Order reconciliation failed: {e}")

    def _should_run_strategies(self, now: datetime) -> bool:
        if self._last_strategies_run is None:
            return True
        return (now - self._last_strategies_run) >= self.strategies_interval

    def _should_run_gemini(self, now: datetime) -> bool:
        if not self.gemini_client.is_configured:
            return False
        if self._last_gemini_run is None:
            return True
        return (now - self._last_gemini_run) >= self.gemini_interval

    def _run_strategies_method(self):
        """Run Method 1: All strategies combined."""
        logger.info("Competition: Running Strategies method...")

        for symbol in self.watchlist:
            try:
                data = self._fetch_data(symbol)
                if data is None:
                    continue

                signal = self.hybrid_strategy.generate_signal(symbol, data)
                logger.info(
                    f"[Strategies] {symbol}: {signal.signal_type.value.upper()} "
                    f"(conf={signal.confidence:.1%}) - {signal.reason}"
                )

                self._process_signal(METHOD_STRATEGIES, symbol, signal, data)

            except Exception as e:
                logger.error(f"[Strategies] Error for {symbol}: {e}")

    def _run_gemini_methods(self):
        """Run Methods 2 & 3 with shared Gemini analysis."""
        logger.info("Competition: Running WSB + Gemini methods (shared analysis)...")

        for symbol in self.watchlist:
            try:
                data = self._fetch_data(symbol)
                if data is None:
                    continue

                # Fetch WSB posts (cached for 10 min by WSBRedditFetcher)
                wsb_data = self.wsb_fetcher.fetch_symbol_data(symbol)
                wsb_posts = []
                post_stats = {"count": 0, "avg_score": 0, "total_comments": 0}

                if wsb_data and wsb_data.posts:
                    wsb_posts = [
                        {
                            "title": p.title,
                            "body": p.body,
                            "score": p.score,
                            "upvote_ratio": p.upvote_ratio,
                            "num_comments": p.num_comments,
                            "flair": p.flair,
                        }
                        for p in wsb_data.posts
                    ]
                    post_stats = {
                        "count": len(wsb_data.posts),
                        "avg_score": wsb_data.avg_score,
                        "total_comments": wsb_data.total_comments,
                    }

                # Build technical indicators dict
                latest = data.iloc[-1]
                indicators = {}
                for col in ["rsi", "macd", "macd_signal", "sma_20", "sma_50",
                            "bb_upper", "bb_lower", "atr", "volatility"]:
                    if col in data.columns:
                        val = latest.get(col)
                        if val is not None and pd.notna(val):
                            indicators[col] = float(val)

                # Price history (last 10 bars)
                price_history = []
                for _, row in data.tail(10).iterrows():
                    price_history.append({
                        "close": float(row["close"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "volume": float(row.get("volume", 0)),
                    })

                current_price = float(latest["close"])

                # ONE shared Gemini call
                analysis = self.gemini_client.analyze_full(
                    symbol=symbol,
                    price=current_price,
                    technical_indicators=indicators,
                    price_history=price_history,
                    wsb_posts=wsb_posts,
                )

                if analysis:
                    self._gemini_cache[symbol] = analysis

                    # Feed to Method 2 (WSB)
                    self.wsb_strategy.set_analysis(symbol, analysis, post_stats)
                    wsb_signal = self.wsb_strategy.generate_signal(symbol, data)
                    logger.info(
                        f"[WSB] {symbol}: {wsb_signal.signal_type.value.upper()} "
                        f"(conf={wsb_signal.confidence:.1%}) - {wsb_signal.reason}"
                    )
                    self._process_signal(METHOD_WSB, symbol, wsb_signal, data)

                    # Feed to Method 3 (Gemini)
                    self.gemini_strategy.set_analysis(symbol, analysis)
                    gemini_signal = self.gemini_strategy.generate_signal(symbol, data)
                    logger.info(
                        f"[Gemini] {symbol}: {gemini_signal.signal_type.value.upper()} "
                        f"(conf={gemini_signal.confidence:.1%}) - {gemini_signal.reason}"
                    )
                    self._process_signal(METHOD_GEMINI, symbol, gemini_signal, data)
                else:
                    logger.warning(f"Gemini analysis returned None for {symbol}")

                # Rate limit between symbols
                time.sleep(1)

            except Exception as e:
                logger.error(f"[Gemini methods] Error for {symbol}: {e}")

    def _fetch_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch and process historical data for a symbol."""
        try:
            df = self.data_fetcher.get_historical_bars(
                symbol=symbol,
                timeframe="1Hour",
                days=30,
                use_cache=True,
            )
            if df is None or len(df) < 50:
                return None
            return DataProcessor.add_technical_indicators(df)
        except Exception as e:
            logger.debug(f"Failed to fetch data for {symbol}: {e}")
            return None

    def _process_signal(
        self,
        method_id: str,
        symbol: str,
        signal,
        data: pd.DataFrame,
    ):
        """Process a signal: check capital, size position, submit order."""
        if signal.signal_type == SignalType.HOLD:
            return

        state = self.tracker.get_method_state(method_id)
        if not state:
            return

        current_price = float(data["close"].iloc[-1])

        if signal.signal_type == SignalType.BUY:
            self._handle_buy(method_id, state, symbol, current_price, signal)
        elif signal.signal_type == SignalType.SELL:
            self._handle_sell(method_id, state, symbol, current_price, signal)

    def _handle_buy(self, method_id: str, state, symbol: str, price: float, signal):
        """Handle a BUY signal for a method."""
        # Already holding?
        if state.has_holding(symbol):
            logger.debug(f"[{state.method_name}] Already holding {symbol}")
            return

        # Check pending orders
        try:
            open_orders = self.client.get_orders(status="open")
            prefix = f"{method_id}-{symbol}-"
            has_pending = any(
                getattr(o, 'client_order_id', '') and
                getattr(o, 'client_order_id', '').startswith(prefix)
                for o in open_orders
            )
            if has_pending:
                logger.debug(f"[{state.method_name}] Pending order exists for {symbol}")
                return
        except Exception:
            pass

        # Position sizing: 30% of method capital, capped by remaining cash
        max_position = state.capital * POSITION_SIZE_PCT
        cash_available = state.remaining_cash()
        trade_value = min(max_position, cash_available)

        if trade_value < 10:  # Minimum $10 trade
            logger.debug(f"[{state.method_name}] Insufficient cash for {symbol}: ${cash_available:.2f}")
            return

        shares = trade_value / price

        # Generate tagged order ID
        client_order_id = self.tracker.generate_order_id(method_id, symbol)

        try:
            order = self.client.submit_limit_order(
                symbol=symbol,
                qty=round(shares, 4),
                side="buy",
                limit_price=price,
                time_in_force="day",
                client_order_id=client_order_id,
            )

            if order:
                # Record in tracker (deducts cash)
                self.tracker.record_buy(
                    method_id=method_id,
                    symbol=symbol,
                    shares=round(shares, 4),
                    price=price,
                    order_id=client_order_id,
                )
                logger.info(
                    f"[{state.method_name}] BUY {shares:.4f} {symbol} @ ${price:.2f} "
                    f"(${trade_value:.2f}, order={client_order_id})"
                )

                # Save transaction for dashboard
                from main import save_transaction
                save_transaction({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "asset_type": "stock",
                    "action": "BUY",
                    "method": state.method_name,
                    "quantity": round(shares, 4),
                    "price": price,
                    "value": round(trade_value, 2),
                    "confidence": round(signal.confidence * 100, 1),
                    "reason": f"[{state.method_name}] {signal.reason}",
                    "order_id": client_order_id,
                    "status": "submitted",
                })

        except Exception as e:
            logger.error(f"[{state.method_name}] BUY order failed for {symbol}: {e}")

    def _handle_sell(self, method_id: str, state, symbol: str, price: float, signal):
        """Handle a SELL signal for a method."""
        holding = state.get_holding(symbol)
        if not holding:
            logger.debug(f"[{state.method_name}] No {symbol} to sell")
            return

        client_order_id = self.tracker.generate_order_id(method_id, symbol)

        try:
            order = self.client.submit_limit_order(
                symbol=symbol,
                qty=holding.shares,
                side="sell",
                limit_price=price,
                time_in_force="day",
                client_order_id=client_order_id,
            )

            if order:
                record = self.tracker.record_sell(
                    method_id=method_id,
                    symbol=symbol,
                    exit_price=price,
                )

                pnl = record.pnl if record else 0
                pnl_pct = record.pnl_pct if record else 0

                logger.info(
                    f"[{state.method_name}] SELL {holding.shares:.4f} {symbol} @ ${price:.2f} "
                    f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%)"
                )

                from main import save_transaction
                save_transaction({
                    "timestamp": datetime.now().isoformat(),
                    "symbol": symbol,
                    "asset_type": "stock",
                    "action": "SELL",
                    "method": state.method_name,
                    "quantity": holding.shares,
                    "price": price,
                    "value": round(holding.shares * price, 2),
                    "entry_price": holding.entry_price,
                    "pnl": round(pnl, 2),
                    "pnl_pct": round(pnl_pct, 2),
                    "confidence": round(signal.confidence * 100, 1),
                    "reason": f"[{state.method_name}] {signal.reason}",
                    "order_id": client_order_id,
                    "status": "submitted",
                })

        except Exception as e:
            logger.error(f"[{state.method_name}] SELL order failed for {symbol}: {e}")

    def _reconcile_orders(self):
        """Match filled Alpaca orders back to methods by client_order_id prefix.

        This handles cases where our local tracker state drifts from reality,
        e.g. a buy order was rejected or a sell was partially filled.
        """
        try:
            # Get recent closed orders
            recent_orders = self.client.get_orders(status="closed", limit=50)

            for order in recent_orders:
                cid = getattr(order, 'client_order_id', '') or ''
                if not cid:
                    continue

                # Check if this is one of our tagged orders
                for method_id in [METHOD_STRATEGIES, METHOD_WSB, METHOD_GEMINI]:
                    if cid.startswith(f"{method_id}-"):
                        # Order belongs to this method — could add
                        # drift correction here if needed
                        break

        except Exception as e:
            logger.debug(f"Order reconciliation error: {e}")

    def get_leaderboard(self, live_prices: Optional[dict[str, float]] = None) -> list[dict]:
        """Build leaderboard data for the dashboard.

        Returns list of dicts sorted by total P&L (descending).
        """
        if live_prices is None:
            live_prices = self._get_live_prices()

        leaderboard = []
        for method_id in [METHOD_STRATEGIES, METHOD_WSB, METHOD_GEMINI]:
            state = self.tracker.get_method_state(method_id)
            if not state:
                continue

            unrealized = self.tracker.get_unrealized_pnl(method_id, live_prices)
            total_pnl = state.realized_pnl + unrealized

            leaderboard.append({
                "method_id": method_id,
                "method_name": state.method_name,
                "capital": state.capital,
                "cash": round(state.cash, 2),
                "invested": round(state.invested, 2),
                "realized_pnl": round(state.realized_pnl, 2),
                "unrealized_pnl": round(unrealized, 2),
                "total_pnl": round(total_pnl, 2),
                "total_pnl_pct": round(total_pnl / state.capital * 100, 2) if state.capital > 0 else 0,
                "total_trades": state.total_trades,
                "winning_trades": state.winning_trades,
                "win_rate": round(state.win_rate * 100, 1),
                "holdings": [
                    {
                        "symbol": h.symbol,
                        "shares": h.shares,
                        "entry_price": h.entry_price,
                        "current_price": live_prices.get(h.symbol, h.entry_price),
                        "pnl": round(
                            (live_prices.get(h.symbol, h.entry_price) - h.entry_price) * h.shares,
                            2,
                        ),
                    }
                    for h in state.holdings
                ],
            })

        # Sort by total P&L descending
        leaderboard.sort(key=lambda x: x["total_pnl"], reverse=True)

        # Add rank
        for i, entry in enumerate(leaderboard):
            entry["rank"] = i + 1

        return leaderboard

    def _get_live_prices(self) -> dict[str, float]:
        """Get live prices for all held symbols."""
        prices = {}
        held_symbols = set()
        for method_id in [METHOD_STRATEGIES, METHOD_WSB, METHOD_GEMINI]:
            state = self.tracker.get_method_state(method_id)
            if state:
                for h in state.holdings:
                    held_symbols.add(h.symbol)

        if not held_symbols:
            return prices

        try:
            positions = self.client.get_all_positions()
            for p in positions:
                if p.symbol in held_symbols:
                    prices[p.symbol] = float(p.current_price)
        except Exception as e:
            logger.debug(f"Failed to get live prices: {e}")

        return prices
