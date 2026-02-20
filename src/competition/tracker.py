"""Per-method P&L tracking for the competition system.

Tracks holdings, realized/unrealized P&L, and trade history for each
trading method independently. Uses a JSON file as persistent storage.
"""

import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
TRACKER_FILE = PROJECT_ROOT / "logs" / "method_tracker.json"


@dataclass
class MethodHolding:
    """A single stock holding for a method."""
    symbol: str
    shares: float
    entry_price: float
    entry_time: str
    order_id: str


@dataclass
class TradeRecord:
    """Record of a completed (closed) trade."""
    symbol: str
    shares: float
    entry_price: float
    exit_price: float
    entry_time: str
    exit_time: str
    pnl: float
    pnl_pct: float


@dataclass
class MethodState:
    """Full state for one trading method."""
    method_id: str
    method_name: str
    capital: float  # Starting capital allocation
    cash: float  # Current cash available
    holdings: list[MethodHolding] = field(default_factory=list)
    realized_pnl: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    trade_history: list[TradeRecord] = field(default_factory=list)

    @property
    def invested(self) -> float:
        """Total cost basis of current holdings."""
        return sum(h.shares * h.entry_price for h in self.holdings)

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    def get_holding(self, symbol: str) -> Optional[MethodHolding]:
        """Get holding for a symbol, or None."""
        for h in self.holdings:
            if h.symbol == symbol:
                return h
        return None

    def has_holding(self, symbol: str) -> bool:
        return self.get_holding(symbol) is not None

    def remaining_cash(self) -> float:
        return self.cash


class MethodTracker:
    """Tracks per-method positions, P&L, and trade history.

    Persists to JSON file so state survives bot restarts.
    """

    def __init__(self, tracker_file: Path = TRACKER_FILE):
        self.tracker_file = tracker_file
        self.methods: dict[str, MethodState] = {}
        self._load()

    def init_method(self, method_id: str, method_name: str, capital: float):
        """Initialize a method if not already tracked."""
        if method_id not in self.methods:
            self.methods[method_id] = MethodState(
                method_id=method_id,
                method_name=method_name,
                capital=capital,
                cash=capital,
            )
            self._save()
            logger.info(f"Initialized method '{method_name}' with ${capital:,.2f}")
        else:
            logger.debug(f"Method '{method_name}' already initialized")

    def record_buy(
        self,
        method_id: str,
        symbol: str,
        shares: float,
        price: float,
        order_id: str,
    ) -> bool:
        """Record a buy and deduct cash. Returns True if successful."""
        state = self.methods.get(method_id)
        if not state:
            logger.error(f"Unknown method: {method_id}")
            return False

        cost = shares * price
        if cost > state.cash + 0.01:  # small tolerance for rounding
            logger.warning(
                f"{state.method_name}: Insufficient cash for {symbol} "
                f"(need ${cost:.2f}, have ${state.cash:.2f})"
            )
            return False

        # Don't buy if already holding this symbol
        if state.has_holding(symbol):
            logger.debug(f"{state.method_name}: Already holding {symbol}")
            return False

        state.holdings.append(MethodHolding(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            entry_time=datetime.now().isoformat(),
            order_id=order_id,
        ))
        state.cash -= cost
        self._save()

        logger.info(
            f"{state.method_name}: BUY {shares:.4f} {symbol} @ ${price:.2f} "
            f"(${cost:.2f}, cash remaining: ${state.cash:.2f})"
        )
        return True

    def record_sell(
        self,
        method_id: str,
        symbol: str,
        exit_price: float,
    ) -> Optional[TradeRecord]:
        """Record a sell. Returns TradeRecord or None if no holding found."""
        state = self.methods.get(method_id)
        if not state:
            logger.error(f"Unknown method: {method_id}")
            return None

        holding = state.get_holding(symbol)
        if not holding:
            logger.debug(f"{state.method_name}: No holding for {symbol}")
            return None

        # Calculate P&L
        proceeds = holding.shares * exit_price
        cost = holding.shares * holding.entry_price
        pnl = proceeds - cost
        pnl_pct = (exit_price / holding.entry_price - 1) * 100

        record = TradeRecord(
            symbol=symbol,
            shares=holding.shares,
            entry_price=holding.entry_price,
            exit_price=exit_price,
            entry_time=holding.entry_time,
            exit_time=datetime.now().isoformat(),
            pnl=round(pnl, 2),
            pnl_pct=round(pnl_pct, 2),
        )

        # Update state
        state.holdings = [h for h in state.holdings if h.symbol != symbol]
        state.cash += proceeds
        state.realized_pnl += pnl
        state.total_trades += 1
        if pnl > 0:
            state.winning_trades += 1

        # Keep last 100 trades
        state.trade_history.append(record)
        state.trade_history = state.trade_history[-100:]

        self._save()

        logger.info(
            f"{state.method_name}: SELL {record.shares:.4f} {symbol} @ ${exit_price:.2f} "
            f"P&L: ${pnl:+.2f} ({pnl_pct:+.1f}%), cash: ${state.cash:.2f}"
        )
        return record

    def get_method_state(self, method_id: str) -> Optional[MethodState]:
        return self.methods.get(method_id)

    def get_unrealized_pnl(self, method_id: str, live_prices: dict[str, float]) -> float:
        """Calculate unrealized P&L using live prices."""
        state = self.methods.get(method_id)
        if not state:
            return 0.0
        total = 0.0
        for h in state.holdings:
            if h.symbol in live_prices:
                total += (live_prices[h.symbol] - h.entry_price) * h.shares
        return total

    def get_total_pnl(self, method_id: str, live_prices: dict[str, float]) -> float:
        """Realized + unrealized P&L."""
        state = self.methods.get(method_id)
        if not state:
            return 0.0
        return state.realized_pnl + self.get_unrealized_pnl(method_id, live_prices)

    def get_all_method_ids(self) -> list[str]:
        return list(self.methods.keys())

    def generate_order_id(self, method_id: str, symbol: str) -> str:
        """Generate a tagged client_order_id for Alpaca."""
        short_uuid = uuid.uuid4().hex[:8]
        return f"{method_id}-{symbol}-{short_uuid}"

    def _save(self):
        """Persist state to JSON."""
        self.tracker_file.parent.mkdir(parents=True, exist_ok=True)
        data = {}
        for mid, state in self.methods.items():
            data[mid] = {
                "method_id": state.method_id,
                "method_name": state.method_name,
                "capital": state.capital,
                "cash": round(state.cash, 4),
                "realized_pnl": round(state.realized_pnl, 4),
                "total_trades": state.total_trades,
                "winning_trades": state.winning_trades,
                "holdings": [asdict(h) for h in state.holdings],
                "trade_history": [asdict(t) for t in state.trade_history],
            }
        with open(self.tracker_file, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load(self):
        """Load state from JSON if it exists."""
        if not self.tracker_file.exists():
            return
        try:
            with open(self.tracker_file, "r") as f:
                data = json.load(f)
            for mid, d in data.items():
                holdings = [MethodHolding(**h) for h in d.get("holdings", [])]
                history = [TradeRecord(**t) for t in d.get("trade_history", [])]
                self.methods[mid] = MethodState(
                    method_id=d["method_id"],
                    method_name=d["method_name"],
                    capital=d["capital"],
                    cash=d["cash"],
                    holdings=holdings,
                    realized_pnl=d.get("realized_pnl", 0.0),
                    total_trades=d.get("total_trades", 0),
                    winning_trades=d.get("winning_trades", 0),
                    trade_history=history,
                )
            logger.info(f"Loaded tracker state: {len(self.methods)} methods")
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            logger.warning(f"Failed to load tracker state: {e}")
            self.methods = {}
