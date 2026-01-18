"""
Strategy performance tracking for adaptive weight adjustment.
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Record of a trade outcome."""
    strategy_name: str
    symbol: str
    signal_type: str  # 'buy' or 'sell'
    confidence: float
    entry_price: float
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    is_closed: bool = False


class PerformanceTracker:
    """
    Tracks strategy performance over time for adaptive weighting.

    Maintains rolling metrics:
    - Win rate
    - Average return
    - Sharpe ratio
    - Signal accuracy
    """

    def __init__(self, lookback_trades: int = 50, min_trades: int = 10):
        """
        Args:
            lookback_trades: Number of recent trades to consider
            min_trades: Minimum trades before adjusting weights
        """
        self.lookback_trades = lookback_trades
        self.min_trades = min_trades

        # Trade history per strategy
        self._trades: dict[str, deque[TradeResult]] = {}

        # Signal history (for accuracy tracking)
        self._signals: dict[str, deque[dict]] = {}

    def record_signal(
        self,
        strategy_name: str,
        symbol: str,
        signal_type: str,
        confidence: float,
        price: float,
    ) -> None:
        """Record a signal for later accuracy evaluation."""
        if strategy_name not in self._signals:
            self._signals[strategy_name] = deque(maxlen=self.lookback_trades * 2)

        self._signals[strategy_name].append({
            "symbol": symbol,
            "signal_type": signal_type,
            "confidence": confidence,
            "price": price,
            "timestamp": datetime.now(),
            "future_price": None,  # To be filled later
            "was_correct": None,
        })

    def update_signal_outcomes(
        self,
        symbol: str,
        current_price: float,
        lookback_bars: int = 5,
    ) -> None:
        """
        Update past signals with their outcomes.

        Checks if price moved in the predicted direction.
        """
        for strategy_name, signals in self._signals.items():
            for signal in signals:
                if signal["symbol"] != symbol:
                    continue
                if signal["was_correct"] is not None:
                    continue

                # Check if enough time has passed
                signal["future_price"] = current_price

                price_change = (current_price - signal["price"]) / signal["price"]

                if signal["signal_type"] == "buy":
                    signal["was_correct"] = price_change > 0
                elif signal["signal_type"] == "sell":
                    signal["was_correct"] = price_change < 0
                else:
                    signal["was_correct"] = abs(price_change) < 0.01  # Hold was correct if little movement

    def record_trade(self, trade: TradeResult) -> None:
        """Record a completed trade."""
        if trade.strategy_name not in self._trades:
            self._trades[trade.strategy_name] = deque(maxlen=self.lookback_trades)

        self._trades[trade.strategy_name].append(trade)

    def close_trade(
        self,
        strategy_name: str,
        symbol: str,
        exit_price: float,
    ) -> Optional[TradeResult]:
        """Close an open trade and calculate PnL."""
        if strategy_name not in self._trades:
            return None

        # Find the most recent open trade for this symbol
        for trade in reversed(self._trades[strategy_name]):
            if trade.symbol == symbol and not trade.is_closed:
                trade.exit_price = exit_price
                trade.is_closed = True

                if trade.signal_type == "buy":
                    trade.pnl_pct = (exit_price - trade.entry_price) / trade.entry_price
                else:  # sell/short
                    trade.pnl_pct = (trade.entry_price - exit_price) / trade.entry_price

                trade.pnl = trade.pnl_pct * trade.entry_price  # Simplified, ignores position size

                logger.debug(f"Closed trade: {strategy_name} {symbol} PnL: {trade.pnl_pct:.2%}")
                return trade

        return None

    def get_win_rate(self, strategy_name: str) -> Optional[float]:
        """Get win rate for a strategy."""
        if strategy_name not in self._trades:
            return None

        closed_trades = [t for t in self._trades[strategy_name] if t.is_closed]

        if len(closed_trades) < self.min_trades:
            return None

        wins = sum(1 for t in closed_trades if t.pnl_pct and t.pnl_pct > 0)
        return wins / len(closed_trades)

    def get_average_return(self, strategy_name: str) -> Optional[float]:
        """Get average return per trade."""
        if strategy_name not in self._trades:
            return None

        closed_trades = [t for t in self._trades[strategy_name] if t.is_closed and t.pnl_pct is not None]

        if len(closed_trades) < self.min_trades:
            return None

        return np.mean([t.pnl_pct for t in closed_trades])

    def get_sharpe_ratio(self, strategy_name: str, risk_free_rate: float = 0.02) -> Optional[float]:
        """Calculate rolling Sharpe ratio for a strategy."""
        if strategy_name not in self._trades:
            return None

        closed_trades = [t for t in self._trades[strategy_name] if t.is_closed and t.pnl_pct is not None]

        if len(closed_trades) < self.min_trades:
            return None

        returns = [t.pnl_pct for t in closed_trades]

        mean_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        # Annualize assuming ~252 trading days
        trades_per_year = 252 / (len(closed_trades) / 365) if len(closed_trades) > 0 else 252
        sharpe = (mean_return - risk_free_rate / trades_per_year) / std_return * np.sqrt(trades_per_year)

        return float(sharpe)

    def get_signal_accuracy(self, strategy_name: str) -> Optional[float]:
        """Get signal prediction accuracy."""
        if strategy_name not in self._signals:
            return None

        evaluated = [s for s in self._signals[strategy_name] if s["was_correct"] is not None]

        if len(evaluated) < self.min_trades:
            return None

        correct = sum(1 for s in evaluated if s["was_correct"])
        return correct / len(evaluated)

    def get_strategy_score(self, strategy_name: str) -> float:
        """
        Calculate overall performance score for a strategy.

        Combines multiple metrics into a single score (0-1).
        Higher is better.
        """
        scores = []
        weights = []

        # Win rate (target: 55%+)
        win_rate = self.get_win_rate(strategy_name)
        if win_rate is not None:
            # Normalize: 40% -> 0, 60% -> 1
            normalized = (win_rate - 0.4) / 0.2
            scores.append(max(0, min(1, normalized)))
            weights.append(0.3)

        # Average return (target: 1%+ per trade)
        avg_return = self.get_average_return(strategy_name)
        if avg_return is not None:
            # Normalize: -2% -> 0, 2% -> 1
            normalized = (avg_return + 0.02) / 0.04
            scores.append(max(0, min(1, normalized)))
            weights.append(0.3)

        # Sharpe ratio (target: 1.0+)
        sharpe = self.get_sharpe_ratio(strategy_name)
        if sharpe is not None:
            # Normalize: -1 -> 0, 2 -> 1
            normalized = (sharpe + 1) / 3
            scores.append(max(0, min(1, normalized)))
            weights.append(0.25)

        # Signal accuracy (target: 55%+)
        accuracy = self.get_signal_accuracy(strategy_name)
        if accuracy is not None:
            # Normalize: 40% -> 0, 60% -> 1
            normalized = (accuracy - 0.4) / 0.2
            scores.append(max(0, min(1, normalized)))
            weights.append(0.15)

        if not scores:
            return 0.5  # Default neutral score

        # Weighted average
        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight

        return weighted_score

    def get_adaptive_weight(
        self,
        strategy_name: str,
        base_weight: float,
        min_weight: float = 0.1,
        max_weight: float = 0.6,
    ) -> float:
        """
        Calculate adaptive weight based on performance.

        Args:
            strategy_name: Strategy to get weight for
            base_weight: Original configured weight
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight

        Returns:
            Adjusted weight
        """
        score = self.get_strategy_score(strategy_name)

        # Adjust weight based on score
        # Score 0.5 -> base_weight (neutral)
        # Score 0.0 -> min_weight (poor performance)
        # Score 1.0 -> max_weight (excellent performance)

        if score >= 0.5:
            # Scale up from base to max
            adjustment = (score - 0.5) * 2  # 0 to 1
            weight = base_weight + adjustment * (max_weight - base_weight)
        else:
            # Scale down from base to min
            adjustment = score * 2  # 0 to 1
            weight = min_weight + adjustment * (base_weight - min_weight)

        return weight

    def get_all_metrics(self, strategy_name: str) -> dict:
        """Get all performance metrics for a strategy."""
        return {
            "win_rate": self.get_win_rate(strategy_name),
            "avg_return": self.get_average_return(strategy_name),
            "sharpe_ratio": self.get_sharpe_ratio(strategy_name),
            "signal_accuracy": self.get_signal_accuracy(strategy_name),
            "overall_score": self.get_strategy_score(strategy_name),
            "trade_count": len(self._trades.get(strategy_name, [])),
        }
