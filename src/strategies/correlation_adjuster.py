"""
Strategy correlation adjustment to prevent redundant weighting.
"""

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from .base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class CorrelationMatrix:
    """Strategy signal correlation matrix."""
    strategy_names: list[str]
    correlations: np.ndarray  # NxN matrix
    sample_count: int


class CorrelationAdjuster:
    """
    Adjusts strategy weights based on signal correlation.

    If two strategies consistently give the same signals, their combined
    weight should be reduced to prevent double-counting the same information.
    """

    def __init__(
        self,
        lookback_signals: int = 100,
        min_samples: int = 20,
        correlation_penalty: float = 0.5,
    ):
        """
        Args:
            lookback_signals: Number of signals to track for correlation
            min_samples: Minimum signals before applying correlation adjustment
            correlation_penalty: How much to penalize correlated strategies (0-1)
        """
        self.lookback_signals = lookback_signals
        self.min_samples = min_samples
        self.correlation_penalty = correlation_penalty

        # Signal history per strategy: strategy_name -> deque of signal values
        self._signal_history: dict[str, deque[float]] = {}

    def record_signal(self, strategy_name: str, signal: Signal) -> None:
        """Record a signal for correlation tracking."""
        if strategy_name not in self._signal_history:
            self._signal_history[strategy_name] = deque(maxlen=self.lookback_signals)

        # Convert signal to numeric value
        signal_value = self._signal_to_numeric(signal)
        self._signal_history[strategy_name].append(signal_value)

    def record_signals(self, signals: dict[str, Signal]) -> None:
        """Record multiple signals at once."""
        for strategy_name, signal in signals.items():
            self.record_signal(strategy_name, signal)

    def _signal_to_numeric(self, signal: Signal) -> float:
        """Convert signal to numeric value for correlation."""
        if signal.signal_type == SignalType.BUY:
            return signal.confidence
        elif signal.signal_type == SignalType.SELL:
            return -signal.confidence
        else:
            return 0.0

    def calculate_correlation_matrix(self) -> CorrelationMatrix:
        """Calculate pairwise correlation between all strategies."""
        strategy_names = list(self._signal_history.keys())
        n = len(strategy_names)

        if n < 2:
            return CorrelationMatrix(
                strategy_names=strategy_names,
                correlations=np.eye(n) if n > 0 else np.array([[]]),
                sample_count=0,
            )

        # Find minimum common length
        min_length = min(len(self._signal_history[s]) for s in strategy_names)

        if min_length < self.min_samples:
            return CorrelationMatrix(
                strategy_names=strategy_names,
                correlations=np.eye(n),
                sample_count=min_length,
            )

        # Build signal matrix
        signal_matrix = np.zeros((min_length, n))
        for i, name in enumerate(strategy_names):
            signals = list(self._signal_history[name])[-min_length:]
            signal_matrix[:, i] = signals

        # Calculate correlation matrix
        correlations = np.corrcoef(signal_matrix.T)

        # Handle NaN (can occur with constant signals)
        correlations = np.nan_to_num(correlations, nan=0.0)

        return CorrelationMatrix(
            strategy_names=strategy_names,
            correlations=correlations,
            sample_count=min_length,
        )

    def get_adjusted_weights(
        self,
        base_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Adjust weights based on strategy correlations.

        Reduces weight of highly correlated strategies to prevent
        double-counting similar information.

        Args:
            base_weights: Original strategy weights

        Returns:
            Adjusted weights
        """
        corr_matrix = self.calculate_correlation_matrix()

        if corr_matrix.sample_count < self.min_samples:
            logger.debug("Insufficient samples for correlation adjustment")
            return base_weights

        adjusted = base_weights.copy()
        strategy_names = corr_matrix.strategy_names

        # For each strategy, calculate correlation penalty
        for i, name_i in enumerate(strategy_names):
            if name_i not in adjusted:
                continue

            # Sum of correlations with other strategies
            total_correlation = 0.0
            count = 0

            for j, name_j in enumerate(strategy_names):
                if i == j or name_j not in adjusted:
                    continue

                corr = abs(corr_matrix.correlations[i, j])
                if corr > 0.5:  # Only penalize high correlations
                    total_correlation += corr
                    count += 1

            if count > 0:
                avg_correlation = total_correlation / count

                # Apply penalty based on correlation
                # High correlation = reduce weight
                penalty = 1 - (avg_correlation * self.correlation_penalty)
                penalty = max(0.3, penalty)  # Don't reduce below 30%

                adjusted[name_i] *= penalty

                if avg_correlation > 0.7:
                    logger.info(
                        f"Strategy {name_i} has high correlation "
                        f"({avg_correlation:.2f}), weight reduced by {(1-penalty)*100:.0f}%"
                    )

        # Normalize weights
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v / total for k, v in adjusted.items()}

        return adjusted

    def get_diversification_score(self) -> float:
        """
        Calculate overall portfolio diversification score (0-1).

        Higher score = more diversified (less correlated) strategies.
        """
        corr_matrix = self.calculate_correlation_matrix()

        if corr_matrix.sample_count < self.min_samples:
            return 1.0  # Assume diversified if insufficient data

        n = len(corr_matrix.strategy_names)
        if n < 2:
            return 1.0

        # Average off-diagonal correlation
        correlations = corr_matrix.correlations
        off_diagonal = correlations[np.triu_indices(n, k=1)]
        avg_correlation = np.mean(np.abs(off_diagonal))

        # Convert to diversification score (0 correlation = 1.0 score)
        return float(1 - avg_correlation)

    def get_correlation_report(self) -> dict:
        """Get detailed correlation analysis."""
        corr_matrix = self.calculate_correlation_matrix()

        report = {
            "sample_count": corr_matrix.sample_count,
            "diversification_score": self.get_diversification_score(),
            "strategy_pairs": [],
        }

        n = len(corr_matrix.strategy_names)
        for i in range(n):
            for j in range(i + 1, n):
                report["strategy_pairs"].append({
                    "strategy_1": corr_matrix.strategy_names[i],
                    "strategy_2": corr_matrix.strategy_names[j],
                    "correlation": float(corr_matrix.correlations[i, j]),
                })

        # Sort by absolute correlation (highest first)
        report["strategy_pairs"].sort(
            key=lambda x: abs(x["correlation"]),
            reverse=True,
        )

        return report


class SignalAgreementAnalyzer:
    """
    Analyzes when strategies agree vs disagree.

    Useful for understanding strategy behavior and potential conflicts.
    """

    def __init__(self):
        self._agreement_history: list[dict] = []

    def record_signals(
        self,
        signals: dict[str, Signal],
        symbol: str,
    ) -> dict:
        """
        Record and analyze signal agreement.

        Returns analysis of current signal agreement.
        """
        # Convert to directions
        directions = {}
        for name, signal in signals.items():
            if signal.signal_type == SignalType.BUY:
                directions[name] = 1
            elif signal.signal_type == SignalType.SELL:
                directions[name] = -1
            else:
                directions[name] = 0

        # Calculate agreement metrics
        values = list(directions.values())
        buy_count = sum(1 for v in values if v == 1)
        sell_count = sum(1 for v in values if v == -1)
        hold_count = sum(1 for v in values if v == 0)
        total = len(values)

        # Agreement score: 1 if all agree, 0 if split
        if total == 0:
            agreement_score = 0
        elif buy_count == total or sell_count == total or hold_count == total:
            agreement_score = 1.0
        else:
            max_agreement = max(buy_count, sell_count, hold_count)
            agreement_score = max_agreement / total

        analysis = {
            "symbol": symbol,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "agreement_score": agreement_score,
            "has_conflict": buy_count > 0 and sell_count > 0,
            "unanimous": agreement_score == 1.0,
            "directions": directions,
        }

        self._agreement_history.append(analysis)

        return analysis

    def get_conflict_resolution(
        self,
        signals: dict[str, Signal],
        weights: dict[str, float],
    ) -> Signal:
        """
        Resolve conflicting signals using weighted voting.

        When strategies disagree, this provides a principled way to
        determine the final signal.
        """
        if not signals:
            # Return a neutral signal with no specific symbol
            return Signal(
                symbol="",
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="No signals to resolve",
            )

        # Calculate weighted scores
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0

        total_weight = sum(weights.get(name, 1.0) for name in signals)

        for name, signal in signals.items():
            weight = weights.get(name, 1.0) / total_weight
            weighted_confidence = signal.confidence * weight

            if signal.signal_type == SignalType.BUY:
                buy_score += weighted_confidence
            elif signal.signal_type == SignalType.SELL:
                sell_score += weighted_confidence
            else:
                hold_score += weighted_confidence

        # Determine winner
        scores = {"buy": buy_score, "sell": sell_score, "hold": hold_score}
        winner = max(scores, key=scores.get)
        winner_score = scores[winner]

        # Check for strong conflict
        if buy_score > 0.3 and sell_score > 0.3:
            # Strong disagreement - default to hold with reduced confidence
            return Signal(
                symbol=list(signals.values())[0].symbol,
                signal_type=SignalType.HOLD,
                confidence=winner_score * 0.5,
                reason=f"Conflict: BUY={buy_score:.2f} vs SELL={sell_score:.2f}",
            )

        signal_type = {
            "buy": SignalType.BUY,
            "sell": SignalType.SELL,
            "hold": SignalType.HOLD,
        }[winner]

        # Find best stop/take profit from agreeing strategies
        agreeing_signals = [
            s for s in signals.values()
            if s.signal_type == signal_type
        ]

        stop_loss = None
        take_profit = None
        if agreeing_signals:
            stop_losses = [s.stop_loss for s in agreeing_signals if s.stop_loss]
            take_profits = [s.take_profit for s in agreeing_signals if s.take_profit]

            if stop_losses:
                stop_loss = np.mean(stop_losses)
            if take_profits:
                take_profit = np.mean(take_profits)

        return Signal(
            symbol=list(signals.values())[0].symbol,
            signal_type=signal_type,
            confidence=winner_score,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"Resolved: {winner.upper()}={winner_score:.2f}",
        )
