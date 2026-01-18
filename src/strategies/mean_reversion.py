"""
Mean reversion trading strategy.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from config import StrategyConfig

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy that trades price deviations from equilibrium.

    Works best in ranging/sideways markets where prices oscillate
    around a mean value.

    Key indicators:
    - Z-score of price vs moving average
    - Bollinger Band position
    - RSI extremes
    - Distance from VWAP
    """

    def __init__(
        self,
        name: str = "mean_reversion",
        weight: float = 1.0,
        config: Optional[StrategyConfig] = None,
        zscore_threshold: float = 2.0,
        zscore_extreme: float = 2.5,
        lookback_period: int = 20,
        holding_period: int = 5,
    ):
        """
        Args:
            name: Strategy name
            weight: Strategy weight
            config: Strategy configuration
            zscore_threshold: Z-score level to consider mean reversion
            zscore_extreme: Extreme z-score for high confidence
            lookback_period: Period for calculating mean and std
            holding_period: Expected holding period in bars
        """
        super().__init__(name, weight)
        self.config = config or StrategyConfig()
        self.zscore_threshold = zscore_threshold
        self.zscore_extreme = zscore_extreme
        self.lookback_period = lookback_period
        self.holding_period = holding_period
        self._is_trained = True

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate mean reversion signal."""
        if len(data) < self.lookback_period + 10:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Insufficient data for mean reversion",
            )

        # Calculate indicators
        indicators = self._calculate_indicators(data)

        # Get current values
        price = float(data["close"].iloc[-1])
        zscore = indicators["zscore"]
        bb_position = indicators["bb_position"]
        rsi = indicators["rsi"]
        distance_from_mean = indicators["distance_from_mean"]

        # Calculate mean reversion score
        mr_score, reasons = self._calculate_mr_score(
            zscore, bb_position, rsi, distance_from_mean
        )

        # Determine signal
        if mr_score > 0.6:
            # Strong oversold - expect bounce
            signal_type = SignalType.BUY
            confidence = min(mr_score, 1.0)

            # Target: return to mean
            target = indicators["mean"]
            stop_loss = price - (price - indicators["lower_band"]) * 0.5
            take_profit = target

        elif mr_score < -0.6:
            # Strong overbought - expect pullback
            signal_type = SignalType.SELL
            confidence = min(abs(mr_score), 1.0)

            target = indicators["mean"]
            stop_loss = price + (indicators["upper_band"] - price) * 0.5
            take_profit = target

        else:
            signal_type = SignalType.HOLD
            confidence = 1 - abs(mr_score)
            stop_loss = None
            take_profit = None
            reasons.append("No mean reversion opportunity")

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="; ".join(reasons),
        )

    def _calculate_indicators(self, data: pd.DataFrame) -> dict:
        """Calculate mean reversion indicators."""
        close = data["close"]

        # Moving average and standard deviation
        mean = close.rolling(self.lookback_period).mean().iloc[-1]
        std = close.rolling(self.lookback_period).std().iloc[-1]

        # Z-score
        current_price = close.iloc[-1]
        zscore = (current_price - mean) / std if std > 0 else 0

        # Bollinger Band position (0 = lower band, 1 = upper band)
        if "bb_lower" in data.columns and "bb_upper" in data.columns:
            bb_lower = data["bb_lower"].iloc[-1]
            bb_upper = data["bb_upper"].iloc[-1]
            bb_range = bb_upper - bb_lower
            bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5
        else:
            bb_lower = mean - 2 * std
            bb_upper = mean + 2 * std
            bb_range = bb_upper - bb_lower
            bb_position = (current_price - bb_lower) / bb_range if bb_range > 0 else 0.5

        # RSI
        if "rsi" in data.columns:
            rsi = data["rsi"].iloc[-1]
        else:
            rsi = self._calculate_rsi(close)

        # Distance from mean (percentage)
        distance_from_mean = (current_price - mean) / mean if mean > 0 else 0

        return {
            "zscore": zscore,
            "bb_position": bb_position,
            "rsi": rsi,
            "distance_from_mean": distance_from_mean,
            "mean": mean,
            "std": std,
            "lower_band": bb_lower,
            "upper_band": bb_upper,
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta).where(delta < 0, 0).rolling(period).mean()

        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs))

        return float(rsi)

    def _calculate_mr_score(
        self,
        zscore: float,
        bb_position: float,
        rsi: float,
        distance_from_mean: float,
    ) -> tuple[float, list[str]]:
        """
        Calculate mean reversion score.

        Positive score = oversold (expect bounce)
        Negative score = overbought (expect pullback)
        """
        scores = []
        reasons = []

        # Z-score component
        if zscore < -self.zscore_extreme:
            scores.append(1.0)
            reasons.append(f"Extreme oversold (Z={zscore:.2f})")
        elif zscore < -self.zscore_threshold:
            scores.append(0.7)
            reasons.append(f"Oversold (Z={zscore:.2f})")
        elif zscore > self.zscore_extreme:
            scores.append(-1.0)
            reasons.append(f"Extreme overbought (Z={zscore:.2f})")
        elif zscore > self.zscore_threshold:
            scores.append(-0.7)
            reasons.append(f"Overbought (Z={zscore:.2f})")
        else:
            scores.append(0)

        # Bollinger Band component
        if bb_position < 0.05:
            scores.append(0.8)
            reasons.append("Price at lower BB")
        elif bb_position < 0.2:
            scores.append(0.5)
        elif bb_position > 0.95:
            scores.append(-0.8)
            reasons.append("Price at upper BB")
        elif bb_position > 0.8:
            scores.append(-0.5)
        else:
            scores.append(0)

        # RSI component
        if rsi < 25:
            scores.append(0.9)
            reasons.append(f"RSI extreme oversold ({rsi:.0f})")
        elif rsi < 30:
            scores.append(0.6)
            reasons.append(f"RSI oversold ({rsi:.0f})")
        elif rsi > 75:
            scores.append(-0.9)
            reasons.append(f"RSI extreme overbought ({rsi:.0f})")
        elif rsi > 70:
            scores.append(-0.6)
            reasons.append(f"RSI overbought ({rsi:.0f})")
        else:
            scores.append(0)

        # Distance from mean component
        if distance_from_mean < -0.05:  # 5% below mean
            scores.append(0.5)
        elif distance_from_mean > 0.05:  # 5% above mean
            scores.append(-0.5)
        else:
            scores.append(0)

        # Weighted average
        weights = [0.35, 0.25, 0.25, 0.15]  # Z-score, BB, RSI, distance
        final_score = sum(s * w for s, w in zip(scores, weights))

        return final_score, reasons

    def calculate_expected_return(self, data: pd.DataFrame) -> float:
        """
        Calculate expected return based on mean reversion.

        Estimates how much price might move back toward the mean.
        """
        indicators = self._calculate_indicators(data)

        current_price = float(data["close"].iloc[-1])
        mean = indicators["mean"]

        # Expected return is move toward mean
        expected_move = (mean - current_price) / current_price

        # Dampen expectation (price rarely returns fully to mean)
        dampening = 0.5
        expected_return = expected_move * dampening

        return expected_return

    def get_mean_reversion_metrics(self, data: pd.DataFrame) -> dict:
        """Get detailed mean reversion metrics."""
        indicators = self._calculate_indicators(data)

        # Historical mean reversion rate
        close = data["close"]
        mean = close.rolling(self.lookback_period).mean()

        # Calculate how often price reverts after extreme deviations
        zscore_history = (close - mean) / close.rolling(self.lookback_period).std()

        extreme_oversold = zscore_history < -self.zscore_threshold
        extreme_overbought = zscore_history > self.zscore_threshold

        # Check reversion after extremes
        reversions_from_oversold = 0
        total_oversold = 0

        for i in range(len(close) - self.holding_period):
            if extreme_oversold.iloc[i]:
                total_oversold += 1
                future_return = (close.iloc[i + self.holding_period] - close.iloc[i]) / close.iloc[i]
                if future_return > 0:
                    reversions_from_oversold += 1

        reversion_rate = reversions_from_oversold / total_oversold if total_oversold > 0 else 0

        return {
            "current_zscore": indicators["zscore"],
            "bb_position": indicators["bb_position"],
            "rsi": indicators["rsi"],
            "distance_from_mean_pct": indicators["distance_from_mean"] * 100,
            "historical_reversion_rate": reversion_rate,
            "mean_price": indicators["mean"],
            "std_price": indicators["std"],
        }
