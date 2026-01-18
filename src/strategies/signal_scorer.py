"""
Continuous signal strength scoring system.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class SignalScore:
    """
    Continuous signal score with detailed breakdown.

    Score ranges from -1.0 (strong sell) to 1.0 (strong buy).
    """
    symbol: str
    raw_score: float  # -1 to 1
    confidence: float  # 0 to 1

    # Component scores
    direction_score: float  # Buy/sell direction
    magnitude_score: float  # How strong the signal is
    timing_score: float  # Entry timing quality
    risk_reward_score: float  # Risk/reward assessment

    # Metadata
    strategy_name: str
    reasons: list[str]

    @property
    def signal_type(self) -> SignalType:
        """Convert score to discrete signal type."""
        if self.raw_score > 0.2 and self.confidence > 0.5:
            return SignalType.BUY
        elif self.raw_score < -0.2 and self.confidence > 0.5:
            return SignalType.SELL
        return SignalType.HOLD

    @property
    def is_actionable(self) -> bool:
        """Check if signal is strong enough to act on."""
        return abs(self.raw_score) > 0.3 and self.confidence > 0.5

    def to_signal(
        self,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Signal:
        """Convert to traditional Signal object."""
        return Signal(
            symbol=self.symbol,
            signal_type=self.signal_type,
            confidence=self.confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="; ".join(self.reasons),
        )


class SignalScorer:
    """
    Calculates continuous signal scores with multiple components.

    Instead of binary buy/sell/hold, provides a continuous score
    that captures signal conviction and quality.
    """

    def __init__(
        self,
        direction_weight: float = 0.4,
        magnitude_weight: float = 0.25,
        timing_weight: float = 0.2,
        risk_reward_weight: float = 0.15,
    ):
        """
        Args:
            direction_weight: Weight for direction component
            magnitude_weight: Weight for signal magnitude
            timing_weight: Weight for entry timing
            risk_reward_weight: Weight for risk/reward ratio
        """
        self.direction_weight = direction_weight
        self.magnitude_weight = magnitude_weight
        self.timing_weight = timing_weight
        self.risk_reward_weight = risk_reward_weight

    def score_signal(
        self,
        symbol: str,
        data: pd.DataFrame,
        strategy_name: str,
        indicators: Optional[dict] = None,
    ) -> SignalScore:
        """
        Calculate comprehensive signal score.

        Args:
            symbol: Ticker symbol
            data: Price data with indicators
            strategy_name: Name of the strategy generating signal
            indicators: Additional indicator values to consider

        Returns:
            SignalScore with detailed breakdown
        """
        indicators = indicators or {}
        reasons = []

        # 1. Direction score (-1 to 1)
        direction_score, direction_reasons = self._calculate_direction_score(data, indicators)
        reasons.extend(direction_reasons)

        # 2. Magnitude score (0 to 1) - how extreme the indicators are
        magnitude_score, magnitude_reasons = self._calculate_magnitude_score(data, indicators)
        reasons.extend(magnitude_reasons)

        # 3. Timing score (0 to 1) - quality of entry point
        timing_score, timing_reasons = self._calculate_timing_score(data, indicators)
        reasons.extend(timing_reasons)

        # 4. Risk/reward score (0 to 1)
        rr_score, rr_reasons = self._calculate_risk_reward_score(data)
        reasons.extend(rr_reasons)

        # Calculate raw score (weighted direction * magnitude)
        raw_score = direction_score * (
            self.magnitude_weight * magnitude_score +
            self.timing_weight * timing_score +
            self.risk_reward_weight * rr_score +
            self.direction_weight
        )

        # Normalize to -1 to 1
        raw_score = np.clip(raw_score, -1.0, 1.0)

        # Confidence based on indicator agreement and data quality
        confidence = self._calculate_confidence(
            direction_score,
            magnitude_score,
            timing_score,
            rr_score,
            data,
        )

        return SignalScore(
            symbol=symbol,
            raw_score=float(raw_score),
            confidence=float(confidence),
            direction_score=float(direction_score),
            magnitude_score=float(magnitude_score),
            timing_score=float(timing_score),
            risk_reward_score=float(rr_score),
            strategy_name=strategy_name,
            reasons=reasons,
        )

    def _calculate_direction_score(
        self,
        data: pd.DataFrame,
        indicators: dict,
    ) -> tuple[float, list[str]]:
        """
        Calculate direction score from -1 (sell) to 1 (buy).
        """
        scores = []
        reasons = []

        latest = data.iloc[-1]

        # RSI
        if "rsi" in data.columns:
            rsi = float(latest["rsi"])
            if rsi < 30:
                scores.append(0.8)
                reasons.append(f"RSI oversold ({rsi:.1f})")
            elif rsi < 40:
                scores.append(0.3)
            elif rsi > 70:
                scores.append(-0.8)
                reasons.append(f"RSI overbought ({rsi:.1f})")
            elif rsi > 60:
                scores.append(-0.3)
            else:
                scores.append(0)

        # MACD
        if "macd" in data.columns and "macd_signal" in data.columns:
            macd = float(latest["macd"])
            signal = float(latest["macd_signal"])
            histogram = macd - signal

            if histogram > 0:
                scores.append(min(histogram * 10, 1.0))
                if histogram > 0.5:
                    reasons.append("MACD bullish")
            else:
                scores.append(max(histogram * 10, -1.0))
                if histogram < -0.5:
                    reasons.append("MACD bearish")

        # Moving averages
        if "sma_20" in data.columns and "sma_50" in data.columns:
            sma_20 = float(latest["sma_20"])
            sma_50 = float(latest["sma_50"])
            price = float(latest["close"])

            if price > sma_20 > sma_50:
                scores.append(0.6)
                reasons.append("Price above aligned MAs")
            elif price < sma_20 < sma_50:
                scores.append(-0.6)
                reasons.append("Price below aligned MAs")
            elif price > sma_20:
                scores.append(0.2)
            elif price < sma_20:
                scores.append(-0.2)

        # Bollinger Bands
        if "bb_lower" in data.columns and "bb_upper" in data.columns:
            price = float(latest["close"])
            bb_lower = float(latest["bb_lower"])
            bb_upper = float(latest["bb_upper"])
            bb_middle = (bb_upper + bb_lower) / 2

            if price < bb_lower:
                scores.append(0.7)
                reasons.append("Price below lower BB")
            elif price > bb_upper:
                scores.append(-0.7)
                reasons.append("Price above upper BB")
            elif price < bb_middle:
                scores.append(0.1)
            else:
                scores.append(-0.1)

        # Additional indicators from dict
        if "ml_prediction" in indicators:
            pred = indicators["ml_prediction"]
            conf = indicators.get("ml_confidence", 0.5)
            if pred == 1:
                scores.append(conf)
                reasons.append(f"ML predicts UP ({conf:.0%})")
            else:
                scores.append(-conf)
                reasons.append(f"ML predicts DOWN ({conf:.0%})")

        if not scores:
            return 0.0, []

        return float(np.mean(scores)), reasons

    def _calculate_magnitude_score(
        self,
        data: pd.DataFrame,
        indicators: dict,
    ) -> tuple[float, list[str]]:
        """
        Calculate how extreme/strong the signal is (0 to 1).
        """
        scores = []
        reasons = []

        latest = data.iloc[-1]

        # RSI extremity
        if "rsi" in data.columns:
            rsi = float(latest["rsi"])
            if rsi < 20 or rsi > 80:
                scores.append(1.0)
                reasons.append("Extreme RSI")
            elif rsi < 30 or rsi > 70:
                scores.append(0.7)
            elif rsi < 40 or rsi > 60:
                scores.append(0.4)
            else:
                scores.append(0.2)

        # Bollinger Band position
        if "bb_lower" in data.columns and "bb_upper" in data.columns:
            price = float(latest["close"])
            bb_lower = float(latest["bb_lower"])
            bb_upper = float(latest["bb_upper"])
            bb_width = bb_upper - bb_lower

            if bb_width > 0:
                position = (price - bb_lower) / bb_width
                # Further from middle = stronger signal
                extremity = abs(position - 0.5) * 2
                scores.append(extremity)

        # Volume confirmation
        if "volume_ratio" in data.columns:
            vol_ratio = float(latest["volume_ratio"])
            if vol_ratio > 2.0:
                scores.append(1.0)
                reasons.append("High volume confirmation")
            elif vol_ratio > 1.5:
                scores.append(0.7)
            elif vol_ratio > 1.0:
                scores.append(0.5)
            else:
                scores.append(0.3)

        if not scores:
            return 0.5, []

        return float(np.mean(scores)), reasons

    def _calculate_timing_score(
        self,
        data: pd.DataFrame,
        indicators: dict,
    ) -> tuple[float, list[str]]:
        """
        Calculate entry timing quality (0 to 1).
        """
        scores = []
        reasons = []

        if len(data) < 5:
            return 0.5, []

        latest = data.iloc[-1]
        recent = data.iloc[-5:]

        # Reversal detection
        closes = recent["close"].values
        if len(closes) >= 3:
            # Check for potential reversal
            if closes[-1] > closes[-2] < closes[-3]:
                scores.append(0.8)
                reasons.append("Potential reversal point")
            elif closes[-1] < closes[-2] > closes[-3]:
                scores.append(0.8)
                reasons.append("Potential reversal point")
            else:
                scores.append(0.4)

        # Proximity to support/resistance (using recent highs/lows)
        if len(data) >= 20:
            recent_high = data["high"].iloc[-20:].max()
            recent_low = data["low"].iloc[-20:].min()
            price = float(latest["close"])

            range_size = recent_high - recent_low
            if range_size > 0:
                position = (price - recent_low) / range_size

                # Better entry near support (for buy) or resistance (for sell)
                if position < 0.2:
                    scores.append(0.9)
                    reasons.append("Near support")
                elif position > 0.8:
                    scores.append(0.9)
                    reasons.append("Near resistance")
                else:
                    scores.append(0.4)

        # Pullback in trend
        if "sma_20" in data.columns:
            sma = data["sma_20"]
            price = data["close"]

            # In uptrend, pullback to MA is good entry
            if price.iloc[-1] > sma.iloc[-1] and price.iloc[-2] < sma.iloc[-2]:
                scores.append(0.8)
                reasons.append("Pullback to MA in uptrend")

        if not scores:
            return 0.5, []

        return float(np.mean(scores)), reasons

    def _calculate_risk_reward_score(self, data: pd.DataFrame) -> tuple[float, list[str]]:
        """
        Calculate risk/reward potential (0 to 1).
        """
        reasons = []

        if len(data) < 20:
            return 0.5, []

        latest = data.iloc[-1]
        price = float(latest["close"])

        # Use ATR for risk estimation
        if "atr" in data.columns:
            atr = float(latest["atr"])
        else:
            # Estimate ATR
            tr = data["high"] - data["low"]
            atr = tr.rolling(14).mean().iloc[-1]

        # Estimate potential reward using recent range
        recent_high = data["high"].iloc[-20:].max()
        recent_low = data["low"].iloc[-20:].min()

        # Potential reward (distance to opposite extreme)
        potential_upside = recent_high - price
        potential_downside = price - recent_low

        # Risk is typically 2x ATR for stop loss
        risk = 2 * atr

        if risk > 0:
            upside_rr = potential_upside / risk
            downside_rr = potential_downside / risk

            # Score based on better R:R
            best_rr = max(upside_rr, downside_rr)

            if best_rr >= 3:
                score = 1.0
                reasons.append(f"Excellent R:R ({best_rr:.1f}:1)")
            elif best_rr >= 2:
                score = 0.8
                reasons.append(f"Good R:R ({best_rr:.1f}:1)")
            elif best_rr >= 1.5:
                score = 0.6
            elif best_rr >= 1:
                score = 0.4
            else:
                score = 0.2
                reasons.append(f"Poor R:R ({best_rr:.1f}:1)")

            return score, reasons

        return 0.5, []

    def _calculate_confidence(
        self,
        direction: float,
        magnitude: float,
        timing: float,
        risk_reward: float,
        data: pd.DataFrame,
    ) -> float:
        """Calculate overall confidence in the signal."""
        # Start with average of component scores
        base_confidence = (
            abs(direction) * 0.3 +
            magnitude * 0.3 +
            timing * 0.2 +
            risk_reward * 0.2
        )

        # Penalize for conflicting signals
        if abs(direction) < 0.2:
            base_confidence *= 0.7

        # Bonus for strong agreement
        if abs(direction) > 0.5 and magnitude > 0.6 and timing > 0.6:
            base_confidence *= 1.2

        # Data quality factor
        data_factor = min(len(data) / 100, 1.0)
        base_confidence *= (0.7 + 0.3 * data_factor)

        return float(np.clip(base_confidence, 0, 1))


def combine_signal_scores(scores: list[SignalScore], weights: dict[str, float]) -> SignalScore:
    """
    Combine multiple signal scores into one.

    Args:
        scores: List of SignalScore objects
        weights: Dictionary of strategy_name -> weight

    Returns:
        Combined SignalScore
    """
    if not scores:
        raise ValueError("No scores to combine")

    if len(scores) == 1:
        return scores[0]

    total_weight = 0
    weighted_raw = 0
    weighted_direction = 0
    weighted_magnitude = 0
    weighted_timing = 0
    weighted_rr = 0
    all_reasons = []

    for score in scores:
        weight = weights.get(score.strategy_name, 1.0)
        total_weight += weight

        weighted_raw += score.raw_score * weight
        weighted_direction += score.direction_score * weight
        weighted_magnitude += score.magnitude_score * weight
        weighted_timing += score.timing_score * weight
        weighted_rr += score.risk_reward_score * weight
        all_reasons.extend([f"[{score.strategy_name}] {r}" for r in score.reasons])

    # Normalize
    if total_weight > 0:
        weighted_raw /= total_weight
        weighted_direction /= total_weight
        weighted_magnitude /= total_weight
        weighted_timing /= total_weight
        weighted_rr /= total_weight

    # Combined confidence - penalize disagreement
    direction_variance = np.var([s.direction_score for s in scores])
    agreement_factor = 1 - min(direction_variance, 0.5)

    avg_confidence = np.mean([s.confidence for s in scores])
    combined_confidence = avg_confidence * agreement_factor

    return SignalScore(
        symbol=scores[0].symbol,
        raw_score=float(weighted_raw),
        confidence=float(combined_confidence),
        direction_score=float(weighted_direction),
        magnitude_score=float(weighted_magnitude),
        timing_score=float(weighted_timing),
        risk_reward_score=float(weighted_rr),
        strategy_name="combined",
        reasons=all_reasons,
    )
