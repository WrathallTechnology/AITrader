"""
Multi-timeframe analysis for signal confirmation.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Supported timeframes."""
    M1 = "1Min"
    M5 = "5Min"
    M15 = "15Min"
    M30 = "30Min"
    H1 = "1Hour"
    H4 = "4Hour"
    D1 = "1Day"
    W1 = "1Week"


@dataclass
class TimeframeAnalysis:
    """Analysis result for a single timeframe."""
    timeframe: Timeframe
    trend: int  # 1 (up), -1 (down), 0 (neutral)
    trend_strength: float  # 0 to 1
    momentum: float  # Rate of change
    support: Optional[float]
    resistance: Optional[float]
    signal: SignalType
    confidence: float


@dataclass
class MTFAnalysis:
    """Complete multi-timeframe analysis."""
    analyses: dict[Timeframe, TimeframeAnalysis]
    alignment_score: float  # How well timeframes agree (-1 to 1)
    primary_trend: int
    recommended_signal: SignalType
    confidence: float


class TimeframeResampler:
    """
    Resamples OHLCV data to different timeframes.
    """

    RESAMPLE_MAP = {
        Timeframe.M1: "1min",
        Timeframe.M5: "5min",
        Timeframe.M15: "15min",
        Timeframe.M30: "30min",
        Timeframe.H1: "1h",
        Timeframe.H4: "4h",
        Timeframe.D1: "1D",
        Timeframe.W1: "1W",
    }

    @staticmethod
    def resample(data: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
        """
        Resample data to a different timeframe.

        Args:
            data: OHLCV DataFrame with datetime index
            timeframe: Target timeframe

        Returns:
            Resampled DataFrame
        """
        rule = TimeframeResampler.RESAMPLE_MAP[timeframe]

        resampled = data.resample(rule).agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()

        return resampled


class MultiTimeframeAnalyzer:
    """
    Analyzes price action across multiple timeframes.

    Higher timeframe determines trend direction.
    Lower timeframe provides entry timing.
    """

    def __init__(
        self,
        primary_tf: Timeframe = Timeframe.D1,
        secondary_tf: Timeframe = Timeframe.H4,
        entry_tf: Timeframe = Timeframe.H1,
    ):
        """
        Args:
            primary_tf: Highest timeframe for trend direction
            secondary_tf: Middle timeframe for confirmation
            entry_tf: Lowest timeframe for entry timing
        """
        self.primary_tf = primary_tf
        self.secondary_tf = secondary_tf
        self.entry_tf = entry_tf
        self.timeframes = [primary_tf, secondary_tf, entry_tf]

    def analyze(
        self,
        data: pd.DataFrame,
        custom_timeframes: Optional[list[Timeframe]] = None,
    ) -> MTFAnalysis:
        """
        Perform multi-timeframe analysis.

        Args:
            data: Base OHLCV data (should be at entry_tf or lower)
            custom_timeframes: Optional list of timeframes to analyze

        Returns:
            MTFAnalysis with complete results
        """
        timeframes = custom_timeframes or self.timeframes
        analyses = {}

        for tf in timeframes:
            try:
                tf_data = TimeframeResampler.resample(data, tf)
                if len(tf_data) < 20:
                    continue

                analysis = self._analyze_timeframe(tf_data, tf)
                analyses[tf] = analysis

            except Exception as e:
                logger.warning(f"Error analyzing {tf.value}: {e}")

        if not analyses:
            return MTFAnalysis(
                analyses={},
                alignment_score=0.0,
                primary_trend=0,
                recommended_signal=SignalType.HOLD,
                confidence=0.0,
            )

        # Calculate alignment
        alignment_score = self._calculate_alignment(analyses)

        # Determine primary trend from highest timeframe
        primary_analysis = analyses.get(self.primary_tf)
        primary_trend = primary_analysis.trend if primary_analysis else 0

        # Generate recommendation
        signal, confidence = self._generate_recommendation(analyses, alignment_score)

        return MTFAnalysis(
            analyses=analyses,
            alignment_score=alignment_score,
            primary_trend=primary_trend,
            recommended_signal=signal,
            confidence=confidence,
        )

    def _analyze_timeframe(
        self,
        data: pd.DataFrame,
        timeframe: Timeframe,
    ) -> TimeframeAnalysis:
        """Analyze a single timeframe."""
        close = data["close"]

        # Calculate trend using EMAs
        ema_fast = close.ewm(span=8, adjust=False).mean()
        ema_slow = close.ewm(span=21, adjust=False).mean()
        ema_trend = close.ewm(span=50, adjust=False).mean()

        current_price = close.iloc[-1]
        current_fast = ema_fast.iloc[-1]
        current_slow = ema_slow.iloc[-1]
        current_trend = ema_trend.iloc[-1]

        # Determine trend
        if current_fast > current_slow > current_trend and current_price > current_fast:
            trend = 1
            trend_strength = min((current_price - current_trend) / current_trend * 10, 1.0)
        elif current_fast < current_slow < current_trend and current_price < current_fast:
            trend = -1
            trend_strength = min((current_trend - current_price) / current_trend * 10, 1.0)
        else:
            trend = 0
            trend_strength = 0.3

        # Calculate momentum
        if len(close) >= 10:
            momentum = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
        else:
            momentum = 0.0

        # Find support/resistance (recent swing points)
        support, resistance = self._find_sr_levels(data)

        # Generate signal
        if trend == 1 and momentum > 0:
            signal = SignalType.BUY
            confidence = trend_strength * 0.7 + abs(momentum) * 10 * 0.3
        elif trend == -1 and momentum < 0:
            signal = SignalType.SELL
            confidence = trend_strength * 0.7 + abs(momentum) * 10 * 0.3
        else:
            signal = SignalType.HOLD
            confidence = 0.5

        confidence = min(confidence, 1.0)

        return TimeframeAnalysis(
            timeframe=timeframe,
            trend=trend,
            trend_strength=trend_strength,
            momentum=momentum,
            support=support,
            resistance=resistance,
            signal=signal,
            confidence=confidence,
        )

    def _find_sr_levels(
        self,
        data: pd.DataFrame,
        lookback: int = 20,
    ) -> tuple[Optional[float], Optional[float]]:
        """Find nearest support and resistance levels."""
        if len(data) < lookback:
            return None, None

        recent = data.iloc[-lookback:]
        current_price = data["close"].iloc[-1]

        # Simple approach: recent swing lows = support, swing highs = resistance
        highs = recent["high"]
        lows = recent["low"]

        # Find local maxima/minima
        resistance_candidates = []
        support_candidates = []

        for i in range(2, len(recent) - 2):
            if highs.iloc[i] > highs.iloc[i-1] and highs.iloc[i] > highs.iloc[i+1]:
                if highs.iloc[i] > current_price:
                    resistance_candidates.append(highs.iloc[i])

            if lows.iloc[i] < lows.iloc[i-1] and lows.iloc[i] < lows.iloc[i+1]:
                if lows.iloc[i] < current_price:
                    support_candidates.append(lows.iloc[i])

        support = max(support_candidates) if support_candidates else None
        resistance = min(resistance_candidates) if resistance_candidates else None

        return support, resistance

    def _calculate_alignment(
        self,
        analyses: dict[Timeframe, TimeframeAnalysis],
    ) -> float:
        """
        Calculate how well timeframes are aligned.

        Returns score from -1 (all bearish) to 1 (all bullish).
        """
        if not analyses:
            return 0.0

        # Weight higher timeframes more
        weights = {
            Timeframe.W1: 3.0,
            Timeframe.D1: 2.5,
            Timeframe.H4: 2.0,
            Timeframe.H1: 1.5,
            Timeframe.M30: 1.0,
            Timeframe.M15: 0.8,
            Timeframe.M5: 0.5,
            Timeframe.M1: 0.3,
        }

        total_weight = 0
        weighted_trend = 0

        for tf, analysis in analyses.items():
            weight = weights.get(tf, 1.0)
            weighted_trend += analysis.trend * analysis.trend_strength * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_trend / total_weight

    def _generate_recommendation(
        self,
        analyses: dict[Timeframe, TimeframeAnalysis],
        alignment: float,
    ) -> tuple[SignalType, float]:
        """Generate final signal recommendation."""
        # Need strong alignment for actionable signal
        if abs(alignment) < 0.3:
            return SignalType.HOLD, abs(alignment)

        # Check if entry timeframe agrees with higher timeframes
        entry_analysis = analyses.get(self.entry_tf)
        primary_analysis = analyses.get(self.primary_tf)

        if not entry_analysis or not primary_analysis:
            return SignalType.HOLD, 0.3

        # Require primary and entry to agree
        if primary_analysis.trend != entry_analysis.trend:
            return SignalType.HOLD, 0.4

        if alignment > 0.5:
            signal = SignalType.BUY
            confidence = min(alignment + 0.2, 1.0)
        elif alignment < -0.5:
            signal = SignalType.SELL
            confidence = min(abs(alignment) + 0.2, 1.0)
        else:
            signal = SignalType.HOLD
            confidence = 0.5

        return signal, confidence


class MultiTimeframeStrategy(BaseStrategy):
    """
    Trading strategy using multi-timeframe confirmation.

    Only trades when multiple timeframes align.
    """

    def __init__(
        self,
        name: str = "mtf",
        weight: float = 1.0,
        primary_tf: Timeframe = Timeframe.D1,
        secondary_tf: Timeframe = Timeframe.H4,
        entry_tf: Timeframe = Timeframe.H1,
        min_alignment: float = 0.5,
    ):
        super().__init__(name, weight)
        self.analyzer = MultiTimeframeAnalyzer(primary_tf, secondary_tf, entry_tf)
        self.min_alignment = min_alignment
        self._is_trained = True

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate signal based on multi-timeframe analysis."""
        if len(data) < 100:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Insufficient data for MTF analysis",
            )

        # Ensure datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Data requires datetime index for MTF",
            )

        mtf = self.analyzer.analyze(data)

        price = float(data["close"].iloc[-1])

        # Check alignment threshold
        if abs(mtf.alignment_score) < self.min_alignment:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=abs(mtf.alignment_score),
                price=price,
                reason=f"Weak MTF alignment ({mtf.alignment_score:.2f})",
            )

        # Build reason from timeframe analyses
        reasons = []
        for tf, analysis in mtf.analyses.items():
            trend_str = "↑" if analysis.trend == 1 else "↓" if analysis.trend == -1 else "→"
            reasons.append(f"{tf.value}:{trend_str}")

        # Get S/R levels from entry timeframe for stops
        entry_analysis = mtf.analyses.get(self.analyzer.entry_tf)
        stop_loss = None
        take_profit = None

        if entry_analysis:
            if mtf.recommended_signal == SignalType.BUY and entry_analysis.support:
                stop_loss = entry_analysis.support * 0.99
                if entry_analysis.resistance:
                    take_profit = entry_analysis.resistance * 0.99
            elif mtf.recommended_signal == SignalType.SELL and entry_analysis.resistance:
                stop_loss = entry_analysis.resistance * 1.01
                if entry_analysis.support:
                    take_profit = entry_analysis.support * 1.01

        return Signal(
            symbol=symbol,
            signal_type=mtf.recommended_signal,
            confidence=mtf.confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"MTF [{', '.join(reasons)}] align={mtf.alignment_score:.2f}",
        )

    def get_mtf_summary(self, data: pd.DataFrame) -> dict:
        """Get detailed MTF analysis summary."""
        mtf = self.analyzer.analyze(data)

        return {
            "alignment_score": mtf.alignment_score,
            "primary_trend": mtf.primary_trend,
            "recommended_signal": mtf.recommended_signal.value,
            "confidence": mtf.confidence,
            "timeframes": {
                tf.value: {
                    "trend": analysis.trend,
                    "trend_strength": analysis.trend_strength,
                    "momentum": analysis.momentum,
                    "signal": analysis.signal.value,
                    "support": analysis.support,
                    "resistance": analysis.resistance,
                }
                for tf, analysis in mtf.analyses.items()
            },
        }
