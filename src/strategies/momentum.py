"""
Momentum/trend following strategy.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from config import StrategyConfig

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum/trend following strategy.

    Works best in trending markets where prices continue in the
    current direction.

    Key indicators:
    - Price momentum (rate of change)
    - Moving average trends
    - ADX for trend strength
    - Relative strength vs benchmark
    """

    def __init__(
        self,
        name: str = "momentum",
        weight: float = 1.0,
        config: Optional[StrategyConfig] = None,
        momentum_period: int = 20,
        trend_period: int = 50,
        adx_threshold: float = 25.0,
        min_momentum: float = 0.02,  # 2% minimum momentum
    ):
        """
        Args:
            name: Strategy name
            weight: Strategy weight
            config: Strategy configuration
            momentum_period: Period for momentum calculation
            trend_period: Period for trend determination
            adx_threshold: ADX value to confirm trend
            min_momentum: Minimum momentum for signal
        """
        super().__init__(name, weight)
        self.config = config or StrategyConfig()
        self.momentum_period = momentum_period
        self.trend_period = trend_period
        self.adx_threshold = adx_threshold
        self.min_momentum = min_momentum
        self._is_trained = True

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate momentum signal."""
        if len(data) < max(self.momentum_period, self.trend_period) + 10:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Insufficient data for momentum",
            )

        # Calculate indicators
        indicators = self._calculate_indicators(data)

        price = float(data["close"].iloc[-1])
        momentum = indicators["momentum"]
        trend_direction = indicators["trend_direction"]
        trend_strength = indicators["trend_strength"]
        adx = indicators["adx"]

        # Calculate momentum score
        mom_score, reasons = self._calculate_momentum_score(
            momentum, trend_direction, trend_strength, adx, indicators
        )

        # ATR for stops
        atr = float(data["atr"].iloc[-1]) if "atr" in data.columns else price * 0.02

        # Determine signal
        if mom_score > 0.5 and adx > self.adx_threshold:
            signal_type = SignalType.BUY
            confidence = min(mom_score, 1.0)

            # Trend following: wider stops, trailing approach
            stop_loss = price - (2.5 * atr)
            take_profit = price + (4 * atr)  # Larger target for trends

        elif mom_score < -0.5 and adx > self.adx_threshold:
            signal_type = SignalType.SELL
            confidence = min(abs(mom_score), 1.0)

            stop_loss = price + (2.5 * atr)
            take_profit = price - (4 * atr)

        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
            stop_loss = None
            take_profit = None

            if adx < self.adx_threshold:
                reasons.append(f"Weak trend (ADX={adx:.1f})")
            else:
                reasons.append("No clear momentum")

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
        """Calculate momentum indicators."""
        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Price momentum (rate of change)
        momentum = (close.iloc[-1] - close.iloc[-self.momentum_period]) / close.iloc[-self.momentum_period]

        # Trend direction using EMAs
        ema_short = close.ewm(span=12, adjust=False).mean()
        ema_medium = close.ewm(span=26, adjust=False).mean()
        ema_long = close.ewm(span=self.trend_period, adjust=False).mean()

        # Trend direction: 1 (up), -1 (down), 0 (unclear)
        if ema_short.iloc[-1] > ema_medium.iloc[-1] > ema_long.iloc[-1]:
            trend_direction = 1
        elif ema_short.iloc[-1] < ema_medium.iloc[-1] < ema_long.iloc[-1]:
            trend_direction = -1
        else:
            trend_direction = 0

        # Trend strength using price vs MAs
        price = close.iloc[-1]
        ma_distances = [
            (price - ema_short.iloc[-1]) / ema_short.iloc[-1],
            (price - ema_medium.iloc[-1]) / ema_medium.iloc[-1],
            (price - ema_long.iloc[-1]) / ema_long.iloc[-1],
        ]
        trend_strength = abs(sum(ma_distances) / 3)

        # ADX calculation
        adx = self._calculate_adx(data)

        # Rate of change at different timeframes
        roc_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] if len(close) > 5 else 0
        roc_10 = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] if len(close) > 10 else 0
        roc_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if len(close) > 20 else 0

        # Higher highs / lower lows
        recent_high = high.iloc[-20:].max()
        recent_low = low.iloc[-20:].min()
        prev_high = high.iloc[-40:-20].max() if len(high) > 40 else recent_high
        prev_low = low.iloc[-40:-20].min() if len(low) > 40 else recent_low

        higher_highs = recent_high > prev_high
        higher_lows = recent_low > prev_low
        lower_highs = recent_high < prev_high
        lower_lows = recent_low < prev_low

        # Momentum acceleration
        momentum_5_ago = (close.iloc[-6] - close.iloc[-6-self.momentum_period]) / close.iloc[-6-self.momentum_period] if len(close) > self.momentum_period + 6 else 0
        momentum_acceleration = momentum - momentum_5_ago

        return {
            "momentum": momentum,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
            "adx": adx,
            "roc_5": roc_5,
            "roc_10": roc_10,
            "roc_20": roc_20,
            "higher_highs": higher_highs,
            "higher_lows": higher_lows,
            "lower_highs": lower_highs,
            "lower_lows": lower_lows,
            "momentum_acceleration": momentum_acceleration,
            "ema_short": ema_short.iloc[-1],
            "ema_medium": ema_medium.iloc[-1],
            "ema_long": ema_long.iloc[-1],
        }

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index."""
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # Smoothed
        atr = pd.Series(tr).rolling(window=period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0

    def _calculate_momentum_score(
        self,
        momentum: float,
        trend_direction: int,
        trend_strength: float,
        adx: float,
        indicators: dict,
    ) -> tuple[float, list[str]]:
        """
        Calculate momentum score.

        Positive = bullish momentum
        Negative = bearish momentum
        """
        scores = []
        reasons = []

        # Primary momentum
        if momentum > self.min_momentum * 2:
            scores.append(1.0)
            reasons.append(f"Strong upward momentum ({momentum:.1%})")
        elif momentum > self.min_momentum:
            scores.append(0.6)
            reasons.append(f"Upward momentum ({momentum:.1%})")
        elif momentum < -self.min_momentum * 2:
            scores.append(-1.0)
            reasons.append(f"Strong downward momentum ({momentum:.1%})")
        elif momentum < -self.min_momentum:
            scores.append(-0.6)
            reasons.append(f"Downward momentum ({momentum:.1%})")
        else:
            scores.append(0)

        # Trend direction
        if trend_direction == 1:
            scores.append(0.7)
            reasons.append("Uptrend (EMAs aligned)")
        elif trend_direction == -1:
            scores.append(-0.7)
            reasons.append("Downtrend (EMAs aligned)")
        else:
            scores.append(0)

        # ADX confirmation
        if adx > 40:
            # Strong trend - boost score in trend direction
            scores.append(0.5 * np.sign(momentum))
            reasons.append(f"Strong trend (ADX={adx:.1f})")
        elif adx > self.adx_threshold:
            scores.append(0.3 * np.sign(momentum))

        # Higher highs/lows pattern
        if indicators["higher_highs"] and indicators["higher_lows"]:
            scores.append(0.5)
            reasons.append("Higher highs & lows")
        elif indicators["lower_highs"] and indicators["lower_lows"]:
            scores.append(-0.5)
            reasons.append("Lower highs & lows")

        # Momentum acceleration
        accel = indicators["momentum_acceleration"]
        if accel > 0.01:
            scores.append(0.3)
            reasons.append("Momentum accelerating")
        elif accel < -0.01:
            scores.append(-0.3)
            reasons.append("Momentum decelerating")

        # Multi-timeframe agreement
        rocs = [indicators["roc_5"], indicators["roc_10"], indicators["roc_20"]]
        if all(r > 0 for r in rocs):
            scores.append(0.4)
            reasons.append("Multi-TF bullish")
        elif all(r < 0 for r in rocs):
            scores.append(-0.4)
            reasons.append("Multi-TF bearish")

        # Weighted average
        if not scores:
            return 0.0, ["No momentum signals"]

        final_score = np.mean(scores)
        return float(np.clip(final_score, -1, 1)), reasons

    def calculate_momentum_metrics(self, data: pd.DataFrame) -> dict:
        """Get detailed momentum metrics."""
        indicators = self._calculate_indicators(data)

        return {
            "momentum_20d": indicators["momentum"] * 100,
            "roc_5d": indicators["roc_5"] * 100,
            "roc_10d": indicators["roc_10"] * 100,
            "roc_20d": indicators["roc_20"] * 100,
            "trend_direction": indicators["trend_direction"],
            "trend_strength": indicators["trend_strength"],
            "adx": indicators["adx"],
            "higher_highs": indicators["higher_highs"],
            "higher_lows": indicators["higher_lows"],
            "momentum_acceleration": indicators["momentum_acceleration"] * 100,
        }


class DualMomentumStrategy(MomentumStrategy):
    """
    Dual momentum strategy combining absolute and relative momentum.

    - Absolute momentum: Is the asset trending up?
    - Relative momentum: Is it outperforming alternatives?

    Only takes positions when both conditions are met.
    """

    def __init__(
        self,
        name: str = "dual_momentum",
        weight: float = 1.0,
        benchmark_data: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.02,
        **kwargs,
    ):
        super().__init__(name, weight, **kwargs)
        self.benchmark_data = benchmark_data
        self.risk_free_rate = risk_free_rate / 252  # Daily rate

    def set_benchmark(self, benchmark_data: pd.DataFrame) -> None:
        """Set benchmark data (e.g., SPY for stocks)."""
        self.benchmark_data = benchmark_data

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate dual momentum signal."""
        # First check absolute momentum
        base_signal = super().generate_signal(symbol, data)

        if base_signal.signal_type == SignalType.HOLD:
            return base_signal

        # Check relative momentum if benchmark available
        if self.benchmark_data is not None and len(self.benchmark_data) >= self.momentum_period:
            relative_strength = self._calculate_relative_strength(data)

            if base_signal.signal_type == SignalType.BUY and relative_strength < 0:
                # Asset has positive momentum but underperforming benchmark
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    confidence=base_signal.confidence * 0.5,
                    price=base_signal.price,
                    reason=f"Underperforming benchmark (RS={relative_strength:.2%})",
                )

            # Boost confidence if outperforming
            if relative_strength > 0.02:
                base_signal = Signal(
                    symbol=base_signal.symbol,
                    signal_type=base_signal.signal_type,
                    confidence=min(base_signal.confidence * 1.2, 1.0),
                    price=base_signal.price,
                    stop_loss=base_signal.stop_loss,
                    take_profit=base_signal.take_profit,
                    reason=f"{base_signal.reason}; Outperforming benchmark (+{relative_strength:.1%})",
                )

        return base_signal

    def _calculate_relative_strength(self, data: pd.DataFrame) -> float:
        """Calculate relative strength vs benchmark."""
        if self.benchmark_data is None:
            return 0.0

        asset_return = (
            data["close"].iloc[-1] - data["close"].iloc[-self.momentum_period]
        ) / data["close"].iloc[-self.momentum_period]

        benchmark_return = (
            self.benchmark_data["close"].iloc[-1] -
            self.benchmark_data["close"].iloc[-self.momentum_period]
        ) / self.benchmark_data["close"].iloc[-self.momentum_period]

        return asset_return - benchmark_return
