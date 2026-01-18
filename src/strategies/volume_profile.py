"""
Volume profile and order flow analysis.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class VolumeProfileLevel:
    """A price level in the volume profile."""
    price: float
    volume: float
    buy_volume: float
    sell_volume: float

    @property
    def delta(self) -> float:
        """Buy volume minus sell volume."""
        return self.buy_volume - self.sell_volume

    @property
    def imbalance_ratio(self) -> float:
        """Ratio of buy to sell volume."""
        if self.sell_volume == 0:
            return float('inf') if self.buy_volume > 0 else 1.0
        return self.buy_volume / self.sell_volume


@dataclass
class VolumeProfile:
    """Complete volume profile analysis."""
    levels: list[VolumeProfileLevel]
    poc: float  # Point of Control (highest volume price)
    value_area_high: float
    value_area_low: float
    total_volume: float
    total_delta: float  # Net buying pressure

    @property
    def value_area_range(self) -> float:
        return self.value_area_high - self.value_area_low


class VolumeProfileAnalyzer:
    """
    Analyzes volume distribution across price levels.

    Key concepts:
    - POC (Point of Control): Price with highest volume - acts as magnet
    - Value Area: Price range containing 70% of volume - key S/R zone
    - Volume Delta: Buy vs sell pressure at each level
    """

    def __init__(
        self,
        num_levels: int = 24,
        value_area_pct: float = 0.70,
    ):
        """
        Args:
            num_levels: Number of price bins
            value_area_pct: Percentage of volume for value area (default 70%)
        """
        self.num_levels = num_levels
        self.value_area_pct = value_area_pct

    def calculate_profile(self, data: pd.DataFrame) -> VolumeProfile:
        """
        Calculate volume profile from OHLCV data.

        Args:
            data: DataFrame with OHLC and volume

        Returns:
            VolumeProfile with analysis
        """
        high = data["high"].max()
        low = data["low"].min()

        # Create price bins
        price_range = high - low
        bin_size = price_range / self.num_levels

        levels = []

        for i in range(self.num_levels):
            bin_low = low + i * bin_size
            bin_high = low + (i + 1) * bin_size
            bin_mid = (bin_low + bin_high) / 2

            # Find bars that traded through this price level
            mask = (data["low"] <= bin_high) & (data["high"] >= bin_low)
            bars_at_level = data[mask]

            if len(bars_at_level) == 0:
                levels.append(VolumeProfileLevel(
                    price=bin_mid,
                    volume=0,
                    buy_volume=0,
                    sell_volume=0,
                ))
                continue

            # Distribute volume across levels the bar touched
            total_volume = 0
            buy_volume = 0
            sell_volume = 0

            for _, bar in bars_at_level.iterrows():
                bar_range = bar["high"] - bar["low"]
                if bar_range == 0:
                    bar_range = 1

                # Proportion of this bar in our bin
                overlap_low = max(bar["low"], bin_low)
                overlap_high = min(bar["high"], bin_high)
                overlap = max(0, overlap_high - overlap_low)
                proportion = overlap / bar_range

                vol_at_level = bar["volume"] * proportion
                total_volume += vol_at_level

                # Estimate buy/sell based on close position
                if bar["close"] >= bar["open"]:
                    # Bullish bar - more buying
                    buy_volume += vol_at_level * 0.6
                    sell_volume += vol_at_level * 0.4
                else:
                    # Bearish bar - more selling
                    buy_volume += vol_at_level * 0.4
                    sell_volume += vol_at_level * 0.6

            levels.append(VolumeProfileLevel(
                price=bin_mid,
                volume=total_volume,
                buy_volume=buy_volume,
                sell_volume=sell_volume,
            ))

        # Find POC (Point of Control)
        poc_level = max(levels, key=lambda x: x.volume)
        poc = poc_level.price

        # Calculate Value Area (70% of volume)
        total_volume = sum(l.volume for l in levels)
        target_volume = total_volume * self.value_area_pct

        # Start from POC and expand outward
        poc_idx = levels.index(poc_level)
        va_low_idx = poc_idx
        va_high_idx = poc_idx
        accumulated_volume = poc_level.volume

        while accumulated_volume < target_volume:
            # Check which direction to expand
            low_vol = levels[va_low_idx - 1].volume if va_low_idx > 0 else 0
            high_vol = levels[va_high_idx + 1].volume if va_high_idx < len(levels) - 1 else 0

            if low_vol >= high_vol and va_low_idx > 0:
                va_low_idx -= 1
                accumulated_volume += levels[va_low_idx].volume
            elif va_high_idx < len(levels) - 1:
                va_high_idx += 1
                accumulated_volume += levels[va_high_idx].volume
            else:
                break

        value_area_low = levels[va_low_idx].price - bin_size / 2
        value_area_high = levels[va_high_idx].price + bin_size / 2

        # Total delta
        total_delta = sum(l.delta for l in levels)

        return VolumeProfile(
            levels=levels,
            poc=poc,
            value_area_high=value_area_high,
            value_area_low=value_area_low,
            total_volume=total_volume,
            total_delta=total_delta,
        )

    def find_high_volume_nodes(
        self,
        profile: VolumeProfile,
        threshold_pct: float = 0.8,
    ) -> list[float]:
        """Find price levels with unusually high volume (support/resistance)."""
        if not profile.levels:
            return []

        avg_volume = profile.total_volume / len(profile.levels)
        threshold = avg_volume * (1 + threshold_pct)

        return [l.price for l in profile.levels if l.volume > threshold]

    def find_low_volume_nodes(
        self,
        profile: VolumeProfile,
        threshold_pct: float = 0.3,
    ) -> list[float]:
        """Find price levels with low volume (fast move areas)."""
        if not profile.levels:
            return []

        avg_volume = profile.total_volume / len(profile.levels)
        threshold = avg_volume * threshold_pct

        return [l.price for l in profile.levels if 0 < l.volume < threshold]


class VWAPCalculator:
    """
    Volume Weighted Average Price calculations.

    VWAP is institutional benchmark - price tends to revert to it.
    """

    @staticmethod
    def calculate_vwap(data: pd.DataFrame) -> pd.Series:
        """Calculate VWAP for the session."""
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        cumulative_tp_vol = (typical_price * data["volume"]).cumsum()
        cumulative_vol = data["volume"].cumsum()

        return cumulative_tp_vol / cumulative_vol

    @staticmethod
    def calculate_vwap_bands(
        data: pd.DataFrame,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate VWAP with standard deviation bands.

        Returns:
            Tuple of (vwap, upper_band, lower_band)
        """
        vwap = VWAPCalculator.calculate_vwap(data)
        typical_price = (data["high"] + data["low"] + data["close"]) / 3

        # Calculate variance
        squared_diff = ((typical_price - vwap) ** 2 * data["volume"]).cumsum()
        cumulative_vol = data["volume"].cumsum()
        variance = squared_diff / cumulative_vol
        std = np.sqrt(variance)

        upper_band = vwap + (std * num_std)
        lower_band = vwap - (std * num_std)

        return vwap, upper_band, lower_band


class AccumulationDistribution:
    """
    Accumulation/Distribution analysis.

    Detects whether smart money is accumulating (buying) or
    distributing (selling) based on volume and price action.
    """

    @staticmethod
    def calculate_ad_line(data: pd.DataFrame) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line.

        Rising AD with rising price = healthy uptrend
        Falling AD with rising price = distribution (bearish divergence)
        """
        clv = ((data["close"] - data["low"]) - (data["high"] - data["close"])) / (data["high"] - data["low"] + 1e-10)
        ad = (clv * data["volume"]).cumsum()
        return ad

    @staticmethod
    def calculate_obv(data: pd.DataFrame) -> pd.Series:
        """
        Calculate On-Balance Volume.

        Volume is added on up days, subtracted on down days.
        """
        direction = np.sign(data["close"].diff())
        obv = (direction * data["volume"]).cumsum()
        return obv

    @staticmethod
    def calculate_mfi(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Money Flow Index (volume-weighted RSI).

        MFI < 20: Oversold
        MFI > 80: Overbought
        """
        typical_price = (data["high"] + data["low"] + data["close"]) / 3
        money_flow = typical_price * data["volume"]

        # Separate positive and negative money flow
        tp_diff = typical_price.diff()
        positive_flow = money_flow.where(tp_diff > 0, 0)
        negative_flow = money_flow.where(tp_diff < 0, 0)

        positive_sum = positive_flow.rolling(period).sum()
        negative_sum = negative_flow.rolling(period).sum()

        money_ratio = positive_sum / (negative_sum + 1e-10)
        mfi = 100 - (100 / (1 + money_ratio))

        return mfi

    @staticmethod
    def detect_divergence(
        price: pd.Series,
        indicator: pd.Series,
        lookback: int = 20,
    ) -> tuple[bool, bool]:
        """
        Detect bullish/bearish divergence.

        Returns:
            Tuple of (bullish_divergence, bearish_divergence)
        """
        if len(price) < lookback:
            return False, False

        recent_price = price.iloc[-lookback:]
        recent_indicator = indicator.iloc[-lookback:]

        # Find local extremes
        price_high_idx = recent_price.idxmax()
        price_low_idx = recent_price.idxmin()
        ind_high_idx = recent_indicator.idxmax()
        ind_low_idx = recent_indicator.idxmin()

        # Bearish divergence: price makes higher high, indicator makes lower high
        bearish = False
        if price.iloc[-1] > price.iloc[-lookback]:  # Price trending up
            if recent_indicator.iloc[-1] < recent_indicator.iloc[0]:
                bearish = True

        # Bullish divergence: price makes lower low, indicator makes higher low
        bullish = False
        if price.iloc[-1] < price.iloc[-lookback]:  # Price trending down
            if recent_indicator.iloc[-1] > recent_indicator.iloc[0]:
                bullish = True

        return bullish, bearish


class VolumeStrategy(BaseStrategy):
    """
    Trading strategy based on volume analysis.
    """

    def __init__(
        self,
        name: str = "volume",
        weight: float = 1.0,
        profile_lookback: int = 20,
    ):
        super().__init__(name, weight)
        self.profile_lookback = profile_lookback
        self.profile_analyzer = VolumeProfileAnalyzer()
        self._is_trained = True

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate signal based on volume analysis."""
        if len(data) < self.profile_lookback:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Insufficient data for volume analysis",
            )

        recent_data = data.iloc[-self.profile_lookback:]
        price = float(data["close"].iloc[-1])

        # Calculate volume profile
        profile = self.profile_analyzer.calculate_profile(recent_data)

        # Calculate VWAP
        vwap, vwap_upper, vwap_lower = VWAPCalculator.calculate_vwap_bands(data)
        current_vwap = float(vwap.iloc[-1])

        # Calculate MFI
        mfi = AccumulationDistribution.calculate_mfi(data)
        current_mfi = float(mfi.iloc[-1])

        # Calculate A/D line for divergence
        ad_line = AccumulationDistribution.calculate_ad_line(data)
        bullish_div, bearish_div = AccumulationDistribution.detect_divergence(
            data["close"], ad_line
        )

        # Build signal
        scores = []
        reasons = []

        # 1. Price vs VWAP
        vwap_distance = (price - current_vwap) / current_vwap
        if price < float(vwap_lower.iloc[-1]):
            scores.append(0.7)
            reasons.append("Price below VWAP lower band")
        elif price > float(vwap_upper.iloc[-1]):
            scores.append(-0.7)
            reasons.append("Price above VWAP upper band")
        elif price < current_vwap:
            scores.append(0.3)
        else:
            scores.append(-0.3)

        # 2. Price vs Value Area
        if price < profile.value_area_low:
            scores.append(0.6)
            reasons.append("Price below value area")
        elif price > profile.value_area_high:
            scores.append(-0.6)
            reasons.append("Price above value area")

        # 3. Volume delta (buying vs selling pressure)
        if profile.total_delta > 0:
            delta_score = min(profile.total_delta / profile.total_volume, 0.5)
            scores.append(delta_score)
            if delta_score > 0.3:
                reasons.append("Strong buying pressure")
        else:
            delta_score = max(profile.total_delta / profile.total_volume, -0.5)
            scores.append(delta_score)
            if delta_score < -0.3:
                reasons.append("Strong selling pressure")

        # 4. MFI
        if current_mfi < 20:
            scores.append(0.7)
            reasons.append(f"MFI oversold ({current_mfi:.0f})")
        elif current_mfi > 80:
            scores.append(-0.7)
            reasons.append(f"MFI overbought ({current_mfi:.0f})")

        # 5. Divergences
        if bullish_div:
            scores.append(0.5)
            reasons.append("Bullish A/D divergence")
        if bearish_div:
            scores.append(-0.5)
            reasons.append("Bearish A/D divergence")

        # Combine scores
        if not scores:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.3,
                price=price,
                reason="No clear volume signals",
            )

        final_score = np.mean(scores)
        confidence = min(abs(final_score) + 0.3, 1.0)

        # Determine signal type
        if final_score > 0.3:
            signal_type = SignalType.BUY
            atr = self._estimate_atr(data)
            stop_loss = price - (2 * atr)
            take_profit = profile.poc  # Target POC
        elif final_score < -0.3:
            signal_type = SignalType.SELL
            atr = self._estimate_atr(data)
            stop_loss = price + (2 * atr)
            take_profit = profile.poc
        else:
            signal_type = SignalType.HOLD
            stop_loss = None
            take_profit = None

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="; ".join(reasons) if reasons else "Neutral volume",
        )

    def _estimate_atr(self, data: pd.DataFrame, period: int = 14) -> float:
        """Estimate ATR for stop loss calculation."""
        if "atr" in data.columns:
            return float(data["atr"].iloc[-1])

        tr = pd.concat([
            data["high"] - data["low"],
            abs(data["high"] - data["close"].shift(1)),
            abs(data["low"] - data["close"].shift(1)),
        ], axis=1).max(axis=1)

        return float(tr.rolling(period).mean().iloc[-1])
