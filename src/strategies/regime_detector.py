"""
Market regime detection for adaptive strategy selection.
"""

import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    """Market regime classifications."""
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    primary_regime: MarketRegime
    trend_strength: float  # -1 (strong down) to 1 (strong up)
    volatility_regime: MarketRegime
    volatility_percentile: float  # 0-100
    adx_value: float
    is_trending: bool
    confidence: float


class RegimeDetector:
    """
    Detects current market regime using multiple indicators.

    Identifies:
    - Trend direction and strength
    - Volatility regime
    - Ranging vs trending markets
    """

    def __init__(
        self,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        volatility_lookback: int = 60,
    ):
        """
        Args:
            adx_period: Period for ADX calculation
            adx_threshold: ADX value above which market is considered trending
            volatility_lookback: Days to consider for volatility percentile
        """
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
        self.volatility_lookback = volatility_lookback

    def detect_regime(self, data: pd.DataFrame) -> RegimeAnalysis:
        """
        Analyze market data to determine current regime.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            RegimeAnalysis with full regime information
        """
        if len(data) < max(self.adx_period * 2, self.volatility_lookback):
            logger.warning("Insufficient data for regime detection")
            return RegimeAnalysis(
                primary_regime=MarketRegime.RANGING,
                trend_strength=0.0,
                volatility_regime=MarketRegime.LOW_VOLATILITY,
                volatility_percentile=50.0,
                adx_value=0.0,
                is_trending=False,
                confidence=0.0,
            )

        # Calculate ADX for trend strength
        adx = self._calculate_adx(data)
        is_trending = adx > self.adx_threshold

        # Calculate trend direction
        trend_strength = self._calculate_trend_strength(data)

        # Determine primary regime
        primary_regime = self._classify_trend_regime(trend_strength, adx)

        # Calculate volatility regime
        volatility_percentile = self._calculate_volatility_percentile(data)
        volatility_regime = self._classify_volatility_regime(volatility_percentile)

        # Calculate confidence based on indicator agreement
        confidence = self._calculate_confidence(data, adx, trend_strength)

        return RegimeAnalysis(
            primary_regime=primary_regime,
            trend_strength=trend_strength,
            volatility_regime=volatility_regime,
            volatility_percentile=volatility_percentile,
            adx_value=adx,
            is_trending=is_trending,
            confidence=confidence,
        )

    def _calculate_adx(self, data: pd.DataFrame) -> float:
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

        # Smoothed averages
        atr = pd.Series(tr).rolling(window=self.adx_period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(window=self.adx_period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=self.adx_period).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=self.adx_period).mean()

        return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 0.0

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """
        Calculate trend strength from -1 (strong downtrend) to 1 (strong uptrend).

        Uses multiple indicators:
        - Price vs moving averages
        - Moving average slopes
        - Higher highs/lower lows pattern
        """
        close = data["close"]

        scores = []

        # Price vs SMAs
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()

        current_price = close.iloc[-1]

        # Score based on price position relative to MAs
        if len(data) >= 20:
            if current_price > sma_20.iloc[-1]:
                scores.append(0.2)
            else:
                scores.append(-0.2)

        if len(data) >= 50:
            if current_price > sma_50.iloc[-1]:
                scores.append(0.3)
            else:
                scores.append(-0.3)

        if len(data) >= 200:
            if current_price > sma_200.iloc[-1]:
                scores.append(0.2)
            else:
                scores.append(-0.2)

        # MA alignment (bullish: 20 > 50 > 200)
        if len(data) >= 200:
            if sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1]:
                scores.append(0.3)
            elif sma_20.iloc[-1] < sma_50.iloc[-1] < sma_200.iloc[-1]:
                scores.append(-0.3)

        # Recent price momentum (20-day return)
        if len(data) >= 20:
            momentum = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
            # Clamp to [-0.3, 0.3]
            momentum_score = max(-0.3, min(0.3, momentum * 3))
            scores.append(momentum_score)

        # Higher highs / lower lows (last 10 bars)
        if len(data) >= 20:
            recent_highs = data["high"].iloc[-20:]
            recent_lows = data["low"].iloc[-20:]

            # Check for higher highs
            mid_high = recent_highs.iloc[:10].max()
            late_high = recent_highs.iloc[10:].max()
            mid_low = recent_lows.iloc[:10].min()
            late_low = recent_lows.iloc[10:].min()

            if late_high > mid_high and late_low > mid_low:
                scores.append(0.2)  # Uptrend pattern
            elif late_high < mid_high and late_low < mid_low:
                scores.append(-0.2)  # Downtrend pattern

        if not scores:
            return 0.0

        return float(np.clip(sum(scores), -1.0, 1.0))

    def _classify_trend_regime(self, trend_strength: float, adx: float) -> MarketRegime:
        """Classify the trend regime based on strength and ADX."""
        if adx < self.adx_threshold:
            return MarketRegime.RANGING

        if trend_strength > 0.6:
            return MarketRegime.STRONG_UPTREND
        elif trend_strength > 0.2:
            return MarketRegime.UPTREND
        elif trend_strength < -0.6:
            return MarketRegime.STRONG_DOWNTREND
        elif trend_strength < -0.2:
            return MarketRegime.DOWNTREND
        else:
            return MarketRegime.RANGING

    def _calculate_volatility_percentile(self, data: pd.DataFrame) -> float:
        """Calculate current volatility percentile (0-100)."""
        returns = data["close"].pct_change().dropna()

        if len(returns) < self.volatility_lookback:
            return 50.0

        # Rolling volatility
        current_vol = returns.iloc[-20:].std() * np.sqrt(252)

        # Historical volatility distribution
        rolling_vol = returns.rolling(20).std() * np.sqrt(252)
        rolling_vol = rolling_vol.dropna()

        if len(rolling_vol) == 0:
            return 50.0

        # Percentile rank
        percentile = (rolling_vol < current_vol).sum() / len(rolling_vol) * 100

        return float(percentile)

    def _classify_volatility_regime(self, percentile: float) -> MarketRegime:
        """Classify volatility regime based on percentile."""
        if percentile > 80:
            return MarketRegime.HIGH_VOLATILITY
        elif percentile < 20:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.RANGING  # Normal volatility

    def _calculate_confidence(
        self,
        data: pd.DataFrame,
        adx: float,
        trend_strength: float,
    ) -> float:
        """Calculate confidence in the regime classification."""
        confidence_factors = []

        # ADX clarity (higher ADX = clearer trend)
        adx_confidence = min(adx / 50, 1.0)
        confidence_factors.append(adx_confidence)

        # Trend strength magnitude
        trend_confidence = abs(trend_strength)
        confidence_factors.append(trend_confidence)

        # Data sufficiency
        data_confidence = min(len(data) / 200, 1.0)
        confidence_factors.append(data_confidence)

        return float(np.mean(confidence_factors))

    def get_strategy_weights_for_regime(
        self,
        regime: RegimeAnalysis,
        base_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Adjust strategy weights based on current regime.

        Args:
            regime: Current regime analysis
            base_weights: Original strategy weights

        Returns:
            Adjusted weights dictionary
        """
        weights = base_weights.copy()

        # In trending markets: favor momentum/trend-following
        # In ranging markets: favor mean reversion
        # In high volatility: reduce position sizes (handled elsewhere), favor shorter-term

        if regime.is_trending:
            # Boost trend-following strategies
            if "momentum" in weights:
                weights["momentum"] *= 1.3
            if "technical" in weights:
                weights["technical"] *= 1.2

            # Reduce mean reversion
            if "mean_reversion" in weights:
                weights["mean_reversion"] *= 0.5

        else:  # Ranging
            # Boost mean reversion
            if "mean_reversion" in weights:
                weights["mean_reversion"] *= 1.5

            # Reduce trend-following
            if "momentum" in weights:
                weights["momentum"] *= 0.6

        # High volatility adjustments
        if regime.volatility_regime == MarketRegime.HIGH_VOLATILITY:
            # Reduce ML confidence (models often struggle in high vol)
            if "ml" in weights:
                weights["ml"] *= 0.7

            # Boost sentiment (news-driven markets)
            if "sentiment" in weights:
                weights["sentiment"] *= 1.3

        # Low volatility
        elif regime.volatility_regime == MarketRegime.LOW_VOLATILITY:
            # Mean reversion works well
            if "mean_reversion" in weights:
                weights["mean_reversion"] *= 1.2

        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights
