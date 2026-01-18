"""
Technical analysis strategy using classic indicators.
"""

import logging
from typing import Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from config import StrategyConfig

logger = logging.getLogger(__name__)


class TechnicalStrategy(BaseStrategy):
    """
    Strategy based on technical indicators:
    - RSI for overbought/oversold conditions
    - MACD for momentum and trend
    - Moving average crossovers for trend direction
    - Bollinger Bands for volatility breakouts
    """

    def __init__(
        self,
        name: str = "technical",
        weight: float = 1.0,
        config: Optional[StrategyConfig] = None,
    ):
        super().__init__(name, weight)
        self.config = config or StrategyConfig()
        self._is_trained = True  # No training required

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Generate trading signal based on technical indicators.
        """
        if len(data) < 50:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="Insufficient data",
            )

        # Ensure we have the required indicators
        required_cols = ["rsi", "macd", "macd_signal", "sma_20", "sma_50", "bb_lower", "bb_upper"]
        missing = [c for c in required_cols if c not in data.columns]
        if missing:
            logger.warning(f"Missing indicators: {missing}")
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason=f"Missing indicators: {missing}",
            )

        # Get latest values
        latest = data.iloc[-1]
        prev = data.iloc[-2] if len(data) > 1 else latest

        price = float(latest["close"])
        rsi = float(latest["rsi"])
        macd = float(latest["macd"])
        macd_signal = float(latest["macd_signal"])
        sma_20 = float(latest["sma_20"])
        sma_50 = float(latest["sma_50"])
        bb_lower = float(latest["bb_lower"])
        bb_upper = float(latest["bb_upper"])
        atr = float(latest["atr"]) if "atr" in data.columns else price * 0.02

        # Calculate individual signals
        signals = []

        # RSI Signal
        rsi_signal = self._rsi_signal(rsi)
        signals.append(("RSI", rsi_signal))

        # MACD Signal
        macd_signal_val = self._macd_signal(macd, macd_signal, float(prev["macd"]), float(prev["macd_signal"]))
        signals.append(("MACD", macd_signal_val))

        # Moving Average Signal
        ma_signal = self._ma_signal(sma_20, sma_50, float(prev["sma_20"]), float(prev["sma_50"]))
        signals.append(("MA", ma_signal))

        # Bollinger Band Signal
        bb_signal = self._bb_signal(price, bb_lower, bb_upper)
        signals.append(("BB", bb_signal))

        # Combine signals
        buy_signals = sum(1 for _, s in signals if s == SignalType.BUY)
        sell_signals = sum(1 for _, s in signals if s == SignalType.SELL)

        # Determine final signal and confidence
        if buy_signals > sell_signals and buy_signals >= 2:
            signal_type = SignalType.BUY
            confidence = buy_signals / len(signals)
            stop_loss = price - (2 * atr)
            take_profit = price + (3 * atr)
        elif sell_signals > buy_signals and sell_signals >= 2:
            signal_type = SignalType.SELL
            confidence = sell_signals / len(signals)
            stop_loss = price + (2 * atr)
            take_profit = price - (3 * atr)
        else:
            signal_type = SignalType.HOLD
            confidence = 0.5
            stop_loss = None
            take_profit = None

        # Build reason
        reason_parts = [f"{name}:{sig.value}" for name, sig in signals]
        reason = f"Technical: {', '.join(reason_parts)}"

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    def _rsi_signal(self, rsi: float) -> SignalType:
        """Generate signal from RSI."""
        if rsi < self.config.rsi_oversold:
            return SignalType.BUY
        elif rsi > self.config.rsi_overbought:
            return SignalType.SELL
        return SignalType.HOLD

    def _macd_signal(
        self,
        macd: float,
        signal: float,
        prev_macd: float,
        prev_signal: float,
    ) -> SignalType:
        """Generate signal from MACD crossover."""
        # Bullish crossover: MACD crosses above signal line
        if prev_macd <= prev_signal and macd > signal:
            return SignalType.BUY
        # Bearish crossover: MACD crosses below signal line
        elif prev_macd >= prev_signal and macd < signal:
            return SignalType.SELL
        return SignalType.HOLD

    def _ma_signal(
        self,
        sma_short: float,
        sma_long: float,
        prev_short: float,
        prev_long: float,
    ) -> SignalType:
        """Generate signal from moving average crossover."""
        # Golden cross: short MA crosses above long MA
        if prev_short <= prev_long and sma_short > sma_long:
            return SignalType.BUY
        # Death cross: short MA crosses below long MA
        elif prev_short >= prev_long and sma_short < sma_long:
            return SignalType.SELL
        # Trend following
        elif sma_short > sma_long:
            return SignalType.BUY
        elif sma_short < sma_long:
            return SignalType.SELL
        return SignalType.HOLD

    def _bb_signal(self, price: float, lower: float, upper: float) -> SignalType:
        """Generate signal from Bollinger Bands."""
        if price < lower:
            return SignalType.BUY  # Price below lower band - potential reversal
        elif price > upper:
            return SignalType.SELL  # Price above upper band - potential reversal
        return SignalType.HOLD
