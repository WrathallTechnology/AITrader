"""Gemini AI comprehensive strategy for stocks (Method 3).

Reads the trade_signal and trade_confidence from the shared
GeminiFullAnalysis cache. Does NOT call Gemini itself.
"""

import logging

import pandas as pd

from src.strategies.base import BaseStrategy, Signal, SignalType
from src.options.gemini_client import GeminiFullAnalysis

logger = logging.getLogger(__name__)


class GeminiComprehensiveStrategy(BaseStrategy):
    """Converts Gemini's comprehensive trade recommendation into stock signals.

    Reads trade_signal and trade_confidence from the shared analysis.
    This is Gemini's holistic view combining technicals + WSB + price data.
    """

    def __init__(
        self,
        name: str = "gemini_ai",
        min_confidence: float = 0.45,
    ):
        super().__init__(name=name, weight=1.0)
        self.min_confidence = min_confidence

        # Shared analysis cache — set by the competition manager
        self._analysis_cache: dict[str, GeminiFullAnalysis] = {}

    def set_analysis(self, symbol: str, analysis: GeminiFullAnalysis):
        """Called by competition manager after shared Gemini call."""
        self._analysis_cache[symbol] = analysis

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate signal from cached Gemini analysis."""
        analysis = self._analysis_cache.get(symbol)
        if not analysis:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD,
                          confidence=0.0, reason="No Gemini analysis available")

        confidence = analysis.trade_confidence
        trade_signal = analysis.trade_signal

        if trade_signal == "buy" and confidence >= self.min_confidence:
            signal_type = SignalType.BUY
        elif trade_signal == "sell" and confidence >= self.min_confidence:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        current_price = float(data["close"].iloc[-1]) if len(data) > 0 else None

        factors_str = ", ".join(analysis.key_factors[:3]) if analysis.key_factors else "none"
        reason = (
            f"Gemini AI: {trade_signal} ({confidence:.0%} confidence). "
            f"Factors: {factors_str}"
        )

        if analysis.risk_warning:
            reason += f". Risk: {analysis.risk_warning}"

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=current_price,
            reason=reason,
        )
