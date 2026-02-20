"""WSB Reddit sentiment strategy for stocks (Method 2).

Reads WSB sentiment from the shared GeminiFullAnalysis cache.
Does NOT call Gemini itself — the competition manager handles that.
Generates BUY/SELL stock signals based on Reddit crowd sentiment.
"""

import logging
from typing import Optional

import pandas as pd

from src.strategies.base import BaseStrategy, Signal, SignalType
from src.options.gemini_client import GeminiFullAnalysis

logger = logging.getLogger(__name__)


class WSBStockStrategy(BaseStrategy):
    """Converts WSB Reddit sentiment into stock BUY/SELL signals.

    Reads the wsb_sentiment and wsb_confidence fields from the shared
    GeminiFullAnalysis result. Combines with post-level metrics
    (count, upvotes, comments) for confidence scaling.
    """

    def __init__(
        self,
        name: str = "wsb_stock",
        min_posts: int = 3,
        min_confidence: float = 0.40,
    ):
        super().__init__(name=name, weight=1.0)
        self.min_posts = min_posts
        self.min_confidence = min_confidence

        # Shared analysis cache — set by the competition manager
        self._analysis_cache: dict[str, GeminiFullAnalysis] = {}
        self._post_stats: dict[str, dict] = {}  # {symbol: {count, avg_score, total_comments}}

    def set_analysis(self, symbol: str, analysis: GeminiFullAnalysis, post_stats: dict):
        """Called by competition manager after shared Gemini call."""
        self._analysis_cache[symbol] = analysis
        self._post_stats[symbol] = post_stats

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate signal from cached WSB analysis."""
        analysis = self._analysis_cache.get(symbol)
        if not analysis:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD,
                          confidence=0.0, reason="No WSB analysis available")

        stats = self._post_stats.get(symbol, {})
        post_count = stats.get("count", 0)
        avg_score = stats.get("avg_score", 0)
        total_comments = stats.get("total_comments", 0)

        # Need minimum post count
        if post_count < self.min_posts:
            return Signal(symbol=symbol, signal_type=SignalType.HOLD,
                          confidence=0.0,
                          reason=f"Only {post_count} WSB posts (need {self.min_posts})")

        # Build composite confidence:
        # 50% from Gemini's WSB confidence
        # 20% from post count (scaled 3-25 posts)
        # 15% from avg upvote score (scaled 0-500)
        # 15% from total comments (scaled 0-1000)
        gemini_conf = analysis.wsb_confidence

        post_count_score = min(1.0, max(0.0, (post_count - 3) / 22))
        upvote_score = min(1.0, max(0.0, avg_score / 500))
        comment_score = min(1.0, max(0.0, total_comments / 1000))

        composite = (
            gemini_conf * 0.50 +
            post_count_score * 0.20 +
            upvote_score * 0.15 +
            comment_score * 0.15
        )
        composite = max(0.0, min(1.0, composite))

        # Determine signal direction
        sentiment = analysis.wsb_sentiment
        if sentiment == "bullish" and composite >= self.min_confidence:
            signal_type = SignalType.BUY
        elif sentiment == "bearish" and composite >= self.min_confidence:
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD

        current_price = float(data["close"].iloc[-1]) if len(data) > 0 else None

        themes_str = ", ".join(analysis.key_themes[:3]) if analysis.key_themes else "none"
        reason = (
            f"WSB {sentiment} ({post_count} posts, "
            f"avg_score={avg_score:.0f}, {total_comments} comments). "
            f"Themes: {themes_str}"
        )

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=composite,
            price=current_price,
            reason=reason,
        )
