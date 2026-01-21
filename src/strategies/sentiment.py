"""
Sentiment analysis strategy using news and social data.
"""

import logging
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .base import BaseStrategy, Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class SentimentData:
    """Sentiment analysis result for a piece of content."""
    source: str  # 'news', 'social', 'analyst'
    symbol: str
    text: str
    sentiment_score: float  # -1 (bearish) to 1 (bullish)
    confidence: float  # 0 to 1
    timestamp: datetime
    url: Optional[str] = None


class SentimentAnalyzer:
    """
    Analyzes text sentiment for trading signals.

    Uses keyword-based analysis with financial context.
    For production, consider integrating with:
    - OpenAI/Anthropic for LLM-based analysis
    - Specialized financial sentiment APIs
    - Twitter/Reddit API for social sentiment
    """

    # Financial sentiment keywords (expanded)
    BULLISH_KEYWORDS = {
        # Strong bullish
        "surge": 0.8, "soar": 0.8, "skyrocket": 0.9, "breakthrough": 0.7,
        "record high": 0.8, "all-time high": 0.9, "rally": 0.7, "boom": 0.7,
        "outperform": 0.6, "beat expectations": 0.7, "upgrade": 0.6,
        "strong buy": 0.8, "bullish": 0.7, "optimistic": 0.5,

        # Moderate bullish
        "growth": 0.4, "gain": 0.4, "rise": 0.3, "increase": 0.3,
        "profit": 0.4, "positive": 0.3, "improve": 0.3, "recover": 0.4,
        "expansion": 0.4, "momentum": 0.3, "upside": 0.5, "buy": 0.3,
        "accumulate": 0.4, "opportunity": 0.3, "undervalued": 0.5,
    }

    BEARISH_KEYWORDS = {
        # Strong bearish
        "crash": -0.9, "plunge": -0.8, "collapse": -0.9, "tank": -0.7,
        "record low": -0.8, "sell-off": -0.7, "panic": -0.8, "crisis": -0.7,
        "downgrade": -0.6, "miss expectations": -0.7, "warning": -0.5,
        "strong sell": -0.8, "bearish": -0.7, "pessimistic": -0.5,

        # Moderate bearish
        "decline": -0.4, "loss": -0.4, "fall": -0.3, "decrease": -0.3,
        "negative": -0.3, "weak": -0.4, "concern": -0.3, "risk": -0.3,
        "slowdown": -0.4, "headwind": -0.4, "downside": -0.5, "sell": -0.3,
        "overvalued": -0.5, "bubble": -0.6, "correction": -0.4,
    }

    INTENSITY_MODIFIERS = {
        "very": 1.3, "extremely": 1.5, "highly": 1.3, "significantly": 1.2,
        "slightly": 0.7, "somewhat": 0.8, "marginally": 0.6,
        "potentially": 0.8, "possibly": 0.7, "likely": 0.9,
    }

    NEGATION_WORDS = {"not", "no", "never", "neither", "hardly", "barely", "without"}

    def analyze_text(self, text: str, symbol: str) -> SentimentData:
        """
        Analyze text for financial sentiment.

        Args:
            text: Text to analyze
            symbol: Symbol being analyzed (for context)

        Returns:
            SentimentData with analysis results
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)

        scores = []
        confidences = []

        # Check for negation context
        negation_positions = set()
        for i, word in enumerate(words):
            if word in self.NEGATION_WORDS:
                # Mark next 3 words as negated
                for j in range(i + 1, min(i + 4, len(words))):
                    negation_positions.add(j)

        # Score keywords
        for i, word in enumerate(words):
            score = 0
            conf = 0

            if word in self.BULLISH_KEYWORDS:
                score = self.BULLISH_KEYWORDS[word]
                conf = 0.7
            elif word in self.BEARISH_KEYWORDS:
                score = self.BEARISH_KEYWORDS[word]
                conf = 0.7

            # Apply negation
            if i in negation_positions and score != 0:
                score = -score * 0.8  # Flip but reduce confidence
                conf *= 0.8

            # Check for intensity modifiers
            if i > 0 and words[i - 1] in self.INTENSITY_MODIFIERS:
                modifier = self.INTENSITY_MODIFIERS[words[i - 1]]
                score *= modifier

            if score != 0:
                scores.append(score)
                confidences.append(conf)

        # Check for multi-word phrases
        for phrase, score in list(self.BULLISH_KEYWORDS.items()) + list(self.BEARISH_KEYWORDS.items()):
            if " " in phrase and phrase in text_lower:
                scores.append(score if score > 0 else self.BEARISH_KEYWORDS.get(phrase, 0))
                confidences.append(0.8)

        # Calculate final sentiment
        if scores:
            # Weighted average by confidence
            total_weight = sum(confidences)
            if total_weight > 0:
                sentiment_score = sum(s * c for s, c in zip(scores, confidences)) / total_weight
            else:
                sentiment_score = 0

            # Confidence based on agreement and count
            avg_confidence = sum(confidences) / len(confidences)
            count_factor = min(len(scores) / 5, 1.0)  # More keywords = more confident
            confidence = avg_confidence * count_factor
        else:
            sentiment_score = 0
            confidence = 0

        return SentimentData(
            source="text_analysis",
            symbol=symbol,
            text=text[:200],  # Truncate for storage
            sentiment_score=float(max(-1, min(1, sentiment_score))),
            confidence=float(confidence),
            timestamp=datetime.now(),
        )

    def analyze_headlines(self, headlines: list[str], symbol: str) -> SentimentData:
        """Analyze multiple headlines and combine."""
        if not headlines:
            return SentimentData(
                source="headlines",
                symbol=symbol,
                text="",
                sentiment_score=0,
                confidence=0,
                timestamp=datetime.now(),
            )

        sentiments = [self.analyze_text(h, symbol) for h in headlines]

        # Recency-weighted average (more recent = higher weight)
        total_score = 0
        total_weight = 0

        for i, sent in enumerate(sentiments):
            weight = sent.confidence * (0.5 + 0.5 * i / len(sentiments))
            total_score += sent.sentiment_score * weight
            total_weight += weight

        if total_weight > 0:
            avg_score = total_score / total_weight
        else:
            avg_score = 0

        avg_confidence = sum(s.confidence for s in sentiments) / len(sentiments)

        return SentimentData(
            source="headlines",
            symbol=symbol,
            text=f"Combined {len(headlines)} headlines",
            sentiment_score=float(avg_score),
            confidence=float(avg_confidence),
            timestamp=datetime.now(),
        )


class SentimentStrategy(BaseStrategy):
    """
    Trading strategy based on sentiment analysis.

    Combines news sentiment, social sentiment, and analyst ratings
    to generate trading signals.
    """

    def __init__(
        self,
        name: str = "sentiment",
        weight: float = 1.0,
        lookback_hours: int = 24,
        min_confidence: float = 0.4,
        bullish_threshold: float = 0.3,
        bearish_threshold: float = -0.3,
    ):
        """
        Args:
            name: Strategy name
            weight: Strategy weight
            lookback_hours: Hours of sentiment data to consider
            min_confidence: Minimum confidence to generate signal
            bullish_threshold: Score threshold for buy signal
            bearish_threshold: Score threshold for sell signal
        """
        super().__init__(name, weight)
        self.lookback_hours = lookback_hours
        self.min_confidence = min_confidence
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold

        self.analyzer = SentimentAnalyzer()
        self._sentiment_cache: dict[str, deque[SentimentData]] = {}
        self._is_trained = True  # No training required
        self._news_provider = None  # Lazy loaded

    @property
    def news_provider(self):
        """Lazy-load news provider."""
        if self._news_provider is None:
            self._news_provider = get_news_provider()
        return self._news_provider

    def fetch_and_analyze_news(self, symbol: str, count: int = 10) -> tuple[float, float]:
        """
        Fetch latest news for a symbol and analyze sentiment.

        Args:
            symbol: Stock symbol
            count: Number of headlines to fetch

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        try:
            headlines = self.news_provider.get_headlines(symbol, count=count)
            if headlines and headlines[0] != f"No news available for {symbol}":
                self.add_headlines(symbol, headlines)
                logger.info(f"Analyzed {len(headlines)} headlines for {symbol}")
            return self.get_current_sentiment(symbol)
        except Exception as e:
            logger.error(f"Failed to fetch news for {symbol}: {e}")
            return 0.0, 0.0

    def add_sentiment_data(self, data: SentimentData) -> None:
        """Add sentiment data point for a symbol."""
        if data.symbol not in self._sentiment_cache:
            self._sentiment_cache[data.symbol] = deque(maxlen=100)
        self._sentiment_cache[data.symbol].append(data)

    def add_headlines(self, symbol: str, headlines: list[str]) -> None:
        """Add headlines for analysis."""
        sentiment = self.analyzer.analyze_headlines(headlines, symbol)
        self.add_sentiment_data(sentiment)

    def get_current_sentiment(self, symbol: str) -> tuple[float, float]:
        """
        Get current aggregate sentiment for a symbol.

        Returns:
            Tuple of (sentiment_score, confidence)
        """
        if symbol not in self._sentiment_cache:
            return 0.0, 0.0

        cutoff = datetime.now() - timedelta(hours=self.lookback_hours)
        recent = [s for s in self._sentiment_cache[symbol] if s.timestamp > cutoff]

        if not recent:
            return 0.0, 0.0

        # Time-weighted average (more recent = higher weight)
        total_score = 0
        total_weight = 0

        now = datetime.now()
        for sent in recent:
            age_hours = (now - sent.timestamp).total_seconds() / 3600
            time_weight = 1 / (1 + age_hours / self.lookback_hours)  # Decay
            weight = sent.confidence * time_weight

            total_score += sent.sentiment_score * weight
            total_weight += weight

        if total_weight > 0:
            avg_score = total_score / total_weight
            avg_confidence = sum(s.confidence for s in recent) / len(recent)
        else:
            avg_score = 0
            avg_confidence = 0

        return float(avg_score), float(avg_confidence)

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """Generate trading signal based on sentiment."""
        sentiment_score, confidence = self.get_current_sentiment(symbol)

        # Get current price
        price = float(data["close"].iloc[-1]) if not data.empty else None
        atr = float(data["atr"].iloc[-1]) if "atr" in data.columns and not data.empty else (price or 0) * 0.02

        # Check confidence threshold
        if confidence < self.min_confidence:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=confidence,
                price=price,
                reason=f"Sentiment confidence too low ({confidence:.2%})",
            )

        # Generate signal based on sentiment
        if sentiment_score > self.bullish_threshold:
            signal_type = SignalType.BUY
            stop_loss = price - (2 * atr) if price else None
            take_profit = price + (3 * atr) if price else None
            reason = f"Bullish sentiment ({sentiment_score:.2f})"

        elif sentiment_score < self.bearish_threshold:
            signal_type = SignalType.SELL
            stop_loss = price + (2 * atr) if price else None
            take_profit = price - (3 * atr) if price else None
            reason = f"Bearish sentiment ({sentiment_score:.2f})"

        else:
            signal_type = SignalType.HOLD
            stop_loss = None
            take_profit = None
            reason = f"Neutral sentiment ({sentiment_score:.2f})"

        # Adjust confidence based on sentiment strength
        signal_confidence = confidence * min(abs(sentiment_score) * 2, 1.0)

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=signal_confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=reason,
        )

    def get_sentiment_summary(self, symbol: str) -> dict:
        """Get detailed sentiment summary for a symbol."""
        if symbol not in self._sentiment_cache:
            return {"symbol": symbol, "data_points": 0}

        cutoff = datetime.now() - timedelta(hours=self.lookback_hours)
        recent = [s for s in self._sentiment_cache[symbol] if s.timestamp > cutoff]

        if not recent:
            return {"symbol": symbol, "data_points": 0}

        sentiment_score, confidence = self.get_current_sentiment(symbol)

        # Breakdown by source
        by_source = {}
        for sent in recent:
            if sent.source not in by_source:
                by_source[sent.source] = []
            by_source[sent.source].append(sent.sentiment_score)

        source_averages = {
            source: sum(scores) / len(scores)
            for source, scores in by_source.items()
        }

        return {
            "symbol": symbol,
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "data_points": len(recent),
            "by_source": source_averages,
            "latest": recent[-1].text if recent else None,
            "trend": self._calculate_sentiment_trend(recent),
        }

    def _calculate_sentiment_trend(self, data: list[SentimentData]) -> str:
        """Calculate if sentiment is improving or deteriorating."""
        if len(data) < 3:
            return "insufficient_data"

        # Compare first half to second half
        mid = len(data) // 2
        first_half = [d.sentiment_score for d in data[:mid]]
        second_half = [d.sentiment_score for d in data[mid:]]

        first_avg = sum(first_half) / len(first_half)
        second_avg = sum(second_half) / len(second_half)

        diff = second_avg - first_avg

        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "deteriorating"
        else:
            return "stable"


class MockNewsProvider:
    """
    Mock news provider for testing.

    In production, replace with actual news API integration:
    - Alpha Vantage News
    - Polygon.io News
    - Benzinga
    - NewsAPI
    """

    SAMPLE_HEADLINES = {
        "AAPL": [
            "Apple reports record quarterly revenue, beats expectations",
            "iPhone sales surge in emerging markets",
            "Apple expands AI features across product line",
        ],
        "MSFT": [
            "Microsoft Azure growth accelerates amid AI boom",
            "Microsoft stock hits all-time high on cloud strength",
            "Analysts upgrade Microsoft citing AI momentum",
        ],
        "TSLA": [
            "Tesla deliveries miss estimates, shares fall",
            "Competition intensifies in EV market",
            "Tesla cuts prices to maintain market share",
        ],
    }

    def get_headlines(self, symbol: str, count: int = 5) -> list[str]:
        """Get mock headlines for a symbol."""
        if symbol in self.SAMPLE_HEADLINES:
            return self.SAMPLE_HEADLINES[symbol][:count]
        return [f"No news available for {symbol}"]


def get_news_provider():
    """
    Get the appropriate news provider.

    Returns real Alpaca news provider if available, falls back to mock.
    """
    try:
        from src.data.news import NewsProvider
        return NewsProvider()
    except ImportError:
        logger.warning("Alpaca news provider not available, using mock")
        return MockNewsProvider()
