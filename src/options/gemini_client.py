"""Gemini AI client for analyzing WSB sentiment."""

import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


class GeminiRateLimiter:
    """Rate limiter for Gemini free tier (10 RPM, 1500 RPD).

    Tracks both local call counts AND server-side 429 errors.
    When a 429 is received, stops all calls until the next hour.
    """

    def __init__(self, max_rpm: int = 8, max_rpd: int = 1400):
        self.max_rpm = max_rpm
        self.max_rpd = max_rpd
        self._minute_timestamps: deque[float] = deque()
        self._daily_count = 0
        self._daily_date: Optional[date] = None
        # Server-side quota exhaustion tracking
        self._quota_exhausted_until: float = 0  # timestamp

    def _reset_daily_if_needed(self):
        today = date.today()
        if self._daily_date != today:
            self._daily_count = 0
            self._daily_date = today
            self._quota_exhausted_until = 0  # Reset on new day

    def report_429(self):
        """Called when a 429 error is received from the API.

        Stops all calls for 1 hour to let the quota recover.
        """
        self._quota_exhausted_until = time.time() + 3600  # 1 hour backoff
        logger.warning("Gemini 429 received — pausing all Gemini calls for 1 hour")

    def wait_if_needed(self) -> bool:
        """Check if we can make a call. Returns False if quota exhausted."""
        self._reset_daily_if_needed()

        # Check if server-side quota is exhausted
        if time.time() < self._quota_exhausted_until:
            remaining = int(self._quota_exhausted_until - time.time())
            logger.debug(f"Gemini quota paused — {remaining}s remaining")
            return False

        if self._daily_count >= self.max_rpd:
            logger.warning(f"Gemini daily limit reached ({self._daily_count}/{self.max_rpd})")
            return False

        now = time.time()
        while self._minute_timestamps and self._minute_timestamps[0] < now - 60:
            self._minute_timestamps.popleft()
        if len(self._minute_timestamps) >= self.max_rpm:
            wait_time = 60 - (now - self._minute_timestamps[0]) + 0.5
            logger.info(f"Gemini rate limit: waiting {wait_time:.1f}s")
            time.sleep(wait_time)
        return True

    def record_call(self):
        """Record that a call was made."""
        self._reset_daily_if_needed()
        self._minute_timestamps.append(time.time())
        self._daily_count += 1

    @property
    def daily_remaining(self) -> int:
        self._reset_daily_if_needed()
        return max(0, self.max_rpd - self._daily_count)


# Global rate limiter shared across all GeminiSentimentClient instances
_rate_limiter = GeminiRateLimiter()


@dataclass
class GeminiSentimentResult:
    """Parsed result from Gemini sentiment analysis."""
    sentiment: str           # "bullish", "bearish", "neutral"
    confidence: float        # 0.0 - 1.0
    suggested_strategy: str  # "long_call", "long_put", etc.
    reasoning: str
    key_themes: list[str] = field(default_factory=list)
    risk_warning: str = ""


@dataclass
class GeminiFullAnalysis:
    """Result from shared comprehensive Gemini analysis.

    Contains both WSB sentiment fields (for Method 2) and
    comprehensive trade signal fields (for Method 3).
    """
    # WSB sentiment (Method 2 reads these)
    wsb_sentiment: str       # "bullish", "bearish", "neutral"
    wsb_confidence: float    # 0.0 - 1.0

    # Comprehensive trade signal (Method 3 reads these)
    trade_signal: str        # "buy", "sell", "hold"
    trade_confidence: float  # 0.0 - 1.0

    # Optional fields (with defaults)
    key_themes: list[str] = field(default_factory=list)
    reasoning: str = ""
    key_factors: list[str] = field(default_factory=list)
    risk_warning: str = ""


class GeminiSentimentClient:
    """
    Uses Google Gemini API to analyze WSB post data
    and return structured sentiment for options trading.
    """

    SYSTEM_PROMPT = """You are a financial sentiment analyzer specializing in options trading.
You analyze Reddit WallStreetBets (WSB) posts about stocks and return structured
JSON assessments. You understand that WSB is a retail investor community known for
aggressive, speculative positions, memes, and sometimes irrational exuberance.

IMPORTANT GUIDELINES:
- WSB sentiment can be a contrarian indicator; extreme hype may precede dumps
- Weight posts by upvotes and comment count (high engagement = stronger signal)
- Distinguish between genuine DD (due diligence) posts and meme/shitposts
- Posts with "DD" or "Technical Analysis" flair carry more weight
- Be skeptical of posts from low-engagement threads
- Factor in the recency of posts (newer posts matter more)
- If there are fewer than 3 meaningful posts, report low confidence
- ALWAYS return valid JSON"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name
        self._model = None

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _get_model(self):
        """Lazy-initialize the Gemini model."""
        if self._model is None:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.SYSTEM_PROMPT,
            )
        return self._model

    def analyze_wsb_sentiment(
        self,
        symbol: str,
        posts_data: list[dict],
        underlying_price: float,
    ) -> Optional[GeminiSentimentResult]:
        """
        Send WSB post data to Gemini and get structured sentiment.

        Args:
            symbol: Stock ticker
            posts_data: List of dicts with title, body, score, upvote_ratio, num_comments, flair
            underlying_price: Current stock price

        Returns:
            GeminiSentimentResult or None on failure
        """
        if not self.is_configured:
            logger.warning("Gemini API key not configured")
            return None

        if not posts_data:
            return None

        if not _rate_limiter.wait_if_needed():
            return None

        prompt = self._build_prompt(symbol, posts_data, underlying_price)

        try:
            model = self._get_model()
            _rate_limiter.record_call()

            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                    "response_mime_type": "application/json",
                },
            )

            return self._parse_response(response.text)

        except Exception as e:
            if "429" in str(e):
                _rate_limiter.report_429()
            logger.error(f"Gemini analysis failed for {symbol}: {e}")
            return None

    def _build_prompt(
        self,
        symbol: str,
        posts_data: list[dict],
        underlying_price: float,
    ) -> str:
        """Build the analysis prompt with WSB data."""
        formatted_posts = []
        for i, post in enumerate(posts_data[:15], 1):
            formatted_posts.append(
                f"Post #{i}:\n"
                f"  Title: {post['title']}\n"
                f"  Body: {post.get('body', 'N/A')[:300]}\n"
                f"  Upvotes: {post['score']}, "
                f"Upvote Ratio: {post.get('upvote_ratio', 'N/A')}, "
                f"Comments: {post['num_comments']}\n"
                f"  Flair: {post.get('flair', 'None')}"
            )

        posts_text = "\n\n".join(formatted_posts)

        return (
            f"Analyze the following {len(posts_data)} WallStreetBets posts about "
            f"{symbol} (current price: ${underlying_price:.2f}).\n\n"
            f"{posts_text}\n\n"
            f"Based on these posts, provide your analysis as JSON with exactly this structure:\n"
            f'{{\n'
            f'    "sentiment": "bullish" | "bearish" | "neutral",\n'
            f'    "confidence": <float 0.0-1.0>,\n'
            f'    "suggested_strategy": "<one of: long_call, long_put, bull_call_spread, '
            f'bear_put_spread, iron_condor, straddle, strangle>",\n'
            f'    "reasoning": "<2-3 sentence explanation>",\n'
            f'    "key_themes": ["<theme1>", "<theme2>", ...],\n'
            f'    "risk_warning": "<brief risk note>"\n'
            f'}}\n\n'
            f"Rules for confidence scoring:\n"
            f"- 0.0-0.3: Few posts, mixed signals, low engagement, mostly memes\n"
            f"- 0.3-0.6: Moderate post count, some agreement, decent engagement\n"
            f"- 0.6-0.8: Many posts, strong agreement, high engagement, DD-backed\n"
            f"- 0.8-1.0: Overwhelming consensus with high-quality DD (very rare)\n\n"
            f"Rules for suggested_strategy:\n"
            f"- Strong bullish + high confidence -> bull_call_spread or long_call\n"
            f"- Strong bearish + high confidence -> bear_put_spread or long_put\n"
            f"- Mixed signals or low confidence -> iron_condor or neutral\n"
            f"- Expect big move but direction unclear -> straddle or strangle"
        )

    def analyze_full(
        self,
        symbol: str,
        price: float,
        technical_indicators: dict,
        price_history: list[dict],
        wsb_posts: list[dict],
    ) -> Optional[GeminiFullAnalysis]:
        """Shared comprehensive analysis for competition methods 2 & 3.

        One Gemini call returns both WSB sentiment analysis AND a
        comprehensive trade recommendation. Each method reads its
        relevant fields from the result.

        Args:
            symbol: Stock ticker
            price: Current stock price
            technical_indicators: Dict of indicator values (rsi, macd, sma_20, etc.)
            price_history: Recent price bars [{close, high, low, volume}, ...]
            wsb_posts: WSB post dicts [{title, body, score, ...}, ...]

        Returns:
            GeminiFullAnalysis or None on failure
        """
        if not self.is_configured:
            logger.warning("Gemini API key not configured")
            return None

        if not _rate_limiter.wait_if_needed():
            return None

        prompt = self._build_full_prompt(symbol, price, technical_indicators, price_history, wsb_posts)

        try:
            model = self._get_model()
            _rate_limiter.record_call()
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.2,
                    "max_output_tokens": 1024,
                    "response_mime_type": "application/json",
                },
            )
            return self._parse_full_response(response.text)

        except Exception as e:
            if "429" in str(e):
                _rate_limiter.report_429()
            logger.error(f"Gemini full analysis failed for {symbol}: {e}")
            return None

    def _build_full_prompt(
        self,
        symbol: str,
        price: float,
        indicators: dict,
        price_history: list[dict],
        wsb_posts: list[dict],
    ) -> str:
        """Build prompt for shared comprehensive analysis."""
        # Technical indicators section
        ind_lines = []
        for key, val in indicators.items():
            if val is not None:
                if isinstance(val, float):
                    ind_lines.append(f"  {key}: {val:.4f}")
                else:
                    ind_lines.append(f"  {key}: {val}")
        indicators_text = "\n".join(ind_lines) if ind_lines else "  (no indicators available)"

        # Price history section (last 5 bars)
        price_lines = []
        for bar in price_history[-5:]:
            price_lines.append(
                f"  Close: ${bar.get('close', 0):.2f}, "
                f"High: ${bar.get('high', 0):.2f}, "
                f"Low: ${bar.get('low', 0):.2f}, "
                f"Vol: {bar.get('volume', 0):,.0f}"
            )
        price_text = "\n".join(price_lines) if price_lines else "  (no price history)"

        # WSB posts section
        if wsb_posts:
            post_lines = []
            for i, post in enumerate(wsb_posts[:10], 1):
                post_lines.append(
                    f"  Post #{i}: \"{post.get('title', '')}\"\n"
                    f"    Score: {post.get('score', 0)}, "
                    f"Comments: {post.get('num_comments', 0)}, "
                    f"Flair: {post.get('flair', 'None')}"
                )
            wsb_text = "\n".join(post_lines)
        else:
            wsb_text = "  (no WSB posts found for this symbol)"

        return (
            f"Analyze {symbol} (current price: ${price:.2f}) comprehensively.\n\n"
            f"TECHNICAL INDICATORS:\n{indicators_text}\n\n"
            f"RECENT PRICE BARS:\n{price_text}\n\n"
            f"WALLSTREETBETS REDDIT POSTS (last 7 days):\n{wsb_text}\n\n"
            f"Provide TWO separate assessments as JSON:\n"
            f"1. WSB SENTIMENT: What is the Reddit crowd's sentiment on this stock?\n"
            f"2. TRADE RECOMMENDATION: Based on ALL data (technicals + price + WSB), "
            f"should someone buy, sell, or hold?\n\n"
            f"Return JSON with exactly this structure:\n"
            f'{{\n'
            f'    "wsb_sentiment": "bullish" | "bearish" | "neutral",\n'
            f'    "wsb_confidence": <float 0.0-1.0>,\n'
            f'    "key_themes": ["<theme1>", "<theme2>"],\n'
            f'    "trade_signal": "buy" | "sell" | "hold",\n'
            f'    "trade_confidence": <float 0.0-1.0>,\n'
            f'    "reasoning": "<2-3 sentence explanation of trade recommendation>",\n'
            f'    "key_factors": ["<factor1>", "<factor2>"],\n'
            f'    "risk_warning": "<brief risk note>"\n'
            f'}}\n\n'
            f"IMPORTANT RULES:\n"
            f"- WSB sentiment should reflect what Reddit thinks, even if you disagree\n"
            f"- Trade signal should be YOUR best recommendation using all data\n"
            f"- If WSB posts are absent, set wsb_sentiment to neutral with low confidence\n"
            f"- Confidence 0.0-0.3 = weak signal, 0.3-0.6 = moderate, 0.6-1.0 = strong\n"
            f"- Be conservative: only high confidence for very clear setups"
        )

    def _parse_full_response(self, response_text: str) -> Optional[GeminiFullAnalysis]:
        """Parse the shared analysis JSON response."""
        try:
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            data = json.loads(text)

            wsb_sent = data.get("wsb_sentiment", "neutral")
            if wsb_sent not in ("bullish", "bearish", "neutral"):
                wsb_sent = "neutral"

            trade_sig = data.get("trade_signal", "hold")
            if trade_sig not in ("buy", "sell", "hold"):
                trade_sig = "hold"

            return GeminiFullAnalysis(
                wsb_sentiment=wsb_sent,
                wsb_confidence=max(0.0, min(1.0, float(data.get("wsb_confidence", 0.0)))),
                key_themes=data.get("key_themes", []),
                trade_signal=trade_sig,
                trade_confidence=max(0.0, min(1.0, float(data.get("trade_confidence", 0.0)))),
                reasoning=data.get("reasoning", ""),
                key_factors=data.get("key_factors", []),
                risk_warning=data.get("risk_warning", ""),
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse Gemini full response: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return None

    def _parse_response(self, response_text: str) -> Optional[GeminiSentimentResult]:
        """Parse Gemini's JSON response."""
        try:
            text = response_text.strip()
            # Remove markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()

            data = json.loads(text)

            sentiment = data.get("sentiment", "neutral")
            if sentiment not in ("bullish", "bearish", "neutral"):
                sentiment = "neutral"

            confidence = float(data.get("confidence", 0.0))
            confidence = max(0.0, min(1.0, confidence))

            suggested = data.get("suggested_strategy", "iron_condor")
            valid_strategies = {
                "long_call", "long_put",
                "bull_call_spread", "bear_put_spread",
                "iron_condor", "straddle", "strangle",
            }
            if suggested not in valid_strategies:
                suggested = "iron_condor"

            return GeminiSentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                suggested_strategy=suggested,
                reasoning=data.get("reasoning", ""),
                key_themes=data.get("key_themes", []),
                risk_warning=data.get("risk_warning", ""),
            )

        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            logger.debug(f"Raw response: {response_text[:500]}")
            return None
