"""Gemini AI client for analyzing WSB sentiment."""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class GeminiSentimentResult:
    """Parsed result from Gemini sentiment analysis."""
    sentiment: str           # "bullish", "bearish", "neutral"
    confidence: float        # 0.0 - 1.0
    suggested_strategy: str  # "long_call", "long_put", etc.
    reasoning: str
    key_themes: list[str] = field(default_factory=list)
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

        prompt = self._build_prompt(symbol, posts_data, underlying_price)

        try:
            model = self._get_model()

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
