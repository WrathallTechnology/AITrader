"""WSB Sentiment Strategy using Reddit + Gemini AI."""

import logging
from typing import Optional

from .strategies import OptionsStrategy, StrategySignal
from .models import OptionType, SpreadType
from .client import OptionsClient
from .wsb_reddit import WSBRedditFetcher
from .gemini_client import GeminiSentimentClient

logger = logging.getLogger(__name__)

# Map Gemini's suggested_strategy string to SpreadType enum
STRATEGY_MAP = {
    "long_call": SpreadType.LONG_CALL,
    "long_put": SpreadType.LONG_PUT,
    "bull_call_spread": SpreadType.BULL_CALL_SPREAD,
    "bear_put_spread": SpreadType.BEAR_PUT_SPREAD,
    "iron_condor": SpreadType.IRON_CONDOR,
    "straddle": SpreadType.STRADDLE,
    "strangle": SpreadType.STRANGLE,
}


class WSBSentimentStrategy(OptionsStrategy):
    """
    Options strategy driven by WallStreetBets sentiment analyzed by Gemini AI.

    Flow:
    1. Fetch recent WSB posts for the symbol (cached 10min)
    2. Send posts to Gemini AI for sentiment analysis
    3. Scale confidence based on post quality/engagement
    4. Select appropriate option contracts from the chain
    5. Return StrategySignal (or None if no signal)
    """

    def __init__(
        self,
        client: OptionsClient,
        min_confidence: float = 0.35,
        min_posts: int = 3,
        reddit_fetcher: Optional[WSBRedditFetcher] = None,
        gemini_client: Optional[GeminiSentimentClient] = None,
    ):
        super().__init__(client)
        self.min_confidence = min_confidence
        self.min_posts = min_posts
        self.reddit_fetcher = reddit_fetcher or WSBRedditFetcher()
        self.gemini_client = gemini_client or GeminiSentimentClient()

    @property
    def name(self) -> str:
        return "WSB_Sentiment"

    @property
    def is_available(self) -> bool:
        """Check if both APIs are configured."""
        return self.reddit_fetcher.is_configured and self.gemini_client.is_configured

    def analyze(
        self,
        underlying: str,
        underlying_price: float,
        volatility: float,
        trend: str,
    ) -> Optional[StrategySignal]:
        """Analyze WSB sentiment for a symbol and generate options signal."""
        if not self.is_available:
            return None

        # Step 1: Fetch WSB data
        wsb_data = self.reddit_fetcher.fetch_symbol_data(underlying)
        if wsb_data is None or len(wsb_data.posts) < self.min_posts:
            return None

        # Step 2: Send to Gemini for analysis
        posts_for_gemini = [
            {
                "title": p.title,
                "body": p.body,
                "score": p.score,
                "upvote_ratio": p.upvote_ratio,
                "num_comments": p.num_comments,
                "flair": p.flair,
            }
            for p in wsb_data.posts
        ]

        gemini_result = self.gemini_client.analyze_wsb_sentiment(
            symbol=underlying,
            posts_data=posts_for_gemini,
            underlying_price=underlying_price,
        )

        if gemini_result is None:
            logger.warning(f"WSB strategy: Gemini analysis failed for {underlying}")
            return None

        # Step 3: Scale confidence
        scaled_confidence = self._scale_confidence(gemini_result, wsb_data)

        if scaled_confidence < self.min_confidence:
            logger.info(
                f"WSB strategy: {underlying} confidence too low "
                f"({scaled_confidence:.2f} < {self.min_confidence})"
            )
            return None

        logger.info(
            f"WSB strategy: {underlying} sentiment={gemini_result.sentiment}, "
            f"confidence={scaled_confidence:.2f}, "
            f"strategy={gemini_result.suggested_strategy}, "
            f"posts={len(wsb_data.posts)}"
        )

        # Step 4: Build signal with contracts
        return self._build_signal(
            underlying=underlying,
            underlying_price=underlying_price,
            gemini_result=gemini_result,
            wsb_data=wsb_data,
            confidence=scaled_confidence,
        )

    def _scale_confidence(self, gemini_result, wsb_data) -> float:
        """
        Scale Gemini's confidence based on WSB data quality.

        Weights: Gemini 50%, post count 20%, upvote ratio 15%, comments 15%
        """
        base = gemini_result.confidence

        # Post count factor: 3 posts = 0.5x, 10+ posts = 1.0x
        post_factor = min(len(wsb_data.posts) / 10, 1.0)
        post_factor = max(post_factor, 0.5)

        # Engagement factor: upvote ratio (already 0-1)
        engagement_factor = wsb_data.avg_upvote_ratio

        # Comment factor: 50+ avg comments = high engagement
        avg_comments = wsb_data.total_comments / max(len(wsb_data.posts), 1)
        comment_factor = min(avg_comments / 50, 1.0)
        comment_factor = max(comment_factor, 0.5)

        scaled = base * 0.5 + post_factor * 0.2 + engagement_factor * 0.15 + comment_factor * 0.15
        return max(0.0, min(1.0, scaled))

    def _build_signal(
        self,
        underlying: str,
        underlying_price: float,
        gemini_result,
        wsb_data,
        confidence: float,
    ) -> Optional[StrategySignal]:
        """Build a StrategySignal with actual option contracts."""
        spread_type = STRATEGY_MAP.get(
            gemini_result.suggested_strategy,
            SpreadType.LONG_CALL,
        )

        if spread_type in (SpreadType.LONG_CALL, SpreadType.BULL_CALL_SPREAD):
            return self._build_bullish_signal(
                underlying, underlying_price, spread_type,
                gemini_result, wsb_data, confidence,
            )
        elif spread_type in (SpreadType.LONG_PUT, SpreadType.BEAR_PUT_SPREAD):
            return self._build_bearish_signal(
                underlying, underlying_price, spread_type,
                gemini_result, wsb_data, confidence,
            )
        elif spread_type in (SpreadType.STRADDLE, SpreadType.STRANGLE):
            return self._build_volatility_signal(
                underlying, underlying_price, spread_type,
                gemini_result, wsb_data, confidence,
            )
        else:
            # iron_condor - skip for now, complex multi-leg
            # Fall back to directional based on sentiment
            if gemini_result.sentiment == "bullish":
                return self._build_bullish_signal(
                    underlying, underlying_price, SpreadType.LONG_CALL,
                    gemini_result, wsb_data, confidence,
                )
            elif gemini_result.sentiment == "bearish":
                return self._build_bearish_signal(
                    underlying, underlying_price, SpreadType.LONG_PUT,
                    gemini_result, wsb_data, confidence,
                )
            return None

    def _build_bullish_signal(
        self, underlying, price, spread_type, gemini_result, wsb_data, confidence,
    ) -> Optional[StrategySignal]:
        """Select contracts for bullish plays (long call or bull call spread)."""
        contracts = self.client.find_contracts(
            underlying=underlying,
            option_type=OptionType.CALL,
            min_days=14,
            max_days=45,
            delta_target=0.35,
            min_open_interest=10,
            max_spread_pct=0.20,
        )

        if not contracts:
            logger.info(f"WSB strategy: no suitable CALL contracts for {underlying}")
            return None

        best = contracts[0]

        if spread_type == SpreadType.BULL_CALL_SPREAD:
            # Try building a spread
            spread_signal = self._build_call_spread(
                underlying, price, best, gemini_result, wsb_data, confidence,
            )
            if spread_signal:
                return spread_signal
            # Fall through to simple long call

        contract_value = best.contract_value
        if contract_value is None:
            return None

        return StrategySignal(
            strategy_name=self.name,
            underlying=underlying,
            action="open",
            spread_type=SpreadType.LONG_CALL,
            contracts=[best],
            confidence=confidence,
            expected_profit=None,
            max_loss=contract_value,
            rationale=self._build_rationale(gemini_result, wsb_data, "bullish"),
        )

    def _build_bearish_signal(
        self, underlying, price, spread_type, gemini_result, wsb_data, confidence,
    ) -> Optional[StrategySignal]:
        """Select contracts for bearish plays (long put or bear put spread)."""
        contracts = self.client.find_contracts(
            underlying=underlying,
            option_type=OptionType.PUT,
            min_days=14,
            max_days=45,
            delta_target=0.35,
            min_open_interest=10,
            max_spread_pct=0.20,
        )

        if not contracts:
            logger.info(f"WSB strategy: no suitable PUT contracts for {underlying}")
            return None

        best = contracts[0]

        if spread_type == SpreadType.BEAR_PUT_SPREAD:
            spread_signal = self._build_put_spread(
                underlying, price, best, gemini_result, wsb_data, confidence,
            )
            if spread_signal:
                return spread_signal

        contract_value = best.contract_value
        if contract_value is None:
            return None

        return StrategySignal(
            strategy_name=self.name,
            underlying=underlying,
            action="open",
            spread_type=SpreadType.LONG_PUT,
            contracts=[best],
            confidence=confidence,
            expected_profit=None,
            max_loss=contract_value,
            rationale=self._build_rationale(gemini_result, wsb_data, "bearish"),
        )

    def _build_volatility_signal(
        self, underlying, price, spread_type, gemini_result, wsb_data, confidence,
    ) -> Optional[StrategySignal]:
        """Select contracts for volatility plays (straddle)."""
        # Get ATM call
        calls = self.client.find_contracts(
            underlying=underlying,
            option_type=OptionType.CALL,
            min_days=14,
            max_days=45,
            delta_target=0.50,  # ATM
            min_open_interest=10,
            max_spread_pct=0.20,
        )
        # Get ATM put
        puts = self.client.find_contracts(
            underlying=underlying,
            option_type=OptionType.PUT,
            min_days=14,
            max_days=45,
            delta_target=0.50,
            min_open_interest=10,
            max_spread_pct=0.20,
        )

        if not calls or not puts:
            return None

        call = calls[0]
        put = puts[0]

        call_value = call.contract_value
        put_value = put.contract_value
        if call_value is None or put_value is None:
            return None

        total_cost = call_value + put_value

        return StrategySignal(
            strategy_name=self.name,
            underlying=underlying,
            action="open",
            spread_type=SpreadType.STRADDLE,
            contracts=[call, put],
            confidence=confidence,
            expected_profit=None,
            max_loss=total_cost,
            rationale=self._build_rationale(gemini_result, wsb_data, "volatile"),
        )

    def _build_call_spread(
        self, underlying, price, long_call, gemini_result, wsb_data, confidence,
    ) -> Optional[StrategySignal]:
        """Build a bull call spread."""
        try:
            chain = self.get_chain(underlying, min_days=14, max_days=45)
            if not chain:
                return None

            short_strike_target = long_call.strike * 1.05
            calls = chain.get_calls(long_call.expiration)
            if not calls:
                return None

            short_call = min(calls, key=lambda c: abs(c.strike - short_strike_target))
            if short_call.strike <= long_call.strike:
                return None

            long_mid = long_call.mid_price
            short_mid = short_call.mid_price
            if long_mid is None or short_mid is None:
                return None

            net_debit = long_mid - short_mid
            if net_debit <= 0:
                return None

            max_profit = (short_call.strike - long_call.strike - net_debit) * 100
            max_loss = net_debit * 100

            return StrategySignal(
                strategy_name=self.name,
                underlying=underlying,
                action="open",
                spread_type=SpreadType.BULL_CALL_SPREAD,
                contracts=[long_call, short_call],
                confidence=confidence,
                expected_profit=max_profit,
                max_loss=max_loss,
                rationale=self._build_rationale(gemini_result, wsb_data, "bullish"),
            )
        except Exception as e:
            logger.debug(f"WSB strategy: failed to build call spread for {underlying}: {e}")
            return None

    def _build_put_spread(
        self, underlying, price, long_put, gemini_result, wsb_data, confidence,
    ) -> Optional[StrategySignal]:
        """Build a bear put spread."""
        try:
            chain = self.get_chain(underlying, min_days=14, max_days=45)
            if not chain:
                return None

            short_strike_target = long_put.strike * 0.95
            puts = chain.get_puts(long_put.expiration)
            if not puts:
                return None

            short_put = min(puts, key=lambda c: abs(c.strike - short_strike_target))
            if short_put.strike >= long_put.strike:
                return None

            long_mid = long_put.mid_price
            short_mid = short_put.mid_price
            if long_mid is None or short_mid is None:
                return None

            net_debit = long_mid - short_mid
            if net_debit <= 0:
                return None

            max_profit = (long_put.strike - short_put.strike - net_debit) * 100
            max_loss = net_debit * 100

            return StrategySignal(
                strategy_name=self.name,
                underlying=underlying,
                action="open",
                spread_type=SpreadType.BEAR_PUT_SPREAD,
                contracts=[long_put, short_put],
                confidence=confidence,
                expected_profit=max_profit,
                max_loss=max_loss,
                rationale=self._build_rationale(gemini_result, wsb_data, "bearish"),
            )
        except Exception as e:
            logger.debug(f"WSB strategy: failed to build put spread for {underlying}: {e}")
            return None

    def _build_rationale(self, gemini_result, wsb_data, direction: str) -> str:
        """Build a human-readable rationale string."""
        themes = ", ".join(gemini_result.key_themes[:3]) if gemini_result.key_themes else "general sentiment"
        return (
            f"WSB {direction} ({gemini_result.confidence:.0%}): "
            f"{gemini_result.reasoning[:120]}. "
            f"Themes: {themes}. "
            f"Based on {len(wsb_data.posts)} WSB posts "
            f"(avg score: {wsb_data.avg_score:.0f})."
        )
