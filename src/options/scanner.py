"""
Options opportunity scanner.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .models import OptionType, OptionContract, OptionChain
from .client import OptionsClient
from .strategies import StrategySignal, OptionsStrategyManager

logger = logging.getLogger(__name__)


def get_sentiment_strategy():
    """Lazy-load sentiment strategy to avoid circular imports."""
    try:
        from src.strategies.sentiment import SentimentStrategy
        return SentimentStrategy()
    except ImportError as e:
        logger.debug(f"Sentiment strategy not available: {e}")
        return None


@dataclass
class OptionOpportunity:
    """A scanned options opportunity."""
    symbol: str
    underlying_price: float
    opportunity_type: str  # "high_iv", "unusual_volume", "earnings_play", etc.
    signal: Optional[StrategySignal]
    score: float  # 0-100
    details: dict


@dataclass
class ScanCriteria:
    """Criteria for scanning options opportunities."""
    # IV criteria
    min_iv_rank: float = 0.0
    max_iv_rank: float = 1.0
    min_iv: float = 0.0
    max_iv: float = 2.0

    # Volume criteria
    min_volume: int = 0
    min_open_interest: int = 100
    unusual_volume_threshold: float = 2.0  # X times average

    # Price criteria
    min_underlying_price: float = 5.0
    max_underlying_price: float = 500.0

    # Liquidity
    max_spread_pct: float = 0.15

    # DTE
    min_dte: int = 7
    max_dte: int = 60


class OptionsScanner:
    """
    Scans for options trading opportunities.

    Features:
    - High IV rank opportunities (premium selling)
    - Low IV opportunities (volatility buying)
    - Unusual options activity
    - Earnings plays
    """

    def __init__(
        self,
        client: OptionsClient,
        watchlist: Optional[list[str]] = None,
        use_sentiment: bool = True,
    ):
        self.client = client
        self.watchlist = watchlist or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META",
            "NVDA", "TSLA", "AMD", "SPY", "QQQ",
        ]
        self.strategy_manager = OptionsStrategyManager(client)

        # Cache for IV history (for IV rank calculation)
        self._iv_history: dict[str, list[float]] = {}

        # Sentiment analysis
        self.use_sentiment = use_sentiment
        self._sentiment_strategy = None
        if use_sentiment:
            self._sentiment_strategy = get_sentiment_strategy()
            if self._sentiment_strategy:
                logger.info("Sentiment analysis enabled for options scanner")

    def set_watchlist(self, symbols: list[str]) -> None:
        """Update the watchlist."""
        self.watchlist = symbols

    def get_sentiment(self, symbol: str) -> tuple[float, float, str]:
        """
        Get sentiment analysis for a symbol.

        Args:
            symbol: Stock symbol

        Returns:
            Tuple of (sentiment_score, confidence, trend_direction)
            trend_direction is "bullish", "bearish", or "neutral"
        """
        if not self._sentiment_strategy:
            return 0.0, 0.0, "neutral"

        try:
            score, confidence = self._sentiment_strategy.fetch_and_analyze_news(symbol)

            # Determine trend from sentiment
            if score > 0.2 and confidence > 0.3:
                trend = "bullish"
            elif score < -0.2 and confidence > 0.3:
                trend = "bearish"
            else:
                trend = "neutral"

            logger.info(f"Sentiment for {symbol}: score={score:.2f}, confidence={confidence:.2f}, trend={trend}")
            return score, confidence, trend

        except Exception as e:
            logger.debug(f"Failed to get sentiment for {symbol}: {e}")
            return 0.0, 0.0, "neutral"

    def scan_all(
        self,
        criteria: Optional[ScanCriteria] = None,
        market_trend: str = "neutral",
    ) -> list[OptionOpportunity]:
        """
        Scan all watchlist symbols for opportunities.

        Args:
            criteria: Scan criteria (uses defaults if None)
            market_trend: Overall market trend assessment

        Returns:
            List of opportunities sorted by score
        """
        criteria = criteria or ScanCriteria()
        opportunities = []

        for symbol in self.watchlist:
            try:
                symbol_opportunities = self.scan_symbol(
                    symbol=symbol,
                    criteria=criteria,
                    market_trend=market_trend,
                )
                opportunities.extend(symbol_opportunities)
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")

        # Sort by score
        opportunities.sort(key=lambda x: x.score, reverse=True)

        return opportunities

    def scan_symbol(
        self,
        symbol: str,
        criteria: ScanCriteria,
        market_trend: str = "neutral",
    ) -> list[OptionOpportunity]:
        """Scan a single symbol for opportunities."""
        opportunities = []

        # Get option chain
        chain = self.client.get_option_chain(
            underlying=symbol,
            expiration_date_gte=date.today() + timedelta(days=criteria.min_dte),
            expiration_date_lte=date.today() + timedelta(days=criteria.max_dte),
        )

        logger.debug(f"scan_symbol: {symbol} - got {len(chain.contracts)} contracts, price=${chain.underlying_price:.2f}")

        if not chain.contracts:
            logger.debug(f"scan_symbol: {symbol} - no contracts found")
            return []

        # Check underlying price
        if not (criteria.min_underlying_price <= chain.underlying_price <= criteria.max_underlying_price):
            logger.debug(f"scan_symbol: {symbol} - price ${chain.underlying_price:.2f} outside range")
            return []

        # Calculate IV metrics
        avg_iv = self._calculate_avg_iv(chain)
        iv_rank = self._calculate_iv_rank(symbol, avg_iv)

        # Get news sentiment for this symbol
        sentiment_score, sentiment_conf, sentiment_trend = self.get_sentiment(symbol)

        # Combine technical trend with sentiment
        # If sentiment has high confidence and disagrees with technical, use sentiment
        # If sentiment agrees with technical, boost confidence
        effective_trend = market_trend
        sentiment_boost = 0.0

        if sentiment_conf > 0.4:
            if sentiment_trend != "neutral":
                if market_trend == "neutral":
                    # Technical neutral but sentiment has direction - use sentiment
                    effective_trend = sentiment_trend
                    logger.debug(f"scan_symbol: {symbol} - using sentiment trend '{sentiment_trend}' (technical was neutral)")
                elif sentiment_trend == market_trend:
                    # Agreement - boost opportunity scores
                    sentiment_boost = sentiment_conf * 10  # Up to 10 point boost
                    logger.debug(f"scan_symbol: {symbol} - sentiment agrees with technical, boost={sentiment_boost:.1f}")
                else:
                    # Disagreement - reduce confidence, stick with technical
                    sentiment_boost = -5  # Penalty for conflicting signals
                    logger.debug(f"scan_symbol: {symbol} - sentiment conflicts with technical (sentiment={sentiment_trend}, technical={market_trend})")

        logger.info(f"scan_symbol: {symbol} - avg_iv={avg_iv:.2%}, iv_rank={iv_rank:.2f}, trend={effective_trend}, sentiment={sentiment_score:.2f}")

        # High IV opportunity (premium selling) - relaxed from 0.7 to 0.5
        if iv_rank >= 0.5 and criteria.min_iv_rank <= iv_rank <= criteria.max_iv_rank:
            logger.info(f"scan_symbol: {symbol} - IV rank {iv_rank:.2f} qualifies for high IV opportunity")
            opp = self._create_high_iv_opportunity(
                symbol, chain, avg_iv, iv_rank, effective_trend
            )
            if opp:
                opp.score = max(0, min(100, opp.score + sentiment_boost))
                opp.details["sentiment_score"] = sentiment_score
                opp.details["sentiment_confidence"] = sentiment_conf
                opportunities.append(opp)
                logger.info(f"scan_symbol: {symbol} - Added high IV opportunity, score={opp.score:.0f}")
            else:
                logger.warning(f"scan_symbol: {symbol} - _create_high_iv_opportunity returned None")

        # Low IV opportunity (volatility buying) - relaxed from 0.3 to 0.4
        if iv_rank <= 0.4:
            logger.info(f"scan_symbol: {symbol} - IV rank {iv_rank:.2f} qualifies for low IV opportunity")
            opp = self._create_low_iv_opportunity(
                symbol, chain, avg_iv, iv_rank, effective_trend
            )
            if opp:
                opp.score = max(0, min(100, opp.score + sentiment_boost))
                opp.details["sentiment_score"] = sentiment_score
                opp.details["sentiment_confidence"] = sentiment_conf
                opportunities.append(opp)
                logger.info(f"scan_symbol: {symbol} - Added low IV opportunity, score={opp.score:.0f}")

        # Unusual volume
        unusual_contracts = self._find_unusual_volume(chain, criteria.unusual_volume_threshold)
        if unusual_contracts:
            opp = self._create_unusual_volume_opportunity(
                symbol, chain, unusual_contracts
            )
            if opp:
                opp.details["sentiment_score"] = sentiment_score
                opp.details["sentiment_confidence"] = sentiment_conf
                opportunities.append(opp)

        # News-driven opportunity (strong sentiment with high confidence)
        if abs(sentiment_score) > 0.4 and sentiment_conf > 0.5:
            opp = self._create_sentiment_opportunity(
                symbol, chain, sentiment_score, sentiment_conf, sentiment_trend
            )
            if opp:
                opportunities.append(opp)

        # Check for strategy signals
        signals = self.strategy_manager.analyze_all(
            underlying=symbol,
            underlying_price=chain.underlying_price,
            volatility=avg_iv,
            trend=effective_trend,  # Use sentiment-adjusted trend
        )

        logger.info(f"scan_symbol: {symbol} - strategy_manager.analyze_all returned {len(signals)} signals")

        for signal in signals:
            # Apply sentiment boost to strategy scores
            score = max(0, min(100, (signal.confidence * 100) + sentiment_boost))
            opportunities.append(OptionOpportunity(
                symbol=symbol,
                underlying_price=chain.underlying_price,
                opportunity_type=f"strategy_{signal.strategy_name.lower()}",
                signal=signal,
                score=score,
                details={
                    "strategy": signal.strategy_name,
                    "spread_type": signal.spread_type.value,
                    "rationale": signal.rationale,
                    "expected_profit": signal.expected_profit,
                    "max_loss": signal.max_loss,
                    "sentiment_score": sentiment_score,
                    "sentiment_confidence": sentiment_conf,
                },
            ))

        return opportunities

    def _calculate_avg_iv(self, chain: OptionChain) -> float:
        """Calculate average IV for the chain."""
        ivs = [
            c.implied_volatility
            for c in chain.contracts
            if c.implied_volatility and c.implied_volatility > 0
        ]

        if not ivs:
            return 0.30  # Default 30%

        return sum(ivs) / len(ivs)

    def _calculate_iv_rank(self, symbol: str, current_iv: float) -> float:
        """
        Calculate IV rank (0-1).

        IV Rank = (Current IV - 52 Week Low) / (52 Week High - 52 Week Low)
        """
        # Update history
        if symbol not in self._iv_history:
            self._iv_history[symbol] = []

        self._iv_history[symbol].append(current_iv)

        # Keep only last 252 trading days (roughly 1 year)
        if len(self._iv_history[symbol]) > 252:
            self._iv_history[symbol] = self._iv_history[symbol][-252:]

        history = self._iv_history[symbol]
        if len(history) < 20:
            # Not enough history, estimate based on current IV
            if current_iv < 0.20:
                return 0.2
            elif current_iv < 0.30:
                return 0.4
            elif current_iv < 0.40:
                return 0.6
            else:
                return 0.8

        iv_min = min(history)
        iv_max = max(history)

        if iv_max == iv_min:
            return 0.5

        return (current_iv - iv_min) / (iv_max - iv_min)

    def _find_unusual_volume(
        self,
        chain: OptionChain,
        threshold: float,
    ) -> list[OptionContract]:
        """Find contracts with unusual volume."""
        unusual = []

        for contract in chain.contracts:
            # Ensure numeric types for comparison (API may return strings)
            try:
                volume = int(contract.volume) if contract.volume else 0
                open_interest = int(contract.open_interest) if contract.open_interest else 0
            except (ValueError, TypeError):
                continue

            if open_interest > 0 and volume > 0:
                volume_ratio = volume / open_interest
                if volume_ratio >= threshold:
                    unusual.append(contract)

        return unusual

    def _create_high_iv_opportunity(
        self,
        symbol: str,
        chain: OptionChain,
        avg_iv: float,
        iv_rank: float,
        market_trend: str,
    ) -> Optional[OptionOpportunity]:
        """Create opportunity for high IV environment."""
        signal = self.strategy_manager.get_best_signal(
            underlying=symbol,
            underlying_price=chain.underlying_price,
            volatility=avg_iv,
            trend=market_trend,
        )

        logger.info(f"_create_high_iv_opportunity: {symbol} - get_best_signal returned {'signal: ' + signal.strategy_name if signal else 'None'}")

        # Prefer income strategies in high IV
        score = iv_rank * 100

        return OptionOpportunity(
            symbol=symbol,
            underlying_price=chain.underlying_price,
            opportunity_type="high_iv",
            signal=signal,
            score=score,
            details={
                "iv": avg_iv,
                "iv_rank": iv_rank,
                "suggestion": "Premium selling opportunity - consider iron condors, credit spreads, or cash-secured puts",
            },
        )

    def _create_low_iv_opportunity(
        self,
        symbol: str,
        chain: OptionChain,
        avg_iv: float,
        iv_rank: float,
        market_trend: str,
    ) -> Optional[OptionOpportunity]:
        """Create opportunity for low IV environment."""
        signal = self.strategy_manager.get_best_signal(
            underlying=symbol,
            underlying_price=chain.underlying_price,
            volatility=avg_iv,
            trend=market_trend,
        )

        score = (1 - iv_rank) * 80  # Lower weight than high IV

        return OptionOpportunity(
            symbol=symbol,
            underlying_price=chain.underlying_price,
            opportunity_type="low_iv",
            signal=signal,
            score=score,
            details={
                "iv": avg_iv,
                "iv_rank": iv_rank,
                "suggestion": "Volatility buying opportunity - consider long straddles or strangles",
            },
        )

    def _create_unusual_volume_opportunity(
        self,
        symbol: str,
        chain: OptionChain,
        unusual_contracts: list[OptionContract],
    ) -> Optional[OptionOpportunity]:
        """Create opportunity for unusual options activity."""
        # Analyze the unusual contracts
        call_volume = sum(c.volume for c in unusual_contracts if c.option_type == OptionType.CALL)
        put_volume = sum(c.volume for c in unusual_contracts if c.option_type == OptionType.PUT)

        if call_volume + put_volume == 0:
            return None

        call_put_ratio = call_volume / put_volume if put_volume > 0 else float("inf")

        # Determine sentiment
        if call_put_ratio > 2:
            sentiment = "bullish"
        elif call_put_ratio < 0.5:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        # Find the most unusual contract
        most_unusual = max(
            unusual_contracts,
            key=lambda c: c.volume / c.open_interest if c.open_interest > 0 else 0,
        )

        score = min(80, len(unusual_contracts) * 10)

        return OptionOpportunity(
            symbol=symbol,
            underlying_price=chain.underlying_price,
            opportunity_type="unusual_volume",
            signal=None,
            score=score,
            details={
                "unusual_contracts": len(unusual_contracts),
                "call_put_ratio": call_put_ratio,
                "sentiment": sentiment,
                "most_unusual": str(most_unusual),
                "suggestion": f"Unusual {sentiment} activity detected - investigate before trading",
            },
        )

    def _create_sentiment_opportunity(
        self,
        symbol: str,
        chain: OptionChain,
        sentiment_score: float,
        sentiment_conf: float,
        sentiment_trend: str,
    ) -> Optional[OptionOpportunity]:
        """Create opportunity based on strong news sentiment."""
        # Get a directional signal based on sentiment
        signal = self.strategy_manager.get_best_signal(
            underlying=symbol,
            underlying_price=chain.underlying_price,
            volatility=0.30,  # Use moderate IV assumption
            trend=sentiment_trend,
        )

        # Score based on sentiment strength and confidence
        score = abs(sentiment_score) * sentiment_conf * 100

        direction = "bullish" if sentiment_score > 0 else "bearish"

        if sentiment_trend == "bullish":
            suggestion = "Strong bullish news sentiment - consider bull call spreads or long calls"
        else:
            suggestion = "Strong bearish news sentiment - consider bear put spreads or long puts"

        return OptionOpportunity(
            symbol=symbol,
            underlying_price=chain.underlying_price,
            opportunity_type="news_sentiment",
            signal=signal,
            score=score,
            details={
                "sentiment_score": sentiment_score,
                "sentiment_confidence": sentiment_conf,
                "direction": direction,
                "suggestion": suggestion,
            },
        )

    def scan_for_earnings(
        self,
        earnings_dates: dict[str, date],
    ) -> list[OptionOpportunity]:
        """
        Scan for earnings play opportunities.

        Args:
            earnings_dates: Dictionary of symbol to earnings date

        Returns:
            List of earnings opportunities
        """
        opportunities = []
        today = date.today()

        for symbol, earnings_date in earnings_dates.items():
            days_to_earnings = (earnings_date - today).days

            # Only consider earnings in next 14 days
            if not (1 <= days_to_earnings <= 14):
                continue

            try:
                chain = self.client.get_option_chain(
                    underlying=symbol,
                    expiration_date_gte=earnings_date,
                    expiration_date_lte=earnings_date + timedelta(days=7),
                )

                if not chain.contracts:
                    continue

                avg_iv = self._calculate_avg_iv(chain)

                # Earnings plays benefit from high IV
                score = min(90, days_to_earnings * 5 + avg_iv * 100)

                opportunities.append(OptionOpportunity(
                    symbol=symbol,
                    underlying_price=chain.underlying_price,
                    opportunity_type="earnings",
                    signal=None,
                    score=score,
                    details={
                        "earnings_date": earnings_date.isoformat(),
                        "days_to_earnings": days_to_earnings,
                        "iv": avg_iv,
                        "suggestion": f"Earnings in {days_to_earnings} days. "
                                     f"Consider straddles/strangles before or iron condors after.",
                    },
                ))

            except Exception as e:
                logger.error(f"Error scanning {symbol} for earnings: {e}")

        return sorted(opportunities, key=lambda x: x.score, reverse=True)

    def get_top_opportunities(
        self,
        count: int = 10,
        criteria: Optional[ScanCriteria] = None,
        market_trend: str = "neutral",
    ) -> list[OptionOpportunity]:
        """Get top N opportunities from scan."""
        all_opportunities = self.scan_all(criteria, market_trend)
        return all_opportunities[:count]

    def get_opportunities_by_type(
        self,
        opportunity_type: str,
        criteria: Optional[ScanCriteria] = None,
        market_trend: str = "neutral",
    ) -> list[OptionOpportunity]:
        """Get opportunities of a specific type."""
        all_opportunities = self.scan_all(criteria, market_trend)
        return [o for o in all_opportunities if o.opportunity_type == opportunity_type]
