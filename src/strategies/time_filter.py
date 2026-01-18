"""
Time-based filters for avoiding high-risk trading periods.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class TradingPeriod(Enum):
    """Trading session periods."""
    PRE_MARKET = "pre_market"
    MARKET_OPEN = "market_open"  # First 30 min
    MORNING = "morning"
    MIDDAY = "midday"
    AFTERNOON = "afternoon"
    MARKET_CLOSE = "market_close"  # Last 30 min
    AFTER_HOURS = "after_hours"
    CLOSED = "closed"
    CRYPTO_24H = "crypto_24h"


@dataclass
class TimeFilterResult:
    """Result of time-based filter analysis."""
    can_trade: bool
    period: TradingPeriod
    confidence_multiplier: float  # 0-1, reduces signal confidence
    reasons: list[str]
    next_good_period: Optional[datetime] = None


class TimeFilter:
    """
    Filters trading based on time of day and market conditions.

    Avoids:
    - Market open volatility (first 30 min)
    - Market close volatility (last 30 min)
    - Major news release times
    - Low liquidity periods
    """

    # US market hours (Eastern Time)
    MARKET_OPEN = time(9, 30)
    MARKET_CLOSE = time(16, 0)
    PRE_MARKET_START = time(4, 0)
    AFTER_HOURS_END = time(20, 0)

    # Volatile periods to avoid or reduce
    OPEN_VOLATILITY_MINUTES = 30
    CLOSE_VOLATILITY_MINUTES = 30

    # Common economic release times (ET)
    ECONOMIC_RELEASE_TIMES = [
        time(8, 30),   # Jobs report, CPI, etc.
        time(10, 0),   # ISM, consumer confidence
        time(14, 0),   # FOMC announcements
    ]

    def __init__(
        self,
        timezone: str = "America/New_York",
        avoid_open: bool = True,
        avoid_close: bool = True,
        avoid_economic_releases: bool = True,
        trade_extended_hours: bool = False,
    ):
        """
        Args:
            timezone: Market timezone
            avoid_open: Avoid first 30 minutes
            avoid_close: Avoid last 30 minutes
            avoid_economic_releases: Avoid major release times
            trade_extended_hours: Allow pre/post market trading
        """
        self.tz = ZoneInfo(timezone)
        self.avoid_open = avoid_open
        self.avoid_close = avoid_close
        self.avoid_economic_releases = avoid_economic_releases
        self.trade_extended_hours = trade_extended_hours

        # Upcoming known events (can be updated dynamically)
        self._known_events: list[tuple[datetime, str]] = []

    def add_event(self, event_time: datetime, description: str) -> None:
        """Add a known event to avoid."""
        self._known_events.append((event_time, description))
        self._known_events.sort(key=lambda x: x[0])

    def filter(
        self,
        symbol: str,
        current_time: Optional[datetime] = None,
    ) -> TimeFilterResult:
        """
        Check if now is a good time to trade.

        Args:
            symbol: Trading symbol (to determine if crypto)
            current_time: Time to check (defaults to now)

        Returns:
            TimeFilterResult with trading recommendation
        """
        if current_time is None:
            current_time = datetime.now(self.tz)
        elif current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=self.tz)

        is_crypto = "/" in symbol  # Crypto symbols like 'BTC/USD'

        if is_crypto:
            return self._filter_crypto(current_time)
        else:
            return self._filter_stock(current_time, symbol)

    def _filter_crypto(self, current_time: datetime) -> TimeFilterResult:
        """Filter for crypto (24/7 market)."""
        reasons = []
        confidence_multiplier = 1.0

        # Check for known events
        event_penalty = self._check_events(current_time)
        if event_penalty < 1.0:
            confidence_multiplier *= event_penalty
            reasons.append("Near known event")

        # Weekend liquidity is lower
        if current_time.weekday() >= 5:
            confidence_multiplier *= 0.9
            reasons.append("Weekend - lower liquidity")

        # Late night US time has lower liquidity
        hour = current_time.hour
        if 2 <= hour <= 6:
            confidence_multiplier *= 0.85
            reasons.append("Low liquidity hours")

        return TimeFilterResult(
            can_trade=True,
            period=TradingPeriod.CRYPTO_24H,
            confidence_multiplier=confidence_multiplier,
            reasons=reasons,
        )

    def _filter_stock(self, current_time: datetime, symbol: str) -> TimeFilterResult:
        """Filter for stock market hours."""
        reasons = []
        confidence_multiplier = 1.0
        current_time_only = current_time.time()

        # Determine period
        period = self._get_period(current_time_only)

        # Check if market is closed
        if period == TradingPeriod.CLOSED:
            return TimeFilterResult(
                can_trade=False,
                period=period,
                confidence_multiplier=0.0,
                reasons=["Market closed"],
                next_good_period=self._next_market_open(current_time),
            )

        # Pre/post market
        if period in (TradingPeriod.PRE_MARKET, TradingPeriod.AFTER_HOURS):
            if not self.trade_extended_hours:
                return TimeFilterResult(
                    can_trade=False,
                    period=period,
                    confidence_multiplier=0.0,
                    reasons=["Extended hours trading disabled"],
                    next_good_period=self._next_market_open(current_time),
                )
            else:
                confidence_multiplier *= 0.7
                reasons.append("Extended hours - lower liquidity")

        # Market open volatility
        if period == TradingPeriod.MARKET_OPEN:
            if self.avoid_open:
                confidence_multiplier *= 0.5
                reasons.append("Market open volatility")

        # Market close volatility
        if period == TradingPeriod.MARKET_CLOSE:
            if self.avoid_close:
                confidence_multiplier *= 0.6
                reasons.append("Market close volatility")

        # Economic release times
        if self.avoid_economic_releases:
            release_penalty = self._check_economic_releases(current_time_only)
            if release_penalty < 1.0:
                confidence_multiplier *= release_penalty
                reasons.append("Near economic release time")

        # Known events
        event_penalty = self._check_events(current_time)
        if event_penalty < 1.0:
            confidence_multiplier *= event_penalty
            reasons.append("Near known event")

        # Monday morning - often volatile
        if current_time.weekday() == 0 and period == TradingPeriod.MARKET_OPEN:
            confidence_multiplier *= 0.8
            reasons.append("Monday open")

        # Friday afternoon - lower conviction
        if current_time.weekday() == 4 and period in (TradingPeriod.AFTERNOON, TradingPeriod.MARKET_CLOSE):
            confidence_multiplier *= 0.9
            reasons.append("Friday afternoon")

        can_trade = confidence_multiplier > 0.3

        return TimeFilterResult(
            can_trade=can_trade,
            period=period,
            confidence_multiplier=confidence_multiplier,
            reasons=reasons,
        )

    def _get_period(self, t: time) -> TradingPeriod:
        """Determine current trading period."""
        if t < self.PRE_MARKET_START:
            return TradingPeriod.CLOSED
        elif t < self.MARKET_OPEN:
            return TradingPeriod.PRE_MARKET
        elif t < time(10, 0):
            return TradingPeriod.MARKET_OPEN
        elif t < time(12, 0):
            return TradingPeriod.MORNING
        elif t < time(14, 0):
            return TradingPeriod.MIDDAY
        elif t < time(15, 30):
            return TradingPeriod.AFTERNOON
        elif t < self.MARKET_CLOSE:
            return TradingPeriod.MARKET_CLOSE
        elif t < self.AFTER_HOURS_END:
            return TradingPeriod.AFTER_HOURS
        else:
            return TradingPeriod.CLOSED

    def _check_economic_releases(self, current_time: time) -> float:
        """Check proximity to economic release times."""
        for release_time in self.ECONOMIC_RELEASE_TIMES:
            # Convert to minutes for comparison
            current_minutes = current_time.hour * 60 + current_time.minute
            release_minutes = release_time.hour * 60 + release_time.minute

            diff = abs(current_minutes - release_minutes)

            if diff <= 15:  # Within 15 minutes
                return 0.4
            elif diff <= 30:  # Within 30 minutes
                return 0.7

        return 1.0

    def _check_events(self, current_time: datetime) -> float:
        """Check proximity to known events."""
        for event_time, description in self._known_events:
            if event_time.tzinfo is None:
                event_time = event_time.replace(tzinfo=self.tz)

            diff = abs((current_time - event_time).total_seconds() / 60)

            if diff <= 30:  # Within 30 minutes
                logger.info(f"Near event: {description}")
                return 0.3
            elif diff <= 60:  # Within 1 hour
                return 0.6

        return 1.0

    def _next_market_open(self, current_time: datetime) -> datetime:
        """Calculate next market open time."""
        next_day = current_time + timedelta(days=1)
        next_open = datetime.combine(next_day.date(), self.MARKET_OPEN, tzinfo=self.tz)

        # Skip weekends
        while next_open.weekday() >= 5:
            next_open += timedelta(days=1)

        return next_open


class VolatilityFilter:
    """
    Filters based on current market volatility.

    Reduces position sizes or avoids trading during extreme volatility.
    """

    def __init__(
        self,
        vix_high_threshold: float = 30.0,
        vix_extreme_threshold: float = 40.0,
        atr_multiplier_threshold: float = 2.0,
    ):
        """
        Args:
            vix_high_threshold: VIX level considered high
            vix_extreme_threshold: VIX level considered extreme
            atr_multiplier_threshold: ATR multiple vs average to flag high vol
        """
        self.vix_high = vix_high_threshold
        self.vix_extreme = vix_extreme_threshold
        self.atr_multiplier = atr_multiplier_threshold

    def filter(
        self,
        data: pd.DataFrame,
        vix_value: Optional[float] = None,
    ) -> TimeFilterResult:
        """
        Check current volatility conditions.

        Args:
            data: Price data with ATR indicator
            vix_value: Current VIX value (optional)

        Returns:
            TimeFilterResult with volatility assessment
        """
        reasons = []
        confidence_multiplier = 1.0

        # Check VIX if provided
        if vix_value is not None:
            if vix_value > self.vix_extreme:
                confidence_multiplier *= 0.3
                reasons.append(f"Extreme VIX ({vix_value:.1f})")
            elif vix_value > self.vix_high:
                confidence_multiplier *= 0.6
                reasons.append(f"High VIX ({vix_value:.1f})")

        # Check ATR vs average
        if "atr" in data.columns and len(data) >= 50:
            current_atr = data["atr"].iloc[-1]
            avg_atr = data["atr"].iloc[-50:].mean()

            if avg_atr > 0:
                atr_ratio = current_atr / avg_atr

                if atr_ratio > self.atr_multiplier * 1.5:
                    confidence_multiplier *= 0.4
                    reasons.append(f"Very high ATR ({atr_ratio:.1f}x avg)")
                elif atr_ratio > self.atr_multiplier:
                    confidence_multiplier *= 0.7
                    reasons.append(f"High ATR ({atr_ratio:.1f}x avg)")

        # Check recent price gaps
        if len(data) >= 2:
            gap = abs(data["open"].iloc[-1] - data["close"].iloc[-2])
            avg_range = (data["high"] - data["low"]).iloc[-20:].mean()

            if avg_range > 0 and gap > avg_range * 2:
                confidence_multiplier *= 0.7
                reasons.append("Large gap detected")

        can_trade = confidence_multiplier > 0.2

        return TimeFilterResult(
            can_trade=can_trade,
            period=TradingPeriod.CRYPTO_24H,  # Not time-specific
            confidence_multiplier=confidence_multiplier,
            reasons=reasons,
        )


def get_optimal_trading_windows(
    timezone: str = "America/New_York",
) -> list[tuple[time, time, float]]:
    """
    Get optimal trading windows with confidence scores.

    Returns list of (start_time, end_time, confidence_score).
    Based on historical liquidity and volatility patterns.
    """
    return [
        # Morning session (after open volatility)
        (time(10, 0), time(11, 30), 1.0),
        # Midday (can be slow but stable)
        (time(11, 30), time(14, 0), 0.8),
        # Afternoon (good liquidity)
        (time(14, 0), time(15, 30), 0.95),
    ]
