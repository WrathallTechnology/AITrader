"""
Economic and corporate event calendar integration.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


class EventImpact(Enum):
    """Impact level of an event."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(Enum):
    """Types of market events."""
    ECONOMIC = "economic"
    EARNINGS = "earnings"
    DIVIDEND = "dividend"
    SPLIT = "split"
    IPO = "ipo"
    FED = "fed"
    EXPIRATION = "expiration"  # Options expiration
    HOLIDAY = "holiday"


@dataclass
class MarketEvent:
    """A market event."""
    event_type: EventType
    name: str
    datetime: datetime
    impact: EventImpact
    symbol: Optional[str] = None  # None for macro events
    description: str = ""
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None


@dataclass
class EventFilterResult:
    """Result of event-based filtering."""
    should_avoid: bool
    events: list[MarketEvent]
    confidence_multiplier: float
    reason: str


class EconomicCalendar:
    """
    Tracks economic events that affect market volatility.

    Events include:
    - Fed meetings (FOMC)
    - Jobs reports
    - CPI/Inflation data
    - GDP releases
    - Earnings announcements
    """

    # Static calendar of recurring events (approximate dates)
    RECURRING_EVENTS = {
        # Monthly events
        "nfp": {
            "name": "Non-Farm Payrolls",
            "day_of_week": 4,  # Friday
            "week_of_month": 1,  # First week
            "time": "08:30",
            "impact": EventImpact.HIGH,
        },
        "cpi": {
            "name": "CPI (Inflation)",
            "day_of_month_range": (10, 15),
            "time": "08:30",
            "impact": EventImpact.HIGH,
        },
        "retail_sales": {
            "name": "Retail Sales",
            "day_of_month_range": (13, 17),
            "time": "08:30",
            "impact": EventImpact.MEDIUM,
        },
    }

    # FOMC meeting dates (2024-2025 example)
    FOMC_DATES = [
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
        "2026-01-28", "2026-03-18",
    ]

    def __init__(self):
        self._events: list[MarketEvent] = []
        self._earnings_cache: dict[str, list[MarketEvent]] = {}
        self._load_static_events()

    def _load_static_events(self) -> None:
        """Load static/known events."""
        # Add FOMC dates
        for date_str in self.FOMC_DATES:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(hour=14, minute=0)
                self._events.append(MarketEvent(
                    event_type=EventType.FED,
                    name="FOMC Rate Decision",
                    datetime=dt,
                    impact=EventImpact.CRITICAL,
                    description="Federal Reserve interest rate decision",
                ))
            except ValueError:
                continue

        # Add options expiration (third Friday of each month)
        self._add_options_expirations()

    def _add_options_expirations(self) -> None:
        """Add monthly options expiration dates."""
        now = datetime.now()

        for month_offset in range(12):
            month = (now.month + month_offset - 1) % 12 + 1
            year = now.year + (now.month + month_offset - 1) // 12

            # Find third Friday
            first_day = datetime(year, month, 1)
            first_friday = first_day + timedelta(days=(4 - first_day.weekday()) % 7)
            third_friday = first_friday + timedelta(days=14)
            third_friday = third_friday.replace(hour=16, minute=0)

            self._events.append(MarketEvent(
                event_type=EventType.EXPIRATION,
                name="Monthly Options Expiration",
                datetime=third_friday,
                impact=EventImpact.MEDIUM,
                description="Monthly options and futures expiration",
            ))

    def add_event(self, event: MarketEvent) -> None:
        """Add a custom event."""
        self._events.append(event)
        self._events.sort(key=lambda x: x.datetime)

    def add_earnings(
        self,
        symbol: str,
        date: datetime,
        time_of_day: str = "after_close",
    ) -> None:
        """
        Add an earnings announcement.

        Args:
            symbol: Stock symbol
            date: Earnings date
            time_of_day: "before_open", "after_close", or "during_market"
        """
        if time_of_day == "before_open":
            dt = date.replace(hour=7, minute=0)
        elif time_of_day == "after_close":
            dt = date.replace(hour=16, minute=30)
        else:
            dt = date.replace(hour=12, minute=0)

        event = MarketEvent(
            event_type=EventType.EARNINGS,
            name=f"{symbol} Earnings",
            datetime=dt,
            impact=EventImpact.HIGH,
            symbol=symbol,
            description=f"Quarterly earnings report for {symbol}",
        )

        self._events.append(event)
        self._events.sort(key=lambda x: x.datetime)

        # Cache by symbol
        if symbol not in self._earnings_cache:
            self._earnings_cache[symbol] = []
        self._earnings_cache[symbol].append(event)

    def get_upcoming_events(
        self,
        hours_ahead: int = 24,
        event_types: Optional[list[EventType]] = None,
        min_impact: EventImpact = EventImpact.LOW,
    ) -> list[MarketEvent]:
        """
        Get upcoming events within the specified window.

        Args:
            hours_ahead: Hours to look ahead
            event_types: Filter by event types (None for all)
            min_impact: Minimum impact level

        Returns:
            List of upcoming events
        """
        now = datetime.now()
        cutoff = now + timedelta(hours=hours_ahead)

        impact_order = [EventImpact.LOW, EventImpact.MEDIUM, EventImpact.HIGH, EventImpact.CRITICAL]
        min_impact_idx = impact_order.index(min_impact)

        events = []
        for event in self._events:
            if event.datetime < now:
                continue
            if event.datetime > cutoff:
                break

            if event_types and event.event_type not in event_types:
                continue

            event_impact_idx = impact_order.index(event.impact)
            if event_impact_idx < min_impact_idx:
                continue

            events.append(event)

        return events

    def get_events_for_symbol(
        self,
        symbol: str,
        days_ahead: int = 7,
    ) -> list[MarketEvent]:
        """Get events affecting a specific symbol."""
        now = datetime.now()
        cutoff = now + timedelta(days=days_ahead)

        events = []

        # Symbol-specific events
        if symbol in self._earnings_cache:
            for event in self._earnings_cache[symbol]:
                if now <= event.datetime <= cutoff:
                    events.append(event)

        # Macro events affect all symbols
        for event in self._events:
            if event.datetime < now:
                continue
            if event.datetime > cutoff:
                break

            if event.symbol is None:  # Macro event
                events.append(event)

        return sorted(events, key=lambda x: x.datetime)

    def is_high_impact_period(
        self,
        hours_window: int = 2,
    ) -> tuple[bool, Optional[MarketEvent]]:
        """
        Check if we're near a high-impact event.

        Returns:
            Tuple of (is_high_impact, nearest_event)
        """
        events = self.get_upcoming_events(
            hours_ahead=hours_window,
            min_impact=EventImpact.HIGH,
        )

        if events:
            return True, events[0]

        # Also check recent past events (market still reacting)
        now = datetime.now()
        for event in self._events:
            if event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]:
                time_since = now - event.datetime
                if timedelta(0) <= time_since <= timedelta(hours=1):
                    return True, event

        return False, None


class EventFilter:
    """
    Filters trading signals based on upcoming events.
    """

    def __init__(
        self,
        calendar: Optional[EconomicCalendar] = None,
        avoid_high_impact_hours: int = 1,
        avoid_critical_hours: int = 4,
        reduce_near_earnings_days: int = 2,
    ):
        """
        Args:
            calendar: Economic calendar instance
            avoid_high_impact_hours: Hours before/after high impact events to avoid
            avoid_critical_hours: Hours before/after critical events to avoid
            reduce_near_earnings_days: Days before earnings to reduce confidence
        """
        self.calendar = calendar or EconomicCalendar()
        self.avoid_high_impact_hours = avoid_high_impact_hours
        self.avoid_critical_hours = avoid_critical_hours
        self.reduce_near_earnings_days = reduce_near_earnings_days

    def filter(
        self,
        symbol: Optional[str] = None,
        check_time: Optional[datetime] = None,
    ) -> EventFilterResult:
        """
        Check if current time is suitable for trading.

        Args:
            symbol: Symbol to check (for earnings)
            check_time: Time to check (defaults to now)

        Returns:
            EventFilterResult with recommendation
        """
        check_time = check_time or datetime.now()
        should_avoid = False
        confidence_multiplier = 1.0
        reasons = []
        relevant_events = []

        # Check for critical events (FOMC, etc.)
        critical_events = self.calendar.get_upcoming_events(
            hours_ahead=self.avoid_critical_hours,
            min_impact=EventImpact.CRITICAL,
        )

        if critical_events:
            should_avoid = True
            confidence_multiplier = 0.0
            reasons.append(f"Critical event: {critical_events[0].name}")
            relevant_events.extend(critical_events)

        # Check for high impact events
        high_events = self.calendar.get_upcoming_events(
            hours_ahead=self.avoid_high_impact_hours,
            min_impact=EventImpact.HIGH,
        )

        # Filter out already counted critical
        high_events = [e for e in high_events if e.impact == EventImpact.HIGH]

        if high_events and not should_avoid:
            confidence_multiplier *= 0.5
            reasons.append(f"High impact event: {high_events[0].name}")
            relevant_events.extend(high_events)

        # Check symbol-specific events
        if symbol:
            symbol_events = self.calendar.get_events_for_symbol(
                symbol,
                days_ahead=self.reduce_near_earnings_days,
            )

            earnings = [e for e in symbol_events if e.event_type == EventType.EARNINGS]

            if earnings:
                nearest = earnings[0]
                hours_until = (nearest.datetime - check_time).total_seconds() / 3600

                if hours_until < 24:
                    should_avoid = True
                    confidence_multiplier = 0.0
                    reasons.append(f"Earnings within 24h: {symbol}")
                elif hours_until < 48:
                    confidence_multiplier *= 0.3
                    reasons.append(f"Earnings within 48h: {symbol}")
                else:
                    confidence_multiplier *= 0.7
                    reasons.append(f"Earnings upcoming: {symbol}")

                relevant_events.extend(earnings)

        # Check options expiration
        expirations = self.calendar.get_upcoming_events(
            hours_ahead=8,
            event_types=[EventType.EXPIRATION],
        )

        if expirations:
            confidence_multiplier *= 0.8
            reasons.append("Near options expiration")
            relevant_events.extend(expirations)

        return EventFilterResult(
            should_avoid=should_avoid,
            events=relevant_events,
            confidence_multiplier=confidence_multiplier,
            reason="; ".join(reasons) if reasons else "No significant events",
        )

    def get_safe_trading_windows(
        self,
        date: datetime,
        symbol: Optional[str] = None,
    ) -> list[tuple[datetime, datetime]]:
        """
        Get safe trading windows for a given date.

        Returns:
            List of (start, end) datetime tuples
        """
        market_open = date.replace(hour=9, minute=30, second=0)
        market_close = date.replace(hour=16, minute=0, second=0)

        # Get all events for the day
        day_start = date.replace(hour=0, minute=0)
        day_end = date.replace(hour=23, minute=59)

        events = []
        for event in self.calendar._events:
            if day_start <= event.datetime <= day_end:
                if event.impact in [EventImpact.HIGH, EventImpact.CRITICAL]:
                    events.append(event)

        if not events:
            # Full trading day is safe
            return [(market_open, market_close)]

        # Sort events by time
        events.sort(key=lambda x: x.datetime)

        # Build safe windows around events
        windows = []
        current_start = market_open

        for event in events:
            if event.impact == EventImpact.CRITICAL:
                buffer_hours = self.avoid_critical_hours
            else:
                buffer_hours = self.avoid_high_impact_hours

            event_start = event.datetime - timedelta(hours=buffer_hours)
            event_end = event.datetime + timedelta(hours=buffer_hours / 2)

            if current_start < event_start:
                windows.append((
                    max(current_start, market_open),
                    min(event_start, market_close),
                ))

            current_start = max(event_end, current_start)

        # Add final window after last event
        if current_start < market_close:
            windows.append((current_start, market_close))

        # Filter out too-short windows (less than 30 min)
        windows = [(s, e) for s, e in windows if (e - s).seconds >= 1800]

        return windows


# Convenience function to populate common earnings
def add_mega_cap_earnings(calendar: EconomicCalendar, quarter_end: datetime) -> None:
    """
    Add approximate earnings dates for mega-cap stocks.

    Earnings typically occur 2-6 weeks after quarter end.
    """
    mega_caps = [
        ("AAPL", 4),  # weeks after quarter
        ("MSFT", 3),
        ("GOOGL", 4),
        ("AMZN", 4),
        ("META", 4),
        ("NVDA", 3),
        ("TSLA", 3),
    ]

    for symbol, weeks_after in mega_caps:
        earnings_date = quarter_end + timedelta(weeks=weeks_after)
        calendar.add_earnings(symbol, earnings_date, "after_close")
