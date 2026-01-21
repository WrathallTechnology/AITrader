"""
Options data models and types.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional


class OptionType(Enum):
    """Option type - call or put."""
    CALL = "call"
    PUT = "put"


class SpreadType(Enum):
    """Types of option spreads."""
    # Single leg
    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_CALL = "short_call"  # Naked - high risk
    SHORT_PUT = "short_put"  # Cash secured put

    # Vertical spreads
    BULL_CALL_SPREAD = "bull_call_spread"
    BEAR_CALL_SPREAD = "bear_call_spread"
    BULL_PUT_SPREAD = "bull_put_spread"
    BEAR_PUT_SPREAD = "bear_put_spread"

    # Neutral strategies
    IRON_CONDOR = "iron_condor"
    IRON_BUTTERFLY = "iron_butterfly"
    STRADDLE = "straddle"
    STRANGLE = "strangle"

    # Covered strategies
    COVERED_CALL = "covered_call"
    PROTECTIVE_PUT = "protective_put"
    COLLAR = "collar"


class OrderAction(Enum):
    """Order action for options."""
    BUY_TO_OPEN = "buy_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_OPEN = "sell_to_open"
    SELL_TO_CLOSE = "sell_to_close"


@dataclass
class OptionGreeks:
    """Option Greeks for risk analysis."""
    delta: float  # Price sensitivity to underlying
    gamma: float  # Rate of change of delta
    theta: float  # Time decay (daily)
    vega: float  # Sensitivity to volatility
    rho: float  # Sensitivity to interest rates

    @property
    def is_valid(self) -> bool:
        """Check if Greeks are reasonable."""
        return (
            -1 <= self.delta <= 1
            and self.gamma >= 0
            and self.vega >= 0
        )


@dataclass
class OptionContract:
    """Represents a single option contract."""
    symbol: str  # OCC symbol (e.g., "AAPL230120C00150000")
    underlying: str  # Underlying symbol (e.g., "AAPL")
    option_type: OptionType
    strike: float
    expiration: date
    multiplier: int = 100  # Standard contract = 100 shares

    # Pricing
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0

    # Greeks (optional, may not always be available)
    greeks: Optional[OptionGreeks] = None

    # Implied volatility
    implied_volatility: Optional[float] = None

    @property
    def mid_price(self) -> Optional[float]:
        """Get mid-point between bid and ask.

        Returns None if bid/ask data is invalid or missing.
        """
        if self.bid > 0 and self.ask > 0 and self.bid < self.ask:
            return (self.bid + self.ask) / 2
        return None

    @property
    def is_tradeable(self) -> bool:
        """Check if contract has valid quote data for trading.

        A contract is tradeable if it has:
        - Valid bid > 0
        - Valid ask > 0
        - Bid < ask (proper market)
        """
        return self.bid > 0 and self.ask > 0 and self.bid < self.ask

    @property
    def spread(self) -> float:
        """Get bid-ask spread."""
        return self.ask - self.bid if self.ask > self.bid else 0.0

    @property
    def spread_pct(self) -> Optional[float]:
        """Get spread as percentage of mid price."""
        mid = self.mid_price
        if mid is not None and mid > 0:
            return self.spread / mid
        return None

    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        return (self.expiration - date.today()).days

    @property
    def is_itm(self) -> bool:
        """Check if option is in-the-money (requires underlying price)."""
        # This needs underlying price - placeholder
        return False

    @property
    def contract_value(self) -> Optional[float]:
        """Total value of one contract at mid price."""
        mid = self.mid_price
        if mid is not None:
            return mid * self.multiplier
        return None

    def intrinsic_value(self, underlying_price: float) -> float:
        """Calculate intrinsic value given underlying price."""
        if self.option_type == OptionType.CALL:
            return max(0, underlying_price - self.strike)
        else:
            return max(0, self.strike - underlying_price)

    def extrinsic_value(self, underlying_price: float) -> Optional[float]:
        """Calculate extrinsic (time) value."""
        mid = self.mid_price
        if mid is not None:
            return max(0, mid - self.intrinsic_value(underlying_price))
        return None

    def moneyness(self, underlying_price: float) -> float:
        """
        Calculate moneyness.
        > 1 = ITM for calls, OTM for puts
        < 1 = OTM for calls, ITM for puts
        """
        return underlying_price / self.strike

    def __str__(self) -> str:
        return (
            f"{self.underlying} {self.expiration.strftime('%m/%d')} "
            f"${self.strike:.0f} {self.option_type.value.upper()}"
        )


@dataclass
class OptionChain:
    """Collection of option contracts for an underlying."""
    underlying: str
    underlying_price: float
    expirations: list[date]
    contracts: list[OptionContract]
    timestamp: datetime = field(default_factory=datetime.now)

    def get_expiration(self, exp_date: date) -> list[OptionContract]:
        """Get all contracts for a specific expiration."""
        return [c for c in self.contracts if c.expiration == exp_date]

    def get_calls(self, exp_date: Optional[date] = None) -> list[OptionContract]:
        """Get call options, optionally filtered by expiration."""
        calls = [c for c in self.contracts if c.option_type == OptionType.CALL]
        if exp_date:
            calls = [c for c in calls if c.expiration == exp_date]
        return sorted(calls, key=lambda x: x.strike)

    def get_puts(self, exp_date: Optional[date] = None) -> list[OptionContract]:
        """Get put options, optionally filtered by expiration."""
        puts = [c for c in self.contracts if c.option_type == OptionType.PUT]
        if exp_date:
            puts = [c for c in puts if c.expiration == exp_date]
        return sorted(puts, key=lambda x: x.strike)

    def get_atm_strike(self) -> float:
        """Get the at-the-money strike price."""
        if not self.contracts:
            return self.underlying_price

        strikes = sorted(set(c.strike for c in self.contracts))
        return min(strikes, key=lambda x: abs(x - self.underlying_price))

    def get_contract(
        self,
        option_type: OptionType,
        strike: float,
        expiration: date,
    ) -> Optional[OptionContract]:
        """Get a specific contract."""
        for c in self.contracts:
            if (
                c.option_type == option_type
                and c.strike == strike
                and c.expiration == expiration
            ):
                return c
        return None

    def nearest_expiration(self, min_days: int = 0, max_days: int = 365) -> Optional[date]:
        """Find nearest expiration within range."""
        today = date.today()
        valid_exps = [
            exp for exp in self.expirations
            if min_days <= (exp - today).days <= max_days
        ]
        return min(valid_exps) if valid_exps else None


@dataclass
class OptionPosition:
    """An open option position."""
    contract: OptionContract
    quantity: int  # Positive = long, negative = short
    avg_cost: float  # Per share, not per contract
    opened_at: datetime = field(default_factory=datetime.now)

    @property
    def is_long(self) -> bool:
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        return self.quantity < 0

    @property
    def market_value(self) -> float:
        """Current market value."""
        return self.contract.mid_price * abs(self.quantity) * self.contract.multiplier

    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return self.avg_cost * abs(self.quantity) * self.contract.multiplier

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.is_long:
            return self.market_value - self.cost_basis
        else:
            return self.cost_basis - self.market_value

    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized P&L as percentage."""
        if self.cost_basis == 0:
            return 0
        return self.unrealized_pnl / self.cost_basis

    @property
    def delta_exposure(self) -> float:
        """Total delta exposure."""
        if self.contract.greeks:
            return (
                self.contract.greeks.delta
                * self.quantity
                * self.contract.multiplier
            )
        return 0

    @property
    def theta_exposure(self) -> float:
        """Daily theta decay."""
        if self.contract.greeks:
            return (
                self.contract.greeks.theta
                * self.quantity
                * self.contract.multiplier
            )
        return 0


@dataclass
class OptionOrder:
    """Option order to be submitted."""
    contract: OptionContract
    action: OrderAction
    quantity: int
    order_type: str = "limit"  # limit, market
    limit_price: Optional[float] = None
    time_in_force: str = "day"  # day, gtc, ioc

    # For multi-leg orders
    legs: list["OptionOrder"] = field(default_factory=list)

    @property
    def is_opening(self) -> bool:
        return self.action in (OrderAction.BUY_TO_OPEN, OrderAction.SELL_TO_OPEN)

    @property
    def is_closing(self) -> bool:
        return self.action in (OrderAction.BUY_TO_CLOSE, OrderAction.SELL_TO_CLOSE)

    @property
    def is_buy(self) -> bool:
        return self.action in (OrderAction.BUY_TO_OPEN, OrderAction.BUY_TO_CLOSE)

    @property
    def estimated_cost(self) -> float:
        """Estimated cost/credit for the order."""
        price = self.limit_price or self.contract.mid_price
        cost = price * self.quantity * self.contract.multiplier

        if self.is_buy:
            return cost  # Debit
        else:
            return -cost  # Credit


@dataclass
class OptionSpread:
    """Multi-leg option spread."""
    spread_type: SpreadType
    underlying: str
    legs: list[OptionPosition]
    opened_at: datetime = field(default_factory=datetime.now)

    @property
    def net_debit(self) -> float:
        """Net debit (positive) or credit (negative) paid."""
        return sum(
            leg.avg_cost * leg.quantity * leg.contract.multiplier
            for leg in self.legs
        )

    @property
    def max_profit(self) -> Optional[float]:
        """Maximum profit potential (if calculable)."""
        if self.spread_type == SpreadType.BULL_CALL_SPREAD:
            # Max profit = width of strikes - net debit
            strikes = sorted(leg.contract.strike for leg in self.legs)
            if len(strikes) == 2:
                width = strikes[1] - strikes[0]
                return (width * 100) - self.net_debit
        elif self.spread_type == SpreadType.IRON_CONDOR:
            # Max profit = net credit received
            return -self.net_debit if self.net_debit < 0 else None

        return None

    @property
    def max_loss(self) -> Optional[float]:
        """Maximum loss potential (if calculable)."""
        if self.spread_type == SpreadType.BULL_CALL_SPREAD:
            return self.net_debit
        elif self.spread_type == SpreadType.IRON_CONDOR:
            # Max loss = width of put spread (or call spread) - credit
            put_legs = [l for l in self.legs if l.contract.option_type == OptionType.PUT]
            if len(put_legs) == 2:
                strikes = sorted(l.contract.strike for l in put_legs)
                width = strikes[1] - strikes[0]
                return (width * 100) + self.net_debit

        return None

    @property
    def risk_reward_ratio(self) -> Optional[float]:
        """Risk to reward ratio."""
        max_profit = self.max_profit
        max_loss = self.max_loss
        if max_profit and max_loss and max_loss > 0:
            return max_profit / max_loss
        return None

    @property
    def total_delta(self) -> float:
        """Net delta of the spread."""
        return sum(leg.delta_exposure for leg in self.legs)

    @property
    def total_theta(self) -> float:
        """Net theta of the spread."""
        return sum(leg.theta_exposure for leg in self.legs)

    @property
    def breakeven_prices(self) -> list[float]:
        """Calculate breakeven price(s)."""
        # Simplified - actual calculation depends on spread type
        if self.spread_type == SpreadType.LONG_CALL:
            leg = self.legs[0]
            return [leg.contract.strike + leg.avg_cost]
        elif self.spread_type == SpreadType.LONG_PUT:
            leg = self.legs[0]
            return [leg.contract.strike - leg.avg_cost]

        return []
