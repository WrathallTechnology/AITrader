"""
Options trading strategies.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .models import (
    OptionType,
    OptionContract,
    OptionChain,
    OptionOrder,
    OptionSpread,
    OrderAction,
    SpreadType,
)
from .client import OptionsClient

logger = logging.getLogger(__name__)


@dataclass
class StrategySignal:
    """Signal from an options strategy."""
    strategy_name: str
    underlying: str
    action: str  # "open", "close", "adjust"
    spread_type: SpreadType
    contracts: list[OptionContract]
    confidence: float  # 0-1
    expected_profit: Optional[float] = None
    max_loss: Optional[float] = None
    rationale: str = ""


class OptionsStrategy(ABC):
    """Base class for options strategies."""

    def __init__(self, client: OptionsClient):
        self.client = client

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""
        pass

    @abstractmethod
    def analyze(
        self,
        underlying: str,
        underlying_price: float,
        volatility: float,
        trend: str,  # "bullish", "bearish", "neutral"
    ) -> Optional[StrategySignal]:
        """
        Analyze market conditions and generate signal.

        Args:
            underlying: Stock symbol
            underlying_price: Current stock price
            volatility: Implied or historical volatility
            trend: Market trend assessment

        Returns:
            StrategySignal if opportunity found, None otherwise
        """
        pass

    def get_chain(
        self,
        underlying: str,
        min_days: int = 7,
        max_days: int = 45,
    ) -> OptionChain:
        """Get option chain within DTE range."""
        return self.client.get_option_chain(
            underlying=underlying,
            expiration_date_gte=date.today() + timedelta(days=min_days),
            expiration_date_lte=date.today() + timedelta(days=max_days),
        )


class DirectionalStrategy(OptionsStrategy):
    """
    Directional options strategies for trending markets.

    Strategies:
    - Long calls for bullish
    - Long puts for bearish
    - Bull call spreads for moderately bullish
    - Bear put spreads for moderately bearish
    """

    def __init__(
        self,
        client: OptionsClient,
        delta_target: float = 0.30,
        min_open_interest: int = 10,  # Relaxed from 100 for better contract availability
        max_spread_pct: float = 0.20,  # Relaxed from 0.10 for better contract availability
        prefer_spreads: bool = True,
    ):
        """
        Args:
            delta_target: Target delta for directional plays
            min_open_interest: Minimum open interest for liquidity
            max_spread_pct: Maximum bid-ask spread percentage
            prefer_spreads: Prefer defined-risk spreads over naked options
        """
        super().__init__(client)
        self.delta_target = delta_target
        self.min_open_interest = min_open_interest
        self.max_spread_pct = max_spread_pct
        self.prefer_spreads = prefer_spreads

    @property
    def name(self) -> str:
        return "Directional"

    def analyze(
        self,
        underlying: str,
        underlying_price: float,
        volatility: float,
        trend: str,
    ) -> Optional[StrategySignal]:
        """Generate directional strategy signal."""
        if trend == "neutral":
            return None

        # Determine option type based on trend
        option_type = OptionType.CALL if trend == "bullish" else OptionType.PUT

        # Get suitable contracts
        contracts = self.client.find_contracts(
            underlying=underlying,
            option_type=option_type,
            min_days=14,
            max_days=45,
            delta_target=self.delta_target,
            min_open_interest=self.min_open_interest,
            max_spread_pct=self.max_spread_pct,
        )

        if not contracts:
            logger.debug(f"No suitable {option_type.value} contracts for {underlying}")
            return None

        best_contract = contracts[0]

        if self.prefer_spreads:
            # Try to create a spread
            spread_signal = self._create_spread(
                underlying, underlying_price, best_contract, trend
            )
            if spread_signal:
                return spread_signal

        # Fall back to single leg
        spread_type = (
            SpreadType.LONG_CALL if trend == "bullish" else SpreadType.LONG_PUT
        )

        contract_value = best_contract.contract_value
        if contract_value is None:
            return None

        return StrategySignal(
            strategy_name=self.name,
            underlying=underlying,
            action="open",
            spread_type=spread_type,
            contracts=[best_contract],
            confidence=0.6,
            expected_profit=None,  # Unlimited for long calls
            max_loss=contract_value,
            rationale=f"{trend.capitalize()} trend detected, buying {option_type.value} at ${best_contract.strike}",
        )

    def _create_spread(
        self,
        underlying: str,
        underlying_price: float,
        long_contract: OptionContract,
        trend: str,
    ) -> Optional[StrategySignal]:
        """Create a vertical spread."""
        # Get chain for same expiration
        chain = self.client.get_option_chain(
            underlying=underlying,
            expiration_date=long_contract.expiration,
            option_type=long_contract.option_type,
        )

        if trend == "bullish":
            # Bull call spread: buy lower strike, sell higher
            short_strike = long_contract.strike * 1.05  # 5% OTM
            short_contract = self._find_nearest_strike(
                chain, short_strike, long_contract.option_type
            )

            if short_contract and short_contract.strike > long_contract.strike:
                long_mid = long_contract.mid_price
                short_mid = short_contract.mid_price
                if long_mid is None or short_mid is None:
                    return None
                net_debit = long_mid - short_mid
                max_profit = (short_contract.strike - long_contract.strike) - net_debit
                max_loss = net_debit * 100

                return StrategySignal(
                    strategy_name=self.name,
                    underlying=underlying,
                    action="open",
                    spread_type=SpreadType.BULL_CALL_SPREAD,
                    contracts=[long_contract, short_contract],
                    confidence=0.65,
                    expected_profit=max_profit * 100,
                    max_loss=max_loss,
                    rationale=f"Bull call spread: ${long_contract.strike}/{short_contract.strike}",
                )

        else:  # bearish
            # Bear put spread: buy higher strike, sell lower
            short_strike = long_contract.strike * 0.95  # 5% OTM
            short_contract = self._find_nearest_strike(
                chain, short_strike, long_contract.option_type
            )

            if short_contract and short_contract.strike < long_contract.strike:
                long_mid = long_contract.mid_price
                short_mid = short_contract.mid_price
                if long_mid is None or short_mid is None:
                    return None
                net_debit = long_mid - short_mid
                max_profit = (long_contract.strike - short_contract.strike) - net_debit
                max_loss = net_debit * 100

                return StrategySignal(
                    strategy_name=self.name,
                    underlying=underlying,
                    action="open",
                    spread_type=SpreadType.BEAR_PUT_SPREAD,
                    contracts=[long_contract, short_contract],
                    confidence=0.65,
                    expected_profit=max_profit * 100,
                    max_loss=max_loss,
                    rationale=f"Bear put spread: ${long_contract.strike}/{short_contract.strike}",
                )

        return None

    def _find_nearest_strike(
        self,
        chain: OptionChain,
        target_strike: float,
        option_type: OptionType,
    ) -> Optional[OptionContract]:
        """Find contract with strike nearest to target."""
        contracts = (
            chain.get_calls() if option_type == OptionType.CALL else chain.get_puts()
        )

        if not contracts:
            return None

        return min(contracts, key=lambda c: abs(c.strike - target_strike))


class IncomeStrategy(OptionsStrategy):
    """
    Income-generating options strategies.

    Strategies:
    - Covered calls (requires stock position)
    - Cash-secured puts
    - Iron condors (neutral market)
    """

    def __init__(
        self,
        client: OptionsClient,
        target_premium_pct: float = 0.02,  # 2% premium target
        min_probability_otm: float = 0.70,  # 70% chance of profit
        max_days_to_expiration: int = 45,
    ):
        super().__init__(client)
        self.target_premium_pct = target_premium_pct
        self.min_probability_otm = min_probability_otm
        self.max_days_to_expiration = max_days_to_expiration

    @property
    def name(self) -> str:
        return "Income"

    def analyze(
        self,
        underlying: str,
        underlying_price: float,
        volatility: float,
        trend: str,
    ) -> Optional[StrategySignal]:
        """Generate income strategy signal."""
        # Higher IV = better for premium sellers
        if volatility < 0.15:  # IV below 15% (relaxed from 20%)
            logger.debug(f"IV too low for income strategy on {underlying}")
            return None

        if trend == "neutral":
            # Iron condor for neutral/range-bound
            return self._analyze_iron_condor(underlying, underlying_price, volatility)
        elif trend == "bullish":
            # Cash-secured put
            return self._analyze_cash_secured_put(underlying, underlying_price, volatility)
        else:
            # For bearish, covered calls if we have stock (assume we don't)
            # Fall back to iron condor with bearish skew
            return self._analyze_iron_condor(underlying, underlying_price, volatility)

    def _analyze_cash_secured_put(
        self,
        underlying: str,
        underlying_price: float,
        volatility: float,
    ) -> Optional[StrategySignal]:
        """Analyze cash-secured put opportunity."""
        # Target delta around 0.25-0.30 (OTM)
        target_delta = 0.30

        contracts = self.client.find_contracts(
            underlying=underlying,
            option_type=OptionType.PUT,
            min_days=21,
            max_days=self.max_days_to_expiration,
            delta_target=target_delta,
            min_open_interest=10,  # Relaxed from 50
            max_spread_pct=0.20,  # Relaxed from 0.15
        )

        if not contracts:
            return None

        best = contracts[0]

        # Calculate premium yield
        premium = best.mid_price
        if premium is None:
            return None
        strike = best.strike
        premium_yield = premium / strike

        if premium_yield < self.target_premium_pct:
            logger.debug(f"Premium yield {premium_yield:.2%} below target")
            return None

        # Cash required = strike price * 100
        cash_required = strike * 100
        max_profit = premium * 100

        return StrategySignal(
            strategy_name=self.name,
            underlying=underlying,
            action="open",
            spread_type=SpreadType.SHORT_PUT,
            contracts=[best],
            confidence=0.70,
            expected_profit=max_profit,
            max_loss=cash_required - max_profit,  # If assigned at $0
            rationale=f"Cash-secured put at ${strike:.0f}, {premium_yield:.2%} yield",
        )

    def _analyze_iron_condor(
        self,
        underlying: str,
        underlying_price: float,
        volatility: float,
    ) -> Optional[StrategySignal]:
        """Analyze iron condor opportunity."""
        chain = self.get_chain(underlying, min_days=21, max_days=45)

        if not chain.contracts or not chain.expirations:
            return None

        expiration = chain.nearest_expiration(min_days=21, max_days=45)
        if not expiration:
            return None

        # Define strikes for iron condor
        # Sell OTM put and call, buy further OTM for protection
        atm_strike = chain.get_atm_strike()
        width = atm_strike * 0.05  # 5% wing width

        # Put spread (below price)
        short_put_strike = atm_strike * 0.95  # 5% OTM
        long_put_strike = short_put_strike - width

        # Call spread (above price)
        short_call_strike = atm_strike * 1.05  # 5% OTM
        long_call_strike = short_call_strike + width

        # Get available puts and calls for this expiration
        puts = chain.get_puts(expiration)
        calls = chain.get_calls(expiration)

        if not puts or not calls:
            logger.debug(f"Iron condor: no puts or calls for {underlying} exp {expiration}")
            return None

        # Find nearest strikes (not exact matches)
        short_put = min(puts, key=lambda c: abs(c.strike - short_put_strike))
        long_put = min(puts, key=lambda c: abs(c.strike - long_put_strike))
        short_call = min(calls, key=lambda c: abs(c.strike - short_call_strike))
        long_call = min(calls, key=lambda c: abs(c.strike - long_call_strike))

        # Validate leg relationships (short put > long put, long call > short call)
        if short_put.strike <= long_put.strike:
            logger.debug(f"Iron condor: invalid put spread strikes {long_put.strike}/{short_put.strike}")
            return None
        if long_call.strike <= short_call.strike:
            logger.debug(f"Iron condor: invalid call spread strikes {short_call.strike}/{long_call.strike}")
            return None

        # Check all legs are tradeable
        if not all([short_put.is_tradeable, long_put.is_tradeable, short_call.is_tradeable, long_call.is_tradeable]):
            logger.debug(f"Iron condor: one or more legs not tradeable for {underlying}")
            return None

        contracts = [short_put, long_put, short_call, long_call]

        # Calculate premium received - all contracts must have valid mid prices
        sp_mid = short_put.mid_price
        sc_mid = short_call.mid_price
        lp_mid = long_put.mid_price
        lc_mid = long_call.mid_price
        if any(m is None for m in [sp_mid, sc_mid, lp_mid, lc_mid]):
            logger.debug("Iron condor: missing mid prices for one or more legs")
            return None

        premium = sp_mid + sc_mid - lp_mid - lc_mid

        # Calculate actual width from selected strikes
        put_width = short_put.strike - long_put.strike
        call_width = long_call.strike - short_call.strike
        max_width = max(put_width, call_width)

        max_profit = premium * 100
        max_loss = (max_width * 100) - max_profit

        if max_loss <= 0:
            return None

        return StrategySignal(
            strategy_name=self.name,
            underlying=underlying,
            action="open",
            spread_type=SpreadType.IRON_CONDOR,
            contracts=contracts,
            confidence=0.65,
            expected_profit=max_profit,
            max_loss=max_loss,
            rationale=f"Iron condor: puts {long_put.strike}/{short_put.strike}, calls {short_call.strike}/{long_call.strike}",
        )


class VolatilityStrategy(OptionsStrategy):
    """
    Volatility-based options strategies.

    Strategies:
    - Long straddles/strangles for expected volatility expansion
    - Short straddles/strangles for expected volatility contraction
    """

    def __init__(
        self,
        client: OptionsClient,
        iv_percentile_threshold: float = 0.40,  # IV rank threshold (relaxed from 0.50)
        expected_move_multiplier: float = 1.5,  # Expected move vs implied
    ):
        super().__init__(client)
        self.iv_percentile_threshold = iv_percentile_threshold
        self.expected_move_multiplier = expected_move_multiplier

    @property
    def name(self) -> str:
        return "Volatility"

    def analyze(
        self,
        underlying: str,
        underlying_price: float,
        volatility: float,
        trend: str,
    ) -> Optional[StrategySignal]:
        """Generate volatility strategy signal."""
        # This is simplified - real implementation would use IV rank/percentile
        iv_rank = self._estimate_iv_rank(volatility)

        if iv_rank < self.iv_percentile_threshold:
            # Low IV - buy volatility (long straddle/strangle)
            return self._analyze_long_volatility(underlying, underlying_price)
        else:
            # High IV - sell volatility (short strangle, more conservative)
            return self._analyze_short_volatility(underlying, underlying_price)

    def _estimate_iv_rank(self, current_iv: float) -> float:
        """
        Estimate IV rank (0-1).
        This is simplified - real implementation would compare to historical IV.
        """
        # Rough estimate: 20% IV = low, 40% IV = medium, 60%+ = high
        if current_iv < 0.20:
            return 0.2
        elif current_iv < 0.30:
            return 0.4
        elif current_iv < 0.40:
            return 0.6
        else:
            return 0.8

    def _analyze_long_volatility(
        self,
        underlying: str,
        underlying_price: float,
    ) -> Optional[StrategySignal]:
        """Analyze long straddle opportunity."""
        chain = self.get_chain(underlying, min_days=21, max_days=45)

        if not chain.contracts:
            return None

        expiration = chain.nearest_expiration(min_days=21, max_days=45)
        if not expiration:
            return None

        atm_strike = chain.get_atm_strike()

        # Get ATM call and put
        call = chain.get_contract(OptionType.CALL, atm_strike, expiration)
        put = chain.get_contract(OptionType.PUT, atm_strike, expiration)

        if not call or not put:
            return None

        call_mid = call.mid_price
        put_mid = put.mid_price
        if call_mid is None or put_mid is None:
            return None

        # Cost of straddle
        total_cost = (call_mid + put_mid) * 100

        # Breakeven points
        upper_breakeven = atm_strike + call_mid + put_mid
        lower_breakeven = atm_strike - call_mid - put_mid

        return StrategySignal(
            strategy_name=self.name,
            underlying=underlying,
            action="open",
            spread_type=SpreadType.STRADDLE,
            contracts=[call, put],
            confidence=0.55,
            expected_profit=None,  # Unlimited
            max_loss=total_cost,
            rationale=f"Long straddle at ${atm_strike:.0f}, breakevens ${lower_breakeven:.2f}-${upper_breakeven:.2f}",
        )

    def _analyze_short_volatility(
        self,
        underlying: str,
        underlying_price: float,
    ) -> Optional[StrategySignal]:
        """Analyze short strangle opportunity (defined risk with wings)."""
        chain = self.get_chain(underlying, min_days=21, max_days=45)

        if not chain.contracts:
            return None

        expiration = chain.nearest_expiration(min_days=21, max_days=45)
        if not expiration:
            return None

        atm_strike = chain.get_atm_strike()

        # Sell OTM options (strangle)
        put_strike = atm_strike * 0.95  # 5% OTM
        call_strike = atm_strike * 1.05  # 5% OTM

        # Find nearest available strikes
        puts = chain.get_puts(expiration)
        calls = chain.get_calls(expiration)

        if not puts or not calls:
            return None

        short_put = min(puts, key=lambda c: abs(c.strike - put_strike))
        short_call = min(calls, key=lambda c: abs(c.strike - call_strike))

        put_mid = short_put.mid_price
        call_mid = short_call.mid_price
        if put_mid is None or call_mid is None:
            return None

        # Premium received
        premium = (put_mid + call_mid) * 100

        return StrategySignal(
            strategy_name=self.name,
            underlying=underlying,
            action="open",
            spread_type=SpreadType.STRANGLE,
            contracts=[short_put, short_call],
            confidence=0.60,
            expected_profit=premium,
            max_loss=None,  # Undefined for naked strangle
            rationale=f"Short strangle: ${short_put.strike:.0f} put / ${short_call.strike:.0f} call",
        )


class OptionsStrategyManager:
    """
    Manages multiple options strategies and coordinates signals.
    """

    def __init__(self, client: OptionsClient):
        self.client = client
        self.strategies: list[OptionsStrategy] = [
            DirectionalStrategy(client),
            IncomeStrategy(client),
            VolatilityStrategy(client),
        ]

    def analyze_all(
        self,
        underlying: str,
        underlying_price: float,
        volatility: float,
        trend: str,
    ) -> list[StrategySignal]:
        """Run all strategies and return signals."""
        signals = []

        for strategy in self.strategies:
            try:
                signal = strategy.analyze(
                    underlying=underlying,
                    underlying_price=underlying_price,
                    volatility=volatility,
                    trend=trend,
                )
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(f"Strategy {strategy.name} failed: {e}")

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)

        return signals

    def get_best_signal(
        self,
        underlying: str,
        underlying_price: float,
        volatility: float,
        trend: str,
        max_loss: Optional[float] = None,
    ) -> Optional[StrategySignal]:
        """Get the best strategy signal within risk limits."""
        signals = self.analyze_all(underlying, underlying_price, volatility, trend)

        for signal in signals:
            # Filter by max loss if specified
            if max_loss and signal.max_loss and signal.max_loss > max_loss:
                continue

            return signal

        return None
