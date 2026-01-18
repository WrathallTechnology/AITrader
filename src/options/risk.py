"""
Options-specific risk management.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from .models import (
    OptionType,
    OptionContract,
    OptionPosition,
    OptionOrder,
    OptionSpread,
    SpreadType,
    OrderAction,
)

logger = logging.getLogger(__name__)


@dataclass
class OptionsRiskLimits:
    """Risk limits for options trading."""
    # Position limits
    max_single_position_value: float = 5000  # Max per single options position
    max_total_options_exposure: float = 25000  # Max total options value
    max_positions: int = 10  # Max number of options positions

    # Greeks limits
    max_portfolio_delta: float = 500  # Max net delta exposure
    max_portfolio_gamma: float = 100  # Max gamma exposure
    max_portfolio_theta: float = -50  # Max daily theta decay (negative = paying)

    # Strategy limits
    allow_naked_calls: bool = False  # Naked calls = unlimited risk
    allow_naked_puts: bool = True  # Cash-secured puts
    max_undefined_risk_positions: int = 2  # Positions with unlimited loss potential

    # DTE limits
    min_days_to_expiration: int = 7  # Don't hold through final week
    max_days_to_expiration: int = 60  # Max DTE for new positions

    # Concentration limits
    max_single_underlying_pct: float = 0.30  # Max 30% in single underlying
    max_single_expiration_pct: float = 0.40  # Max 40% expiring same date


@dataclass
class PositionRiskMetrics:
    """Risk metrics for a position."""
    position: OptionPosition
    delta_exposure: float
    gamma_exposure: float
    theta_exposure: float
    vega_exposure: float
    max_loss: Optional[float]
    days_to_expiration: int
    at_risk_value: float  # Amount at risk


@dataclass
class PortfolioRiskMetrics:
    """Aggregate risk metrics for options portfolio."""
    total_value: float
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    max_portfolio_loss: Optional[float]
    positions_count: int
    undefined_risk_count: int
    expiring_this_week: int
    concentration_by_underlying: dict[str, float]
    concentration_by_expiration: dict[date, float]


class OptionsRiskManager:
    """
    Risk manager for options positions.

    Monitors:
    - Position size limits
    - Greeks exposure
    - Concentration risk
    - Expiration risk
    """

    def __init__(self, limits: Optional[OptionsRiskLimits] = None):
        self.limits = limits or OptionsRiskLimits()
        self._positions: list[OptionPosition] = []
        self._position_metrics: list[PositionRiskMetrics] = []

    def set_positions(self, positions: list[OptionPosition]) -> None:
        """Update current positions and recalculate metrics."""
        self._positions = positions
        self._calculate_all_metrics()

    def _calculate_all_metrics(self) -> None:
        """Calculate risk metrics for all positions."""
        self._position_metrics = []

        for position in self._positions:
            metrics = self._calculate_position_metrics(position)
            self._position_metrics.append(metrics)

    def _calculate_position_metrics(
        self,
        position: OptionPosition,
    ) -> PositionRiskMetrics:
        """Calculate risk metrics for a single position."""
        contract = position.contract
        qty = position.quantity
        multiplier = contract.multiplier

        # Calculate Greeks exposure
        if contract.greeks:
            delta = contract.greeks.delta * qty * multiplier
            gamma = contract.greeks.gamma * qty * multiplier
            theta = contract.greeks.theta * qty * multiplier
            vega = contract.greeks.vega * qty * multiplier
        else:
            # Estimate if Greeks not available
            delta = self._estimate_delta(contract, position.is_long) * qty * multiplier
            gamma = 0
            theta = -0.05 * contract.mid_price * qty * multiplier  # Rough estimate
            vega = 0

        # Calculate max loss
        max_loss = self._calculate_max_loss(position)

        # Days to expiration
        dte = (contract.expiration - date.today()).days

        # At-risk value (premium paid for long, collateral for short)
        if position.is_long:
            at_risk_value = position.cost_basis
        else:
            # For short positions, at-risk is potential assignment value
            at_risk_value = contract.strike * abs(qty) * multiplier

        return PositionRiskMetrics(
            position=position,
            delta_exposure=delta,
            gamma_exposure=gamma,
            theta_exposure=theta,
            vega_exposure=vega,
            max_loss=max_loss,
            days_to_expiration=dte,
            at_risk_value=at_risk_value,
        )

    def _estimate_delta(
        self,
        contract: OptionContract,
        is_long: bool,
    ) -> float:
        """Estimate delta when Greeks not available."""
        # Very rough estimation based on moneyness
        # Assumes underlying price ~ strike (ATM)
        base_delta = 0.5  # ATM delta

        if is_long:
            return base_delta if contract.option_type == OptionType.CALL else -base_delta
        else:
            return -base_delta if contract.option_type == OptionType.CALL else base_delta

    def _calculate_max_loss(self, position: OptionPosition) -> Optional[float]:
        """Calculate maximum loss for a position."""
        contract = position.contract
        qty = abs(position.quantity)
        multiplier = contract.multiplier

        if position.is_long:
            # Long options: max loss = premium paid
            return position.cost_basis

        else:
            # Short options
            if contract.option_type == OptionType.CALL:
                # Naked call: unlimited loss
                return None
            else:
                # Short put: max loss = strike - premium received
                premium_received = position.avg_cost * qty * multiplier
                max_loss = (contract.strike * qty * multiplier) - premium_received
                return max(0, max_loss)

    def get_portfolio_metrics(self) -> PortfolioRiskMetrics:
        """Get aggregate portfolio risk metrics."""
        if not self._position_metrics:
            return PortfolioRiskMetrics(
                total_value=0,
                total_delta=0,
                total_gamma=0,
                total_theta=0,
                total_vega=0,
                max_portfolio_loss=0,
                positions_count=0,
                undefined_risk_count=0,
                expiring_this_week=0,
                concentration_by_underlying={},
                concentration_by_expiration={},
            )

        # Aggregate metrics
        total_value = sum(m.position.market_value for m in self._position_metrics)
        total_delta = sum(m.delta_exposure for m in self._position_metrics)
        total_gamma = sum(m.gamma_exposure for m in self._position_metrics)
        total_theta = sum(m.theta_exposure for m in self._position_metrics)
        total_vega = sum(m.vega_exposure for m in self._position_metrics)

        # Max loss
        max_losses = [m.max_loss for m in self._position_metrics if m.max_loss is not None]
        undefined_risk = [m for m in self._position_metrics if m.max_loss is None]

        max_portfolio_loss = sum(max_losses) if max_losses else None
        if undefined_risk:
            max_portfolio_loss = None  # Undefined if any position has unlimited risk

        # Count expiring this week
        week_from_now = date.today() + timedelta(days=7)
        expiring_this_week = sum(
            1 for m in self._position_metrics
            if m.position.contract.expiration <= week_from_now
        )

        # Concentration by underlying
        underlying_values: dict[str, float] = {}
        for m in self._position_metrics:
            underlying = m.position.contract.underlying
            underlying_values[underlying] = underlying_values.get(underlying, 0) + m.at_risk_value

        if total_value > 0:
            concentration_underlying = {
                u: v / total_value for u, v in underlying_values.items()
            }
        else:
            concentration_underlying = {}

        # Concentration by expiration
        exp_values: dict[date, float] = {}
        for m in self._position_metrics:
            exp = m.position.contract.expiration
            exp_values[exp] = exp_values.get(exp, 0) + m.at_risk_value

        if total_value > 0:
            concentration_exp = {e: v / total_value for e, v in exp_values.items()}
        else:
            concentration_exp = {}

        return PortfolioRiskMetrics(
            total_value=total_value,
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            max_portfolio_loss=max_portfolio_loss,
            positions_count=len(self._position_metrics),
            undefined_risk_count=len(undefined_risk),
            expiring_this_week=expiring_this_week,
            concentration_by_underlying=concentration_underlying,
            concentration_by_expiration=concentration_exp,
        )

    def check_order(
        self,
        order: OptionOrder,
        account_value: float,
    ) -> tuple[bool, list[str]]:
        """
        Check if an order passes risk checks.

        Returns:
            Tuple of (passes, list of violations)
        """
        violations = []
        contract = order.contract

        # Check DTE
        dte = (contract.expiration - date.today()).days
        if dte < self.limits.min_days_to_expiration:
            violations.append(
                f"DTE ({dte}) below minimum ({self.limits.min_days_to_expiration})"
            )
        if dte > self.limits.max_days_to_expiration:
            violations.append(
                f"DTE ({dte}) above maximum ({self.limits.max_days_to_expiration})"
            )

        # Check for naked positions
        if order.action == OrderAction.SELL_TO_OPEN:
            if contract.option_type == OptionType.CALL and not self.limits.allow_naked_calls:
                violations.append("Naked calls not allowed")
            if contract.option_type == OptionType.PUT and not self.limits.allow_naked_puts:
                violations.append("Naked puts not allowed")

        # Check position size
        order_value = order.estimated_cost
        if abs(order_value) > self.limits.max_single_position_value:
            violations.append(
                f"Position value (${abs(order_value):.0f}) exceeds limit "
                f"(${self.limits.max_single_position_value:.0f})"
            )

        # Check total exposure
        current_metrics = self.get_portfolio_metrics()
        new_total = current_metrics.total_value + abs(order_value)
        if new_total > self.limits.max_total_options_exposure:
            violations.append(
                f"Total options exposure (${new_total:.0f}) would exceed limit "
                f"(${self.limits.max_total_options_exposure:.0f})"
            )

        # Check position count
        if order.is_opening:
            new_count = current_metrics.positions_count + 1
            if new_count > self.limits.max_positions:
                violations.append(
                    f"Position count ({new_count}) would exceed limit "
                    f"({self.limits.max_positions})"
                )

        # Check concentration by underlying
        underlying = contract.underlying
        current_underlying_value = sum(
            m.at_risk_value for m in self._position_metrics
            if m.position.contract.underlying == underlying
        )
        new_underlying_value = current_underlying_value + abs(order_value)
        if account_value > 0:
            underlying_pct = new_underlying_value / account_value
            if underlying_pct > self.limits.max_single_underlying_pct:
                violations.append(
                    f"Concentration in {underlying} ({underlying_pct:.1%}) would exceed "
                    f"limit ({self.limits.max_single_underlying_pct:.1%})"
                )

        return len(violations) == 0, violations

    def check_spread_order(
        self,
        orders: list[OptionOrder],
        account_value: float,
    ) -> tuple[bool, list[str]]:
        """Check if a multi-leg spread order passes risk checks."""
        violations = []

        # Check individual legs
        for order in orders:
            passes, leg_violations = self.check_order(order, account_value)
            violations.extend(leg_violations)

        # For spreads, check if defined risk
        buy_legs = [o for o in orders if o.is_buy]
        sell_legs = [o for o in orders if not o.is_buy]

        if sell_legs and not buy_legs:
            # All short = naked
            for sell_order in sell_legs:
                if sell_order.contract.option_type == OptionType.CALL:
                    if not self.limits.allow_naked_calls:
                        violations.append("Naked call in spread not allowed")

        return len(violations) == 0, violations

    def get_position_size_limit(
        self,
        contract: OptionContract,
        account_value: float,
    ) -> float:
        """Calculate maximum position size for a contract."""
        # Start with single position limit
        max_value = self.limits.max_single_position_value

        # Check remaining exposure capacity
        current_metrics = self.get_portfolio_metrics()
        remaining_exposure = self.limits.max_total_options_exposure - current_metrics.total_value
        max_value = min(max_value, remaining_exposure)

        # Check underlying concentration
        underlying = contract.underlying
        current_underlying_value = sum(
            m.at_risk_value for m in self._position_metrics
            if m.position.contract.underlying == underlying
        )
        max_underlying = (account_value * self.limits.max_single_underlying_pct) - current_underlying_value
        max_value = min(max_value, max_underlying)

        return max(0, max_value)

    def get_expiration_warnings(self) -> list[str]:
        """Get warnings for positions near expiration."""
        warnings = []
        today = date.today()
        week_from_now = today + timedelta(days=7)

        for metrics in self._position_metrics:
            contract = metrics.position.contract
            if contract.expiration <= week_from_now:
                dte = metrics.days_to_expiration
                warnings.append(
                    f"{contract.underlying} {contract.option_type.value} "
                    f"${contract.strike:.0f} expires in {dte} days"
                )

        return warnings

    def get_delta_adjustment_needed(self) -> float:
        """
        Calculate delta adjustment needed to stay within limits.

        Returns:
            Positive = need to reduce long delta
            Negative = need to reduce short delta
        """
        metrics = self.get_portfolio_metrics()

        if metrics.total_delta > self.limits.max_portfolio_delta:
            return metrics.total_delta - self.limits.max_portfolio_delta
        elif metrics.total_delta < -self.limits.max_portfolio_delta:
            return metrics.total_delta + self.limits.max_portfolio_delta

        return 0

    def suggest_hedges(
        self,
        underlying_price: float,
    ) -> list[str]:
        """Suggest hedging actions based on current exposure."""
        suggestions = []
        metrics = self.get_portfolio_metrics()

        # Delta hedge suggestions
        if abs(metrics.total_delta) > self.limits.max_portfolio_delta * 0.8:
            if metrics.total_delta > 0:
                suggestions.append(
                    f"Consider buying puts or selling calls to reduce "
                    f"delta exposure ({metrics.total_delta:.0f})"
                )
            else:
                suggestions.append(
                    f"Consider buying calls or selling puts to reduce "
                    f"short delta exposure ({metrics.total_delta:.0f})"
                )

        # Theta warning
        if metrics.total_theta < self.limits.max_portfolio_theta:
            suggestions.append(
                f"High theta decay (${metrics.total_theta:.2f}/day). "
                f"Consider closing losing long positions."
            )

        # Concentration warning
        for underlying, pct in metrics.concentration_by_underlying.items():
            if pct > self.limits.max_single_underlying_pct * 0.8:
                suggestions.append(
                    f"High concentration in {underlying} ({pct:.1%}). "
                    f"Consider diversifying or reducing position."
                )

        return suggestions

    def calculate_buying_power_usage(
        self,
        order: OptionOrder,
    ) -> float:
        """
        Estimate buying power usage for an order.

        Note: Actual margin requirements vary by broker.
        """
        contract = order.contract
        qty = order.quantity
        multiplier = contract.multiplier

        if order.is_buy:
            # Long options: full premium
            return abs(order.estimated_cost)

        else:
            # Short options: margin required
            if contract.option_type == OptionType.PUT:
                # Cash-secured put: strike * 100
                return contract.strike * qty * multiplier
            else:
                # Naked call: complex calculation
                # Simplified: 20% of underlying + premium
                notional = contract.strike * qty * multiplier
                return notional * 0.20 + abs(order.estimated_cost)


def create_risk_limits_conservative() -> OptionsRiskLimits:
    """Create conservative risk limits for beginners."""
    return OptionsRiskLimits(
        max_single_position_value=2000,
        max_total_options_exposure=10000,
        max_positions=5,
        max_portfolio_delta=200,
        max_portfolio_gamma=50,
        max_portfolio_theta=-25,
        allow_naked_calls=False,
        allow_naked_puts=True,
        max_undefined_risk_positions=0,
        min_days_to_expiration=14,
        max_days_to_expiration=45,
        max_single_underlying_pct=0.25,
        max_single_expiration_pct=0.30,
    )


def create_risk_limits_moderate() -> OptionsRiskLimits:
    """Create moderate risk limits."""
    return OptionsRiskLimits(
        max_single_position_value=5000,
        max_total_options_exposure=25000,
        max_positions=10,
        max_portfolio_delta=500,
        max_portfolio_gamma=100,
        max_portfolio_theta=-50,
        allow_naked_calls=False,
        allow_naked_puts=True,
        max_undefined_risk_positions=2,
        min_days_to_expiration=7,
        max_days_to_expiration=60,
        max_single_underlying_pct=0.30,
        max_single_expiration_pct=0.40,
    )


def create_risk_limits_aggressive() -> OptionsRiskLimits:
    """Create aggressive risk limits for experienced traders."""
    return OptionsRiskLimits(
        max_single_position_value=10000,
        max_total_options_exposure=50000,
        max_positions=20,
        max_portfolio_delta=1000,
        max_portfolio_gamma=200,
        max_portfolio_theta=-100,
        allow_naked_calls=True,  # Requires high margin
        allow_naked_puts=True,
        max_undefined_risk_positions=5,
        min_days_to_expiration=3,
        max_days_to_expiration=90,
        max_single_underlying_pct=0.40,
        max_single_expiration_pct=0.50,
    )
