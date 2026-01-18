"""
Drawdown protection and circuit breaker system.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RiskState(Enum):
    """Current risk state of the trading system."""
    NORMAL = "normal"  # Full trading allowed
    CAUTION = "caution"  # Reduced position sizes
    RESTRICTED = "restricted"  # Only closing trades
    HALTED = "halted"  # No trading allowed


@dataclass
class DrawdownMetrics:
    """Current drawdown metrics."""
    current_value: float
    peak_value: float
    current_drawdown: float  # As decimal (0.10 = 10%)
    max_drawdown: float
    drawdown_duration_days: int
    recovery_needed: float  # Percentage needed to recover


@dataclass
class TradingSession:
    """Metrics for current trading session."""
    date: datetime
    starting_value: float
    current_value: float
    trades_count: int
    winning_trades: int
    losing_trades: int
    gross_pnl: float
    consecutive_losses: int

    @property
    def win_rate(self) -> float:
        if self.trades_count == 0:
            return 0
        return self.winning_trades / self.trades_count

    @property
    def session_return(self) -> float:
        if self.starting_value == 0:
            return 0
        return (self.current_value - self.starting_value) / self.starting_value


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    # Daily limits
    max_daily_loss_pct: float = 0.03  # 3% daily loss limit
    max_daily_trades: int = 20
    max_consecutive_losses: int = 5

    # Drawdown limits
    caution_drawdown: float = 0.05  # 5% drawdown -> caution
    restricted_drawdown: float = 0.10  # 10% drawdown -> restricted
    halt_drawdown: float = 0.15  # 15% drawdown -> halt

    # Recovery settings
    recovery_period_days: int = 1  # Days before resuming after halt
    gradual_recovery: bool = True  # Gradually increase position sizes

    # Position sizing adjustments
    caution_size_multiplier: float = 0.5
    restricted_size_multiplier: float = 0.0  # No new positions


class DrawdownProtection:
    """
    Monitors and manages drawdown risk.

    Tracks:
    - Current drawdown from peak
    - Daily P&L
    - Consecutive losses
    - Recovery progress
    """

    def __init__(
        self,
        initial_value: float,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.config = config or CircuitBreakerConfig()
        self.peak_value = initial_value
        self.current_value = initial_value
        self.max_drawdown = 0.0
        self.drawdown_start_date: Optional[datetime] = None

        # History
        self._value_history: list[tuple[datetime, float]] = [
            (datetime.now(), initial_value)
        ]
        self._trade_history: list[dict] = []

        # Current session
        self._current_session: Optional[TradingSession] = None
        self._init_session()

    def _init_session(self) -> None:
        """Initialize a new trading session."""
        self._current_session = TradingSession(
            date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),
            starting_value=self.current_value,
            current_value=self.current_value,
            trades_count=0,
            winning_trades=0,
            losing_trades=0,
            gross_pnl=0,
            consecutive_losses=0,
        )

    def update_value(self, new_value: float) -> None:
        """Update current portfolio value."""
        self.current_value = new_value
        self._value_history.append((datetime.now(), new_value))

        # Update peak
        if new_value > self.peak_value:
            self.peak_value = new_value
            self.drawdown_start_date = None

        # Update session
        if self._current_session:
            self._current_session.current_value = new_value

        # Check if entering drawdown
        drawdown = self.get_current_drawdown()
        if drawdown > 0 and self.drawdown_start_date is None:
            self.drawdown_start_date = datetime.now()

        # Track max drawdown
        if drawdown > self.max_drawdown:
            self.max_drawdown = drawdown

    def record_trade(
        self,
        symbol: str,
        pnl: float,
        entry_price: float,
        exit_price: float,
    ) -> None:
        """Record a completed trade."""
        self._trade_history.append({
            "timestamp": datetime.now(),
            "symbol": symbol,
            "pnl": pnl,
            "entry_price": entry_price,
            "exit_price": exit_price,
        })

        if self._current_session:
            self._current_session.trades_count += 1
            self._current_session.gross_pnl += pnl

            if pnl > 0:
                self._current_session.winning_trades += 1
                self._current_session.consecutive_losses = 0
            else:
                self._current_session.losing_trades += 1
                self._current_session.consecutive_losses += 1

    def get_current_drawdown(self) -> float:
        """Get current drawdown as decimal."""
        if self.peak_value == 0:
            return 0
        return (self.peak_value - self.current_value) / self.peak_value

    def get_metrics(self) -> DrawdownMetrics:
        """Get current drawdown metrics."""
        drawdown = self.get_current_drawdown()

        if self.drawdown_start_date:
            duration = (datetime.now() - self.drawdown_start_date).days
        else:
            duration = 0

        # Calculate recovery needed
        if drawdown > 0:
            recovery_needed = (self.peak_value / self.current_value - 1)
        else:
            recovery_needed = 0

        return DrawdownMetrics(
            current_value=self.current_value,
            peak_value=self.peak_value,
            current_drawdown=drawdown,
            max_drawdown=self.max_drawdown,
            drawdown_duration_days=duration,
            recovery_needed=recovery_needed,
        )

    def get_session_metrics(self) -> Optional[TradingSession]:
        """Get current session metrics."""
        return self._current_session

    def check_new_session(self) -> None:
        """Check if we need to start a new trading session."""
        if self._current_session is None:
            self._init_session()
            return

        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if self._current_session.date < today:
            self._init_session()


class CircuitBreaker:
    """
    Implements circuit breaker logic for risk management.

    Monitors various risk metrics and determines trading state.
    """

    def __init__(
        self,
        drawdown_protection: DrawdownProtection,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.protection = drawdown_protection
        self.config = config or CircuitBreakerConfig()

        self._state = RiskState.NORMAL
        self._state_changed_at: datetime = datetime.now()
        self._halt_until: Optional[datetime] = None

        # Recovery tracking
        self._recovery_start_value: Optional[float] = None
        self._recovery_target: Optional[float] = None

    @property
    def state(self) -> RiskState:
        """Get current risk state."""
        return self._state

    def check_state(self) -> RiskState:
        """
        Evaluate current conditions and update state.

        The most restrictive state wins - we check all conditions and
        apply the most severe state that applies.

        Returns:
            Current RiskState
        """
        self.protection.check_new_session()

        # Check if halted with time remaining
        if self._halt_until and datetime.now() < self._halt_until:
            return RiskState.HALTED

        # Get metrics
        dd_metrics = self.protection.get_metrics()
        session = self.protection.get_session_metrics()

        # Track the most restrictive state and reason
        new_state = RiskState.NORMAL
        reason = ""

        # Check drawdown levels (most severe first)
        drawdown = dd_metrics.current_drawdown

        if drawdown >= self.config.halt_drawdown:
            new_state = RiskState.HALTED
            reason = f"Drawdown {drawdown:.1%} exceeds limit {self.config.halt_drawdown:.1%}"
        elif drawdown >= self.config.restricted_drawdown:
            new_state = RiskState.RESTRICTED
            reason = f"Drawdown {drawdown:.1%} exceeds limit {self.config.restricted_drawdown:.1%}"
        elif drawdown >= self.config.caution_drawdown:
            new_state = RiskState.CAUTION
            reason = f"Drawdown {drawdown:.1%} exceeds limit {self.config.caution_drawdown:.1%}"

        # Check session limits (can make state MORE restrictive, not less)
        if session:
            # Daily loss limit -> RESTRICTED
            if session.session_return < -self.config.max_daily_loss_pct:
                if new_state.value not in ("halted",):  # Don't downgrade from HALTED
                    new_state = RiskState.RESTRICTED
                    reason = f"Daily loss {session.session_return:.1%} exceeds limit {-self.config.max_daily_loss_pct:.1%}"
                    logger.warning(f"CIRCUIT BREAKER: RESTRICTED - {reason}")

            # Max trades -> RESTRICTED
            if session.trades_count >= self.config.max_daily_trades:
                if new_state.value not in ("halted",):
                    new_state = RiskState.RESTRICTED
                    reason = f"Trade count {session.trades_count} exceeds limit {self.config.max_daily_trades}"
                    logger.warning(f"CIRCUIT BREAKER: RESTRICTED - {reason}")

            # Consecutive losses -> CAUTION (only if not already more restrictive)
            if session.consecutive_losses >= self.config.max_consecutive_losses:
                if new_state == RiskState.NORMAL:
                    new_state = RiskState.CAUTION
                    reason = f"{session.consecutive_losses} consecutive losses"
                    logger.warning(f"CIRCUIT BREAKER: CAUTION - {reason}")

        # Apply halt timing if needed
        if new_state == RiskState.HALTED and self._state != RiskState.HALTED:
            self._halt_until = datetime.now() + timedelta(days=self.config.recovery_period_days)
            logger.warning(f"CIRCUIT BREAKER: HALTED - {reason}")

        # Check recovery progress only if we'd otherwise return to normal
        if new_state == RiskState.NORMAL and self._state != RiskState.NORMAL:
            if not self._check_recovery():
                # Stay at current state until recovery threshold met
                return self._state

        self._transition_to(new_state)
        return self._state

    def _transition_to(self, new_state: RiskState) -> None:
        """Transition to a new state."""
        if new_state != self._state:
            logger.info(f"Risk state: {self._state.value} -> {new_state.value}")
            self._state = new_state
            self._state_changed_at = datetime.now()

            if new_state == RiskState.NORMAL:
                self._recovery_start_value = None
                self._recovery_target = None
            elif self._recovery_start_value is None:
                self._recovery_start_value = self.protection.current_value
                # Set recovery target at 50% of drawdown recovered
                dd = self.protection.get_metrics()
                recovery_amount = (dd.peak_value - dd.current_value) * 0.5
                self._recovery_target = self.protection.current_value + recovery_amount

    def _check_recovery(self) -> bool:
        """Check if we've recovered enough to return to normal."""
        if self._recovery_target is None:
            return True

        return self.protection.current_value >= self._recovery_target

    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on current state.

        Returns:
            Multiplier (0 to 1) to apply to position sizes
        """
        if self._state == RiskState.NORMAL:
            return 1.0
        elif self._state == RiskState.CAUTION:
            multiplier = self.config.caution_size_multiplier

            # Gradual recovery
            if self.config.gradual_recovery and self._recovery_start_value:
                recovery_progress = self._get_recovery_progress()
                multiplier = self.config.caution_size_multiplier + (
                    (1 - self.config.caution_size_multiplier) * recovery_progress
                )

            return multiplier
        elif self._state == RiskState.RESTRICTED:
            return self.config.restricted_size_multiplier
        else:  # HALTED
            return 0.0

    def _get_recovery_progress(self) -> float:
        """Get recovery progress from 0 to 1."""
        if self._recovery_start_value is None or self._recovery_target is None:
            return 0

        total_recovery = self._recovery_target - self._recovery_start_value
        if total_recovery <= 0:
            return 1

        current_recovery = self.protection.current_value - self._recovery_start_value
        return max(0, min(1, current_recovery / total_recovery))

    def can_open_position(self) -> tuple[bool, str]:
        """Check if new positions can be opened."""
        state = self.check_state()

        if state == RiskState.HALTED:
            return False, "Trading halted due to excessive drawdown"
        elif state == RiskState.RESTRICTED:
            return False, "Only closing positions allowed"
        else:
            return True, "OK"

    def can_trade(self) -> tuple[bool, str]:
        """Check if any trading is allowed."""
        state = self.check_state()

        if state == RiskState.HALTED:
            remaining = ""
            if self._halt_until:
                remaining = f" (resumes {self._halt_until.strftime('%Y-%m-%d %H:%M')})"
            return False, f"Trading halted{remaining}"

        return True, "OK"

    def get_status(self) -> dict:
        """Get comprehensive status report."""
        dd_metrics = self.protection.get_metrics()
        session = self.protection.get_session_metrics()

        status = {
            "state": self._state.value,
            "state_since": self._state_changed_at.isoformat(),
            "position_size_multiplier": self.get_position_size_multiplier(),
            "drawdown": {
                "current": dd_metrics.current_drawdown,
                "max": dd_metrics.max_drawdown,
                "duration_days": dd_metrics.drawdown_duration_days,
                "recovery_needed": dd_metrics.recovery_needed,
            },
            "limits": {
                "caution_threshold": self.config.caution_drawdown,
                "restricted_threshold": self.config.restricted_drawdown,
                "halt_threshold": self.config.halt_drawdown,
            },
        }

        if session:
            status["session"] = {
                "date": session.date.isoformat(),
                "return": session.session_return,
                "trades": session.trades_count,
                "win_rate": session.win_rate,
                "consecutive_losses": session.consecutive_losses,
            }

        if self._halt_until:
            status["halt_until"] = self._halt_until.isoformat()

        if self._recovery_target:
            status["recovery"] = {
                "target": self._recovery_target,
                "progress": self._get_recovery_progress(),
            }

        return status


class PositionSizer:
    """
    Intelligent position sizing based on risk conditions.
    """

    def __init__(
        self,
        circuit_breaker: CircuitBreaker,
        base_risk_pct: float = 0.01,  # 1% base risk per trade
        max_position_pct: float = 0.05,  # 5% max position size
    ):
        self.circuit_breaker = circuit_breaker
        self.base_risk_pct = base_risk_pct
        self.max_position_pct = max_position_pct

    def calculate_position_size(
        self,
        portfolio_value: float,
        entry_price: float,
        stop_loss: float,
        signal_confidence: float = 1.0,
    ) -> float:
        """
        Calculate position size considering all risk factors.

        Args:
            portfolio_value: Current portfolio value
            entry_price: Planned entry price
            stop_loss: Stop loss price
            signal_confidence: Signal confidence (0 to 1)

        Returns:
            Number of shares/units to trade
        """
        # Check if trading allowed
        can_trade, _ = self.circuit_breaker.can_open_position()
        if not can_trade:
            return 0

        # Base risk amount
        risk_amount = portfolio_value * self.base_risk_pct

        # Apply circuit breaker multiplier
        cb_multiplier = self.circuit_breaker.get_position_size_multiplier()
        risk_amount *= cb_multiplier

        # Apply signal confidence
        risk_amount *= signal_confidence

        # Calculate shares based on risk
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share <= 0:
            return 0

        shares = risk_amount / risk_per_share

        # Apply maximum position size
        max_shares = (portfolio_value * self.max_position_pct) / entry_price
        shares = min(shares, max_shares)

        return shares

    def should_reduce_position(
        self,
        symbol: str,
        current_value: float,
        portfolio_value: float,
    ) -> tuple[bool, float]:
        """
        Check if an existing position should be reduced.

        Returns:
            Tuple of (should_reduce, suggested_reduction_pct)
        """
        state = self.circuit_breaker.check_state()

        if state == RiskState.RESTRICTED:
            # Reduce all positions by 50%
            return True, 0.5

        if state == RiskState.HALTED:
            # Close all positions
            return True, 1.0

        # Check if position is oversized
        position_pct = current_value / portfolio_value
        if position_pct > self.max_position_pct * 1.5:  # 50% buffer
            excess = position_pct - self.max_position_pct
            reduction = excess / position_pct
            return True, reduction

        return False, 0
