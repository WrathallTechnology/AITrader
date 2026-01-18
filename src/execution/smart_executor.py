"""
Smart order execution and management.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class ExecutionStyle(Enum):
    """Execution style preferences."""
    AGGRESSIVE = "aggressive"  # Market orders, fast fills
    PASSIVE = "passive"  # Limit orders at favorable prices
    BALANCED = "balanced"  # Mix based on conditions
    SCALED = "scaled"  # Scale in/out of positions


@dataclass
class ExecutionPlan:
    """Plan for executing an order."""
    symbol: str
    side: str  # "buy" or "sell"
    total_quantity: float
    order_type: OrderType
    limit_price: Optional[float]
    urgency: float  # 0 to 1
    max_slippage_pct: float
    time_limit_minutes: int
    tranches: int  # Number of orders to split into
    tranche_interval_seconds: int


@dataclass
class ExecutionResult:
    """Result of order execution."""
    symbol: str
    side: str
    requested_quantity: float
    filled_quantity: float
    avg_fill_price: float
    total_cost: float
    slippage_pct: float
    execution_time_seconds: float
    num_fills: int
    success: bool
    message: str


class SpreadAnalyzer:
    """
    Analyzes bid-ask spread for execution decisions.
    """

    @staticmethod
    def calculate_effective_spread(
        bid: float,
        ask: float,
        mid: Optional[float] = None,
    ) -> dict:
        """
        Calculate spread metrics.

        Returns:
            Dictionary with spread metrics
        """
        mid = mid or (bid + ask) / 2
        absolute_spread = ask - bid
        relative_spread = absolute_spread / mid if mid > 0 else 0

        return {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "absolute_spread": absolute_spread,
            "relative_spread": relative_spread,
            "half_spread": relative_spread / 2,
        }

    @staticmethod
    def is_spread_acceptable(
        spread_pct: float,
        max_spread_pct: float = 0.005,  # 0.5% default
    ) -> bool:
        """Check if spread is acceptable for trading."""
        return spread_pct <= max_spread_pct

    @staticmethod
    def estimate_market_impact(
        quantity: float,
        avg_volume: float,
        spread_pct: float,
    ) -> float:
        """
        Estimate market impact of an order.

        Uses square-root model: impact ∝ √(Q/V) * spread

        Returns:
            Estimated impact as percentage
        """
        if avg_volume <= 0:
            return spread_pct

        participation_rate = quantity / avg_volume
        impact = np.sqrt(participation_rate) * spread_pct * 2

        return min(impact, 0.05)  # Cap at 5%


class SmartOrderRouter:
    """
    Determines optimal order type and execution strategy.
    """

    def __init__(
        self,
        default_style: ExecutionStyle = ExecutionStyle.BALANCED,
        max_spread_pct: float = 0.005,
        large_order_threshold: float = 0.01,  # 1% of daily volume
    ):
        self.default_style = default_style
        self.max_spread_pct = max_spread_pct
        self.large_order_threshold = large_order_threshold

    def create_execution_plan(
        self,
        symbol: str,
        side: str,
        quantity: float,
        current_price: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        avg_daily_volume: Optional[float] = None,
        urgency: float = 0.5,
        style: Optional[ExecutionStyle] = None,
    ) -> ExecutionPlan:
        """
        Create an execution plan for an order.

        Args:
            symbol: Trading symbol
            side: "buy" or "sell"
            quantity: Number of shares/units
            current_price: Current market price
            bid: Current bid price
            ask: Current ask price
            avg_daily_volume: Average daily volume
            urgency: How urgent (0 = patient, 1 = immediate)
            style: Execution style override

        Returns:
            ExecutionPlan with recommended approach
        """
        style = style or self.default_style

        # Analyze spread if available
        if bid and ask:
            spread_info = SpreadAnalyzer.calculate_effective_spread(bid, ask)
            spread_pct = spread_info["relative_spread"]
        else:
            spread_pct = 0.002  # Assume 0.2% spread

        # Determine if order is large
        is_large_order = False
        if avg_daily_volume and avg_daily_volume > 0:
            participation = quantity / avg_daily_volume
            is_large_order = participation > self.large_order_threshold

        # Calculate market impact
        market_impact = 0
        if avg_daily_volume:
            market_impact = SpreadAnalyzer.estimate_market_impact(
                quantity, avg_daily_volume, spread_pct
            )

        # Determine order type and parameters
        if style == ExecutionStyle.AGGRESSIVE or urgency > 0.8:
            order_type = OrderType.MARKET
            limit_price = None
            tranches = 1
            max_slippage = spread_pct + market_impact + 0.002

        elif style == ExecutionStyle.PASSIVE or urgency < 0.3:
            order_type = OrderType.LIMIT
            if side == "buy":
                limit_price = bid if bid else current_price * 0.998
            else:
                limit_price = ask if ask else current_price * 1.002
            tranches = 1
            max_slippage = spread_pct / 2

        elif style == ExecutionStyle.SCALED or is_large_order:
            order_type = OrderType.LIMIT
            limit_price = current_price
            # Split into tranches based on size
            tranches = min(5, max(2, int(quantity / (avg_daily_volume * 0.001) + 1))) if avg_daily_volume else 3
            max_slippage = spread_pct + market_impact / 2

        else:  # BALANCED
            # Choose based on conditions
            if spread_pct > self.max_spread_pct:
                # Wide spread - use limit
                order_type = OrderType.LIMIT
                if side == "buy":
                    limit_price = current_price * (1 - spread_pct / 4)
                else:
                    limit_price = current_price * (1 + spread_pct / 4)
            else:
                # Tight spread - market is fine
                order_type = OrderType.MARKET
                limit_price = None

            tranches = 2 if is_large_order else 1
            max_slippage = spread_pct + 0.001

        # Calculate time limit based on urgency
        if urgency > 0.7:
            time_limit = 5
        elif urgency > 0.4:
            time_limit = 30
        else:
            time_limit = 120

        # Tranche interval
        tranche_interval = int(time_limit * 60 / tranches) if tranches > 1 else 0

        return ExecutionPlan(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            order_type=order_type,
            limit_price=limit_price,
            urgency=urgency,
            max_slippage_pct=max_slippage,
            time_limit_minutes=time_limit,
            tranches=tranches,
            tranche_interval_seconds=tranche_interval,
        )


class OrderManager:
    """
    Manages order lifecycle and tracking.
    """

    def __init__(self):
        self._pending_orders: dict[str, dict] = {}  # order_id -> order info
        self._execution_history: list[ExecutionResult] = []

    def track_order(
        self,
        order_id: str,
        plan: ExecutionPlan,
        submitted_at: datetime,
    ) -> None:
        """Start tracking an order."""
        self._pending_orders[order_id] = {
            "plan": plan,
            "submitted_at": submitted_at,
            "status": "pending",
            "fills": [],
            "filled_quantity": 0,
            "avg_fill_price": 0,
        }

    def record_fill(
        self,
        order_id: str,
        quantity: float,
        price: float,
        timestamp: datetime,
    ) -> None:
        """Record a fill for an order."""
        if order_id not in self._pending_orders:
            logger.warning(f"Unknown order: {order_id}")
            return

        order = self._pending_orders[order_id]
        order["fills"].append({
            "quantity": quantity,
            "price": price,
            "timestamp": timestamp,
        })

        # Update running average
        total_qty = order["filled_quantity"] + quantity
        total_value = (order["avg_fill_price"] * order["filled_quantity"]) + (price * quantity)

        order["filled_quantity"] = total_qty
        order["avg_fill_price"] = total_value / total_qty if total_qty > 0 else 0

    def complete_order(
        self,
        order_id: str,
        final_status: str = "filled",
    ) -> Optional[ExecutionResult]:
        """Complete tracking of an order."""
        if order_id not in self._pending_orders:
            return None

        order = self._pending_orders.pop(order_id)
        plan = order["plan"]

        # Calculate execution quality
        if order["filled_quantity"] > 0:
            expected_price = plan.limit_price or order["avg_fill_price"]
            actual_slippage = abs(order["avg_fill_price"] - expected_price) / expected_price

            if plan.side == "buy":
                # Positive slippage = paid more than expected
                actual_slippage = (order["avg_fill_price"] - expected_price) / expected_price
            else:
                # Positive slippage = received less than expected
                actual_slippage = (expected_price - order["avg_fill_price"]) / expected_price
        else:
            actual_slippage = 0

        exec_time = (datetime.now() - order["submitted_at"]).total_seconds()

        result = ExecutionResult(
            symbol=plan.symbol,
            side=plan.side,
            requested_quantity=plan.total_quantity,
            filled_quantity=order["filled_quantity"],
            avg_fill_price=order["avg_fill_price"],
            total_cost=order["avg_fill_price"] * order["filled_quantity"],
            slippage_pct=actual_slippage,
            execution_time_seconds=exec_time,
            num_fills=len(order["fills"]),
            success=order["filled_quantity"] >= plan.total_quantity * 0.95,
            message=final_status,
        )

        self._execution_history.append(result)
        return result

    def get_execution_stats(self) -> dict:
        """Get execution quality statistics."""
        if not self._execution_history:
            return {"total_orders": 0}

        results = self._execution_history

        total_orders = len(results)
        successful = sum(1 for r in results if r.success)
        avg_slippage = np.mean([r.slippage_pct for r in results])
        avg_exec_time = np.mean([r.execution_time_seconds for r in results])

        return {
            "total_orders": total_orders,
            "success_rate": successful / total_orders,
            "avg_slippage_pct": avg_slippage,
            "avg_execution_time_seconds": avg_exec_time,
            "total_slippage_cost": sum(r.slippage_pct * r.total_cost for r in results),
        }


class PositionScaler:
    """
    Manages scaling in and out of positions.
    """

    def __init__(
        self,
        scale_in_tranches: int = 3,
        scale_out_tranches: int = 3,
        scale_in_spacing_pct: float = 0.01,  # 1% between entries
        scale_out_spacing_pct: float = 0.02,  # 2% between exits
    ):
        self.scale_in_tranches = scale_in_tranches
        self.scale_out_tranches = scale_out_tranches
        self.scale_in_spacing = scale_in_spacing_pct
        self.scale_out_spacing = scale_out_spacing_pct

    def create_scale_in_plan(
        self,
        symbol: str,
        side: str,
        total_quantity: float,
        entry_price: float,
        stop_loss: float,
    ) -> list[dict]:
        """
        Create a plan for scaling into a position.

        Returns list of orders with prices and quantities.
        """
        orders = []
        quantity_per_tranche = total_quantity / self.scale_in_tranches

        for i in range(self.scale_in_tranches):
            if side == "buy":
                # Buy more at lower prices
                price = entry_price * (1 - i * self.scale_in_spacing)
            else:
                # Sell more at higher prices
                price = entry_price * (1 + i * self.scale_in_spacing)

            orders.append({
                "symbol": symbol,
                "side": side,
                "quantity": quantity_per_tranche,
                "limit_price": price,
                "order_type": OrderType.LIMIT,
                "tranche": i + 1,
            })

        return orders

    def create_scale_out_plan(
        self,
        symbol: str,
        side: str,  # "sell" for long position, "buy" for short
        total_quantity: float,
        entry_price: float,
        take_profit: float,
    ) -> list[dict]:
        """
        Create a plan for scaling out of a position.

        Returns list of orders with prices and quantities.
        """
        orders = []

        # Non-uniform sizing: take more profit early
        weights = [0.4, 0.35, 0.25][:self.scale_out_tranches]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        price_range = abs(take_profit - entry_price)

        for i, weight in enumerate(weights):
            quantity = total_quantity * weight

            if side == "sell":  # Closing long
                price = entry_price + (price_range * (i + 1) / len(weights))
            else:  # Closing short
                price = entry_price - (price_range * (i + 1) / len(weights))

            orders.append({
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "limit_price": price,
                "order_type": OrderType.LIMIT,
                "tranche": i + 1,
            })

        return orders

    def calculate_break_even(
        self,
        fills: list[dict],  # List of {"quantity": x, "price": y}
    ) -> float:
        """Calculate break-even price from multiple fills."""
        if not fills:
            return 0

        total_cost = sum(f["quantity"] * f["price"] for f in fills)
        total_quantity = sum(f["quantity"] for f in fills)

        return total_cost / total_quantity if total_quantity > 0 else 0


class TrailingStopManager:
    """
    Manages trailing stop orders.
    """

    def __init__(
        self,
        initial_stop_pct: float = 0.02,  # 2% initial stop
        trail_pct: float = 0.015,  # 1.5% trailing
        activation_profit_pct: float = 0.01,  # Activate after 1% profit
    ):
        self.initial_stop_pct = initial_stop_pct
        self.trail_pct = trail_pct
        self.activation_profit_pct = activation_profit_pct

        self._positions: dict[str, dict] = {}

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        side: str,
        initial_stop: Optional[float] = None,
    ) -> float:
        """Register a new position for trailing stop management."""
        if initial_stop is None:
            if side == "long":
                initial_stop = entry_price * (1 - self.initial_stop_pct)
            else:
                initial_stop = entry_price * (1 + self.initial_stop_pct)

        self._positions[symbol] = {
            "entry_price": entry_price,
            "side": side,
            "current_stop": initial_stop,
            "highest_price": entry_price,
            "lowest_price": entry_price,
            "trailing_active": False,
        }

        return initial_stop

    def update_price(self, symbol: str, current_price: float) -> Optional[float]:
        """
        Update price and return new stop level if changed.

        Returns:
            New stop price if updated, None otherwise
        """
        if symbol not in self._positions:
            return None

        pos = self._positions[symbol]

        # Update price extremes
        if current_price > pos["highest_price"]:
            pos["highest_price"] = current_price
        if current_price < pos["lowest_price"]:
            pos["lowest_price"] = current_price

        # Check if trailing should activate
        if not pos["trailing_active"]:
            if pos["side"] == "long":
                profit_pct = (current_price - pos["entry_price"]) / pos["entry_price"]
            else:
                profit_pct = (pos["entry_price"] - current_price) / pos["entry_price"]

            if profit_pct >= self.activation_profit_pct:
                pos["trailing_active"] = True
                logger.debug(f"Trailing stop activated for {symbol}")

        # Calculate new trailing stop
        if pos["trailing_active"]:
            if pos["side"] == "long":
                new_stop = pos["highest_price"] * (1 - self.trail_pct)
                if new_stop > pos["current_stop"]:
                    pos["current_stop"] = new_stop
                    return new_stop
            else:
                new_stop = pos["lowest_price"] * (1 + self.trail_pct)
                if new_stop < pos["current_stop"]:
                    pos["current_stop"] = new_stop
                    return new_stop

        return None

    def check_stop_hit(self, symbol: str, current_price: float) -> bool:
        """Check if stop has been hit."""
        if symbol not in self._positions:
            return False

        pos = self._positions[symbol]

        if pos["side"] == "long":
            return current_price <= pos["current_stop"]
        else:
            return current_price >= pos["current_stop"]

    def get_current_stop(self, symbol: str) -> Optional[float]:
        """Get current stop level for a position."""
        if symbol in self._positions:
            return self._positions[symbol]["current_stop"]
        return None

    def close_position(self, symbol: str) -> None:
        """Remove position from tracking."""
        self._positions.pop(symbol, None)
