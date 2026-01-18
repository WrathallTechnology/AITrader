"""
Portfolio and risk management module.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional

from src.client import AlpacaClient
from src.strategies.base import Signal, SignalType
from config import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class PositionInfo:
    """Information about a position."""
    symbol: str
    quantity: float
    market_value: Decimal
    cost_basis: Decimal
    unrealized_pl: Decimal
    unrealized_pl_pct: float
    side: str  # 'long' or 'short'


@dataclass
class TradeOrder:
    """Proposed trade order."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    order_type: str  # 'market', 'limit', 'bracket'
    limit_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class PortfolioManager:
    """
    Manages portfolio state and risk controls.

    Responsibilities:
    - Track positions and portfolio value
    - Calculate position sizes
    - Enforce risk limits
    - Generate trade orders from signals
    """

    def __init__(self, client: AlpacaClient, config: TradingConfig):
        self.client = client
        self.config = config

    # ==================== Portfolio State ====================

    def get_portfolio_value(self) -> Decimal:
        """Get total portfolio value."""
        return self.client.get_portfolio_value()

    def get_buying_power(self) -> Decimal:
        """Get available buying power."""
        return self.client.get_buying_power()

    def get_cash(self) -> Decimal:
        """Get available cash."""
        return self.client.get_cash()

    def get_positions(self) -> list[PositionInfo]:
        """Get all current positions."""
        positions = self.client.get_all_positions()

        return [
            PositionInfo(
                symbol=p.symbol,
                quantity=float(p.qty),
                market_value=Decimal(p.market_value),
                cost_basis=Decimal(p.cost_basis),
                unrealized_pl=Decimal(p.unrealized_pl),
                unrealized_pl_pct=float(p.unrealized_plpc) * 100,
                side="long" if float(p.qty) > 0 else "short",
            )
            for p in positions
        ]

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position for a specific symbol."""
        position = self.client.get_position(symbol)
        if position is None:
            return None

        return PositionInfo(
            symbol=position.symbol,
            quantity=float(position.qty),
            market_value=Decimal(position.market_value),
            cost_basis=Decimal(position.cost_basis),
            unrealized_pl=Decimal(position.unrealized_pl),
            unrealized_pl_pct=float(position.unrealized_plpc) * 100,
            side="long" if float(position.qty) > 0 else "short",
        )

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.client.get_all_positions())

    # ==================== Position Sizing ====================

    def calculate_position_size(
        self,
        symbol: str,
        price: float,
        stop_loss: Optional[float] = None,
    ) -> float:
        """
        Calculate position size based on risk parameters.

        Uses either:
        1. Fixed percentage of portfolio
        2. Risk-based sizing (if stop_loss provided)
        """
        portfolio_value = float(self.get_portfolio_value())

        if stop_loss and stop_loss > 0:
            # Risk-based position sizing
            risk_amount = portfolio_value * self.config.risk_per_trade
            risk_per_share = abs(price - stop_loss)

            if risk_per_share > 0:
                shares = risk_amount / risk_per_share
            else:
                shares = 0
        else:
            # Fixed percentage sizing
            position_value = portfolio_value * self.config.default_position_size_pct
            shares = position_value / price

        # Round down to whole shares for stocks
        if not self.client.is_crypto(symbol):
            shares = int(shares)

        return max(0, shares)

    def calculate_max_position_value(self) -> Decimal:
        """Calculate maximum value for a single position."""
        portfolio_value = self.get_portfolio_value()
        return portfolio_value * Decimal(str(self.config.default_position_size_pct))

    # ==================== Risk Checks ====================

    def can_open_position(self, symbol: str) -> tuple[bool, str]:
        """
        Check if we can open a new position.

        Returns:
            Tuple of (can_open, reason)
        """
        # Check max positions
        current_positions = self.get_position_count()
        if current_positions >= self.config.max_positions:
            return False, f"Max positions reached ({self.config.max_positions})"

        # Check if already have position in this symbol
        existing = self.get_position(symbol)
        if existing is not None:
            return False, f"Already have position in {symbol}"

        # Check buying power
        buying_power = float(self.get_buying_power())
        min_required = float(self.get_portfolio_value()) * 0.01  # At least 1%
        if buying_power < min_required:
            return False, "Insufficient buying power"

        # Check if asset is tradable
        if not self.client.is_tradable(symbol):
            return False, f"{symbol} is not tradable"

        return True, "OK"

    def validate_order(self, order: TradeOrder) -> tuple[bool, str]:
        """
        Validate a proposed order.

        Returns:
            Tuple of (is_valid, reason)
        """
        if order.quantity <= 0:
            return False, "Invalid quantity"

        if order.side == "buy":
            # Check buying power
            buying_power = float(self.get_buying_power())
            if order.limit_price:
                required = order.quantity * order.limit_price
            else:
                # Estimate market order cost with buffer
                required = order.quantity * (order.limit_price or 0) * 1.05

            if required > buying_power:
                return False, f"Insufficient buying power (need {required:.2f})"

        return True, "OK"

    # ==================== Order Generation ====================

    def signal_to_order(
        self,
        signal: Signal,
        use_bracket: bool = True,
    ) -> Optional[TradeOrder]:
        """
        Convert a trading signal to a trade order.

        Args:
            signal: Trading signal from strategy
            use_bracket: Whether to use bracket orders with SL/TP

        Returns:
            TradeOrder if actionable, None otherwise
        """
        if not signal.is_actionable:
            return None

        # Check if we can open position
        if signal.signal_type == SignalType.BUY:
            can_open, reason = self.can_open_position(signal.symbol)
            if not can_open:
                logger.warning(f"Cannot open position for {signal.symbol}: {reason}")
                return None

        # Calculate position size
        quantity = self.calculate_position_size(
            symbol=signal.symbol,
            price=signal.price or 0,
            stop_loss=signal.stop_loss,
        )

        if quantity <= 0:
            logger.warning(f"Position size is 0 for {signal.symbol}")
            return None

        # Determine order type
        if use_bracket and signal.stop_loss and signal.take_profit:
            order_type = "bracket"
        else:
            order_type = "market"

        return TradeOrder(
            symbol=signal.symbol,
            side=signal.signal_type.value,
            quantity=quantity,
            order_type=order_type,
            limit_price=signal.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )

    # ==================== Order Execution ====================

    def execute_order(self, order: TradeOrder) -> Optional[str]:
        """
        Execute a trade order.

        Returns:
            Order ID if successful, None otherwise
        """
        # Validate order
        is_valid, reason = self.validate_order(order)
        if not is_valid:
            logger.error(f"Invalid order: {reason}")
            return None

        try:
            if order.order_type == "bracket":
                result = self.client.submit_bracket_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=order.side,
                    take_profit_price=order.take_profit,
                    stop_loss_price=order.stop_loss,
                )
            elif order.order_type == "limit" and order.limit_price:
                result = self.client.submit_limit_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=order.side,
                    limit_price=order.limit_price,
                )
            else:
                result = self.client.submit_market_order(
                    symbol=order.symbol,
                    qty=order.quantity,
                    side=order.side,
                )

            logger.info(f"Order submitted: {result.id}")
            return str(result.id)

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return None

    # ==================== Position Management ====================

    def close_position(self, symbol: str) -> bool:
        """Close a position entirely."""
        try:
            self.client.close_position(symbol)
            logger.info(f"Closed position: {symbol}")
            return True
        except Exception as e:
            logger.error(f"Failed to close {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        """Close all positions."""
        try:
            self.client.close_all_positions()
            logger.info("Closed all positions")
            return True
        except Exception as e:
            logger.error(f"Failed to close all positions: {e}")
            return False
