"""
Utility helper functions.
"""

from decimal import Decimal
from typing import Union


def calculate_position_size(
    portfolio_value: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
) -> float:
    """
    Calculate position size based on risk percentage.

    Args:
        portfolio_value: Total portfolio value
        risk_pct: Percentage of portfolio to risk (e.g., 0.01 for 1%)
        entry_price: Entry price for the trade
        stop_loss: Stop loss price

    Returns:
        Number of shares to buy
    """
    risk_amount = portfolio_value * risk_pct
    risk_per_share = abs(entry_price - stop_loss)

    if risk_per_share <= 0:
        return 0

    return risk_amount / risk_per_share


def format_currency(amount: Union[float, Decimal], symbol: str = "$") -> str:
    """Format a number as currency."""
    return f"{symbol}{float(amount):,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage."""
    return f"{value * 100:.{decimals}f}%"


def calculate_returns(start_value: float, end_value: float) -> float:
    """Calculate percentage returns."""
    if start_value == 0:
        return 0
    return (end_value - start_value) / start_value


def calculate_sharpe_ratio(
    returns: list[float],
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: List of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    import numpy as np

    if len(returns) < 2:
        return 0

    returns_array = np.array(returns)
    excess_returns = returns_array - (risk_free_rate / periods_per_year)

    mean_return = np.mean(excess_returns)
    std_return = np.std(excess_returns)

    if std_return == 0:
        return 0

    sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
    return float(sharpe)


def calculate_max_drawdown(equity_curve: list[float]) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.20 for 20% drawdown)
    """
    import numpy as np

    if len(equity_curve) < 2:
        return 0

    equity = np.array(equity_curve)
    peak = np.maximum.accumulate(equity)
    drawdown = (peak - equity) / peak

    return float(np.max(drawdown))
