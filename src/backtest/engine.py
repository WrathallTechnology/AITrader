"""
Backtesting engine for strategy evaluation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.strategies.base import Signal, SignalType

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    symbol: str
    entry_date: datetime
    entry_price: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 0
    side: str = "long"  # "long" or "short"
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: float = 0
    pnl_pct: float = 0
    exit_reason: str = ""
    commission: float = 0

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 100000
    commission_per_trade: float = 0  # Alpaca is commission-free
    slippage_pct: float = 0.001  # 0.1% slippage
    position_size_pct: float = 0.02  # 2% per position
    max_positions: int = 10
    use_stops: bool = True
    risk_free_rate: float = 0.02  # For Sharpe calculation


@dataclass
class BacktestResults:
    """Results from a backtest run."""
    # Returns
    total_return: float
    annual_return: float
    benchmark_return: float

    # Risk metrics
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    volatility: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    avg_trade_duration: float  # Days

    # Time series
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    trades: list[Trade]

    # Benchmark comparison
    alpha: float
    beta: float


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Simulates trading strategy execution on historical data.
    """

    def __init__(self, config: Optional[BacktestConfig] = None):
        self.config = config or BacktestConfig()
        self._reset()

    def _reset(self) -> None:
        """Reset engine state."""
        self.cash = self.config.initial_capital
        self.positions: dict[str, Trade] = {}  # Open positions
        self.closed_trades: list[Trade] = []
        self.equity_history: list[tuple[datetime, float]] = []
        self.current_date: Optional[datetime] = None

    def run(
        self,
        data: dict[str, pd.DataFrame],
        strategy_fn: Callable[[str, pd.DataFrame, datetime], Optional[Signal]],
        benchmark_symbol: str = "SPY",
    ) -> BacktestResults:
        """
        Run backtest on historical data.

        Args:
            data: Dictionary of symbol -> OHLCV DataFrame
            strategy_fn: Function that takes (symbol, data, date) and returns Signal
            benchmark_symbol: Symbol for benchmark comparison

        Returns:
            BacktestResults with performance metrics
        """
        self._reset()

        # Get all unique dates across all symbols
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())

        dates = sorted(all_dates)

        if not dates:
            raise ValueError("No data provided")

        logger.info(f"Running backtest from {dates[0]} to {dates[-1]}")
        logger.info(f"Symbols: {list(data.keys())}")

        # Main backtest loop
        for date in dates:
            self.current_date = date

            # Update open positions and check stops
            self._update_positions(data, date)

            # Generate signals for each symbol
            for symbol, df in data.items():
                if date not in df.index:
                    continue

                # Get data up to current date (no look-ahead)
                historical = df.loc[:date]

                # Generate signal
                try:
                    signal = strategy_fn(symbol, historical, date)
                    if signal and signal.is_actionable:
                        self._process_signal(signal, df.loc[date])
                except Exception as e:
                    logger.debug(f"Signal error for {symbol} on {date}: {e}")

            # Record equity
            equity = self._calculate_equity(data, date)
            self.equity_history.append((date, equity))

        # Close all remaining positions
        self._close_all_positions(data, dates[-1])

        # Calculate results
        return self._calculate_results(data, benchmark_symbol)

    def _update_positions(
        self,
        data: dict[str, pd.DataFrame],
        date: datetime,
    ) -> None:
        """Update positions and check stop/take profit."""
        for symbol, trade in list(self.positions.items()):
            if symbol not in data or date not in data[symbol].index:
                continue

            bar = data[symbol].loc[date]
            high = bar["high"]
            low = bar["low"]
            close = bar["close"]

            # Check stop loss
            if self.config.use_stops and trade.stop_loss:
                if trade.side == "long" and low <= trade.stop_loss:
                    self._close_position(symbol, trade.stop_loss, "stop_loss", date)
                    continue
                elif trade.side == "short" and high >= trade.stop_loss:
                    self._close_position(symbol, trade.stop_loss, "stop_loss", date)
                    continue

            # Check take profit
            if trade.take_profit:
                if trade.side == "long" and high >= trade.take_profit:
                    self._close_position(symbol, trade.take_profit, "take_profit", date)
                    continue
                elif trade.side == "short" and low <= trade.take_profit:
                    self._close_position(symbol, trade.take_profit, "take_profit", date)
                    continue

    def _process_signal(self, signal: Signal, bar: pd.Series) -> None:
        """Process a trading signal."""
        symbol = signal.symbol

        # Check if we already have a position
        if symbol in self.positions:
            existing = self.positions[symbol]

            # Close on opposite signal
            if signal.signal_type == SignalType.BUY and existing.side == "short":
                self._close_position(symbol, bar["close"], "signal_reversal", self.current_date)
            elif signal.signal_type == SignalType.SELL and existing.side == "long":
                self._close_position(symbol, bar["close"], "signal_reversal", self.current_date)
            else:
                return  # Already in position same direction

        # Check position limits
        if len(self.positions) >= self.config.max_positions:
            return

        # Calculate position size
        equity = self._calculate_equity_fast()
        position_value = equity * self.config.position_size_pct
        entry_price = bar["close"] * (1 + self.config.slippage_pct)

        if signal.signal_type == SignalType.SELL:
            entry_price = bar["close"] * (1 - self.config.slippage_pct)

        quantity = position_value / entry_price

        if quantity <= 0:
            return

        # Check cash
        if position_value > self.cash:
            quantity = self.cash / entry_price
            position_value = self.cash

        if quantity <= 0:
            return

        # Open position
        trade = Trade(
            symbol=symbol,
            entry_date=self.current_date,
            entry_price=entry_price,
            quantity=quantity,
            side="long" if signal.signal_type == SignalType.BUY else "short",
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            commission=self.config.commission_per_trade,
        )

        self.positions[symbol] = trade
        self.cash -= position_value + trade.commission

        logger.debug(
            f"Opened {trade.side} {symbol}: {quantity:.2f} @ {entry_price:.2f}"
        )

    def _close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str,
        date: datetime,
    ) -> None:
        """Close a position."""
        if symbol not in self.positions:
            return

        trade = self.positions[symbol]

        # Apply slippage
        if trade.side == "long":
            exit_price *= (1 - self.config.slippage_pct)
        else:
            exit_price *= (1 + self.config.slippage_pct)

        trade.exit_date = date
        trade.exit_price = exit_price
        trade.exit_reason = reason

        # Calculate PnL
        if trade.side == "long":
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity

        trade.pnl -= trade.commission
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.quantity)

        # Update cash
        self.cash += (trade.entry_price * trade.quantity) + trade.pnl

        self.closed_trades.append(trade)
        del self.positions[symbol]

        logger.debug(
            f"Closed {trade.side} {symbol}: PnL ${trade.pnl:.2f} ({trade.pnl_pct:.2%}) - {reason}"
        )

    def _close_all_positions(self, data: dict[str, pd.DataFrame], date: datetime) -> None:
        """Close all open positions at end of backtest."""
        for symbol in list(self.positions.keys()):
            if symbol in data and date in data[symbol].index:
                close_price = data[symbol].loc[date, "close"]
                self._close_position(symbol, close_price, "backtest_end", date)

    def _calculate_equity(self, data: dict[str, pd.DataFrame], date: datetime) -> float:
        """Calculate total equity including open positions."""
        equity = self.cash

        for symbol, trade in self.positions.items():
            if symbol in data and date in data[symbol].index:
                current_price = data[symbol].loc[date, "close"]
                position_value = trade.quantity * current_price
                equity += position_value

        return equity

    def _calculate_equity_fast(self) -> float:
        """Quick equity calculation using last known values."""
        if self.equity_history:
            return self.equity_history[-1][1]
        return self.config.initial_capital

    def _calculate_results(
        self,
        data: dict[str, pd.DataFrame],
        benchmark_symbol: str,
    ) -> BacktestResults:
        """Calculate comprehensive backtest results."""
        # Build equity curve
        dates = [d for d, _ in self.equity_history]
        equities = [e for _, e in self.equity_history]

        equity_curve = pd.Series(equities, index=pd.DatetimeIndex(dates))

        # Calculate returns
        returns = equity_curve.pct_change().dropna()

        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Annualized return
        days = (dates[-1] - dates[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        volatility = returns.std() * np.sqrt(252)

        # Sharpe ratio
        excess_returns = returns - self.config.risk_free_rate / 252
        sharpe_ratio = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                        if excess_returns.std() > 0 else 0)

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - self.config.risk_free_rate) / downside_std if downside_std > 0 else 0

        # Drawdown
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = abs(drawdown.min())

        # Calmar ratio
        calmar_ratio = annual_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        total_trades = len(self.closed_trades)
        winning_trades = sum(1 for t in self.closed_trades if t.is_winner)
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        wins = [t.pnl for t in self.closed_trades if t.is_winner]
        losses = [t.pnl for t in self.closed_trades if not t.is_winner]

        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Average trade duration
        durations = []
        for t in self.closed_trades:
            if t.exit_date and t.entry_date:
                durations.append((t.exit_date - t.entry_date).days)
        avg_trade_duration = np.mean(durations) if durations else 0

        # Benchmark comparison
        benchmark_return = 0
        alpha = 0
        beta = 1

        if benchmark_symbol in data:
            benchmark_data = data[benchmark_symbol]
            benchmark_start = benchmark_data.iloc[0]["close"]
            benchmark_end = benchmark_data.iloc[-1]["close"]
            benchmark_return = (benchmark_end / benchmark_start) - 1

            # Calculate alpha and beta
            benchmark_returns = benchmark_data["close"].pct_change().dropna()
            aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()

            if len(aligned) > 20:
                aligned.columns = ["strategy", "benchmark"]
                covariance = aligned["strategy"].cov(aligned["benchmark"])
                variance = aligned["benchmark"].var()
                beta = covariance / variance if variance > 0 else 1
                alpha = annual_return - (self.config.risk_free_rate + beta * (benchmark_return - self.config.risk_free_rate))

        return BacktestResults(
            total_return=total_return,
            annual_return=annual_return,
            benchmark_return=benchmark_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            calmar_ratio=calmar_ratio,
            volatility=volatility,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            avg_trade_duration=avg_trade_duration,
            equity_curve=equity_curve,
            drawdown_curve=drawdown,
            trades=self.closed_trades,
            alpha=alpha,
            beta=beta,
        )


class WalkForwardOptimizer:
    """
    Walk-forward optimization for robust parameter selection.

    Divides data into training and testing periods, optimizes on training,
    validates on testing, then rolls forward.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        train_period_days: int = 252,  # 1 year
        test_period_days: int = 63,  # 3 months
        min_train_trades: int = 30,
    ):
        self.engine = engine
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.min_train_trades = min_train_trades

    def optimize(
        self,
        data: dict[str, pd.DataFrame],
        strategy_factory: Callable[..., Callable],
        param_grid: dict[str, list],
        metric: str = "sharpe_ratio",
    ) -> tuple[dict, list[BacktestResults]]:
        """
        Perform walk-forward optimization.

        Args:
            data: Historical data
            strategy_factory: Function that takes params and returns strategy_fn
            param_grid: Dictionary of parameter names to lists of values
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)

        Returns:
            Tuple of (best_params, list of out-of-sample results)
        """
        # Get date range
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index.tolist())
        dates = sorted(all_dates)

        results_by_period = []
        best_params_history = []

        # Walk forward
        train_start_idx = 0

        while train_start_idx + self.train_period_days + self.test_period_days < len(dates):
            train_end_idx = train_start_idx + self.train_period_days
            test_end_idx = train_end_idx + self.test_period_days

            train_dates = dates[train_start_idx:train_end_idx]
            test_dates = dates[train_end_idx:test_end_idx]

            logger.info(f"Walk-forward period: Train {train_dates[0]} to {train_dates[-1]}, "
                       f"Test {test_dates[0]} to {test_dates[-1]}")

            # Filter data for training period
            train_data = self._filter_data(data, train_dates[0], train_dates[-1])

            # Optimize on training period
            best_params, _ = self._grid_search(
                train_data, strategy_factory, param_grid, metric
            )
            best_params_history.append(best_params)

            # Test on out-of-sample period
            test_data = self._filter_data(data, test_dates[0], test_dates[-1])
            strategy_fn = strategy_factory(**best_params)
            result = self.engine.run(test_data, strategy_fn)
            results_by_period.append(result)

            # Roll forward
            train_start_idx += self.test_period_days

        # Use most common best params as final recommendation
        final_params = self._aggregate_params(best_params_history)

        return final_params, results_by_period

    def _filter_data(
        self,
        data: dict[str, pd.DataFrame],
        start_date: datetime,
        end_date: datetime,
    ) -> dict[str, pd.DataFrame]:
        """Filter data to date range."""
        filtered = {}
        for symbol, df in data.items():
            mask = (df.index >= start_date) & (df.index <= end_date)
            if mask.any():
                filtered[symbol] = df[mask]
        return filtered

    def _grid_search(
        self,
        data: dict[str, pd.DataFrame],
        strategy_factory: Callable,
        param_grid: dict[str, list],
        metric: str,
    ) -> tuple[dict, float]:
        """Grid search for best parameters."""
        import itertools

        # Generate all combinations
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(itertools.product(*values))

        best_params = None
        best_score = float('-inf')

        for combo in combinations:
            params = dict(zip(keys, combo))

            try:
                strategy_fn = strategy_factory(**params)
                result = self.engine.run(data, strategy_fn)

                score = getattr(result, metric, 0)

                # Require minimum trades
                if result.total_trades < self.min_train_trades:
                    continue

                if score > best_score:
                    best_score = score
                    best_params = params

            except Exception as e:
                logger.debug(f"Error with params {params}: {e}")

        return best_params or dict(zip(keys, [v[0] for v in values])), best_score

    def _aggregate_params(self, params_history: list[dict]) -> dict:
        """Aggregate parameter choices across periods."""
        if not params_history:
            return {}

        # For numeric params, use median
        # For categorical, use mode
        aggregated = {}

        keys = params_history[0].keys()
        for key in keys:
            values = [p[key] for p in params_history]

            if isinstance(values[0], (int, float)):
                aggregated[key] = np.median(values)
                if isinstance(values[0], int):
                    aggregated[key] = int(aggregated[key])
            else:
                # Mode for categorical
                from collections import Counter
                aggregated[key] = Counter(values).most_common(1)[0][0]

        return aggregated


def print_backtest_report(results: BacktestResults) -> None:
    """Print formatted backtest report."""
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print("\n--- Returns ---")
    print(f"Total Return:      {results.total_return:>10.2%}")
    print(f"Annual Return:     {results.annual_return:>10.2%}")
    print(f"Benchmark Return:  {results.benchmark_return:>10.2%}")
    print(f"Alpha:             {results.alpha:>10.2%}")
    print(f"Beta:              {results.beta:>10.2f}")

    print("\n--- Risk Metrics ---")
    print(f"Sharpe Ratio:      {results.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:     {results.sortino_ratio:>10.2f}")
    print(f"Max Drawdown:      {results.max_drawdown:>10.2%}")
    print(f"Calmar Ratio:      {results.calmar_ratio:>10.2f}")
    print(f"Volatility:        {results.volatility:>10.2%}")

    print("\n--- Trade Statistics ---")
    print(f"Total Trades:      {results.total_trades:>10d}")
    print(f"Win Rate:          {results.win_rate:>10.2%}")
    print(f"Profit Factor:     {results.profit_factor:>10.2f}")
    print(f"Avg Win:           ${results.avg_win:>9.2f}")
    print(f"Avg Loss:          ${results.avg_loss:>9.2f}")
    print(f"Avg Duration:      {results.avg_trade_duration:>10.1f} days")

    print("=" * 60)
