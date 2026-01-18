"""
Example: Running a complete backtest with the trading system.

This demonstrates:
1. Loading historical data
2. Running backtest with the hybrid strategy
3. Analyzing results
4. Walk-forward optimization
"""

from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from src.strategies import TechnicalStrategy, MomentumStrategy, MeanReversionStrategy
from src.strategies.hybrid import AdvancedHybridStrategy
from src.data.processor import DataProcessor
from src.backtest import (
    BacktestEngine,
    BacktestConfig,
    WalkForwardOptimizer,
    print_backtest_report,
)


def generate_sample_data(
    symbol: str,
    days: int = 500,
    start_price: float = 100.0,
    volatility: float = 0.02,
    trend: float = 0.0001,
) -> pd.DataFrame:
    """
    Generate sample OHLCV data for testing.

    In production, you'd fetch real data from Alpaca.
    """
    dates = pd.date_range(end=datetime.now(), periods=days, freq='D')

    # Generate returns with trend and volatility
    returns = np.random.normal(trend, volatility, days)

    # Add some mean reversion
    prices = [start_price]
    for r in returns[1:]:
        # Mean reversion factor
        deviation = (prices[-1] - start_price) / start_price
        mean_reversion = -deviation * 0.02
        new_price = prices[-1] * (1 + r + mean_reversion)
        prices.append(max(new_price, 1))  # Floor at $1

    prices = np.array(prices)

    # Generate OHLC from close
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        'high': prices * (1 + np.random.uniform(0.001, 0.015, days)),
        'low': prices * (1 - np.random.uniform(0.001, 0.015, days)),
        'close': prices,
        'volume': np.random.uniform(1000000, 5000000, days),
    }, index=dates)

    # Ensure high >= open, close and low <= open, close
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


def create_strategy():
    """Create a strategy for backtesting."""
    strategy = AdvancedHybridStrategy(
        name="backtest_strategy",
        min_confidence=0.5,
        use_adaptive_weights=False,  # Disable for consistent backtest
        use_regime_detection=True,
        use_correlation_adjustment=False,
        use_time_filter=False,  # Disable time filter for backtest
        use_signal_scoring=True,
    )

    strategy.add_strategy(TechnicalStrategy(weight=0.4))
    strategy.add_strategy(MomentumStrategy(weight=0.35))
    strategy.add_strategy(MeanReversionStrategy(weight=0.25))

    return strategy


def run_simple_backtest():
    """Run a simple backtest example."""
    print("=" * 60)
    print("SIMPLE BACKTEST EXAMPLE")
    print("=" * 60)

    # Generate sample data for multiple symbols
    print("\nGenerating sample data...")
    data = {
        "AAPL": generate_sample_data("AAPL", days=500, start_price=150, trend=0.0003),
        "MSFT": generate_sample_data("MSFT", days=500, start_price=300, trend=0.0002),
        "GOOGL": generate_sample_data("GOOGL", days=500, start_price=140, trend=0.0001),
        "SPY": generate_sample_data("SPY", days=500, start_price=450, trend=0.0002),  # Benchmark
    }

    # Create strategy
    strategy = create_strategy()

    # Strategy function for backtest
    def strategy_fn(symbol: str, hist_data: pd.DataFrame, date: datetime):
        if len(hist_data) < 60:
            return None

        # Add indicators
        df = DataProcessor.add_technical_indicators(hist_data.copy())

        # Generate signal
        return strategy.generate_signal(symbol, df)

    # Create and run backtest
    print("\nRunning backtest...")
    engine = BacktestEngine(BacktestConfig(
        initial_capital=100000,
        commission_per_trade=0,
        slippage_pct=0.001,
        position_size_pct=0.02,
        max_positions=5,
        use_stops=True,
    ))

    results = engine.run(data, strategy_fn, benchmark_symbol="SPY")

    # Print results
    print_backtest_report(results)

    # Additional analysis
    print("\n--- Trade Analysis ---")

    if results.trades:
        # Win/loss by symbol
        trades_by_symbol = {}
        for trade in results.trades:
            if trade.symbol not in trades_by_symbol:
                trades_by_symbol[trade.symbol] = {"wins": 0, "losses": 0, "pnl": 0}

            if trade.is_winner:
                trades_by_symbol[trade.symbol]["wins"] += 1
            else:
                trades_by_symbol[trade.symbol]["losses"] += 1
            trades_by_symbol[trade.symbol]["pnl"] += trade.pnl

        print("\nPerformance by Symbol:")
        for symbol, stats in trades_by_symbol.items():
            total = stats["wins"] + stats["losses"]
            win_rate = stats["wins"] / total if total > 0 else 0
            print(f"  {symbol}: {stats['wins']}W/{stats['losses']}L "
                  f"({win_rate:.1%}) PnL: ${stats['pnl']:,.2f}")

        # Exit reason analysis
        exit_reasons = {}
        for trade in results.trades:
            reason = trade.exit_reason
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print("\nExit Reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason}: {count}")

    return results


def run_walk_forward_optimization():
    """Run walk-forward optimization example."""
    print("\n" + "=" * 60)
    print("WALK-FORWARD OPTIMIZATION EXAMPLE")
    print("=" * 60)

    # Generate longer dataset for walk-forward
    print("\nGenerating extended sample data...")
    data = {
        "SPY": generate_sample_data("SPY", days=1000, start_price=400, trend=0.0002),
        "QQQ": generate_sample_data("QQQ", days=1000, start_price=350, trend=0.0003),
    }

    # Define strategy factory with parameters
    def strategy_factory(
        technical_weight: float = 0.4,
        momentum_weight: float = 0.3,
        min_confidence: float = 0.5,
    ):
        """Create strategy with given parameters."""
        strategy = AdvancedHybridStrategy(
            min_confidence=min_confidence,
            use_adaptive_weights=False,
            use_time_filter=False,
        )

        remaining = 1 - technical_weight - momentum_weight
        strategy.add_strategy(TechnicalStrategy(weight=technical_weight))
        strategy.add_strategy(MomentumStrategy(weight=momentum_weight))
        strategy.add_strategy(MeanReversionStrategy(weight=remaining))

        def strategy_fn(symbol: str, hist_data: pd.DataFrame, date: datetime):
            if len(hist_data) < 60:
                return None
            df = DataProcessor.add_technical_indicators(hist_data.copy())
            return strategy.generate_signal(symbol, df)

        return strategy_fn

    # Parameter grid to search
    param_grid = {
        "technical_weight": [0.3, 0.4, 0.5],
        "momentum_weight": [0.2, 0.3, 0.4],
        "min_confidence": [0.4, 0.5, 0.6],
    }

    # Run walk-forward optimization
    engine = BacktestEngine(BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.02,
    ))

    optimizer = WalkForwardOptimizer(
        engine=engine,
        train_period_days=252,  # 1 year training
        test_period_days=63,    # 3 months testing
        min_train_trades=20,
    )

    print("\nRunning walk-forward optimization...")
    print("(This may take a while...)")

    best_params, oos_results = optimizer.optimize(
        data=data,
        strategy_factory=strategy_factory,
        param_grid=param_grid,
        metric="sharpe_ratio",
    )

    print(f"\nBest Parameters: {best_params}")

    # Aggregate out-of-sample results
    if oos_results:
        total_return = 1
        sharpes = []
        for result in oos_results:
            total_return *= (1 + result.total_return)
            sharpes.append(result.sharpe_ratio)

        print(f"\nOut-of-Sample Performance:")
        print(f"  Cumulative Return: {(total_return - 1):.2%}")
        print(f"  Avg Sharpe Ratio: {np.mean(sharpes):.2f}")
        print(f"  Periods Tested: {len(oos_results)}")


def main():
    """Run all examples."""
    # Simple backtest
    results = run_simple_backtest()

    # Walk-forward (commented out by default as it takes longer)
    # run_walk_forward_optimization()

    print("\n" + "=" * 60)
    print("Backtest examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
