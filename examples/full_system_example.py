"""
Example: Using the full AITrader system with all components.

This shows how to:
1. Set up the advanced hybrid strategy with all sub-strategies
2. Use portfolio correlation management
3. Enable circuit breakers and drawdown protection
4. Run backtests before going live
5. Use smart execution for orders
"""

from datetime import datetime, timedelta

import pandas as pd

# Strategy components
from src.strategies import (
    AdvancedHybridStrategy,
    TechnicalStrategy,
    MLStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    SentimentStrategy,
    SignalType,
)
from src.strategies.volume_profile import VolumeStrategy
from src.strategies.multi_timeframe import MultiTimeframeStrategy, Timeframe

# Risk management
from src.risk import (
    PortfolioCorrelationManager,
    DrawdownProtection,
    CircuitBreaker,
)
from src.risk.drawdown_protection import PositionSizer, CircuitBreakerConfig

# Event calendar
from src.data.event_calendar import EconomicCalendar, EventFilter

# Backtesting
from src.backtest import BacktestEngine, BacktestConfig, print_backtest_report

# Execution
from src.execution import (
    SmartOrderRouter,
    OrderManager,
    TrailingStopManager,
    ExecutionStyle,
)

# Data
from src.data.processor import DataProcessor
from src.client import AlpacaClient
from src.data.fetcher import DataFetcher
from config import load_config


def create_full_strategy() -> AdvancedHybridStrategy:
    """
    Create the complete hybrid strategy with all sub-strategies.
    """
    strategy = AdvancedHybridStrategy(
        name="full_hybrid",
        min_confidence=0.5,
        consensus_required=0.4,
        use_adaptive_weights=True,
        use_regime_detection=True,
        use_correlation_adjustment=True,
        use_time_filter=True,
        use_signal_scoring=True,
    )

    # Add strategies with weights
    # Weights should sum to ~1.0, they'll be normalized

    # Core strategies
    strategy.add_strategy(TechnicalStrategy(name="technical", weight=0.20))
    strategy.add_strategy(MomentumStrategy(name="momentum", weight=0.15))
    strategy.add_strategy(MeanReversionStrategy(name="mean_reversion", weight=0.15))

    # ML strategy (needs training)
    strategy.add_strategy(MLStrategy(name="ml", weight=0.20, min_confidence=0.55))

    # Volume and MTF for confirmation
    strategy.add_strategy(VolumeStrategy(name="volume", weight=0.15))
    strategy.add_strategy(MultiTimeframeStrategy(
        name="mtf",
        weight=0.10,
        primary_tf=Timeframe.D1,
        secondary_tf=Timeframe.H4,
        entry_tf=Timeframe.H1,
    ))

    # Sentiment (lower weight, supplementary)
    strategy.add_strategy(SentimentStrategy(name="sentiment", weight=0.05))

    return strategy


def setup_risk_management(initial_capital: float):
    """
    Set up the complete risk management system.
    """
    # Drawdown protection
    drawdown_protection = DrawdownProtection(
        initial_value=initial_capital,
        config=CircuitBreakerConfig(
            max_daily_loss_pct=0.03,  # 3% daily limit
            max_daily_trades=15,
            max_consecutive_losses=4,
            caution_drawdown=0.05,  # 5% -> reduce size
            restricted_drawdown=0.10,  # 10% -> close only
            halt_drawdown=0.15,  # 15% -> stop trading
        ),
    )

    # Circuit breaker
    circuit_breaker = CircuitBreaker(drawdown_protection)

    # Position sizer
    position_sizer = PositionSizer(
        circuit_breaker=circuit_breaker,
        base_risk_pct=0.01,  # 1% risk per trade
        max_position_pct=0.05,  # 5% max position
    )

    # Portfolio correlation manager
    correlation_manager = PortfolioCorrelationManager(
        max_sector_weight=0.30,  # 30% max in any sector
        max_single_position=0.10,  # 10% max single position
        max_correlation=0.70,  # Don't add highly correlated positions
        max_portfolio_beta=1.5,
    )

    return {
        "drawdown": drawdown_protection,
        "circuit_breaker": circuit_breaker,
        "position_sizer": position_sizer,
        "correlation_manager": correlation_manager,
    }


def setup_execution():
    """
    Set up smart execution components.
    """
    order_router = SmartOrderRouter(
        default_style=ExecutionStyle.BALANCED,
        max_spread_pct=0.005,
    )

    order_manager = OrderManager()

    trailing_stop_manager = TrailingStopManager(
        initial_stop_pct=0.02,
        trail_pct=0.015,
        activation_profit_pct=0.01,
    )

    return {
        "router": order_router,
        "manager": order_manager,
        "trailing_stops": trailing_stop_manager,
    }


def run_backtest_example():
    """
    Example: Run a backtest before going live.
    """
    print("=" * 60)
    print("RUNNING BACKTEST")
    print("=" * 60)

    # Create strategy
    strategy = create_full_strategy()

    # Create backtest engine
    engine = BacktestEngine(BacktestConfig(
        initial_capital=100000,
        commission_per_trade=0,
        slippage_pct=0.001,
        position_size_pct=0.02,
        max_positions=10,
    ))

    # In real usage, you'd fetch historical data
    # For this example, we'll create a strategy function

    def strategy_fn(symbol: str, data: pd.DataFrame, date: datetime):
        """Strategy function for backtest."""
        if len(data) < 50:
            return None

        # Add indicators
        data = DataProcessor.add_technical_indicators(data)

        # Generate signal
        signal = strategy.generate_signal(symbol, data)
        return signal

    # You would load real data like this:
    # config = load_config()
    # client = AlpacaClient(config.alpaca)
    # fetcher = DataFetcher(client)
    # data = {"SPY": fetcher.get_historical_bars("SPY", days=500)}

    print("To run backtest, load historical data and call:")
    print("  results = engine.run(data, strategy_fn)")
    print("  print_backtest_report(results)")


def live_trading_example():
    """
    Example: Live trading with all components integrated.
    """
    print("\n" + "=" * 60)
    print("LIVE TRADING SETUP")
    print("=" * 60)

    # Load config
    config = load_config()

    # Initialize Alpaca client
    client = AlpacaClient(config.alpaca)
    fetcher = DataFetcher(client)

    # Get initial capital
    initial_capital = float(client.get_portfolio_value())
    print(f"Portfolio Value: ${initial_capital:,.2f}")

    # Setup components
    strategy = create_full_strategy()
    risk = setup_risk_management(initial_capital)
    execution = setup_execution()

    # Event calendar for avoiding high-impact events
    calendar = EconomicCalendar()
    event_filter = EventFilter(calendar)

    # Watchlist
    watchlist = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "SPY", "BTC/USD"]

    print(f"Watchlist: {watchlist}")
    print(f"Risk State: {risk['circuit_breaker'].state.value}")

    # Main trading loop (simplified)
    for symbol in watchlist:
        print(f"\n--- Analyzing {symbol} ---")

        # 1. Check circuit breaker
        can_trade, reason = risk["circuit_breaker"].can_open_position()
        if not can_trade:
            print(f"  Cannot trade: {reason}")
            continue

        # 2. Check event filter
        event_result = event_filter.filter(symbol)
        if event_result.should_avoid:
            print(f"  Avoiding due to events: {event_result.reason}")
            continue

        # 3. Check portfolio correlation
        current_positions = {}  # Would get from client.get_all_positions()
        can_add, violations = risk["correlation_manager"].can_add_position(
            symbol,
            proposed_value=initial_capital * 0.02,
            current_positions=current_positions,
        )
        if not can_add:
            print(f"  Portfolio limit: {violations}")
            continue

        # 4. Get data and generate signal
        try:
            data = fetcher.get_historical_bars(symbol, timeframe="1Day", days=100)
            data = DataProcessor.add_technical_indicators(data)

            signal = strategy.generate_signal(symbol, data)
            print(f"  Signal: {signal.signal_type.value} (confidence: {signal.confidence:.2%})")

            if signal.signal_type == SignalType.BUY:
                # 5. Calculate position size
                price = float(data["close"].iloc[-1])
                stop_loss = signal.stop_loss or price * 0.98

                quantity = risk["position_sizer"].calculate_position_size(
                    portfolio_value=initial_capital,
                    entry_price=price,
                    stop_loss=stop_loss,
                    signal_confidence=signal.confidence,
                )

                # Apply event confidence multiplier
                quantity *= event_result.confidence_multiplier

                print(f"  Position size: {quantity:.2f} shares")

                # 6. Create execution plan
                plan = execution["router"].create_execution_plan(
                    symbol=symbol,
                    side="buy",
                    quantity=quantity,
                    current_price=price,
                    urgency=0.5,
                )

                print(f"  Order type: {plan.order_type.value}")
                if plan.limit_price:
                    print(f"  Limit price: ${plan.limit_price:.2f}")

                # In live trading, you'd execute here:
                # order = client.submit_order(...)

        except Exception as e:
            print(f"  Error: {e}")

    # Print diagnostics
    print("\n" + "=" * 60)
    print("STRATEGY DIAGNOSTICS")
    print("=" * 60)

    diagnostics = strategy.get_diagnostics()
    print(f"Market Regime: {diagnostics['current_regime']}")
    print(f"Trend Strength: {diagnostics['trend_strength']}")
    print(f"Diversification Score: {diagnostics['diversification_score']:.2f}")
    print(f"Effective Weights: {diagnostics['effective_weights']}")


def main():
    """Run examples."""
    print("AITrader - Full System Example")
    print("=" * 60)

    # Run backtest example
    run_backtest_example()

    # Uncomment to run live trading example (requires valid API keys)
    # live_trading_example()

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
