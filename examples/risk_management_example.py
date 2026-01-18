"""
Example: Using the risk management system.

Demonstrates:
1. Drawdown protection and circuit breakers
2. Portfolio correlation management
3. Position sizing with risk controls
4. Event-based filtering
"""

from datetime import datetime, timedelta

from src.risk import (
    DrawdownProtection,
    CircuitBreaker,
    PortfolioCorrelationManager,
)
from src.risk.drawdown_protection import (
    CircuitBreakerConfig,
    PositionSizer,
    RiskState,
)
from src.data.event_calendar import EconomicCalendar, EventFilter


def demonstrate_circuit_breakers():
    """Show how circuit breakers work."""
    print("=" * 60)
    print("CIRCUIT BREAKER DEMONSTRATION")
    print("=" * 60)

    # Initialize with $100k
    initial_capital = 100000

    config = CircuitBreakerConfig(
        max_daily_loss_pct=0.03,      # 3% daily limit
        max_consecutive_losses=3,      # 3 losses in a row
        caution_drawdown=0.05,         # 5% drawdown -> caution
        restricted_drawdown=0.10,      # 10% drawdown -> restricted
        halt_drawdown=0.15,            # 15% drawdown -> halt
    )

    protection = DrawdownProtection(initial_capital, config)
    circuit_breaker = CircuitBreaker(protection, config)

    print(f"\nInitial capital: ${initial_capital:,}")
    print(f"State: {circuit_breaker.state.value}")

    # Simulate some losses
    scenarios = [
        (98000, "Small loss"),
        (96000, "Another loss"),
        (94000, "Approaching caution"),
        (92000, "In caution territory"),
        (88000, "Approaching restricted"),
        (85000, "In restricted territory"),
        (84000, "Close to halt"),
    ]

    for new_value, description in scenarios:
        protection.update_value(new_value)
        state = circuit_breaker.check_state()

        print(f"\n{description}")
        print(f"  Portfolio: ${new_value:,}")
        print(f"  Drawdown: {protection.get_current_drawdown():.1%}")
        print(f"  State: {state.value}")
        print(f"  Position multiplier: {circuit_breaker.get_position_size_multiplier():.0%}")

        can_open, reason = circuit_breaker.can_open_position()
        print(f"  Can open new position: {can_open} ({reason})")

    # Show recovery
    print("\n--- Recovery Scenario ---")
    protection.update_value(90000)
    state = circuit_breaker.check_state()
    print(f"After recovery to $90,000:")
    print(f"  State: {state.value}")
    print(f"  Position multiplier: {circuit_breaker.get_position_size_multiplier():.0%}")


def demonstrate_position_sizing():
    """Show intelligent position sizing."""
    print("\n" + "=" * 60)
    print("POSITION SIZING DEMONSTRATION")
    print("=" * 60)

    initial_capital = 100000
    protection = DrawdownProtection(initial_capital)
    circuit_breaker = CircuitBreaker(protection)

    sizer = PositionSizer(
        circuit_breaker=circuit_breaker,
        base_risk_pct=0.01,  # 1% risk per trade
        max_position_pct=0.05,  # 5% max position
    )

    # Scenario 1: Normal conditions
    print("\nScenario 1: Normal Market Conditions")
    entry_price = 150
    stop_loss = 140  # Wider stop to avoid hitting position cap

    quantity = sizer.calculate_position_size(
        portfolio_value=initial_capital,
        entry_price=entry_price,
        stop_loss=stop_loss,
        signal_confidence=0.8,
    )

    position_value = quantity * entry_price
    risk_amount = quantity * (entry_price - stop_loss)

    print(f"  Entry: ${entry_price}, Stop: ${stop_loss}")
    print(f"  Signal confidence: 80%")
    print(f"  Shares: {quantity:.1f}")
    print(f"  Position value: ${position_value:,.2f} ({position_value/initial_capital:.1%} of portfolio)")
    print(f"  Risk amount: ${risk_amount:,.2f} ({risk_amount/initial_capital:.1%} of portfolio)")

    # Scenario 2: After drawdown (caution state)
    # Use fresh instances starting at peak, then simulate gradual drawdown
    print("\nScenario 2: After 6% Drawdown (Caution State)")
    protection2 = DrawdownProtection(initial_capital)
    # Simulate reaching 94k over multiple days (not single session)
    # by setting peak and current value directly
    protection2.peak_value = initial_capital
    protection2.current_value = 94000
    protection2._current_session.starting_value = 94000  # Session starts at current value
    protection2._current_session.current_value = 94000

    circuit_breaker2 = CircuitBreaker(protection2)
    circuit_breaker2.check_state()

    sizer2 = PositionSizer(
        circuit_breaker=circuit_breaker2,
        base_risk_pct=0.01,
        max_position_pct=0.05,
    )

    quantity = sizer2.calculate_position_size(
        portfolio_value=94000,
        entry_price=entry_price,
        stop_loss=stop_loss,
        signal_confidence=0.8,
    )

    print(f"  Current state: {circuit_breaker2.state.value}")
    print(f"  Size multiplier: {circuit_breaker2.get_position_size_multiplier():.0%}")
    print(f"  Shares: {quantity:.1f} (reduced due to drawdown)")

    # Scenario 3: Low confidence signal (use fresh instances to avoid state carryover)
    print("\nScenario 3: Low Confidence Signal")
    protection3 = DrawdownProtection(initial_capital)
    circuit_breaker3 = CircuitBreaker(protection3)
    sizer3 = PositionSizer(
        circuit_breaker=circuit_breaker3,
        base_risk_pct=0.02,  # 2% risk per trade
        max_position_pct=0.25,  # High cap so risk-based sizing dominates
    )

    # Use wider stop so position size is determined by risk, not cap
    wide_stop = 135  # $15 risk per share

    # Compare high vs low confidence
    # With 2% risk ($2000) and $15/share risk: 2000/15 = 133 shares at 100% confidence
    # At 50% confidence: 1000/15 = 66.7 shares
    high_conf_qty = sizer3.calculate_position_size(
        portfolio_value=100000,
        entry_price=entry_price,
        stop_loss=wide_stop,
        signal_confidence=1.0,  # 100% confident
    )

    low_conf_qty = sizer3.calculate_position_size(
        portfolio_value=100000,
        entry_price=entry_price,
        stop_loss=wide_stop,
        signal_confidence=0.5,  # Only 50% confident
    )

    print(f"  Entry: ${entry_price}, Stop: ${wide_stop}")
    print(f"  100% confidence: {high_conf_qty:.1f} shares (${high_conf_qty * entry_price:,.0f})")
    print(f"  50% confidence: {low_conf_qty:.1f} shares (${low_conf_qty * entry_price:,.0f})")
    if high_conf_qty > 0:
        print(f"  Confidence scaling: {low_conf_qty/high_conf_qty:.0%} of high-confidence size")


def demonstrate_portfolio_correlation():
    """Show portfolio correlation management."""
    print("\n" + "=" * 60)
    print("PORTFOLIO CORRELATION MANAGEMENT")
    print("=" * 60)

    manager = PortfolioCorrelationManager(
        max_sector_weight=0.30,
        max_single_position=0.10,
        max_correlation=0.70,
    )

    # Current portfolio
    current_positions = {
        "AAPL": 15000,  # Technology
        "MSFT": 12000,  # Technology
        "JPM": 10000,   # Financials
        "JNJ": 8000,    # Healthcare
    }

    portfolio_value = sum(current_positions.values())

    print(f"\nCurrent Portfolio: ${portfolio_value:,}")
    for symbol, value in current_positions.items():
        print(f"  {symbol}: ${value:,} ({value/portfolio_value:.1%})")

    # Analyze current exposure
    exposure = manager.analyze_portfolio(current_positions)

    print(f"\nSector Weights:")
    for sector, weight in exposure.sector_weights.items():
        print(f"  {sector}: {weight:.1%}")

    print(f"\nConcentration Score: {exposure.concentration_score:.3f}")
    print(f"Diversification Score: {manager.get_diversification_score(current_positions):.1f}/100")

    # Try adding more tech
    print("\n--- Attempting to Add NVDA (Technology) ---")
    can_add, violations = manager.can_add_position(
        symbol="NVDA",
        proposed_value=10000,
        current_positions=current_positions,
    )

    if can_add:
        print("  ✓ Can add position")
    else:
        print("  ✗ Cannot add position:")
        for v in violations:
            print(f"    - {v}")

    # Try adding diversifying position
    print("\n--- Attempting to Add XOM (Energy) ---")
    can_add, violations = manager.can_add_position(
        symbol="XOM",
        proposed_value=10000,
        current_positions=current_positions,
    )

    if can_add:
        print("  ✓ Can add position (diversifying)")
    else:
        print("  ✗ Cannot add position:")
        for v in violations:
            print(f"    - {v}")

    # Get max position size
    max_size, reason = manager.get_position_size_limit(
        symbol="GOOGL",
        current_positions=current_positions,
        portfolio_value=portfolio_value,
    )
    print(f"\n  Max position size for GOOGL: ${max_size:,.2f}")
    print(f"    Reason: {reason}")

    # Try a non-tech stock
    max_size_energy, reason_energy = manager.get_position_size_limit(
        symbol="XOM",
        current_positions=current_positions,
        portfolio_value=portfolio_value,
    )
    print(f"\n  Max position size for XOM (Energy): ${max_size_energy:,.2f}")
    print(f"    Reason: {reason_energy}")


def demonstrate_event_filtering():
    """Show event-based trade filtering."""
    print("\n" + "=" * 60)
    print("EVENT CALENDAR FILTERING")
    print("=" * 60)

    calendar = EconomicCalendar()
    event_filter = EventFilter(calendar)

    # Add some earnings
    calendar.add_earnings("AAPL", datetime.now() + timedelta(hours=20), "after_close")
    calendar.add_earnings("MSFT", datetime.now() + timedelta(days=5), "before_open")

    print("\nUpcoming Events:")
    events = calendar.get_upcoming_events(hours_ahead=168)  # 1 week
    for event in events[:10]:
        print(f"  {event.datetime.strftime('%Y-%m-%d %H:%M')} - {event.name} ({event.impact.value})")

    # Check trading conditions for different symbols
    symbols_to_check = ["AAPL", "MSFT", "GOOGL", "SPY"]

    print("\nTrading Conditions:")
    for symbol in symbols_to_check:
        result = event_filter.filter(symbol)

        status = "AVOID" if result.should_avoid else "OK"
        print(f"\n  {symbol}: {status}")
        print(f"    Confidence multiplier: {result.confidence_multiplier:.0%}")
        if result.reason != "No significant events":
            print(f"    Reason: {result.reason}")

    # Check for high impact period
    is_high_impact, event = calendar.is_high_impact_period(hours_window=4)
    print(f"\nHigh impact period (next 4h): {is_high_impact}")
    if event:
        print(f"  Event: {event.name}")


def main():
    """Run all risk management examples."""
    demonstrate_circuit_breakers()
    demonstrate_position_sizing()
    demonstrate_portfolio_correlation()
    demonstrate_event_filtering()

    print("\n" + "=" * 60)
    print("Risk management examples complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
