"""
Options Trading Example

Demonstrates how to use the options trading module:
- Getting option chains
- Finding suitable contracts
- Analyzing opportunities
- Risk management
- Executing trades
"""

import logging
import os
from datetime import date, timedelta
from dotenv import load_dotenv

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.options import (
    OptionsClient,
    OptionsScanner,
    OptionsStrategyManager,
    OptionsRiskManager,
    OptionType,
    OptionOrder,
    OrderAction,
    ScanCriteria,
    create_risk_limits_conservative,
    create_risk_limits_moderate,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def example_get_option_chain():
    """Example: Get option chain for a symbol."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Getting Option Chain")
    print("=" * 60)

    load_dotenv()

    client = OptionsClient(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper=True,
    )

    # Get option chain for AAPL
    symbol = "AAPL"
    print(f"\nFetching option chain for {symbol}...")

    chain = client.get_option_chain(
        underlying=symbol,
        expiration_date_gte=date.today() + timedelta(days=7),
        expiration_date_lte=date.today() + timedelta(days=45),
    )

    print(f"\nUnderlying: {chain.underlying}")
    print(f"Underlying Price: ${chain.underlying_price:.2f}")
    print(f"Available Expirations: {len(chain.expirations)}")
    print(f"Total Contracts: {len(chain.contracts)}")

    if chain.expirations:
        print(f"\nExpirations:")
        for exp in chain.expirations[:5]:
            print(f"  - {exp}")
        if len(chain.expirations) > 5:
            print(f"  ... and {len(chain.expirations) - 5} more")

    # Get ATM strike
    atm_strike = chain.get_atm_strike()
    print(f"\nATM Strike: ${atm_strike:.2f}")

    # Show some calls and puts near ATM
    if chain.expirations:
        first_exp = chain.expirations[0]
        calls = chain.get_calls(first_exp)
        puts = chain.get_puts(first_exp)

        # Find near ATM options
        near_atm_calls = [c for c in calls if abs(c.strike - atm_strike) <= atm_strike * 0.05]
        near_atm_puts = [p for p in puts if abs(p.strike - atm_strike) <= atm_strike * 0.05]

        print(f"\nNear-ATM Calls (exp {first_exp}):")
        for call in near_atm_calls[:5]:
            print(f"  ${call.strike:.0f} - Bid: ${call.bid:.2f}, Ask: ${call.ask:.2f}, "
                  f"OI: {call.open_interest}")

        print(f"\nNear-ATM Puts (exp {first_exp}):")
        for put in near_atm_puts[:5]:
            print(f"  ${put.strike:.0f} - Bid: ${put.bid:.2f}, Ask: ${put.ask:.2f}, "
                  f"OI: {put.open_interest}")


def example_find_contracts():
    """Example: Find suitable contracts based on criteria."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Finding Suitable Contracts")
    print("=" * 60)

    load_dotenv()

    client = OptionsClient(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper=True,
    )

    symbol = "SPY"
    print(f"\nFinding 30-delta calls for {symbol}...")

    contracts = client.find_contracts(
        underlying=symbol,
        option_type=OptionType.CALL,
        min_days=14,
        max_days=45,
        delta_target=0.30,
        min_open_interest=100,
        max_spread_pct=0.10,
    )

    print(f"\nFound {len(contracts)} suitable contracts:")
    for i, contract in enumerate(contracts[:5]):
        print(f"\n  {i+1}. {contract}")
        print(f"     Bid: ${contract.bid:.2f}, Ask: ${contract.ask:.2f}")
        print(f"     Spread: ${contract.spread:.2f} ({contract.spread_pct:.1%})")
        print(f"     OI: {contract.open_interest}, Volume: {contract.volume}")
        print(f"     DTE: {contract.days_to_expiration}")
        if contract.greeks:
            print(f"     Delta: {contract.greeks.delta:.3f}, Theta: ${contract.greeks.theta:.3f}")


def example_strategy_analysis():
    """Example: Analyze market for strategy signals."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Strategy Analysis")
    print("=" * 60)

    load_dotenv()

    client = OptionsClient(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper=True,
    )

    strategy_manager = OptionsStrategyManager(client)

    # Analyze AAPL with bullish trend and moderate volatility
    symbol = "AAPL"
    underlying_price = 180.0  # Example price
    volatility = 0.35  # 35% IV
    trend = "bullish"

    print(f"\nAnalyzing {symbol}:")
    print(f"  Price: ${underlying_price}")
    print(f"  IV: {volatility:.0%}")
    print(f"  Trend: {trend}")

    signals = strategy_manager.analyze_all(
        underlying=symbol,
        underlying_price=underlying_price,
        volatility=volatility,
        trend=trend,
    )

    print(f"\nStrategy Signals:")
    for signal in signals:
        print(f"\n  Strategy: {signal.strategy_name}")
        print(f"  Type: {signal.spread_type.value}")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Rationale: {signal.rationale}")
        if signal.expected_profit:
            print(f"  Expected Profit: ${signal.expected_profit:.2f}")
        if signal.max_loss:
            print(f"  Max Loss: ${signal.max_loss:.2f}")
        print(f"  Contracts: {len(signal.contracts)}")
        for contract in signal.contracts:
            print(f"    - {contract}")


def example_scan_opportunities():
    """Example: Scan for options opportunities."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Scanning for Opportunities")
    print("=" * 60)

    load_dotenv()

    client = OptionsClient(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        paper=True,
    )

    scanner = OptionsScanner(
        client=client,
        watchlist=["AAPL", "MSFT", "GOOGL", "NVDA", "SPY"],
    )

    criteria = ScanCriteria(
        min_underlying_price=50,
        max_underlying_price=500,
        min_open_interest=100,
        max_spread_pct=0.15,
        min_dte=14,
        max_dte=45,
    )

    print("\nScanning for opportunities...")
    opportunities = scanner.get_top_opportunities(
        count=5,
        criteria=criteria,
        market_trend="neutral",
    )

    print(f"\nTop {len(opportunities)} Opportunities:")
    for i, opp in enumerate(opportunities):
        print(f"\n  {i+1}. {opp.symbol} - {opp.opportunity_type}")
        print(f"     Price: ${opp.underlying_price:.2f}")
        print(f"     Score: {opp.score:.1f}")
        print(f"     Details: {opp.details.get('suggestion', '')[:80]}")


def example_risk_management():
    """Example: Options risk management."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Risk Management")
    print("=" * 60)

    # Create risk manager with conservative limits
    risk_limits = create_risk_limits_conservative()
    risk_manager = OptionsRiskManager(limits=risk_limits)

    print("\nRisk Limits (Conservative):")
    print(f"  Max Single Position: ${risk_limits.max_single_position_value:,.0f}")
    print(f"  Max Total Exposure: ${risk_limits.max_total_options_exposure:,.0f}")
    print(f"  Max Positions: {risk_limits.max_positions}")
    print(f"  Max Portfolio Delta: {risk_limits.max_portfolio_delta}")
    print(f"  Allow Naked Calls: {risk_limits.allow_naked_calls}")
    print(f"  Allow Naked Puts: {risk_limits.allow_naked_puts}")
    print(f"  Min DTE: {risk_limits.min_days_to_expiration}")
    print(f"  Max DTE: {risk_limits.max_days_to_expiration}")

    # Create a mock order to check
    from src.options.models import OptionContract

    mock_contract = OptionContract(
        symbol="AAPL240315C00180000",
        underlying="AAPL",
        option_type=OptionType.CALL,
        strike=180.0,
        expiration=date.today() + timedelta(days=30),
        bid=5.50,
        ask=5.70,
        open_interest=1000,
    )

    # Check a buy order
    order = OptionOrder(
        contract=mock_contract,
        action=OrderAction.BUY_TO_OPEN,
        quantity=10,
        limit_price=5.60,
    )

    print(f"\n\nChecking Order:")
    print(f"  Contract: {mock_contract}")
    print(f"  Action: {order.action.value}")
    print(f"  Quantity: {order.quantity}")
    print(f"  Limit Price: ${order.limit_price:.2f}")
    print(f"  Estimated Cost: ${order.estimated_cost:.2f}")

    account_value = 100000.0
    passes, violations = risk_manager.check_order(order, account_value)

    if passes:
        print("\n  ✓ Order PASSES risk checks")
    else:
        print("\n  ✗ Order FAILS risk checks:")
        for v in violations:
            print(f"    - {v}")

    # Check position size limit
    max_size = risk_manager.get_position_size_limit(mock_contract, account_value)
    print(f"\n  Max Position Size: ${max_size:,.2f}")


def example_submit_order():
    """Example: Submit an options order (paper trading)."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Submit Order (Paper Trading)")
    print("=" * 60)

    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        print("\n⚠ Alpaca API keys not configured. Skipping order example.")
        print("  Set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env to test.")
        return

    client = OptionsClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
    )

    # Get account info
    account = client.get_account_info()
    print(f"\nAccount Info:")
    print(f"  Buying Power: ${account.get('buying_power', 0):,.2f}")
    print(f"  Cash: ${account.get('cash', 0):,.2f}")
    print(f"  Options Buying Power: ${account.get('options_buying_power', 0):,.2f}")
    print(f"  Options Approval Level: {account.get('options_approved_level', 'N/A')}")

    # Find a contract to trade
    symbol = "SPY"
    contracts = client.find_contracts(
        underlying=symbol,
        option_type=OptionType.CALL,
        min_days=14,
        max_days=30,
        delta_target=0.30,
        min_open_interest=500,
        max_spread_pct=0.05,
    )

    if not contracts:
        print(f"\n⚠ No suitable contracts found for {symbol}")
        return

    contract = contracts[0]
    print(f"\nSelected Contract: {contract}")
    print(f"  Bid: ${contract.bid:.2f}, Ask: ${contract.ask:.2f}")
    print(f"  Mid: ${contract.mid_price:.2f}")

    # Create order (but don't submit in example)
    order = OptionOrder(
        contract=contract,
        action=OrderAction.BUY_TO_OPEN,
        quantity=1,
        order_type="limit",
        limit_price=contract.mid_price,
        time_in_force="day",
    )

    print(f"\nOrder to Submit:")
    print(f"  Symbol: {order.contract.symbol}")
    print(f"  Action: {order.action.value}")
    print(f"  Quantity: {order.quantity}")
    print(f"  Type: {order.order_type}")
    print(f"  Limit: ${order.limit_price:.2f}")
    print(f"  Estimated Cost: ${order.estimated_cost:.2f}")

    # Uncomment to actually submit the order:
    # order_id = client.submit_order(order)
    # if order_id:
    #     print(f"\n✓ Order submitted! ID: {order_id}")
    # else:
    #     print("\n✗ Order submission failed")

    print("\n  (Order not submitted - uncomment submit code to execute)")


def example_get_positions():
    """Example: Get current option positions."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Current Positions")
    print("=" * 60)

    load_dotenv()

    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", "")

    if not api_key or not secret_key:
        print("\n⚠ Alpaca API keys not configured. Skipping positions example.")
        return

    client = OptionsClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
    )

    positions = client.get_positions()

    if not positions:
        print("\nNo open option positions.")
        return

    print(f"\nOpen Option Positions: {len(positions)}")
    for pos in positions:
        print(f"\n  {pos.contract}")
        print(f"    Quantity: {pos.quantity}")
        print(f"    Avg Cost: ${pos.avg_cost:.2f}")
        print(f"    Market Value: ${pos.market_value:.2f}")
        print(f"    Unrealized P&L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:.1%})")
        print(f"    Days to Exp: {pos.contract.days_to_expiration}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("OPTIONS TRADING EXAMPLES")
    print("=" * 60)

    # Note: Some examples require valid Alpaca API keys
    print("\nNote: Examples that interact with Alpaca API require valid")
    print("credentials in .env file. Some examples use mock data.")

    try:
        # Example 1: Get option chain
        example_get_option_chain()

        # Example 2: Find contracts
        example_find_contracts()

        # Example 3: Strategy analysis
        example_strategy_analysis()

        # Example 4: Scan opportunities
        example_scan_opportunities()

        # Example 5: Risk management
        example_risk_management()

        # Example 6: Submit order
        example_submit_order()

        # Example 7: Get positions
        example_get_positions()

    except Exception as e:
        logger.error(f"Example failed: {e}")
        raise

    print("\n" + "=" * 60)
    print("EXAMPLES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
