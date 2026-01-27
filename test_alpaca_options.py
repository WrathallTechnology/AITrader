#!/usr/bin/env python3
"""
Test script to check if Alpaca supports options for specific symbols.
Run with: python test_alpaca_options.py
"""
import os
import sys
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest

def test_alpaca_options(symbols=None):
    """Test if Alpaca has options contracts for given symbols."""
    if symbols is None:
        symbols = ["MU", "RKLB", "SPY", "AAPL"]

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    paper = os.getenv("ALPACA_PAPER", "true").lower() == "true"

    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        sys.exit(1)

    print(f"Using {'PAPER' if paper else 'LIVE'} trading")
    print(f"Testing symbols: {symbols}")
    print("="*60)

    client = TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=paper,
    )

    # Check account options approval level
    try:
        account = client.get_account()
        options_level = getattr(account, 'options_approved_level', 'N/A')
        print(f"Account options approval level: {options_level}")
        print()
    except Exception as e:
        print(f"Could not get account info: {e}")
        print()

    for symbol in symbols:
        print(f"\n{symbol}:")
        print("-" * 40)

        try:
            request = GetOptionContractsRequest(
                underlying_symbols=[symbol],
                expiration_date_gte=date.today() + timedelta(days=7),
                expiration_date_lte=date.today() + timedelta(days=45),
                limit=100,
            )

            response = client.get_option_contracts(request)
            contracts = response.option_contracts or []

            if contracts:
                print(f"  Found {len(contracts)} option contracts")

                # Count calls and puts
                calls = [c for c in contracts if c.type.lower() == 'call']
                puts = [c for c in contracts if c.type.lower() == 'put']
                print(f"  Calls: {len(calls)}, Puts: {len(puts)}")

                # Show sample contracts
                if contracts:
                    c = contracts[0]
                    print(f"  Sample: {c.symbol}")
                    print(f"    Strike: ${c.strike_price}")
                    print(f"    Expiration: {c.expiration_date}")
                    print(f"    Type: {c.type}")
                    print(f"    Open Interest: {c.open_interest}")
            else:
                print(f"  NO OPTIONS CONTRACTS FOUND")
                print(f"  Alpaca may not support options for {symbol}")
                print(f"  Or no contracts match the date range (7-45 DTE)")

        except Exception as e:
            print(f"  ERROR: {e}")
            if "not found" in str(e).lower() or "404" in str(e):
                print(f"  {symbol} likely not supported for options trading on Alpaca")

    print("\n" + "="*60)
    print("Done!")

if __name__ == "__main__":
    symbols = sys.argv[1:] if len(sys.argv) > 1 else None
    test_alpaca_options(symbols)
