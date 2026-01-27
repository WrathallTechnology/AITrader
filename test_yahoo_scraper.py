#!/usr/bin/env python3
"""
Test script for Yahoo Finance options scraper.
Run with: python test_yahoo_scraper.py
"""
import logging
import sys

# Setup logging to see debug output
logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(name)s - %(message)s')

from src.options.yahoo_provider import YahooOptionsDataProvider

def test_scraper(symbols=None):
    """Test the Yahoo scraper for given symbols."""
    if symbols is None:
        symbols = ["MU", "RKLB", "SPY"]

    provider = YahooOptionsDataProvider(cache_ttl_minutes=1)

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Testing {symbol}")
        print('='*60)

        # Test underlying price
        price = provider.get_underlying_price(symbol)
        print(f"Underlying price: ${price:.2f}")

        # Test option chain
        chain = provider.get_option_chain(symbol)
        print(f"Expirations: {len(chain.expirations)}")
        print(f"Contracts: {len(chain.contracts)}")

        if chain.contracts:
            # Show sample contracts
            calls = [c for c in chain.contracts if c.option_type.value == 'call']
            puts = [c for c in chain.contracts if c.option_type.value == 'put']
            print(f"  Calls: {len(calls)}, Puts: {len(puts)}")

            if calls:
                c = calls[0]
                print(f"  Sample call: {c.symbol} strike=${c.strike} exp={c.expiration} bid=${c.bid} ask=${c.ask}")
            if puts:
                p = puts[0]
                print(f"  Sample put: {p.symbol} strike=${p.strike} exp={p.expiration} bid=${p.bid} ask=${p.ask}")
        else:
            print("  No contracts found!")

            # Debug: Save HTML for inspection
            print(f"\n  Saving HTML to debug_{symbol}.html for inspection...")
            try:
                url = f"https://finance.yahoo.com/quote/{symbol}/options"
                html = provider._fetch_with_retry(url)
                with open(f"debug_{symbol}.html", "w", encoding="utf-8") as f:
                    f.write(html)
                print(f"  Saved! Check debug_{symbol}.html")
            except Exception as e:
                print(f"  Failed to save HTML: {e}")

if __name__ == "__main__":
    # Get symbols from command line or use defaults
    symbols = sys.argv[1:] if len(sys.argv) > 1 else None
    test_scraper(symbols)
