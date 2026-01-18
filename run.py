#!/usr/bin/env python3
"""
Simple entry point script for running the trading bot.
"""

import argparse
import sys

from src.bot import TradingBot
from src.utils.logger import setup_logger


def main():
    parser = argparse.ArgumentParser(description="AITrader - AI-Powered Trading Bot")

    parser.add_argument(
        "--mode",
        choices=["run", "train", "once", "status"],
        default="once",
        help="Execution mode (default: once)",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Minutes between trading cycles (default: 60)",
    )

    parser.add_argument(
        "--watchlist",
        nargs="+",
        help="Symbols to trade (default: built-in list)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Setup logging
    logger = setup_logger("aitrader", level=args.log_level, log_file="logs/trading.log")

    # Create bot
    bot = TradingBot(watchlist=args.watchlist)

    if args.mode == "status":
        # Just print status
        status = bot.get_status()
        print("\n=== AITrader Status ===")
        print(f"Trading Mode: {status['trading_mode']}")
        print(f"Portfolio Value: ${status['portfolio_value']:,.2f}")
        print(f"Open Positions: {status['position_count']}")
        print(f"Watchlist: {', '.join(status['watchlist'])}")
        print(f"Strategy Weights: {status['strategy_weights']}")

    elif args.mode == "train":
        # Train strategies
        print("Training ML strategies...")
        bot.train_strategies(days=365)
        print("Training complete!")

    elif args.mode == "once":
        # Run single cycle
        print("Running single trading cycle...")
        signals = bot.run_once()

        print("\n=== Signals ===")
        for symbol, signal in signals.items():
            if signal.is_actionable:
                print(f"  {signal}")

    elif args.mode == "run":
        # Continuous run
        print(f"Starting continuous trading (interval: {args.interval} min)")
        print("Press Ctrl+C to stop")
        bot.run(interval_minutes=args.interval)


if __name__ == "__main__":
    main()
