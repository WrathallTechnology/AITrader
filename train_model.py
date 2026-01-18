"""
Train the ML model for price prediction.

Usage:
    python train_model.py                    # Train on BTC/USD (default)
    python train_model.py --symbol ETH/USD   # Train on specific symbol
    python train_model.py --days 90          # Use 90 days of data
    python train_model.py --all              # Train on all watchlist symbols
"""

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from config import config
from src.client import AlpacaClient
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.models.predictor import PricePredictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("TrainModel")


def train_model(
    symbol: str = "BTC/USD",
    days: int = 60,
    save_path: Path = None,
) -> dict:
    """
    Train the ML model on historical data.

    Args:
        symbol: Trading symbol to train on
        days: Number of days of historical data
        save_path: Path to save the trained model

    Returns:
        Training metrics dictionary
    """
    logger.info("=" * 60)
    logger.info(f"Training ML Model")
    logger.info("=" * 60)
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Days of data: {days}")

    # Initialize client
    client = AlpacaClient(config.alpaca)
    fetcher = DataFetcher(client)

    # Fetch historical data
    logger.info(f"Fetching {days} days of hourly data...")
    df = fetcher.get_historical_bars(
        symbol=symbol,
        timeframe="1Hour",
        days=days,
    )

    if df is None or len(df) < 100:
        logger.error(f"Insufficient data for training. Got {len(df) if df is not None else 0} bars.")
        return None

    logger.info(f"Fetched {len(df)} bars")

    # Add technical indicators
    logger.info("Adding technical indicators...")
    df = DataProcessor.add_technical_indicators(df)

    # Initialize predictor
    predictor = PricePredictor(
        lookback=config.strategy.lookback_period,
        prediction_horizon=config.strategy.prediction_horizon,
        model_path=save_path or Path("models/price_predictor.pkl"),
    )

    # Train
    logger.info("Training model...")
    metrics = predictor.train(df, verbose=True)

    # Save
    predictor.save()

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Train Accuracy: {metrics['train_accuracy']:.2%}")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
    logger.info(f"Model saved to: {predictor.model_path}")

    return metrics


def train_on_multiple(symbols: list[str], days: int = 60):
    """Train on multiple symbols and combine data."""
    logger.info("=" * 60)
    logger.info("Training on Multiple Symbols")
    logger.info("=" * 60)

    client = AlpacaClient(config.alpaca)
    fetcher = DataFetcher(client)

    all_data = []

    for symbol in symbols:
        logger.info(f"Fetching data for {symbol}...")
        try:
            df = fetcher.get_historical_bars(
                symbol=symbol,
                timeframe="1Hour",
                days=days,
            )
            if df is not None and len(df) > 100:
                df = DataProcessor.add_technical_indicators(df)
                df["symbol"] = symbol
                all_data.append(df)
                logger.info(f"  Got {len(df)} bars")
        except Exception as e:
            logger.warning(f"  Failed to fetch {symbol}: {e}")

    if not all_data:
        logger.error("No data fetched!")
        return None

    # Combine all data
    import pandas as pd
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} bars")

    # Train
    predictor = PricePredictor(
        lookback=config.strategy.lookback_period,
        prediction_horizon=config.strategy.prediction_horizon,
    )

    metrics = predictor.train(combined, verbose=True)
    predictor.save()

    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train the ML prediction model")
    parser.add_argument(
        "--symbol",
        default="BTC/USD",
        help="Symbol to train on (default: BTC/USD)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="Days of historical data (default: 60)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train on all watchlist symbols",
    )
    args = parser.parse_args()

    # Watchlist for --all option
    watchlist = [
        "BTC/USD",
        "ETH/USD",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "NVDA",
        "TSLA",
    ]

    try:
        if args.all:
            metrics = train_on_multiple(watchlist, days=args.days)
        else:
            metrics = train_model(symbol=args.symbol, days=args.days)

        if metrics:
            logger.info("\nModel is ready! You can now run the trading bot.")
            logger.info("Run: python main.py --mode crypto --dry-run")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
