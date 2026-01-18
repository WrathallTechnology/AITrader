# AITrader - AI-Powered Trading Bot

An AI-powered trading bot for Alpaca Markets that uses a hybrid approach combining technical analysis and machine learning predictions.

## Features

- **Multi-Asset Support**: Trade both stocks and cryptocurrencies
- **Hybrid Strategy**: Combines multiple strategies with weighted voting
  - Technical indicators (RSI, MACD, Moving Averages, Bollinger Bands)
  - Machine learning predictions (Gradient Boosting classifier)
- **Risk Management**: Position sizing based on portfolio risk
- **Paper Trading**: Test strategies without risking real money
- **Modular Design**: Easy to add new strategies

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

1. Sign up at [Alpaca](https://alpaca.markets/)
2. Get your API keys from the dashboard
3. Copy `.env.example` to `.env` and add your keys:

```bash
cp .env.example .env
```

Edit `.env`:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

### 3. Run the Bot

```python
from src.bot import TradingBot

# Create bot instance
bot = TradingBot()

# Optional: Train ML strategies (do this once)
bot.train_strategies(days=365)

# Run single iteration
signals = bot.run_once()

# Or run continuously
bot.run(interval_minutes=60)
```

## Project Structure

```
AITrader/
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
├── .env.example          # Environment template
├── src/
│   ├── bot.py            # Main trading bot orchestrator
│   ├── client.py         # Alpaca API client wrapper
│   ├── portfolio.py      # Portfolio & risk management
│   ├── data/
│   │   ├── fetcher.py    # Market data fetching
│   │   └── processor.py  # Data processing & indicators
│   ├── strategies/
│   │   ├── base.py       # Base strategy class
│   │   ├── technical.py  # Technical analysis strategy
│   │   ├── ml_strategy.py # ML prediction strategy
│   │   └── hybrid.py     # Hybrid strategy combiner
│   ├── models/
│   │   └── predictor.py  # ML price predictor
│   └── utils/
│       ├── logger.py     # Logging setup
│       └── helpers.py    # Utility functions
└── tests/                # Test files
```

## Configuration

Edit `config.py` or set environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `TRADING_MODE` | `paper` or `live` | `paper` |
| `DEFAULT_POSITION_SIZE_PCT` | Position size as % of portfolio | `0.02` (2%) |
| `MAX_POSITIONS` | Maximum concurrent positions | `10` |
| `RISK_PER_TRADE` | Risk per trade as % of portfolio | `0.01` (1%) |

## Strategies

### Technical Strategy
Uses classic indicators:
- **RSI**: Oversold (<30) = Buy, Overbought (>70) = Sell
- **MACD**: Crossover signals
- **Moving Averages**: 20/50 SMA crossovers
- **Bollinger Bands**: Price at bands suggests reversal

### ML Strategy
- Gradient Boosting classifier trained on historical data
- Predicts price direction over configurable horizon
- Uses normalized technical indicator features

### Hybrid Strategy
- Combines strategies with weighted voting
- Requires minimum confidence threshold (default 60%)
- Requires consensus among strategies (default 50%)

## Usage Examples

### Run Single Trading Cycle
```python
from src.bot import TradingBot

bot = TradingBot()
signals = bot.run_once()

for symbol, signal in signals.items():
    if signal.is_actionable:
        print(f"{symbol}: {signal}")
```

### Custom Watchlist
```python
bot = TradingBot(watchlist=["AAPL", "MSFT", "BTC/USD"])
```

### Adjust Strategy Weights
```python
bot.strategy.set_strategy_weight("technical", 0.6)
bot.strategy.set_strategy_weight("ml", 0.4)
```

### Manual Signal Generation
```python
from src.client import AlpacaClient
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.strategies.technical import TechnicalStrategy
from config import load_config

config = load_config()
client = AlpacaClient(config.alpaca)
fetcher = DataFetcher(client)

# Get data
data = fetcher.get_historical_bars("AAPL", timeframe="1Day", days=100)
data = DataProcessor.add_technical_indicators(data)

# Generate signal
strategy = TechnicalStrategy()
signal = strategy.generate_signal("AAPL", data)
print(signal)
```

## Switching to Live Trading

1. Get live API keys from Alpaca dashboard
2. Update `.env`:
   ```
   ALPACA_API_KEY=your_live_api_key
   ALPACA_SECRET_KEY=your_live_secret_key
   ALPACA_BASE_URL=https://api.alpaca.markets
   TRADING_MODE=live
   ```

**Important**:
- Start with small position sizes
- Monitor the bot closely
- Review logs regularly
- Consider regulatory requirements (Pattern Day Trader rules, etc.)

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Always do your own research and consider consulting a financial advisor.

## License

MIT License
