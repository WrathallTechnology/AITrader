"""
Configuration management for AITrader.
Loads settings from environment variables and provides defaults.
"""

import os
from dataclasses import dataclass
from typing import Literal
from dotenv import load_dotenv

load_dotenv()


@dataclass
class AlpacaConfig:
    """Alpaca API configuration."""
    api_key: str
    secret_key: str
    base_url: str

    @property
    def is_paper(self) -> bool:
        return "paper" in self.base_url.lower()


@dataclass
class TradingConfig:
    """Trading parameters configuration."""
    mode: Literal["paper", "live"]
    default_position_size_pct: float
    max_positions: int
    risk_per_trade: float

    # Capital management
    # Set initial_capital to define your starting trading capital
    # The bot will use this + any profits (or minus any losses)
    # None = use full account value
    initial_capital: float | None = None

    # Asset classes to trade
    trade_stocks: bool = True
    trade_crypto: bool = True

    # Trading hours (for stocks)
    trade_extended_hours: bool = False

    def get_effective_capital(
        self,
        account_value: float,
        starting_account_value: float | None = None,
    ) -> float:
        """
        Get the capital amount the bot should use for calculations.

        If initial_capital is set, the bot uses that amount plus any gains/losses.
        For example: if initial_capital=$10k and you've made $2k profit,
        effective capital = $12k (not capped at $10k).

        Args:
            account_value: Current total account value
            starting_account_value: Account value when bot first started (for tracking P&L)
        """
        if self.initial_capital is None:
            return account_value

        # If we don't know starting value, just use initial_capital as floor
        if starting_account_value is None:
            return self.initial_capital

        # Calculate P&L since bot started
        pnl = account_value - starting_account_value

        # Effective capital = initial allocation + P&L from trading
        return max(0, self.initial_capital + pnl)


@dataclass
class StrategyConfig:
    """Strategy-specific configuration."""
    # Technical indicators
    rsi_period: int = 14
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    sma_short: int = 20
    sma_long: int = 50

    # ML model settings
    lookback_period: int = 60
    prediction_horizon: int = 5
    retrain_interval_days: int = 7

    # Hybrid strategy weights
    technical_weight: float = 0.4
    ml_weight: float = 0.4
    sentiment_weight: float = 0.2


@dataclass
class Config:
    """Main configuration container."""
    alpaca: AlpacaConfig
    trading: TradingConfig
    strategy: StrategyConfig
    log_level: str
    log_file: str


def load_config() -> Config:
    """Load configuration from environment variables."""

    alpaca = AlpacaConfig(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets"),
    )

    # Parse initial_capital - can be None, empty string, or a number
    initial_capital_str = os.getenv("INITIAL_CAPITAL", "")
    initial_capital = float(initial_capital_str) if initial_capital_str else None

    trading = TradingConfig(
        mode=os.getenv("TRADING_MODE", "paper"),
        default_position_size_pct=float(os.getenv("DEFAULT_POSITION_SIZE_PCT", "0.02")),
        max_positions=int(os.getenv("MAX_POSITIONS", "10")),
        risk_per_trade=float(os.getenv("RISK_PER_TRADE", "0.01")),
        initial_capital=initial_capital,
    )

    strategy = StrategyConfig()

    return Config(
        alpaca=alpaca,
        trading=trading,
        strategy=strategy,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_file=os.getenv("LOG_FILE", "logs/trading.log"),
    )


# Global config instance
config = load_config()
