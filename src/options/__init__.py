"""Options trading module."""

from .models import (
    OptionType,
    OptionContract,
    OptionChain,
    OptionGreeks,
    OptionPosition,
    OptionOrder,
    OptionSpread,
    SpreadType,
)
from .client import OptionsClient, OptionsDataProvider
from .models import OrderAction
from .yahoo_provider import YahooOptionsDataProvider
from .strategies import (
    OptionsStrategy,
    DirectionalStrategy,
    IncomeStrategy,
    VolatilityStrategy,
    OptionsStrategyManager,
    StrategySignal,
)
from .risk import (
    OptionsRiskManager,
    OptionsRiskLimits,
    PortfolioRiskMetrics,
    create_risk_limits_conservative,
    create_risk_limits_moderate,
    create_risk_limits_aggressive,
)
from .scanner import OptionsScanner, OptionOpportunity, ScanCriteria

# WSB Sentiment Strategy (optional - requires praw, google-generativeai)
try:
    from .wsb_strategy import WSBSentimentStrategy
except ImportError:
    WSBSentimentStrategy = None

__all__ = [
    # Models
    "OptionType",
    "OptionContract",
    "OptionChain",
    "OptionGreeks",
    "OptionPosition",
    "OptionOrder",
    "OptionSpread",
    "SpreadType",
    "OrderAction",
    # Client
    "OptionsClient",
    "OptionsDataProvider",
    # Data Providers
    "YahooOptionsDataProvider",
    # Strategies
    "OptionsStrategy",
    "DirectionalStrategy",
    "IncomeStrategy",
    "VolatilityStrategy",
    "OptionsStrategyManager",
    "StrategySignal",
    # Risk
    "OptionsRiskManager",
    "OptionsRiskLimits",
    "PortfolioRiskMetrics",
    "create_risk_limits_conservative",
    "create_risk_limits_moderate",
    "create_risk_limits_aggressive",
    # Scanner
    "OptionsScanner",
    "OptionOpportunity",
    "ScanCriteria",
    # WSB Sentiment (optional)
    "WSBSentimentStrategy",
]
