"""
Portfolio-level correlation and exposure management.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PositionExposure:
    """Exposure information for a position."""
    symbol: str
    sector: Optional[str]
    market_value: float
    weight: float  # Percentage of portfolio
    beta: float  # Market sensitivity
    correlations: dict[str, float]  # Correlation with other positions


@dataclass
class PortfolioExposure:
    """Overall portfolio exposure analysis."""
    total_value: float
    positions: list[PositionExposure]
    sector_weights: dict[str, float]
    avg_correlation: float
    concentration_score: float  # 0 = diversified, 1 = concentrated
    beta: float  # Portfolio beta
    largest_position_weight: float


# Sector mappings for common stocks
SECTOR_MAP = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "META": "Technology", "NVDA": "Technology",
    "TSLA": "Technology", "AMD": "Technology", "INTC": "Technology",
    "CRM": "Technology", "ADBE": "Technology", "ORCL": "Technology",

    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "C": "Financials",
    "V": "Financials", "MA": "Financials", "AXP": "Financials",

    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "MRK": "Healthcare", "ABBV": "Healthcare", "LLY": "Healthcare",

    # Consumer
    "WMT": "Consumer", "HD": "Consumer", "MCD": "Consumer",
    "NKE": "Consumer", "SBUX": "Consumer", "TGT": "Consumer",
    "COST": "Consumer", "KO": "Consumer", "PEP": "Consumer",

    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy",
    "SLB": "Energy", "EOG": "Energy",

    # Industrials
    "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
    "UPS": "Industrials", "HON": "Industrials",

    # Crypto
    "BTC/USD": "Crypto", "ETH/USD": "Crypto", "SOL/USD": "Crypto",

    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
}


class PortfolioCorrelationManager:
    """
    Manages portfolio-level correlations and exposure limits.

    Prevents:
    - Over-concentration in single sector
    - Taking correlated positions
    - Excessive portfolio beta
    """

    def __init__(
        self,
        max_sector_weight: float = 0.30,
        max_single_position: float = 0.10,
        max_correlation: float = 0.70,
        max_portfolio_beta: float = 1.5,
        lookback_days: int = 60,
    ):
        """
        Args:
            max_sector_weight: Maximum weight in any sector
            max_single_position: Maximum weight in any single position
            max_correlation: Maximum correlation with existing portfolio
            max_portfolio_beta: Maximum portfolio beta
            lookback_days: Days for correlation calculation
        """
        self.max_sector_weight = max_sector_weight
        self.max_single_position = max_single_position
        self.max_correlation = max_correlation
        self.max_portfolio_beta = max_portfolio_beta
        self.lookback_days = lookback_days

        # Cache for price data and correlations
        self._price_cache: dict[str, pd.Series] = {}
        self._correlation_cache: Optional[pd.DataFrame] = None
        self._beta_cache: dict[str, float] = {}

    def update_prices(self, symbol: str, prices: pd.Series) -> None:
        """Update price data for correlation calculations."""
        self._price_cache[symbol] = prices
        self._correlation_cache = None  # Invalidate cache

    def update_prices_bulk(self, price_data: dict[str, pd.Series]) -> None:
        """Update multiple symbols at once."""
        self._price_cache.update(price_data)
        self._correlation_cache = None

    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix for all cached symbols."""
        if self._correlation_cache is not None:
            return self._correlation_cache

        if len(self._price_cache) < 2:
            return pd.DataFrame()

        # Build returns DataFrame
        returns_data = {}
        for symbol, prices in self._price_cache.items():
            if len(prices) > self.lookback_days:
                prices = prices.iloc[-self.lookback_days:]
            returns_data[symbol] = prices.pct_change().dropna()

        returns_df = pd.DataFrame(returns_data).dropna()

        if len(returns_df) < 20:
            return pd.DataFrame()

        self._correlation_cache = returns_df.corr()
        return self._correlation_cache

    def calculate_beta(
        self,
        symbol: str,
        benchmark_symbol: str = "SPY",
    ) -> float:
        """Calculate beta of a symbol vs benchmark."""
        cache_key = f"{symbol}_{benchmark_symbol}"
        if cache_key in self._beta_cache:
            return self._beta_cache[cache_key]

        if symbol not in self._price_cache or benchmark_symbol not in self._price_cache:
            return 1.0  # Default beta

        symbol_returns = self._price_cache[symbol].pct_change().dropna()
        benchmark_returns = self._price_cache[benchmark_symbol].pct_change().dropna()

        # Align data
        combined = pd.concat([symbol_returns, benchmark_returns], axis=1).dropna()
        if len(combined) < 20:
            return 1.0

        combined.columns = ["symbol", "benchmark"]

        covariance = combined["symbol"].cov(combined["benchmark"])
        variance = combined["benchmark"].var()

        if variance == 0:
            return 1.0

        beta = covariance / variance
        self._beta_cache[cache_key] = beta

        return beta

    def analyze_portfolio(
        self,
        positions: dict[str, float],  # symbol -> market_value
    ) -> PortfolioExposure:
        """
        Analyze current portfolio exposure.

        Args:
            positions: Dictionary of symbol to market value

        Returns:
            PortfolioExposure analysis
        """
        total_value = sum(positions.values())
        if total_value == 0:
            return PortfolioExposure(
                total_value=0,
                positions=[],
                sector_weights={},
                avg_correlation=0,
                concentration_score=0,
                beta=1.0,
                largest_position_weight=0,
            )

        # Calculate correlations
        corr_matrix = self.calculate_correlations()

        # Build position exposures
        position_exposures = []
        sector_values = {}
        weights = []
        betas = []

        for symbol, value in positions.items():
            weight = value / total_value
            weights.append(weight)

            sector = SECTOR_MAP.get(symbol, "Other")
            sector_values[sector] = sector_values.get(sector, 0) + value

            beta = self.calculate_beta(symbol)
            betas.append(beta * weight)

            # Get correlations with other positions
            correlations = {}
            if symbol in corr_matrix.index:
                for other in positions:
                    if other != symbol and other in corr_matrix.columns:
                        correlations[other] = corr_matrix.loc[symbol, other]

            position_exposures.append(PositionExposure(
                symbol=symbol,
                sector=sector,
                market_value=value,
                weight=weight,
                beta=beta,
                correlations=correlations,
            ))

        # Calculate sector weights
        sector_weights = {s: v / total_value for s, v in sector_values.items()}

        # Calculate average correlation
        all_correlations = []
        for pos in position_exposures:
            all_correlations.extend(pos.correlations.values())
        avg_correlation = np.mean(all_correlations) if all_correlations else 0

        # Calculate concentration (HHI-like)
        concentration_score = sum(w ** 2 for w in weights)

        # Portfolio beta
        portfolio_beta = sum(betas)

        # Largest position
        largest_weight = max(weights) if weights else 0

        return PortfolioExposure(
            total_value=total_value,
            positions=position_exposures,
            sector_weights=sector_weights,
            avg_correlation=avg_correlation,
            concentration_score=concentration_score,
            beta=portfolio_beta,
            largest_position_weight=largest_weight,
        )

    def can_add_position(
        self,
        symbol: str,
        proposed_value: float,
        current_positions: dict[str, float],
    ) -> tuple[bool, list[str]]:
        """
        Check if a new position can be added within risk limits.

        Returns:
            Tuple of (can_add, list_of_violations)
        """
        violations = []

        # Simulate new portfolio
        new_positions = current_positions.copy()
        new_positions[symbol] = new_positions.get(symbol, 0) + proposed_value

        exposure = self.analyze_portfolio(new_positions)

        # Check single position limit
        symbol_weight = new_positions[symbol] / exposure.total_value
        if symbol_weight > self.max_single_position:
            violations.append(
                f"Position {symbol} would be {symbol_weight:.1%} "
                f"(max: {self.max_single_position:.1%})"
            )

        # Check sector limit
        sector = SECTOR_MAP.get(symbol, "Other")
        sector_weight = exposure.sector_weights.get(sector, 0)
        if sector_weight > self.max_sector_weight:
            violations.append(
                f"Sector {sector} would be {sector_weight:.1%} "
                f"(max: {self.max_sector_weight:.1%})"
            )

        # Check correlation with existing positions
        corr_matrix = self.calculate_correlations()
        if symbol in corr_matrix.index:
            for other_symbol in current_positions:
                if other_symbol in corr_matrix.columns:
                    correlation = corr_matrix.loc[symbol, other_symbol]
                    if abs(correlation) > self.max_correlation:
                        violations.append(
                            f"Correlation with {other_symbol} is {correlation:.2f} "
                            f"(max: {self.max_correlation:.2f})"
                        )

        # Check portfolio beta
        if exposure.beta > self.max_portfolio_beta:
            violations.append(
                f"Portfolio beta would be {exposure.beta:.2f} "
                f"(max: {self.max_portfolio_beta:.2f})"
            )

        return len(violations) == 0, violations

    def get_position_size_limit(
        self,
        symbol: str,
        current_positions: dict[str, float],
        portfolio_value: float,
    ) -> tuple[float, str]:
        """
        Calculate maximum position size within all limits.

        Returns:
            Tuple of (maximum allowed position value, limiting factor description)
        """
        # Start with max single position limit
        max_value = portfolio_value * self.max_single_position

        # Account for existing position
        current_value = current_positions.get(symbol, 0)
        max_additional = max_value - current_value
        limiting_factor = "single position limit"

        # Check sector limit
        sector = SECTOR_MAP.get(symbol, "Other")
        current_sector_value = sum(
            v for s, v in current_positions.items()
            if SECTOR_MAP.get(s, "Other") == sector
        )
        sector_limit = portfolio_value * self.max_sector_weight
        max_for_sector = sector_limit - current_sector_value

        if max_for_sector < max_additional:
            limiting_factor = f"{sector} sector limit ({current_sector_value/portfolio_value:.1%} already allocated)"

        result = max(0, min(max_additional, max_for_sector))

        if result == 0:
            if max_for_sector <= 0:
                limiting_factor = f"{sector} sector already at/over limit ({current_sector_value/portfolio_value:.1%} vs {self.max_sector_weight:.1%} max)"
            elif max_additional <= 0:
                limiting_factor = f"position already at limit ({current_value/portfolio_value:.1%} of portfolio)"

        return result, limiting_factor

    def suggest_rebalance(
        self,
        current_positions: dict[str, float],
    ) -> list[tuple[str, str, float]]:
        """
        Suggest rebalancing actions to improve diversification.

        Returns:
            List of (symbol, action, amount) tuples
        """
        exposure = self.analyze_portfolio(current_positions)
        suggestions = []

        # Check for over-concentrated positions
        for pos in exposure.positions:
            if pos.weight > self.max_single_position * 1.1:  # 10% buffer
                excess = (pos.weight - self.max_single_position) * exposure.total_value
                suggestions.append((pos.symbol, "reduce", excess))

        # Check for over-weighted sectors
        for sector, weight in exposure.sector_weights.items():
            if weight > self.max_sector_weight * 1.1:
                # Find positions in this sector to reduce
                sector_positions = [
                    p for p in exposure.positions if p.sector == sector
                ]
                if sector_positions:
                    # Suggest reducing largest position in sector
                    largest = max(sector_positions, key=lambda x: x.weight)
                    excess = (weight - self.max_sector_weight) * exposure.total_value
                    suggestions.append((largest.symbol, "reduce", excess))

        return suggestions

    def get_diversification_score(
        self,
        current_positions: dict[str, float],
    ) -> float:
        """
        Calculate overall diversification score (0-100).

        Higher is better diversified.
        """
        if not current_positions:
            return 100.0

        exposure = self.analyze_portfolio(current_positions)

        scores = []

        # Position concentration (lower HHI = better)
        # Perfect = 1/n, worst = 1
        n = len(current_positions)
        perfect_hhi = 1 / n if n > 0 else 1
        concentration_score = 1 - (exposure.concentration_score - perfect_hhi) / (1 - perfect_hhi)
        scores.append(max(0, concentration_score) * 100)

        # Sector diversification
        n_sectors = len(exposure.sector_weights)
        if n_sectors > 1:
            sector_score = min(n_sectors / 5, 1) * 100  # Target 5+ sectors
        else:
            sector_score = 20
        scores.append(sector_score)

        # Correlation (lower = better)
        correlation_score = (1 - abs(exposure.avg_correlation)) * 100
        scores.append(correlation_score)

        # Beta (closer to 1 = better for most portfolios)
        beta_deviation = abs(exposure.beta - 1)
        beta_score = max(0, (1 - beta_deviation / 0.5)) * 100
        scores.append(beta_score)

        return np.mean(scores)
