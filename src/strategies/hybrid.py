"""
Advanced hybrid strategy that combines multiple sub-strategies with:
- Adaptive weighting based on performance
- Market regime detection
- Strategy correlation adjustment
- Time-based filters
- Continuous signal scoring
"""

import logging
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalType
from .performance_tracker import PerformanceTracker, TradeResult
from .regime_detector import RegimeDetector, RegimeAnalysis, MarketRegime
from .correlation_adjuster import CorrelationAdjuster, SignalAgreementAnalyzer
from .signal_scorer import SignalScorer, SignalScore, combine_signal_scores
from .time_filter import TimeFilter, VolatilityFilter

logger = logging.getLogger(__name__)


class AdvancedHybridStrategy(BaseStrategy):
    """
    Advanced hybrid strategy with intelligent signal combination.

    Features:
    1. Adaptive weighting - adjusts strategy weights based on recent performance
    2. Market regime awareness - weights strategies based on current market conditions
    3. Correlation adjustment - reduces weight of highly correlated strategies
    4. Time-based filtering - avoids trading during high-risk periods
    5. Continuous signal scoring - uses granular scores instead of binary signals
    6. Conflict resolution - intelligently handles strategy disagreements
    """

    def __init__(
        self,
        name: str = "advanced_hybrid",
        strategies: Optional[list[BaseStrategy]] = None,
        min_confidence: float = 0.5,
        consensus_required: float = 0.4,
        use_adaptive_weights: bool = True,
        use_regime_detection: bool = True,
        use_correlation_adjustment: bool = True,
        use_time_filter: bool = True,
        use_signal_scoring: bool = True,
    ):
        """
        Initialize the advanced hybrid strategy.

        Args:
            name: Strategy identifier
            strategies: List of sub-strategies to combine
            min_confidence: Minimum confidence threshold for signals
            consensus_required: Fraction of strategies that must agree
            use_adaptive_weights: Enable performance-based weight adjustment
            use_regime_detection: Enable market regime awareness
            use_correlation_adjustment: Enable correlation-based weight reduction
            use_time_filter: Enable time-based trade filtering
            use_signal_scoring: Use continuous signal scores
        """
        super().__init__(name)
        self.strategies = strategies or []
        self.min_confidence = min_confidence
        self.consensus_required = consensus_required

        # Feature flags
        self.use_adaptive_weights = use_adaptive_weights
        self.use_regime_detection = use_regime_detection
        self.use_correlation_adjustment = use_correlation_adjustment
        self.use_time_filter = use_time_filter
        self.use_signal_scoring = use_signal_scoring

        # Initialize components
        self.performance_tracker = PerformanceTracker()
        self.regime_detector = RegimeDetector()
        self.correlation_adjuster = CorrelationAdjuster()
        self.signal_scorer = SignalScorer()
        self.agreement_analyzer = SignalAgreementAnalyzer()
        self.time_filter = TimeFilter()
        self.volatility_filter = VolatilityFilter()

        # Cache for current regime
        self._current_regime: Optional[RegimeAnalysis] = None
        self._regime_cache_time: Optional[datetime] = None

        # Base weights (stored separately for adjustment)
        self._base_weights: dict[str, float] = {}

    def add_strategy(self, strategy: BaseStrategy) -> None:
        """Add a sub-strategy."""
        self.strategies.append(strategy)
        self._base_weights[strategy.name] = strategy.weight
        logger.info(f"Added strategy: {strategy.name} (weight={strategy.weight})")

    def remove_strategy(self, name: str) -> None:
        """Remove a sub-strategy by name."""
        self.strategies = [s for s in self.strategies if s.name != name]
        self._base_weights.pop(name, None)

    def generate_signal(self, symbol: str, data: pd.DataFrame) -> Signal:
        """
        Generate a combined signal using all advanced features.
        """
        if not self.strategies:
            logger.warning("No strategies configured!")
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="No strategies configured",
            )

        # Step 1: Apply time filter
        if self.use_time_filter:
            time_result = self.time_filter.filter(symbol)
            if not time_result.can_trade:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    confidence=0.0,
                    reason=f"Time filter: {'; '.join(time_result.reasons)}",
                )

            # Also check volatility
            vol_result = self.volatility_filter.filter(data)
            time_confidence_multiplier = (
                time_result.confidence_multiplier *
                vol_result.confidence_multiplier
            )
        else:
            time_confidence_multiplier = 1.0

        # Step 2: Detect market regime
        if self.use_regime_detection:
            regime = self._get_regime(data)
        else:
            regime = None

        # Step 3: Calculate adjusted weights
        weights = self._calculate_weights(regime)

        # Step 4: Collect signals from all strategies
        raw_signals: dict[str, Signal] = {}
        signal_scores: list[SignalScore] = []

        for strategy in self.strategies:
            try:
                signal = strategy.generate_signal(symbol, data)
                raw_signals[strategy.name] = signal

                # Track for correlation analysis
                self.correlation_adjuster.record_signal(strategy.name, signal)

                # Calculate signal score if enabled
                if self.use_signal_scoring:
                    score = self.signal_scorer.score_signal(
                        symbol=symbol,
                        data=data,
                        strategy_name=strategy.name,
                    )
                    signal_scores.append(score)

                logger.debug(f"{strategy.name}: {signal}")

            except Exception as e:
                logger.error(f"Error in {strategy.name}: {e}")

        if not raw_signals:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=0.0,
                reason="All strategies failed",
            )

        # Step 5: Analyze agreement
        agreement = self.agreement_analyzer.record_signals(raw_signals, symbol)

        # Step 6: Combine signals
        if self.use_signal_scoring and signal_scores:
            final_signal = self._combine_signal_scores(
                symbol, data, signal_scores, weights, agreement
            )
        else:
            final_signal = self._combine_traditional(
                symbol, data, raw_signals, weights, agreement
            )

        # Step 7: Apply time-based confidence adjustment
        if time_confidence_multiplier < 1.0:
            adjusted_confidence = final_signal.confidence * time_confidence_multiplier
            final_signal = Signal(
                symbol=final_signal.symbol,
                signal_type=final_signal.signal_type if adjusted_confidence > self.min_confidence else SignalType.HOLD,
                confidence=adjusted_confidence,
                price=final_signal.price,
                stop_loss=final_signal.stop_loss,
                take_profit=final_signal.take_profit,
                reason=f"{final_signal.reason} (time adj: {time_confidence_multiplier:.2f})",
            )

        # Step 8: Apply regime-based adjustments
        if regime and final_signal.signal_type != SignalType.HOLD:
            final_signal = self._apply_regime_adjustments(final_signal, regime, data)

        return final_signal

    def _get_regime(self, data: pd.DataFrame) -> RegimeAnalysis:
        """Get current market regime (cached)."""
        now = datetime.now()

        # Cache regime for 5 minutes
        if (self._current_regime is not None and
            self._regime_cache_time is not None and
            (now - self._regime_cache_time).seconds < 300):
            return self._current_regime

        self._current_regime = self.regime_detector.detect_regime(data)
        self._regime_cache_time = now

        logger.info(
            f"Market regime: {self._current_regime.primary_regime.value} "
            f"(trend={self._current_regime.trend_strength:.2f}, "
            f"ADX={self._current_regime.adx_value:.1f})"
        )

        return self._current_regime

    def _calculate_weights(self, regime: Optional[RegimeAnalysis]) -> dict[str, float]:
        """Calculate final strategy weights with all adjustments."""
        # Start with base weights
        weights = self._base_weights.copy()

        # Apply adaptive weighting based on performance
        if self.use_adaptive_weights:
            for name in weights:
                adaptive_weight = self.performance_tracker.get_adaptive_weight(
                    name,
                    base_weight=weights[name],
                    min_weight=0.1,
                    max_weight=0.5,
                )
                weights[name] = adaptive_weight

        # Apply regime-based adjustments
        if self.use_regime_detection and regime:
            weights = self.regime_detector.get_strategy_weights_for_regime(
                regime, weights
            )

        # Apply correlation adjustment
        if self.use_correlation_adjustment:
            weights = self.correlation_adjuster.get_adjusted_weights(weights)

        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def _combine_signal_scores(
        self,
        symbol: str,
        data: pd.DataFrame,
        scores: list[SignalScore],
        weights: dict[str, float],
        agreement: dict,
    ) -> Signal:
        """Combine using continuous signal scores."""
        combined = combine_signal_scores(scores, weights)

        # Adjust for agreement
        if agreement["has_conflict"]:
            combined = SignalScore(
                symbol=combined.symbol,
                raw_score=combined.raw_score * 0.7,  # Reduce on conflict
                confidence=combined.confidence * 0.8,
                direction_score=combined.direction_score,
                magnitude_score=combined.magnitude_score,
                timing_score=combined.timing_score,
                risk_reward_score=combined.risk_reward_score,
                strategy_name=combined.strategy_name,
                reasons=combined.reasons + ["Conflict detected"],
            )

        # Check minimum thresholds
        if combined.confidence < self.min_confidence:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=combined.confidence,
                reason=f"Combined confidence {combined.confidence:.2%} below threshold",
            )

        # Convert to signal
        price = float(data["close"].iloc[-1]) if not data.empty else None
        atr = float(data["atr"].iloc[-1]) if "atr" in data.columns and not data.empty else (price or 0) * 0.02

        if combined.signal_type == SignalType.BUY:
            stop_loss = price - (2 * atr) if price else None
            take_profit = price + (3 * atr) if price else None
        elif combined.signal_type == SignalType.SELL:
            stop_loss = price + (2 * atr) if price else None
            take_profit = price - (3 * atr) if price else None
        else:
            stop_loss = None
            take_profit = None

        return Signal(
            symbol=symbol,
            signal_type=combined.signal_type,
            confidence=combined.confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="; ".join(combined.reasons[:5]),  # Limit reasons
        )

    def _combine_traditional(
        self,
        symbol: str,
        data: pd.DataFrame,
        signals: dict[str, Signal],
        weights: dict[str, float],
        agreement: dict,
    ) -> Signal:
        """Traditional weighted voting combination."""
        # Use conflict resolution if there's disagreement
        if agreement["has_conflict"]:
            return self.agreement_analyzer.get_conflict_resolution(signals, weights)

        # Standard weighted combination
        buy_score = 0.0
        sell_score = 0.0
        hold_score = 0.0
        total_weight = sum(weights.get(name, 1.0) for name in signals)

        for name, signal in signals.items():
            weight = weights.get(name, 1.0) / total_weight
            weighted_conf = signal.confidence * weight

            if signal.signal_type == SignalType.BUY:
                buy_score += weighted_conf
            elif signal.signal_type == SignalType.SELL:
                sell_score += weighted_conf
            else:
                hold_score += weighted_conf

        # Determine winner
        max_score = max(buy_score, sell_score, hold_score)

        if max_score < self.min_confidence:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.HOLD,
                confidence=max_score,
                reason=f"Confidence {max_score:.2%} below threshold",
            )

        # Check consensus
        if buy_score == max_score:
            buy_count = sum(1 for s in signals.values() if s.signal_type == SignalType.BUY)
            if buy_count / len(signals) < self.consensus_required:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    confidence=buy_score,
                    reason=f"Insufficient BUY consensus ({buy_count}/{len(signals)})",
                )
            signal_type = SignalType.BUY
            confidence = buy_score

        elif sell_score == max_score:
            sell_count = sum(1 for s in signals.values() if s.signal_type == SignalType.SELL)
            if sell_count / len(signals) < self.consensus_required:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.HOLD,
                    confidence=sell_score,
                    reason=f"Insufficient SELL consensus ({sell_count}/{len(signals)})",
                )
            signal_type = SignalType.SELL
            confidence = sell_score

        else:
            signal_type = SignalType.HOLD
            confidence = hold_score

        # Get price and build stops
        price = float(data["close"].iloc[-1]) if not data.empty else None

        # Aggregate stops from agreeing strategies
        agreeing = [s for s in signals.values() if s.signal_type == signal_type]
        stop_losses = [s.stop_loss for s in agreeing if s.stop_loss]
        take_profits = [s.take_profit for s in agreeing if s.take_profit]

        stop_loss = np.mean(stop_losses) if stop_losses else None
        take_profit = np.mean(take_profits) if take_profits else None

        reasons = [f"{name}:{s.signal_type.value}" for name, s in signals.items()]

        return Signal(
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason=f"Combined: {', '.join(reasons)}",
        )

    def _apply_regime_adjustments(
        self,
        signal: Signal,
        regime: RegimeAnalysis,
        data: pd.DataFrame,
    ) -> Signal:
        """Apply regime-specific adjustments to the signal."""
        price = signal.price or float(data["close"].iloc[-1])
        atr = float(data["atr"].iloc[-1]) if "atr" in data.columns else price * 0.02

        # Trending market: widen stops, larger targets
        if regime.is_trending and regime.trend_strength > 0.5:
            if signal.stop_loss:
                # Widen stop by 20%
                stop_distance = abs(price - signal.stop_loss)
                if signal.signal_type == SignalType.BUY:
                    new_stop = price - (stop_distance * 1.2)
                else:
                    new_stop = price + (stop_distance * 1.2)
            else:
                new_stop = signal.stop_loss

            if signal.take_profit:
                # Extend target by 30%
                target_distance = abs(signal.take_profit - price)
                if signal.signal_type == SignalType.BUY:
                    new_target = price + (target_distance * 1.3)
                else:
                    new_target = price - (target_distance * 1.3)
            else:
                new_target = signal.take_profit

            return Signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                confidence=signal.confidence,
                price=signal.price,
                stop_loss=new_stop,
                take_profit=new_target,
                reason=f"{signal.reason} [trending market adj]",
            )

        # High volatility: tighten position (handled by position sizing, but reduce confidence)
        if regime.volatility_regime == MarketRegime.HIGH_VOLATILITY:
            return Signal(
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                confidence=signal.confidence * 0.85,
                price=signal.price,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
                reason=f"{signal.reason} [high vol adj]",
            )

        return signal

    def record_trade_result(
        self,
        strategy_name: str,
        symbol: str,
        signal_type: str,
        confidence: float,
        entry_price: float,
        exit_price: Optional[float] = None,
    ) -> None:
        """Record a trade result for performance tracking."""
        trade = TradeResult(
            strategy_name=strategy_name,
            symbol=symbol,
            signal_type=signal_type,
            confidence=confidence,
            entry_price=entry_price,
            exit_price=exit_price,
        )

        if exit_price:
            self.performance_tracker.close_trade(strategy_name, symbol, exit_price)
        else:
            self.performance_tracker.record_trade(trade)

    def train(self, data: pd.DataFrame) -> None:
        """Train all sub-strategies."""
        for strategy in self.strategies:
            logger.info(f"Training {strategy.name}...")
            strategy.train(data)
        self._is_trained = True

    def get_strategy_weights(self) -> dict[str, float]:
        """Get current effective strategy weights."""
        regime = self._current_regime
        return self._calculate_weights(regime)

    def get_base_weights(self) -> dict[str, float]:
        """Get base (uncorrected) strategy weights."""
        return self._base_weights.copy()

    def set_strategy_weight(self, name: str, weight: float) -> None:
        """Update base weight for a specific strategy."""
        if name in self._base_weights:
            self._base_weights[name] = weight
            for strategy in self.strategies:
                if strategy.name == name:
                    strategy.weight = weight
                    break
            logger.info(f"Updated {name} base weight to {weight}")
        else:
            logger.warning(f"Strategy {name} not found")

    def get_diagnostics(self) -> dict:
        """Get diagnostic information about the strategy state."""
        regime = self._current_regime

        return {
            "current_regime": regime.primary_regime.value if regime else None,
            "trend_strength": regime.trend_strength if regime else None,
            "volatility_regime": regime.volatility_regime.value if regime else None,
            "base_weights": self._base_weights,
            "effective_weights": self.get_strategy_weights(),
            "diversification_score": self.correlation_adjuster.get_diversification_score(),
            "strategy_performance": {
                name: self.performance_tracker.get_all_metrics(name)
                for name in self._base_weights
            },
            "features_enabled": {
                "adaptive_weights": self.use_adaptive_weights,
                "regime_detection": self.use_regime_detection,
                "correlation_adjustment": self.use_correlation_adjustment,
                "time_filter": self.use_time_filter,
                "signal_scoring": self.use_signal_scoring,
            },
        }


# Keep the original HybridStrategy for backwards compatibility
class HybridStrategy(AdvancedHybridStrategy):
    """
    Backwards-compatible hybrid strategy.

    Uses the advanced implementation with all features enabled.
    """

    def __init__(
        self,
        name: str = "hybrid",
        strategies: Optional[list[BaseStrategy]] = None,
        min_confidence: float = 0.6,
        consensus_required: float = 0.5,
    ):
        super().__init__(
            name=name,
            strategies=strategies,
            min_confidence=min_confidence,
            consensus_required=consensus_required,
            use_adaptive_weights=True,
            use_regime_detection=True,
            use_correlation_adjustment=True,
            use_time_filter=True,
            use_signal_scoring=True,
        )
