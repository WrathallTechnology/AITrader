"""
AITrader Web Dashboard

A simple Flask-based dashboard to monitor the trading bot.
"""

import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from flask import Flask, render_template, jsonify

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from config import config
from src.client import AlpacaClient
from src.data.fetcher import DataFetcher
from src.data.processor import DataProcessor
from src.strategies import TechnicalStrategy, MLStrategy, HybridStrategy
from src.risk import DrawdownProtection, CircuitBreaker

app = Flask(__name__)

# Global state for caching
_cache = {
    "last_update": None,
    "data": None,
    "signals": {},
}

CACHE_DURATION = 30  # seconds


def get_client():
    """Get Alpaca client."""
    return AlpacaClient(config.alpaca)


def get_account_info():
    """Get account information."""
    client = get_client()
    account = client.get_account()

    return {
        "portfolio_value": float(account.portfolio_value),
        "cash": float(account.cash),
        "buying_power": float(account.buying_power),
        "equity": float(account.equity),
        "initial_capital": config.trading.initial_capital or float(account.portfolio_value),
    }


def get_positions():
    """Get current positions."""
    client = get_client()
    positions = client.get_all_positions()

    return [
        {
            "symbol": p.symbol,
            "qty": float(p.qty),
            "market_value": float(p.market_value),
            "avg_entry_price": float(p.avg_entry_price),
            "current_price": float(p.current_price),
            "unrealized_pl": float(p.unrealized_pl),
            "unrealized_plpc": float(p.unrealized_plpc) * 100,
        }
        for p in positions
    ]


def get_recent_orders():
    """Get recent orders."""
    client = get_client()
    orders = client.get_orders(status="all", limit=10)

    return [
        {
            "symbol": o.symbol,
            "side": o.side,
            "qty": float(o.qty) if o.qty else 0,
            "filled_qty": float(o.filled_qty) if o.filled_qty else 0,
            "type": o.type,
            "status": o.status,
            "created_at": o.created_at.isoformat() if o.created_at else None,
        }
        for o in orders
    ]


def analyze_symbol(symbol: str):
    """Analyze a symbol and return signals."""
    client = get_client()
    fetcher = DataFetcher(client)

    try:
        df = fetcher.get_historical_bars(
            symbol=symbol,
            timeframe="1Hour",
            days=30,
        )

        if df is None or len(df) < 50:
            return None

        # Add indicators
        df = DataProcessor.add_technical_indicators(df)

        latest = df.iloc[-1]

        # Get technical signals
        technical = TechnicalStrategy(config=config.strategy)
        tech_signal = technical.generate_signal(symbol, df)

        # Get ML signal (if model exists)
        ml_signal = None
        model_path = Path("models/price_predictor.pkl")
        if model_path.exists():
            ml = MLStrategy(config=config.strategy)
            try:
                ml.load_model(model_path)
                ml_signal = ml.generate_signal(symbol, df)
            except Exception as e:
                ml_signal = {"error": str(e)}

        return {
            "symbol": symbol,
            "current_price": float(latest["close"]),
            "change_24h": float(df["close"].pct_change(24).iloc[-1] * 100) if len(df) > 24 else 0,
            "indicators": {
                "rsi": round(float(latest["rsi"]), 2) if "rsi" in df.columns else None,
                "macd": round(float(latest["macd"]), 4) if "macd" in df.columns else None,
                "macd_signal": round(float(latest["macd_signal"]), 4) if "macd_signal" in df.columns else None,
                "sma_20": round(float(latest["sma_20"]), 2) if "sma_20" in df.columns else None,
                "sma_50": round(float(latest["sma_50"]), 2) if "sma_50" in df.columns else None,
                "bb_upper": round(float(latest["bb_upper"]), 2) if "bb_upper" in df.columns else None,
                "bb_lower": round(float(latest["bb_lower"]), 2) if "bb_lower" in df.columns else None,
                "atr": round(float(latest["atr"]), 2) if "atr" in df.columns else None,
                "volatility": round(float(latest["volatility"]) * 100, 2) if "volatility" in df.columns else None,
            },
            "technical_signal": {
                "type": tech_signal.signal_type.value,
                "confidence": round(tech_signal.confidence * 100, 1),
                "reason": tech_signal.reason,
            },
            "ml_signal": {
                "type": ml_signal.signal_type.value if ml_signal and hasattr(ml_signal, 'signal_type') else "N/A",
                "confidence": round(ml_signal.confidence * 100, 1) if ml_signal and hasattr(ml_signal, 'confidence') else 0,
                "reason": ml_signal.reason if ml_signal and hasattr(ml_signal, 'reason') else str(ml_signal) if ml_signal else "Model not loaded",
            } if ml_signal else {"type": "N/A", "confidence": 0, "reason": "ML model not available"},
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e)}


def read_recent_logs(lines: int = 50):
    """Read recent log entries."""
    log_file = Path("logs/trading.log")
    if not log_file.exists():
        return []

    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            return [line.strip() for line in all_lines[-lines:]]
    except Exception as e:
        return [f"Error reading logs: {e}"]


@app.route("/")
def index():
    """Main dashboard page."""
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    """Get overall bot status."""
    try:
        account = get_account_info()
        positions = get_positions()

        # Calculate P&L
        initial = account["initial_capital"]
        current = account["portfolio_value"]
        pnl = current - initial
        pnl_pct = (pnl / initial) * 100 if initial > 0 else 0

        return jsonify({
            "status": "running",
            "account": account,
            "pnl": {
                "value": round(pnl, 2),
                "percent": round(pnl_pct, 2),
            },
            "positions_count": len(positions),
            "mode": "crypto",  # Could be dynamic
            "paper_trading": config.alpaca.is_paper,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route("/api/positions")
def api_positions():
    """Get current positions."""
    try:
        positions = get_positions()
        return jsonify({"positions": positions})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/orders")
def api_orders():
    """Get recent orders."""
    try:
        orders = get_recent_orders()
        return jsonify({"orders": orders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/signals")
def api_signals():
    """Get current signals for watched symbols."""
    try:
        symbols = ["BTC/USD", "ETH/USD"]
        signals = []

        for symbol in symbols:
            analysis = analyze_symbol(symbol)
            if analysis:
                signals.append(analysis)

        return jsonify({"signals": signals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/logs")
def api_logs():
    """Get recent log entries."""
    try:
        logs = read_recent_logs(100)
        return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/analysis/<symbol>")
def api_analysis(symbol: str):
    """Get detailed analysis for a symbol."""
    try:
        # Handle URL encoding for crypto symbols
        symbol = symbol.replace("-", "/")
        analysis = analyze_symbol(symbol)
        if analysis:
            return jsonify(analysis)
        return jsonify({"error": "Could not analyze symbol"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Run on all interfaces so it's accessible externally
    app.run(host="0.0.0.0", port=5000, debug=False)
