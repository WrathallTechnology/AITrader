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
from src.data.crypto_scanner import CryptoScanner
from src.strategies import TechnicalStrategy, MLStrategy, HybridStrategy
from src.risk import DrawdownProtection, CircuitBreaker
from src.options import OptionsClient, OptionsScanner, OptionsStrategyManager

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


def get_options_client():
    """Get Options client for options trading."""
    return OptionsClient(
        api_key=config.alpaca.api_key,
        secret_key=config.alpaca.secret_key,
        paper=config.alpaca.is_paper,
    )


def get_account_info():
    """Get account information - shows only effective trading capital."""
    client = get_client()
    account = client.get_account()

    full_portfolio = float(account.portfolio_value)
    initial_capital = config.trading.initial_capital or full_portfolio

    # Calculate effective capital (initial + any P&L from bot's trades)
    # For now, we track based on initial_capital setting
    # In a real scenario, you'd track the bot's actual P&L separately
    effective_capital = initial_capital

    # Calculate what portion of cash is allocated to bot
    cash_ratio = initial_capital / full_portfolio if full_portfolio > 0 else 1
    effective_cash = float(account.cash) * cash_ratio
    effective_buying_power = float(account.buying_power) * cash_ratio

    return {
        "portfolio_value": effective_capital,  # Show only bot's capital
        "cash": round(effective_cash, 2),
        "buying_power": round(effective_buying_power, 2),
        "equity": effective_capital,
        "initial_capital": initial_capital,
        "full_account_value": full_portfolio,  # Keep for reference
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

        # Calculate P&L based on positions only (bot's actual trades)
        initial = account["initial_capital"]
        positions_value = sum(p["market_value"] for p in positions)
        positions_pnl = sum(p["unrealized_pl"] for p in positions)

        # Effective value = initial capital + unrealized P&L from positions
        effective_value = initial + positions_pnl
        pnl_pct = (positions_pnl / initial) * 100 if initial > 0 else 0

        return jsonify({
            "status": "running",
            "account": {
                "portfolio_value": round(effective_value, 2),
                "cash": round(initial - positions_value, 2),  # Cash = initial minus invested
                "buying_power": round(initial - positions_value, 2),
                "initial_capital": initial,
                "positions_value": round(positions_value, 2),
            },
            "pnl": {
                "value": round(positions_pnl, 2),
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
    """Get current signals for all watched symbols (stocks + crypto)."""
    try:
        # All watchlist symbols
        stock_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AMD", "SPY", "QQQ"]
        crypto_symbols = ["BTC/USD", "ETH/USD"]

        signals = []
        client = get_client()
        market_open = client.is_market_open()

        # Get crypto signals (always available)
        for symbol in crypto_symbols:
            analysis = analyze_symbol(symbol)
            if analysis and "error" not in analysis:
                analysis["asset_type"] = "crypto"
                signals.append(analysis)

        # Get stock signals (only when market is open, or use cached data)
        if market_open:
            for symbol in stock_symbols:
                analysis = analyze_symbol(symbol)
                if analysis and "error" not in analysis:
                    analysis["asset_type"] = "stock"
                    signals.append(analysis)

        return jsonify({
            "signals": signals,
            "market_open": market_open,
            "timestamp": datetime.now().isoformat(),
        })
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


@app.route("/api/scanner")
def api_scanner():
    """Get crypto scanner opportunities."""
    try:
        client = get_client()
        scanner = CryptoScanner(
            client=client,
            min_volume_usd=1_000,  # Very low - small account doesn't need high liquidity
            max_pairs=15,
        )
        opportunities = scanner.get_top_opportunities(count=15, min_score=30)

        return jsonify({
            "opportunities": [
                {
                    "symbol": o.symbol,
                    "price": round(o.price, 2) if o.price > 1 else round(o.price, 6),
                    "change_24h": round(o.change_24h, 2),
                    "volume_24h": round(o.volume_24h, 0),
                    "volatility": round(o.volatility, 1),
                    "rsi": round(o.rsi, 1),
                    "trend_strength": round(o.trend_strength, 2),
                    "score": round(o.score, 0),
                    "reason": o.reason,
                }
                for o in opportunities
            ],
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/transactions")
def api_transactions():
    """Get transaction history with reasoning."""
    try:
        transactions_file = Path("logs/transactions.json")
        if transactions_file.exists():
            with open(transactions_file, "r") as f:
                transactions = json.load(f)
            # Return last 100 transactions, most recent first
            transactions = list(reversed(transactions[-100:]))
            return jsonify({"transactions": transactions})
        return jsonify({"transactions": []})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Stock watchlist for options scanning
OPTIONS_WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "NVDA", "TSLA", "AMD", "SPY", "QQQ",
]


@app.route("/api/options-scanner")
def api_options_scanner():
    """Get options scanner opportunities."""
    try:
        options_client = get_options_client()
        scanner = OptionsScanner(
            client=options_client,
            watchlist=OPTIONS_WATCHLIST,
        )

        # Get market trend from request args or default to neutral
        market_trend = "neutral"

        # Check if market is open
        client = get_client()
        market_open = client.is_market_open()

        opportunities = scanner.get_top_opportunities(
            count=10,
            market_trend=market_trend,
        )

        # If no opportunities, return helpful message
        if not opportunities:
            return jsonify({
                "opportunities": [],
                "watchlist": OPTIONS_WATCHLIST,
                "market_trend": market_trend,
                "market_open": market_open,
                "message": "No options opportunities found" + (" (market closed)" if not market_open else ""),
                "timestamp": datetime.now().isoformat(),
            })

        return jsonify({
            "opportunities": [
                {
                    "symbol": o.symbol,
                    "underlying_price": round(o.underlying_price, 2),
                    "opportunity_type": o.opportunity_type,
                    "score": round(o.score, 0),
                    "details": o.details,
                    "signal": {
                        "strategy": o.signal.strategy_name,
                        "spread_type": o.signal.spread_type.value if hasattr(o.signal.spread_type, 'value') else str(o.signal.spread_type),
                        "confidence": round(o.signal.confidence * 100, 1),
                        "expected_profit": round(o.signal.expected_profit, 2) if o.signal.expected_profit else None,
                        "max_loss": round(o.signal.max_loss, 2) if o.signal.max_loss else None,
                        "rationale": o.signal.rationale,
                        "contracts": [
                            {
                                "symbol": c.symbol,
                                "strike": c.strike,
                                "expiration": c.expiration.isoformat() if hasattr(c.expiration, 'isoformat') else str(c.expiration),
                                "option_type": c.option_type.value if hasattr(c.option_type, 'value') else str(c.option_type),
                                "bid": c.bid,
                                "ask": c.ask,
                                "delta": round(c.delta, 3) if c.delta else None,
                                "iv": round(c.implied_volatility * 100, 1) if c.implied_volatility else None,
                            }
                            for c in o.signal.contracts
                        ] if o.signal.contracts else [],
                    } if o.signal else None,
                }
                for o in opportunities
            ],
            "watchlist": OPTIONS_WATCHLIST,
            "market_trend": market_trend,
            "market_open": market_open,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "opportunities": []}), 500


@app.route("/api/options-positions")
def api_options_positions():
    """Get current options positions."""
    try:
        options_client = get_options_client()
        positions = options_client.get_positions()

        return jsonify({
            "positions": [
                {
                    "symbol": p.contract.symbol,
                    "underlying": p.contract.underlying,
                    "strike": p.contract.strike,
                    "expiration": p.contract.expiration.isoformat() if hasattr(p.contract.expiration, 'isoformat') else str(p.contract.expiration),
                    "option_type": p.contract.option_type.value if hasattr(p.contract.option_type, 'value') else str(p.contract.option_type),
                    "quantity": p.quantity,
                    "avg_cost": round(p.avg_cost, 2),
                    "market_value": round(p.market_value, 2) if p.market_value else None,
                    "unrealized_pnl": round(p.unrealized_pnl, 2) if p.unrealized_pnl else None,
                    "delta_exposure": round(p.delta_exposure, 2) if p.delta_exposure else None,
                    "days_to_expiration": p.contract.days_to_expiration,
                }
                for p in positions
            ],
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "positions": []})


@app.route("/api/options-chain/<symbol>")
def api_options_chain(symbol: str):
    """Get options chain for a symbol."""
    try:
        options_client = get_options_client()

        # Get chain with 7-60 DTE range
        chain = options_client.get_option_chain(
            underlying=symbol.upper(),
            expiration_date_gte=(datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            expiration_date_lte=(datetime.now() + timedelta(days=60)).strftime("%Y-%m-%d"),
        )

        if not chain:
            return jsonify({"error": f"No options chain found for {symbol}"}), 404

        # Get ATM contracts for summary
        atm_strike = chain.get_atm_strike()
        calls = chain.get_calls()
        puts = chain.get_puts()

        return jsonify({
            "symbol": symbol.upper(),
            "underlying_price": round(chain.underlying_price, 2),
            "atm_strike": atm_strike,
            "expirations": [exp.isoformat() for exp in chain.expirations],
            "contracts_count": len(chain.contracts),
            "calls_count": len(calls),
            "puts_count": len(puts),
            "sample_contracts": [
                {
                    "symbol": c.symbol,
                    "strike": c.strike,
                    "expiration": c.expiration.isoformat() if hasattr(c.expiration, 'isoformat') else str(c.expiration),
                    "option_type": c.option_type.value if hasattr(c.option_type, 'value') else str(c.option_type),
                    "bid": c.bid,
                    "ask": c.ask,
                    "volume": c.volume,
                    "open_interest": c.open_interest,
                    "iv": round(c.implied_volatility * 100, 1) if c.implied_volatility else None,
                    "delta": round(c.delta, 3) if c.delta else None,
                }
                for c in chain.contracts[:20]  # First 20 contracts as sample
            ],
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/options-debug")
def api_options_debug():
    """Debug endpoint to diagnose options scanning issues."""
    try:
        import traceback
        from datetime import date, timedelta

        debug_info = {
            "timestamp": datetime.now().isoformat(),
            "tests": [],
            "errors": [],
        }

        # Test 1: Check if market is open
        client = get_client()
        market_open = client.is_market_open()
        debug_info["market_open"] = market_open
        debug_info["tests"].append({
            "name": "Market Status",
            "passed": True,
            "result": f"Market is {'OPEN' if market_open else 'CLOSED'}",
        })

        # Test 2: Initialize options client
        try:
            options_client = get_options_client()
            debug_info["tests"].append({
                "name": "Options Client Init",
                "passed": True,
                "result": "Options client initialized successfully",
            })
        except Exception as e:
            debug_info["tests"].append({
                "name": "Options Client Init",
                "passed": False,
                "result": f"Failed: {str(e)}",
            })
            debug_info["errors"].append(traceback.format_exc())
            return jsonify(debug_info)

        # Test 3: Get underlying price for a test symbol
        test_symbol = "SPY"
        try:
            underlying_price = options_client._get_underlying_price(test_symbol)
            debug_info["tests"].append({
                "name": f"Get Underlying Price ({test_symbol})",
                "passed": underlying_price > 0,
                "result": f"${underlying_price:.2f}" if underlying_price > 0 else "Failed - returned 0",
            })
        except Exception as e:
            debug_info["tests"].append({
                "name": f"Get Underlying Price ({test_symbol})",
                "passed": False,
                "result": f"Error: {str(e)}",
            })
            debug_info["errors"].append(traceback.format_exc())

        # Test 4: Get option chain for test symbol
        try:
            chain = options_client.get_option_chain(
                underlying=test_symbol,
                expiration_date_gte=date.today() + timedelta(days=7),
                expiration_date_lte=date.today() + timedelta(days=45),
            )
            debug_info["tests"].append({
                "name": f"Get Option Chain ({test_symbol})",
                "passed": len(chain.contracts) > 0,
                "result": f"Got {len(chain.contracts)} contracts, {len(chain.expirations)} expirations, underlying=${chain.underlying_price:.2f}",
            })

            # Additional chain details
            debug_info["chain_sample"] = {
                "underlying": test_symbol,
                "underlying_price": chain.underlying_price,
                "total_contracts": len(chain.contracts),
                "expirations": [exp.isoformat() for exp in chain.expirations[:5]],
                "sample_contracts": [
                    {
                        "symbol": c.symbol,
                        "strike": c.strike,
                        "type": c.option_type.value,
                        "bid": c.bid,
                        "ask": c.ask,
                        "open_interest": c.open_interest,
                        "volume": c.volume,
                    }
                    for c in chain.contracts[:5]
                ]
            }
        except Exception as e:
            debug_info["tests"].append({
                "name": f"Get Option Chain ({test_symbol})",
                "passed": False,
                "result": f"Error: {str(e)}",
            })
            debug_info["errors"].append(traceback.format_exc())

        # Test 5: Find contracts with relaxed criteria
        try:
            from src.options import OptionType
            contracts = options_client.find_contracts(
                underlying=test_symbol,
                option_type=OptionType.CALL,
                min_days=7,
                max_days=45,
                min_open_interest=10,
                max_spread_pct=0.20,
            )
            debug_info["tests"].append({
                "name": f"Find Contracts ({test_symbol} calls)",
                "passed": len(contracts) > 0,
                "result": f"Found {len(contracts)} suitable contracts",
            })

            if contracts:
                debug_info["suitable_contracts"] = [
                    {
                        "symbol": c.symbol,
                        "strike": c.strike,
                        "expiration": c.expiration.isoformat(),
                        "bid": c.bid,
                        "ask": c.ask,
                        "spread_pct": round(c.spread_pct * 100, 1) if c.spread_pct else None,
                        "open_interest": c.open_interest,
                    }
                    for c in contracts[:5]
                ]
        except Exception as e:
            debug_info["tests"].append({
                "name": f"Find Contracts ({test_symbol} calls)",
                "passed": False,
                "result": f"Error: {str(e)}",
            })
            debug_info["errors"].append(traceback.format_exc())

        # Test 6: Run scanner on single symbol
        try:
            scanner = OptionsScanner(client=options_client, watchlist=[test_symbol])
            from src.options.scanner import ScanCriteria
            criteria = ScanCriteria(min_open_interest=10, max_spread_pct=0.20)
            opportunities = scanner.scan_symbol(test_symbol, criteria, market_trend="neutral")

            debug_info["tests"].append({
                "name": f"Scan Single Symbol ({test_symbol})",
                "passed": True,
                "result": f"Found {len(opportunities)} opportunities",
            })

            if opportunities:
                debug_info["scanner_results"] = [
                    {
                        "type": o.opportunity_type,
                        "score": o.score,
                        "signal": o.signal.strategy_name if o.signal else None,
                        "spread_type": o.signal.spread_type.value if o.signal else None,
                    }
                    for o in opportunities[:5]
                ]
        except Exception as e:
            debug_info["tests"].append({
                "name": f"Scan Single Symbol ({test_symbol})",
                "passed": False,
                "result": f"Error: {str(e)}",
            })
            debug_info["errors"].append(traceback.format_exc())

        # Summary
        passed = sum(1 for t in debug_info["tests"] if t["passed"])
        total = len(debug_info["tests"])
        debug_info["summary"] = {
            "passed": passed,
            "failed": total - passed,
            "total": total,
            "all_passed": passed == total,
        }

        return jsonify(debug_info)

    except Exception as e:
        import traceback
        return jsonify({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "timestamp": datetime.now().isoformat(),
        }), 500


@app.route("/api/logs/filter")
def api_logs_filter():
    """Get filtered log entries."""
    try:
        from flask import request

        lines = int(request.args.get('lines', 100))
        filter_text = request.args.get('filter', '').lower()
        level = request.args.get('level', 'all').upper()

        logs = read_recent_logs(lines * 2)  # Read extra to account for filtering

        # Filter by level
        if level != 'ALL':
            logs = [l for l in logs if level in l]

        # Filter by text search
        if filter_text:
            logs = [l for l in logs if filter_text in l.lower()]

        # Limit results
        logs = logs[-lines:]

        return jsonify({
            "logs": logs,
            "count": len(logs),
            "filter": filter_text,
            "level": level,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        return jsonify({"error": str(e), "logs": []}), 500


if __name__ == "__main__":
    # Run on all interfaces so it's accessible externally
    app.run(host="0.0.0.0", port=5000, debug=False)
