#!/bin/bash
# =============================================================================
# View AITrader logs
# =============================================================================

cd ~/aitrader

if [ -f "logs/trading.log" ]; then
    echo "Last 50 lines of trading log:"
    echo "=============================================="
    tail -50 logs/trading.log
    echo ""
    echo "=============================================="
    echo "Use 'tail -f logs/trading.log' for live view"
else
    echo "No log file found yet."
fi
