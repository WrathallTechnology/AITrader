#!/bin/bash
# =============================================================================
# Check AITrader status
# =============================================================================

echo "=============================================="
echo "AITrader Status"
echo "=============================================="

# Check if running
if screen -list | grep -q "trader"; then
    echo "Status: RUNNING"
else
    echo "Status: STOPPED"
fi

echo ""

# Show system resources
echo "System Resources:"
echo "  Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "  Disk: $(df -h ~ | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"

echo ""

# Show recent log entry
if [ -f ~/aitrader/logs/trading.log ]; then
    echo "Last log entry:"
    tail -1 ~/aitrader/logs/trading.log
fi

echo ""
