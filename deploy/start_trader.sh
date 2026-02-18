#!/bin/bash
# =============================================================================
# Start AITrader trading bot
# =============================================================================

echo "Starting AITrader..."
sudo systemctl start aitrader
sleep 2

if systemctl is-active --quiet aitrader; then
    echo "AITrader is running."
    echo "View logs: journalctl -u aitrader -f"
else
    echo "AITrader failed to start."
    echo "Check logs: journalctl -u aitrader -n 50"
    exit 1
fi
