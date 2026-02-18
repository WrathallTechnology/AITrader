#!/bin/bash
# =============================================================================
# Start AITrader Dashboard
# =============================================================================

echo "Starting AITrader Dashboard..."
sudo systemctl start aitrader-dashboard
sleep 2

if systemctl is-active --quiet aitrader-dashboard; then
    EXTERNAL_IP=$(curl -s ifconfig.me)
    echo "Dashboard is running at http://${EXTERNAL_IP}:5000"
    echo "View logs: journalctl -u aitrader-dashboard -f"
else
    echo "Dashboard failed to start."
    echo "Check logs: journalctl -u aitrader-dashboard -n 50"
    exit 1
fi
