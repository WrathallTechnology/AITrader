#!/bin/bash
# =============================================================================
# Start AITrader Dashboard
# =============================================================================

cd ~/aitrader
source venv/bin/activate

# Check if already running
if screen -list | grep -q "dashboard"; then
    echo "Dashboard is already running!"
    echo "Access it at http://$(curl -s ifconfig.me):5000"
    echo ""
    echo "Use 'screen -r dashboard' to attach"
    echo "Or './stop_dashboard.sh' to stop it first"
    exit 1
fi

# Start in screen session
echo "Starting AITrader Dashboard..."
screen -dmS dashboard python dashboard/app.py

# Get external IP
EXTERNAL_IP=$(curl -s ifconfig.me)

echo "=============================================="
echo "Dashboard started!"
echo "=============================================="
echo ""
echo "Access the dashboard at:"
echo "  http://${EXTERNAL_IP}:5000"
echo ""
echo "Commands:"
echo "  screen -r dashboard  - View server output"
echo "  Ctrl+A then D        - Detach from screen"
echo "  ./stop_dashboard.sh  - Stop the dashboard"
echo ""
