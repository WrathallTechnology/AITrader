#!/bin/bash
# =============================================================================
# Start AITrader in a screen session
# =============================================================================

cd ~/aitrader
source venv/bin/activate

# Check if already running
if screen -list | grep -q "trader"; then
    echo "AITrader is already running!"
    echo "Use 'screen -r trader' to attach"
    echo "Or './stop_trader.sh' to stop it first"
    exit 1
fi

# Start in screen session
echo "Starting AITrader..."
screen -dmS trader python main.py --mode all

echo "=============================================="
echo "AITrader started in background!"
echo "=============================================="
echo ""
echo "Commands:"
echo "  screen -r trader     - View live output"
echo "  Ctrl+A then D        - Detach from screen"
echo "  ./stop_trader.sh     - Stop the trader"
echo "  ./logs.sh            - View recent logs"
echo ""
