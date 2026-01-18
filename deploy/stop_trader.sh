#!/bin/bash
# =============================================================================
# Stop AITrader gracefully
# =============================================================================

if screen -list | grep -q "trader"; then
    echo "Stopping AITrader..."
    screen -S trader -X stuff $'\003'  # Send Ctrl+C
    sleep 3

    # Force kill if still running
    if screen -list | grep -q "trader"; then
        screen -S trader -X quit
    fi

    echo "AITrader stopped."
else
    echo "AITrader is not running."
fi
