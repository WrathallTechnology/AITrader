#!/bin/bash
# =============================================================================
# Stop AITrader Dashboard
# =============================================================================

if screen -list | grep -q "dashboard"; then
    echo "Stopping Dashboard..."
    screen -S dashboard -X stuff $'\003'  # Send Ctrl+C
    sleep 2

    # Force kill if still running
    if screen -list | grep -q "dashboard"; then
        screen -S dashboard -X quit
    fi

    echo "Dashboard stopped."
else
    echo "Dashboard is not running."
fi
