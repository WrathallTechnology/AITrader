#!/bin/bash
# =============================================================================
# View AITrader logs from journald
# Usage: ./logs.sh              (last 50 lines of trader)
#        ./logs.sh dashboard    (last 50 lines of dashboard)
#        ./logs.sh -f           (follow trader logs live)
#        ./logs.sh dashboard -f (follow dashboard logs live)
# =============================================================================

SERVICE="aitrader"
ARGS="-n 50 --no-pager"

if [ "$1" = "dashboard" ]; then
    SERVICE="aitrader-dashboard"
    shift
fi

if [ "$1" = "-f" ] || [ "$1" = "--follow" ]; then
    ARGS="-f"
fi

journalctl -u "$SERVICE" $ARGS
