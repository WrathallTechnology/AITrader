#!/bin/bash
# =============================================================================
# Check AITrader status
# =============================================================================

echo "=============================================="
echo "AITrader Status"
echo "=============================================="

echo ""
echo "Services:"
for svc in aitrader aitrader-dashboard; do
    STATUS=$(systemctl is-active "$svc" 2>/dev/null || true)
    echo "  $svc: $STATUS"
done

echo ""
echo "System Resources:"
echo "  Memory: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
echo "  Disk: $(df -h ~ | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"

echo ""
echo "Recent trader log:"
journalctl -u aitrader --no-pager -n 5 2>/dev/null || echo "  No logs available"
echo ""
