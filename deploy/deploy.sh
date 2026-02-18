#!/bin/bash
# =============================================================================
# AITrader - Server-Side Deploy Script
# Idempotent: safe to run multiple times
# Called by GitHub Actions or manually via SSH
# =============================================================================
set -euo pipefail

APP_DIR="$HOME/aitrader"
VENV_DIR="$APP_DIR/venv"

echo "=============================="
echo "AITrader Deploy - $(date)"
echo "=============================="

# 1. Pull latest code
echo "[1/5] Pulling latest code..."
cd "$APP_DIR"
git fetch origin main
git reset --hard origin/main

# 2. Install/update dependencies
echo "[2/5] Installing dependencies..."
source "$VENV_DIR/bin/activate"
pip install -q -r requirements.txt

# 3. Install/update systemd service files
echo "[3/5] Updating systemd services..."
CURRENT_USER=$(whoami)
for svc in aitrader.service aitrader-dashboard.service; do
    sed "s/%USER%/$CURRENT_USER/g" "deploy/$svc" | sudo tee "/etc/systemd/system/$svc" > /dev/null
done
sudo systemctl daemon-reload

# 4. Restart services
echo "[4/5] Restarting services..."
sudo systemctl restart aitrader aitrader-dashboard

# 5. Health check
echo "[5/5] Health check..."
sleep 5

FAILED=0
for svc in aitrader aitrader-dashboard; do
    if systemctl is-active --quiet "$svc"; then
        echo "  $svc: RUNNING"
    else
        echo "  $svc: FAILED"
        journalctl -u "$svc" --no-pager -n 20
        FAILED=1
    fi
done

if [ "$FAILED" -eq 1 ]; then
    echo "DEPLOY FAILED - one or more services did not start"
    exit 1
fi

echo "=============================="
echo "Deploy complete!"
echo "=============================="
