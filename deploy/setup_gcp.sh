#!/bin/bash
# =============================================================================
# AITrader - Google Cloud Platform Setup Script
# Run this on a fresh Ubuntu 22.04 e2-micro VM
# =============================================================================

set -e  # Exit on error

echo "=============================================="
echo "AITrader GCP Setup"
echo "=============================================="

# Update system
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install dependencies
echo "Installing Python and dependencies..."
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    htop \
    nano

# Set up swap space (essential for 1GB e2-micro)
if [ ! -f /swapfile ]; then
    echo "Setting up swap space..."
    sudo fallocate -l 1G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# Clone repo if not already present
if [ ! -d ~/aitrader/.git ]; then
    echo "Cloning repository..."
    git clone https://github.com/WrathallTechnology/AITrader.git ~/aitrader
fi

cd ~/aitrader

# Create virtual environment if it doesn't exist
if [ ! -d venv ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Install systemd service files
echo "Installing systemd services..."
CURRENT_USER=$(whoami)
for svc in aitrader.service aitrader-dashboard.service; do
    sed "s/%USER%/$CURRENT_USER/g" "deploy/$svc" | sudo tee "/etc/systemd/system/$svc" > /dev/null
done
sudo systemctl daemon-reload
sudo systemctl enable aitrader aitrader-dashboard

# Configure passwordless sudo for deploy commands
echo "Configuring sudo for deploy..."
SUDOERS_LINE="$CURRENT_USER ALL=(ALL) NOPASSWD: /usr/bin/tee /etc/systemd/system/aitrader.service, /usr/bin/tee /etc/systemd/system/aitrader-dashboard.service, /bin/systemctl"
echo "$SUDOERS_LINE" | sudo tee "/etc/sudoers.d/aitrader" > /dev/null
sudo chmod 440 /etc/sudoers.d/aitrader

echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Create ~/aitrader/.env with your API keys (see .env.example)"
echo "2. Start services:  sudo systemctl start aitrader aitrader-dashboard"
echo "3. Check status:    ./deploy/status.sh"
echo "4. View logs:       journalctl -u aitrader -f"
echo ""
echo "For CI/CD, add these GitHub Secrets (Settings > Secrets > Actions):"
echo "  GCE_HOST     - This VM's external IP (consider reserving a static IP)"
echo "  GCE_USER     - $CURRENT_USER"
echo "  GCE_SSH_KEY  - Private key for SSH access"
echo ""
echo "To generate an SSH key for GitHub Actions:"
echo "  ssh-keygen -t ed25519 -f ~/.ssh/deploy_key -N ''"
echo "  cat ~/.ssh/deploy_key.pub >> ~/.ssh/authorized_keys"
echo "  cat ~/.ssh/deploy_key  # Copy this as GCE_SSH_KEY secret"
echo ""
