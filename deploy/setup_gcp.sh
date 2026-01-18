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
    screen \
    htop \
    nano

# Create app directory
echo "Setting up application directory..."
mkdir -p ~/aitrader
cd ~/aitrader

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

echo "=============================================="
echo "Base setup complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Upload your code to ~/aitrader/"
echo "2. Run: source venv/bin/activate"
echo "3. Run: pip install -r requirements.txt"
echo "4. Create .env file with your API keys"
echo "5. Run: python train_model.py"
echo "6. Run: ./start_trader.sh"
echo ""
