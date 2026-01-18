# AITrader - Google Cloud Deployment Guide

## Step 1: Create the VM

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **Compute Engine** → **VM Instances**
3. Click **Create Instance**
4. Configure:
   - **Name:** `aitrader`
   - **Region:** `us-central1` (Iowa) - FREE TIER
   - **Zone:** `us-central1-a`
   - **Machine type:** `e2-micro` (2 vCPU, 1 GB memory) - FREE TIER
   - **Boot disk:** Click "Change"
     - Operating system: Ubuntu
     - Version: Ubuntu 22.04 LTS
     - Size: 30 GB (standard persistent disk) - FREE TIER
   - **Firewall:** Check "Allow HTTP traffic" (optional)
5. Click **Create**

## Step 2: Connect to VM

Click **SSH** button next to your VM in the console, or use:
```bash
gcloud compute ssh aitrader --zone=us-central1-a
```

## Step 3: Upload Your Code

**Option A: Using Git (recommended)**
```bash
# On the VM
git clone https://github.com/YOUR_USERNAME/AITrader.git ~/aitrader
```

**Option B: Using SCP from your local machine**
```bash
# From your local Windows machine (PowerShell)
gcloud compute scp --recurse "D:\WrathallTechnologies\AITrader\*" aitrader:~/aitrader/ --zone=us-central1-a
```

**Option C: Using the Cloud Console**
1. Click the gear icon in the SSH window
2. Select "Upload file"
3. Upload a zip of your project, then unzip on the VM

## Step 4: Run Setup Script

```bash
cd ~/aitrader/deploy
chmod +x *.sh
./setup_gcp.sh
```

## Step 5: Install Python Packages

```bash
cd ~/aitrader
source venv/bin/activate
pip install -r requirements.txt
```

## Step 6: Configure Environment

```bash
cd ~/aitrader
nano .env
```

Paste your configuration:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
INITIAL_CAPITAL=1000
LOG_LEVEL=INFO
```

Save: `Ctrl+O`, Enter, `Ctrl+X`

## Step 7: Train the Model

```bash
cd ~/aitrader
source venv/bin/activate
python train_model.py --all
```

## Step 8: Start the Bot

```bash
cd ~/aitrader/deploy
./start_trader.sh
```

## Daily Commands

| Command | Description |
|---------|-------------|
| `./start_trader.sh` | Start the trading bot |
| `./stop_trader.sh` | Stop the trading bot |
| `./status.sh` | Check if bot is running |
| `./logs.sh` | View recent log entries |
| `screen -r trader` | Attach to live output |
| `Ctrl+A` then `D` | Detach from screen |

## Monitoring

**View live logs:**
```bash
tail -f ~/aitrader/logs/trading.log
```

**Check system resources:**
```bash
htop
```

## Troubleshooting

**Bot stopped unexpectedly:**
```bash
# Check logs for errors
cat ~/aitrader/logs/trading.log | tail -100

# Restart
cd ~/aitrader/deploy
./start_trader.sh
```

**Out of memory:**
The e2-micro has only 1GB RAM. If you see memory issues:
```bash
# Check memory usage
free -h

# Add swap space
sudo fallocate -l 1G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

**VM restarted (bot stopped):**
Set up auto-start by adding to crontab:
```bash
crontab -e
# Add this line:
@reboot cd ~/aitrader/deploy && ./start_trader.sh
```

## Costs

With the free tier, you should pay $0/month if you:
- Use `e2-micro` instance
- Use `us-central1`, `us-west1`, or `us-east1` region
- Stay under 30GB disk
- Stay under 1GB egress/month

Monitor your billing at: https://console.cloud.google.com/billing
