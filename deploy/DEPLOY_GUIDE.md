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
   - **Firewall:** Check "Allow HTTP traffic" (for dashboard)
5. Click **Create**
6. **Recommended:** Reserve a static external IP (VPC Network → IP Addresses → Reserve) so the IP doesn't change on VM stop/start.

## Step 2: Connect to VM

Click **SSH** button next to your VM in the console, or use:
```bash
gcloud compute ssh aitrader --zone=us-central1-a
```

## Step 3: Run Setup Script

The setup script clones the repo, installs dependencies, sets up systemd services, and configures swap space:

```bash
# If the repo isn't cloned yet, clone it first:
git clone https://github.com/WrathallTechnology/AITrader.git ~/aitrader

cd ~/aitrader/deploy
chmod +x *.sh
./setup_gcp.sh
```

## Step 4: Configure Environment

```bash
cp ~/aitrader/.env.example ~/aitrader/.env
nano ~/aitrader/.env
```

Add your API keys:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
INITIAL_CAPITAL=1000
LOG_LEVEL=INFO
```

Save: `Ctrl+O`, Enter, `Ctrl+X`

## Step 5: Train the Model

```bash
cd ~/aitrader
source venv/bin/activate
python train_model.py --all
```

## Step 6: Start Services

```bash
sudo systemctl start aitrader aitrader-dashboard
```

Verify they're running:
```bash
./deploy/status.sh
```

## CI/CD Setup (Push-to-Deploy)

Once the VM is set up, configure GitHub Actions for automatic deploys on push to `main`.

### Generate SSH Key

On the VM:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/deploy_key -N ""
cat ~/.ssh/deploy_key.pub >> ~/.ssh/authorized_keys
cat ~/.ssh/deploy_key  # Copy this output
```

### Add GitHub Secrets

Go to your repo's **Settings → Secrets and variables → Actions** and add:

| Secret | Value |
|--------|-------|
| `GCE_HOST` | Your VM's external IP address |
| `GCE_USER` | Your SSH username on the VM |
| `GCE_SSH_KEY` | Contents of `~/.ssh/deploy_key` (the private key) |

### How It Works

Every push to `main` triggers the GitHub Actions workflow (`.github/workflows/deploy.yml`) which:
1. SSHes into your GCE instance
2. Runs `deploy/deploy.sh` which pulls code, installs deps, and restarts services
3. Runs a health check to verify both services started

### Manual Deploy

You can also trigger a deploy manually by SSHing in:
```bash
bash ~/aitrader/deploy/deploy.sh
```

## Daily Commands

| Command | Description |
|---------|-------------|
| `./deploy/start_trader.sh` | Start the trading bot |
| `./deploy/stop_trader.sh` | Stop the trading bot |
| `./deploy/start_dashboard.sh` | Start the web dashboard |
| `./deploy/stop_dashboard.sh` | Stop the web dashboard |
| `./deploy/status.sh` | Check service status |
| `./deploy/logs.sh` | View last 50 log lines |
| `./deploy/logs.sh -f` | Follow logs live |
| `./deploy/logs.sh dashboard` | View dashboard logs |

### systemctl Commands

```bash
# Service management
sudo systemctl start aitrader
sudo systemctl stop aitrader
sudo systemctl restart aitrader
sudo systemctl status aitrader

# View logs
journalctl -u aitrader -f          # Follow live
journalctl -u aitrader -n 100      # Last 100 lines
journalctl -u aitrader --since today
```

## Monitoring

**View live trading logs:**
```bash
journalctl -u aitrader -f
```

**Check system resources:**
```bash
htop
```

## Troubleshooting

**Bot stopped unexpectedly:**
```bash
# Check service status and recent logs
sudo systemctl status aitrader
journalctl -u aitrader -n 100

# Restart
sudo systemctl restart aitrader
```

**Service keeps crashing (restart loop):**

systemd stops restarting after 5 failures in 5 minutes. Check the logs, fix the issue, then:
```bash
sudo systemctl reset-failed aitrader
sudo systemctl start aitrader
```

**Out of memory:**

The setup script adds 1GB swap automatically. If you still see issues:
```bash
free -h
journalctl -u aitrader --since "1 hour ago" | grep -i memory
```

**VM restarted:**

Services are enabled via systemd and start automatically on boot. No crontab needed.

## Costs

With the free tier, you should pay $0/month if you:
- Use `e2-micro` instance
- Use `us-central1`, `us-west1`, or `us-east1` region
- Stay under 30GB disk
- Stay under 1GB egress/month

Monitor your billing at: https://console.cloud.google.com/billing
