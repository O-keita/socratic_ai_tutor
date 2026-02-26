# Backend Deployment Guide

Deploying the Socratic AI Tutor backend on a Linux VPS with Cloudflare handling HTTPS.
No Nginx required — Cloudflare terminates TLS and proxies traffic to your server on port 80.

---

## What You Need

| Item | Notes |
|------|-------|
| Linux VPS | Ubuntu 22.04 recommended. 2+ CPU cores, 4 GB RAM minimum (the model uses ~1.5 GB). |
| Domain name | Any registrar. You'll point its nameservers at Cloudflare. |
| Cloudflare account | Free tier is enough. |
| GGUF model file | `socratic-q4_k_m.gguf` (~350 MB). Not in the repo — see Step 4. |

---

## Step 1 — Set Up the Server

SSH into your server and install Docker.

```bash
# Update packages
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh

# Add your user to the docker group so you don't need sudo every time
sudo usermod -aG docker $USER

# Apply the group change without logging out
newgrp docker

# Verify Docker works
docker run --rm hello-world
```

---

## Step 2 — Clone the Repository

```bash
cd ~
git clone <your-repo-url> socratic_ai_tutor
cd socratic_ai_tutor
```

---

## Step 3 — Create the Environment File

Copy the example and fill in your values.

```bash
cp .env.example .env
nano .env
```

Set every value in `.env`:

```env
# Generate a secret: python3 -c "import secrets; print(secrets.token_hex(32))"
JWT_SECRET_KEY=paste-a-long-random-string-here

# Your Cloudflare domain + the Flutter app URL scheme
CORS_ORIGINS=https://yourdomain.com,app://socratic-tutor

# Admin panel login
ADMIN_EMAIL=admin@yourdomain.com
ADMIN_PASSWORD=a-strong-password-here

# Model settings (defaults are fine for a CPU-only VPS)
MODEL_PATH=/app/models/socratic-q4_k_m.gguf
N_GPU_LAYERS=0
N_THREADS=4
N_CTX=4096
```

> **Never commit `.env` to git.** It is already in `.gitignore`.

---

## Step 4 — Upload the Model File

The GGUF model is not in the repository (~350 MB). You have two options:

**Option A — Upload from your local machine:**

```bash
# Run this on your LOCAL machine, not the server
scp /path/to/socratic-q4_k_m.gguf user@your-server-ip:~/socratic_ai_tutor/models/
```

**Option B — Download directly on the server:**

```bash
mkdir -p ~/socratic_ai_tutor/models
cd ~/socratic_ai_tutor/models

# Download from HuggingFace (replace the URL with your model's actual URL)
wget -O socratic-q4_k_m.gguf "https://huggingface.co/Omar-keita/DSML-Socatic-qwen3-0.6B/resolve/main/Socratic-Qwen3-0.6-Merged-Quality_Data-752M-Q4_K_M%20(1).gguf"
```

Verify the file is there:

```bash
ls -lh ~/socratic_ai_tutor/models/
# Should show: socratic-q4_k_m.gguf  ~350-768 MB
```

---

## Step 5 — Start the Backend

```bash
cd ~/socratic_ai_tutor

# Build the image and start the container in the background
docker compose up -d --build
```

This will take a few minutes the first time (compiling `llama-cpp-python`).

**Check that it started correctly:**

```bash
# Watch live logs (Ctrl+C to stop watching)
docker compose logs -f

# You should see something like:
# INFO:     Started server process
# INFO:     Waiting for application startup.
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:80
```

**Test the health endpoint:**

```bash
curl http://localhost/health
# Expected: {"status":"ok","model_loaded":true}
```

If `model_loaded` is `false`, the model file path is wrong — double-check Step 4.

---

## Step 6 — Configure Cloudflare

### 6a — Point your domain to Cloudflare

1. In your domain registrar, set the nameservers to Cloudflare's (they're shown when you add a site in Cloudflare).
2. In Cloudflare, go to **DNS → Records**.
3. Add an **A record**:

   | Type | Name | Content | Proxy |
   |------|------|---------|-------|
   | A | `@` | `your-server-ip` | Proxied (orange cloud) |

   If you want `api.yourdomain.com` instead of the root domain, set Name to `api`.

### 6b — Set the SSL/TLS mode

Go to **SSL/TLS → Overview** and select **Flexible**.

> **Why Flexible?** Your server runs plain HTTP on port 80. Cloudflare terminates HTTPS on its edge and forwards HTTP to your origin. Users always see HTTPS in their browser — Cloudflare handles the certificate automatically.

### 6c — Force HTTPS (optional but recommended)

Go to **SSL/TLS → Edge Certificates** and turn on **Always Use HTTPS**.
This redirects any `http://` visitor to `https://` automatically.

---

## Step 7 — Update the Flutter App

Open `socratic_app/lib/utils/app_config.dart` and set the backend URL to your domain:

```dart
static const String backendUrl = 'https://yourdomain.com';
```

Then rebuild and redistribute the app.

---

## Ongoing Operations

### View logs

```bash
docker compose logs -f
```

### Restart the backend

```bash
docker compose restart
```

### Deploy a code update

```bash
cd ~/socratic_ai_tutor
git pull
docker compose up -d --build
```

### Stop the backend

```bash
docker compose down
```

### Check container health

```bash
docker compose ps
# STATUS column should show "healthy" after ~60 seconds
```

---

## Persistent Data

The `docker-compose.yml` mounts three directories from the host into the container.
Your data survives container rebuilds and updates automatically.

| Host path | Container path | Contains |
|-----------|---------------|----------|
| `./models/` | `/app/models/` | GGUF model file |
| `./data/` | `/app/data/` | `users.json` (user accounts) |
| `./content/` | `/app/content/` | Admin-uploaded courses |

**Back these up regularly:**

```bash
# Simple backup to a tar archive
tar -czf backup-$(date +%Y%m%d).tar.gz ~/socratic_ai_tutor/data ~/socratic_ai_tutor/content
```

---

## Firewall (Recommended)

Allow only Cloudflare's IP ranges on port 80 so users cannot bypass Cloudflare and hit your server directly over plain HTTP.

```bash
# Install ufw if not present
sudo apt install -y ufw

# Default: deny all incoming, allow all outgoing
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Always allow SSH (do this BEFORE enabling ufw)
sudo ufw allow ssh

# Allow Cloudflare IPv4 ranges on port 80
# Full list: https://www.cloudflare.com/ips-v4
for ip in \
  173.245.48.0/20 \
  103.21.244.0/22 \
  103.22.200.0/22 \
  103.31.4.0/22 \
  141.101.64.0/18 \
  108.162.192.0/18 \
  190.93.240.0/20 \
  188.114.96.0/20 \
  197.234.240.0/22 \
  198.41.128.0/17 \
  162.158.0.0/15 \
  104.16.0.0/13 \
  104.24.0.0/14 \
  172.64.0.0/13 \
  131.0.72.0/22; do
  sudo ufw allow from $ip to any port 80
done

sudo ufw enable
sudo ufw status
```

---

## Troubleshooting

**Container exits immediately after starting:**

```bash
docker compose logs backend
```
Look for `RuntimeError: Missing environment variables`. Open `.env` and make sure `JWT_SECRET_KEY`, `CORS_ORIGINS`, `ADMIN_EMAIL`, and `ADMIN_PASSWORD` are all set.

---

**`model_loaded: false` in `/health`:**

The model file is missing or has the wrong name. Check:

```bash
ls -lh ~/socratic_ai_tutor/models/
# File must be named exactly: socratic-q4_k_m.gguf
```

---

**502 Bad Gateway from Cloudflare:**

The container is not running or not listening on port 80.

```bash
docker compose ps          # is the container up?
curl http://localhost/health  # does it respond locally?
docker compose logs backend   # any crash messages?
```

---

**CORS error in the Flutter app:**

Your `CORS_ORIGINS` in `.env` does not match the origin the app is sending.
Make sure it includes `https://yourdomain.com` (no trailing slash) and the Flutter app scheme.

```bash
# Restart after editing .env
docker compose up -d
```

---

**Out of memory — container killed:**

The model requires ~1.5 GB of RAM. If your VPS has less than 4 GB, add swap:

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

---

## Admin Panel

Once the backend is running, access the admin panel at:

```
https://yourdomain.com/admin
```

Log in with the `ADMIN_EMAIL` and `ADMIN_PASSWORD` you set in `.env`.
Use it to upload courses, manage content, and view registered users.
