# 🚀 Render.com Deployment Guide — City Guardian

This repository is pre-configured for **1-Click / Zero-Configuration Deployment** on [Render](https://render.com).

---

## ⚡ Method 1: Automatic Blueprint (Recommended — 2 Minutes)

1. Sign in or Sign up at **[dashboard.render.com](https://dashboard.render.com)** (sign in with your GitHub account: `ashishbalodia1`).
2. Click the **"New +"** button at top right and select **"Blueprint"**.
3. Select your repository: **`ashishbalodia1/Context-Image-Analyzer`**.
4. Render will automatically detect [`render.yaml`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/render.yaml) and configure:
   - **Service Name:** `city-guardian`
   - **Runtime:** `Python 3.11`
   - **Build Command:** `bash render_build.sh`
   - **Start Command:** `gunicorn backend.app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
5. Click **"Apply"**.
6. Render will build and deploy your app. Once deployed, you will get a public live URL (e.g. `https://city-guardian.onrender.com`).

---

## 🐳 Method 2: Docker Web Service (Most Reliable for ML)

If you prefer a 100% containerized deployment:

1. In Render Dashboard, click **"New +"** -> **"Web Service"**.
2. Connect your repository: `ashishbalodia1/Context-Image-Analyzer`.
3. Select **Docker** as the Environment.
4. Render will automatically read the included [`Dockerfile`](file:///c:/Users/ashis/OneDrive/Desktop/ML/ML_PROJECT/ML-Antigravity/Dockerfile).
5. Choose the **Free** instance type.
6. Click **"Create Web Service"**.

---

## 📋 Pre-Configured Optimizations included in this repo:

- **CPU-Optimized PyTorch (`--index-url https://download.pytorch.org/whl/cpu`):** Reduces build size from 2.5 GB to ~200 MB, preventing Render build timeouts and Out-of-Memory (OOM) errors.
- **Headless OpenCV (`opencv-python-headless`):** Completely eliminates `libGL.so.1` missing errors common on headless Linux servers.
- **Gunicorn Workers & Threads:** Configured with 1 worker and 4 threads to keep RAM usage within Render's 512 MB free tier while maintaining high concurrency.
- **All 5 Fine-Tuned YOLOv8 Models:** Pre-bundled in `models/trained/`, requiring zero external download during deployment.
