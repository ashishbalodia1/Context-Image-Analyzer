#!/usr/bin/env bash
# Exit on any error
set -o errexit

echo "===> Upgrading pip..."
pip install --upgrade pip

echo "===> Installing CPU-optimized PyTorch (saves ~2GB memory/disk on Render)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "===> Installing application requirements..."
pip install -r requirements.txt

echo "===> Ensuring upload directories exist..."
mkdir -p data/uploads/annotated

echo "===> Render build completed successfully!"
