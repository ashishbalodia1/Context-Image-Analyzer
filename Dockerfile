# Lightweight Python 3.11 base image
FROM python:3.11-slim

# Set environment variables for Python & hardware efficiency
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

# Set working directory
WORKDIR /app

# Install minimal OS dependencies for computer vision
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (saves ~2GB memory/disk on Render)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files and trained models
COPY . .

# Ensure upload directories exist
RUN mkdir -p data/uploads/annotated

# Expose Render default port
EXPOSE 10000

# Launch with Gunicorn (1 worker + 4 threads for memory efficiency on free tier)
CMD ["gunicorn", "backend.app:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "4", "--timeout", "120"]

