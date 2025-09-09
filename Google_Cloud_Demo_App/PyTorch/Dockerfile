# Start from Python slim (Debian Bookworm)
FROM python:3.10-slim

# Keep image lean and stdout unbuffered
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    CUDA_VISIBLE_DEVICES=-1

# System deps (Bookworm: use libgl1, add libgomp1 for PyTorch CPU)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Workdir
WORKDIR /app

# Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# App files (make sure model.pth, class_names.txt, templates/ exist)
COPY . .

# Cloud Run expects the app to listen on $PORT (defaults to 8080)
ENV PORT=8080
EXPOSE 8080

# Serve via gunicorn (single worker is fine; model loaded once)
CMD exec gunicorn -b :$PORT app:app --timeout 120 --workers 1 --threads 2
