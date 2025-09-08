FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system libs for TF/Flask + image ops (OpenCV not required here)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt gunicorn

# Copy app and assets (ensure these files exist locally!)
COPY app.py .
COPY model.h5 .
COPY class_names.txt .
# If you have templates and static assets:
COPY templates ./templates
COPY static ./static

# Cloud Run uses PORT
ENV PORT=8080

# Start with gunicorn (more robust than python app.py)
CMD exec gunicorn -b :$PORT app:app --timeout 120 --workers 1 --threads 4
