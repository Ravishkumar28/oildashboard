FROM python:3.12-slim

# Build tag: 2026-06-10-working-effect-flag
# the image instead of just restarting the existing container).
# system tools:
#   curl     — healthchecks
#   libgomp1 — OpenMP runtime required by lightgbm's C++ shared library;
#              without it, the lightgbm import fails silently at runtime
#              even though pip install succeeds
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "import lightgbm; print('lightgbm ' + lightgbm.__version__ + ' imported OK')"

# app code
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Hugging Face Spaces requires non-root execution
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# HF Spaces convention: the app listens on 7860 (must match README app_port)
EXPOSE 7860

# main.py uses sibling-module imports (import config, datafeed, etc.) so
# we run uvicorn with cwd = /app/backend
WORKDIR /app/backend
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
