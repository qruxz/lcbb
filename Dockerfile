# ---------- Base Image ----------
FROM python:3.11-slim

# ---------- Set working directory ----------
WORKDIR /app

# ---------- Install system dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy requirements first (for caching) ----------
COPY requirements.txt .

# ---------- Install Python dependencies ----------
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# ---------- Copy application code ----------
COPY . .

# ---------- Expose port ----------
EXPOSE 8000

# ---------- Start the FastAPI app ----------
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]
