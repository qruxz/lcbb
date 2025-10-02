# ---------- Base Image ----------
FROM python:3.11-slim

# ---------- Set work directory ----------
WORKDIR /app

# ---------- Install system dependencies ----------
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------- Copy dependencies first (better caching) ----------
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# ---------- Copy app code ----------
COPY . .

# ---------- Expose port ----------
EXPOSE 8000

# ---------- Start server ----------
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "main:app"]
