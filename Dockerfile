FROM python:3.11-slim

LABEL maintainer="pablo@example.com"
LABEL description="LLM RAG FastAPI Application"

# Environment variables for Python
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

# Environment variables for the application
ENV PORT=8080
ENV HOST=0.0.0.0
ENV LOG_LEVEL=info

EXPOSE $PORT

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start command with longer timeout and more workers for production
CMD uvicorn main:app --host $HOST --port $PORT --log-level $LOG_LEVEL --timeout-keep-alive 300 --timeout-graceful-shutdown 300