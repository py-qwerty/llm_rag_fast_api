FROM python:3.11-slim

# Metadatos
LABEL maintainer="pablo@example.com"
LABEL description="LLM RAG FastAPI Application"

# Configurar entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir python-dotenv

# Copiar archivos de aplicación
COPY main.py .
COPY start.sh .

# Hacer ejecutable el script
RUN chmod +x start.sh

# Crear usuario no-root
RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Variables de entorno por defecto
ENV PORT=8080
ENV HOST=0.0.0.0
ENV LOG_LEVEL=info

# Exponer puerto
EXPOSE $PORT

# REMOVER health check por ahora para debugging
# HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
#     CMD curl -f http://localhost:$PORT/health || exit 1

# Comando de inicio simplificado para debugging
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]