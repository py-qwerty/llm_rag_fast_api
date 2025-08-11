#!/bin/bash

# Configuración de colores para logs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Función para logs con colores
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner de inicio
echo "=================================="
echo "🚀 Starting LLM RAG FastAPI Server"
echo "=================================="

# Obtener configuración de entorno
PORT=${PORT:-8080}
HOST=${HOST:-0.0.0.0}
LOG_LEVEL=${LOG_LEVEL:-info}
WORKERS=${WORKERS:-1}

log_info "Port: $PORT"
log_info "Host: $HOST"
log_info "Log Level: $LOG_LEVEL"
log_info "Workers: $WORKERS"

# Verificar variables críticas
if [ -z "$OPENAI_API_KEY" ]; then
    log_warning "OPENAI_API_KEY no está configurada"
else
    log_success "OPENAI_API_KEY configurada ✅"
fi

# Verificar que el archivo main.py existe
if [ ! -f "main.py" ]; then
    log_error "main.py no encontrado"
    exit 1
fi

log_success "main.py encontrado ✅"

# Mostrar información de Python y paquetes
log_info "Python version: $(python --version)"
log_info "FastAPI instalado: $(python -c 'import fastapi; print(fastapi.__version__)' 2>/dev/null || echo 'No instalado')"
log_info "Uvicorn instalado: $(python -c 'import uvicorn; print(uvicorn.__version__)' 2>/dev/null || echo 'No instalado')"

# Iniciar el servidor
log_info "Iniciando servidor uvicorn..."
log_success "🌟 Servidor disponible en: http://$HOST:$PORT"
log_info "📚 Documentación en: http://$HOST:$PORT/docs"
log_info "🔍 Health check en: http://$HOST:$PORT/health"

# Ejecutar uvicorn con la configuración
exec uvicorn main:app \
    --host $HOST \
    --port $PORT \
    --log-level $LOG_LEVEL \
    --workers $WORKERS \
    --access-log \
    --loop asyncio