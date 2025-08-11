# Configuración del proyecto
$PROJECT_ID = "opn-test-457508"
$IMAGE_NAME = "llmendpoint"
$REGION = "us-central1"
$SERVICE_NAME = "llmendpoint"

# Variables de entorno para la aplicación
$envVars = Get-Content .env | ConvertFrom-StringData
$OPENAI_API_KEY = $envVars.OPENAI_API_KEY

# Configuración del modelo
$DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
$EXPECTED_DIMENSIONS = "3072"

# Configuración de la API - CORREGIDO: No usar $HOST
$APP_HOST = "0.0.0.0"  # Usar variable diferente
$LOG_LEVEL = "INFO"

# Configuración de rendimiento
$CACHE_SIZE = "1000"
$TIMEOUT_SECONDS = "1200"

Write-Host "Estableciendo proyecto: $PROJECT_ID"
gcloud config set project $PROJECT_ID

Write-Host "Compilando y subiendo imagen a Google Container Registry..."
gcloud builds submit --tag "gcr.io/$PROJECT_ID/$IMAGE_NAME"

Write-Host "Desplegando en Cloud Run..."
gcloud run deploy $SERVICE_NAME `
  --image "gcr.io/$PROJECT_ID/$IMAGE_NAME" `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --port 8080 `
  --timeout 600 `
  --memory 1Gi `
  --cpu 1 `
  --max-instances 10 `
  --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY,DEFAULT_EMBEDDING_MODEL=$DEFAULT_EMBEDDING_MODEL,EXPECTED_DIMENSIONS=$EXPECTED_DIMENSIONS,HOST=$APP_HOST,LOG_LEVEL=$LOG_LEVEL,CACHE_SIZE=$CACHE_SIZE,TIMEOUT_SECONDS=$TIMEOUT_SECONDS"

Write-Host "Despliegue completado."
Write-Host "Obteniendo URL del servicio..."
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"