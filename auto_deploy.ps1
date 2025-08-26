# Autodeploy.ps1

$ErrorActionPreference = "Stop"

# Config del proyecto
$PROJECT_ID   = "opn-test-457508"
$IMAGE_NAME   = "llmendpoint"
$REGION       = "us-central1"
$SERVICE_NAME = "llmendpoint"

# Cargar .env (ignora líneas vacías o con #)
$envVars = @{}
Get-Content .env | ForEach-Object {
  if ($_ -match '^\s*$' -or $_ -match '^\s*#') { return }
  $k,$v = $_ -split '=', 2
  $envVars[$k] = $v
}

# Vars de entorno
$OPENAI_API_KEY = $envVars.OPENAI_API_KEY
$JWT_SIGNATURE  = $envVars.JWT_SIGNATURE
$SUPABASE_URL   = $envVars.SUPABASE_URL
$SUPABASE_KEY   = $envVars.SUPABASE_KEY

# Config del modelo
$DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
$EXPECTED_DIMENSIONS     = "3072"

# App / logging
$APP_HOST  = "0.0.0.0"   # Usa APP_HOST si tu código lo lee; no usar HOST


# Rendimiento
$CACHE_SIZE      = "1000"
$TIMEOUT_SECONDS = "1200"

Write-Host "Estableciendo proyecto: $PROJECT_ID"
gcloud config set project $PROJECT_ID

Write-Host "Compilando y subiendo imagen..."
gcloud builds submit --tag "gcr.io/$PROJECT_ID/$IMAGE_NAME"

Write-Host "Desplegando en Cloud Run..."
gcloud run deploy $SERVICE_NAME `
  --image "gcr.io/$PROJECT_ID/$IMAGE_NAME" `
  --platform managed `
  --region $REGION `
  --allow-unauthenticated `
  --timeout 900 `
  --memory 2Gi `
  --cpu 2 `
  --max-instances 10 `
  --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY,JWT_SIGNATURE=$JWT_SIGNATURE,SUPABASE_URL=$SUPABASE_URL,SUPABASE_KEY=$SUPABASE_KEY,DEFAULT_EMBEDDING_MODEL=$DEFAULT_EMBEDDING_MODEL,EXPECTED_DIMENSIONS=$EXPECTED_DIMENSIONS,APP_HOST=$APP_HOST,CACHE_SIZE=$CACHE_SIZE,TIMEOUT_SECONDS=$TIMEOUT_SECONDS"

Write-Host "Despliegue completado."
Write-Host "URL del servicio:"
gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)"
