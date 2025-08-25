import os
from fastapi import FastAPI
from routes import api

# NO cargar dotenv en Cloud Run por ahora
# from dotenv import load_dotenv
# load_dotenv()

app = FastAPI(
    title="LLM RAG FastAPI",
    description="API para procesamiento de embeddings con OpenAI",
    version="1.0.0"
)

app.include_router(api.router)