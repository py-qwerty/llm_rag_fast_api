import os
from fastapi import FastAPI
from typing import Union
from routes.root import read_root as rr
from routes.llm_create import create as cre, get_questions as gener

# NO cargar dotenv en Cloud Run por ahora
# from dotenv import load_dotenv
# load_dotenv()

app = FastAPI(
    title="LLM RAG FastAPI",
    description="API para procesamiento de embeddings con OpenAI",
    version="1.0.0"
)

@app.get("/")
def read_root():
    """Endpoint raíz"""
    return rr()

@app.get("/create")
def read_item(prompt:str = '', system:str = '', effort: str = "low", model: str = 'gpt-5-2025-08-07'):
    return cre(system=system, prompt=prompt, model=model, effort=effort)


@app.get("/generate_questions")
async def question_endpoint(topic: int, academy: int, has4questions: bool, prompt:str = '', context: str = '', num_of_q: int = 1, model: str = 'gpt-5-2025-08-07'):
    return await gener(topic=topic, academy=academy, has4questions=has4questions, prompt=prompt, num_of_q=num_of_q, model=model, context=context)

@app.get("/health")
def health_check():
    """Endpoint de salud simple"""
    return {"status": "ok", "port": os.environ.get("PORT", "8080")}

@app.get("/env")
def show_env():
    """Debug: mostrar variables de entorno"""
    return {
        "PORT": os.environ.get("PORT"),
        "HOST": os.environ.get("HOST"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL"),
        "OPENAI_KEY_SET": "yes" if os.environ.get("OPENAI_API_KEY") else "no"
    }