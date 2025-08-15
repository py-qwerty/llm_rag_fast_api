import os
from typing import Union
from routes.root import read_root as rr
from routes.llm_create import create as cre, get_questions as gener
from utils.models.generate_question_model import GenerateQuestionsRequest
from fastapi import APIRouter

router = APIRouter()

@router.get("/")
def read_root():
    """Endpoint raíz"""
    return rr()

@router.get("/create")
def read_item(prompt:str = '', system:str = '', effort: str = "low", model: str = 'gpt-5-2025-08-07'):
    return cre(system=system, prompt=prompt, model=model, effort=effort)


@router.post("/generate_questions")
# async def question_endpoint(topic: int, academy: int, has4questions: bool, prompt:str = '', context: str = '', num_of_q: int = 1, model: str = 'gpt-5-2025-08-07'):
#     return await gener(topic=topic, academy=academy, has4questions=has4questions, prompt=prompt, num_of_q=num_of_q, model=model, context=context)
async def question_endpoint(req: GenerateQuestionsRequest):
    return await gener(
        topic=req.topic,
        academy=req.academy,
        has4questions=req.has4questions,
        prompt=req.prompt,
        num_of_q=req.num_of_q,
        model=req.llm_model,
        context=req.context
    )

@router.get("/health")
def health_check():
    """Endpoint de salud simple"""
    return {"status": "ok", "port": os.environ.get("PORT", "8080")}

@router.get("/env")
def show_env():
    """Debug: mostrar variables de entorno"""
    return {
        "PORT": os.environ.get("PORT"),
        "HOST": os.environ.get("HOST"),
        "LOG_LEVEL": os.environ.get("LOG_LEVEL"),
        "OPENAI_KEY_SET": "yes" if os.environ.get("OPENAI_API_KEY") else "no"
    }