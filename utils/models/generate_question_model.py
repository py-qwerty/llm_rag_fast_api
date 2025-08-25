from pydantic import BaseModel


class GenerateQuestionsRequest(BaseModel):
    topic: int
    prompt: str
    academy: int
    has4questions: bool = False
    num_of_q: int = 5
    llm_model: str = "gpt-5-2025-08-07"
    max_tokens_per_chunk: int = 100  # valor por defecto,
    context: str = ''  # contexto opcional