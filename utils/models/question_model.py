from datetime import datetime
from typing import Optional, Union, List
from pydantic import BaseModel, Field

class Question(BaseModel):
    id: Optional[int] = None
    academy: int = 1
    question: str = ""
    answer1: str = ""
    answer2: str = ""
    answer3: str = ""
    answer4: Optional[str] = None
    solution: int = 1
    tip: Optional[str] = None
    topic: int = 1
    createdAt: Optional[datetime] = Field(None, alias="created_at")
    # retro_text: str = ""
    order: Optional[int] = 0
    question_prompt: str = ""
    llm_model: Optional[str] = None

    class Config:
        populate_by_name = True
        json_encoders = {
            datetime: lambda v: v.isoformat() if v else None
        }

    # Métodos auxiliares
    def to_json(self) -> dict:
        return {
            "id": self.id,
            "question": self.question,
            "correct_answer": [self.answer1, self.answer2, self.answer3, self.answer4][self.solution - 1],
            "solution": self.solution,
            "tip": self.tip,
            "category": self.category,
            "publised": self.publised,
            "num_answered": self.num_answered,
            "num_fails": self.num_fails,
            "num_empty": self.num_empty,
            "difficult_rate": self.difficult_rate,
            "challenge_by_tutor": self.challenge_by_tutor,
            "difficult_unique_rate": self.difficult_unique_rate,
            "topic_name": self.topic_name,
            "tema": self.tema,
            
        }
    
    def to_json_without_id(self) -> dict:
        """Devuelve un diccionario JSON con todos los atributos del modelo excepto 'id'"""
        return {
            "academy": self.academy,
            "question": self.question,
            "answer1": self.answer1,
            "answer2": self.answer2,
            "answer3": self.answer3,
            "answer4": self.answer4,
            "solution": self.solution,
            
            "tip": self.tip,
            "topic": self.topic,
            "createdAt": self.createdAt.isoformat() if self.createdAt else None,
            "question_prompt": self.question_prompt,
            "llm_model": self.llm_model,
            "order": self.order,
            "by_llm": True
        }

    def to_json_with_topic_info(self) -> dict:
        data = self.to_json()
        data.update({
            "topic_id": self.topic,
            "topic_name": self.topic_name,
            "tema_numero": self.tema
        })
        return data

    def to_db_dict(self) -> dict:
        exclude_fields = {"topic_name", "tema", "difficult_rate"}
        return self.model_dump(exclude_none=True, exclude=exclude_fields, by_alias=True)

    def copy_with(self, **kwargs) -> "Question":
        return self.model_copy(update=kwargs)

    def get_text_to_embedding(self) -> str:
        answers = [self.answer1, self.answer2, self.answer3, self.answer4]
        correct_answer = answers[self.solution - 1] if 0 < self.solution <= len(answers) else ""
        return f"{self.question} {correct_answer}".strip()

    def get_topic_info(self) -> dict:
        return {
            "topic_id": self.topic,
            "topic_name": self.topic_name,
            "tema_numero": self.tema
        }

    def is_valid_tema_range(self) -> bool:
        return self.tema is not None and 1 <= self.tema <= 45

    def to_json_vector(self) -> dict:
        return {
            "question": self.question,
            "vector": self.vector,
            "embedding_model": self.embedding_model,
            "tema": self.tema,
            "topic_name": self.topic_name,
            "category": self.category
        }

    def get_vector_category(self) -> dict[str, Union[List[float], None, int]]:
        return {
            "vector": self.vector,
            "category": self.category,
            "tema": self.tema,
            "topic_name": self.topic_name
        }


class QuestionList(BaseModel):
    """Modelo para la respuesta del LLM que contiene una lista de preguntas"""
    questions: List[Question]
