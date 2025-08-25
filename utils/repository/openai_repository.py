import json
import os

from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from typing import Optional, List

from utils.models.question_model import Question

from utils.repository.question_repository import QuestionRepository

class OpenAIRepository:
    def __init__(self, model: str = "gpt-5"):
        # Opción A: busca .env hacia arriba automáticamente
        load_dotenv(find_dotenv())

        # Opción B: exactamente 2 carpetas arriba del archivo
        # load_dotenv(Path(__file__).resolve().parents[2] / ".env")

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key)
        self.main_model = model

    def generate_text(self, system: str,  prompt: str, model: str = None, effort: str = "low") -> str:
        """Generates text using OpenAI's API."""
        if model is None:
            model = self.main_model
        response = self.client.responses.create(
            model="gpt-5",
            reasoning={"effort": effort},
            instructions=system,
            input=prompt,
        )
        return response.output_text

    async def generate_questions(
        self,
        topic: int,
        model: str,
        prompt: str,
        academy: int,
        has4questions: bool,
        num_of_q: int,
        context: str,
    ) -> list[Question] | str:
    
        # self.agent_repo = AgentRepository(context=context)
        self.question_repo = QuestionRepository()


        self.result = await self.question_repo.generate_questions_with_feedback(
            topic=topic,
            prompt=prompt,
            academy=academy,
            has4questions=has4questions,
            num_of_q=num_of_q,
            llm_model=model,
            context=context
        )

        return self.result