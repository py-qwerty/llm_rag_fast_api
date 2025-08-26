import json
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
from typing import Optional, List

from utils.models.question_model import Question


class OpenAIRepository:
    def __init__(self, model: str = "gpt-5"):
        # Only load .env if it exists (for local development)
        if os.path.exists('.env'):
            load_dotenv(find_dotenv())

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        self.client = OpenAI(api_key=api_key)
        self.main_model = model

    def generate_text(self, system: str, prompt: str, model: str = None, effort: str = "low") -> str:
        """Generates text using OpenAI's API."""
        if model is None:
            model = self.main_model

        try:
            response = self.client.responses.create(
                model="gpt-5",
                reasoning={"effort": effort},
                instructions=system,
                input=prompt,
            )
            return response.output_text
        except Exception as e:
            print(f"Error in generate_text: {e}")
            # Fallback to regular chat completion if responses API fails
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content

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
        try:
            # Import here to avoid circular imports and startup delays
            from utils.repository.question_repository import QuestionRepository

            question_repo = QuestionRepository()

            result = await question_repo.generate_questions_with_feedback(
                topic=topic,
                prompt=prompt,
                academy=academy,
                has4questions=has4questions,
                num_of_q=num_of_q,
                llm_model=model,
                context=context
            )

            return result
        except Exception as e:
            print(f"Error in generate_questions: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating questions: {e}"