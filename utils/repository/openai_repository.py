import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

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
