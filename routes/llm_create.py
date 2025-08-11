from openai import OpenAI
from utils.repository.openai_repository import OpenAIRepository

def create(system: str, prompt: str, model: str = None, effort: str = "low"):
    try:
        client = OpenAIRepository()
        response = client.generate_text(
            system=system,
            prompt=prompt,
            model=model,
            effort=effort
        )

        return  response

    except Exception as e:
        return {"error": str(e)}, 500
