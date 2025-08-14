from openai import OpenAI
from utils.repository.openai_repository import OpenAIRepository
from utils.repository.supabase_repository import SupabaseRepository

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

async def get_questions(topic: int, prompt: str, academy: int, model: str, has4questions: bool, num_of_q: int, context: str = ''):
    try:
        client = OpenAIRepository()
        SBClient = SupabaseRepository()

        response = await client.generate_questions(
            topic=topic,
            academy=academy,
            has4questions=has4questions,
            prompt=prompt,
            num_of_q=num_of_q,
            model=model,
            context=context
        )

        for question in response:
            SBClient.insert(
                table="questions",
                data=question.to_json_without_id()
            )

        # SBClient.insert(
        #     table="questions",
        #     data={
        #         "topic": topic,
        #         "academy": academy,
        #         "prompt": prompt,
        #         "model": model,
        #         "num_of_q": num_of_q,
        #         "effort": effort,
        #         "questions": [q.to_db_dict() for q in response]
        #     }
        # )

        return response

    except Exception as e:
        return {"error": str(e)}, 500
