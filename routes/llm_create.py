from utils.repository.openai_repository import OpenAIRepository
from utils.repository.rag_respository import RAGRepository
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

async def get_questions(topic: int, prompt: str, academy: int, model: str, has4questions: bool, num_of_q: int):
    try:
        client = OpenAIRepository()
        SBClient = SupabaseRepository()
        rag = RAGRepository(embedding_provider="openai", model_name="text-embedding-3-large")

        # context = "rag_context" # información que has sacado del RAG
        similar_documents = await rag.search_similar_documents(prompt, limit=5)

        # print(f"Documentos similares encontrados: {similar_documents}")

        documents = [doc['content'] for doc in similar_documents]

        documents = ' '.join(documents)

        # print(f"Contexto RAG obtenido: {len(context)} caracteres")
        print(documents[:1000])  # Mostrar solo los primeros 1000 caracteres para no saturar el log

        response = await client.generate_questions(
            topic=topic,
            academy=academy,
            has4questions=has4questions,
            prompt=prompt,
            num_of_q=num_of_q,
            model=model,
            context=documents
        )

        for question in response:
            SBClient.insert(
                table="questions",
                data=question.to_json_without_id()
            ) # guarda en BDD
        # return "Hola"
        return response

    except Exception as e:
        return {"error": str(e)}, 500