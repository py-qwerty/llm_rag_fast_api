from openai import OpenAI
from utils.repository.openai_repository import OpenAIRepository
from utils.repository.supabase_repository import SupabaseRepository
from utils.rag.embedding_service import EmbeddingService
from utils.rag.vector_search import VectorSearchService

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

        # context = "rag_context" # información que has sacado del RAG
        context = await get_rag_context(prompt, top_k=5)

        print(f"Contexto RAG obtenido: {len(context)} caracteres")
        print(context[:1000])  # Mostrar solo los primeros 1000 caracteres para no saturar el log

        # response = await client.generate_questions(
        #     topic=topic,
        #     academy=academy,
        #     has4questions=has4questions,
        #     prompt=prompt,
        #     num_of_q=num_of_q,
        #     model=model,
        #     context=context
        # )

        # for question in response:
        #     SBClient.insert(
        #         table="questions",
        #         data=question.to_json_without_id()
        #     )

        return 'HOla'
        return response

    except Exception as e:
        return {"error": str(e)}, 500
    
async def get_rag_context(prompt: str, top_k: int = 5, similarity_threshold: float = 0.7) -> str:
    """
    Función para obtener contexto relevante usando RAG
    
    Args:
        prompt: El texto a vectorizar y buscar
        top_k: Número de documentos más similares a retornar
        similarity_threshold: Umbral mínimo de similitud
    
    Returns:
        str: Contexto concatenado de los documentos más similares
    """
    try:
        # 1. Inicializar servicios
        embedding_service = EmbeddingService(provider="openai")  # o "sentence_transformers"
        vector_search = VectorSearchService()
        
        # 2. Vectorizar el prompt
        embedding_vector = await embedding_service.generate_embedding(prompt)
        
        # 3. Buscar documentos similares
        similar_docs = await vector_search.search_similar_vectors(
            embedding=embedding_vector,
            limit=top_k
        )
        
        # 4. Filtrar por umbral de similitud y concatenar contexto
        context_parts = []
        relevant_docs = 0
        
        for doc in similar_docs:
            if doc.get('similarity', 0) >= similarity_threshold:
                context_parts.append(doc.get('content', ''))
                relevant_docs += 1
        
        # 5. Unir todo el contexto
        context = "\n\n---\n\n".join(context_parts)
        
        print(f"RAG context: {len(context)} chars de {relevant_docs}/{len(similar_docs)} docs relevantes")
        return context
        
    except Exception as e:
        print.error(f"Error obteniendo contexto RAG: {str(e)}")
        return ""

