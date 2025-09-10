from utils.repository.openai_repository import OpenAIRepository
from utils.repository.rag_repository import ProductionRAGService
from utils.repository.supabase_repository import SupabaseRepository


def create(system: str, prompt: str, model: str = None, effort: str = "low"):
    """
    Genera texto usando OpenAI con los parámetros especificados

    Args:
        system: Prompt del sistema
        prompt: Prompt del usuario
        model: Modelo a utilizar (opcional)
        effort: Nivel de esfuerzo (opcional)

    Returns:
        Respuesta generada o diccionario con error
    """
    try:
        client = OpenAIRepository()
        response = client.generate_text(
            system=system,
            prompt=prompt,
            model=model,
            effort=effort
        )

        return response

    except Exception as e:
        return {"error": str(e)}, 500


async def get_questions(topic: int, prompt: str, academy: int, model: str, has4questions: bool, num_of_q: int):
    """
    Genera preguntas usando RAG + OpenAI y las guarda en Supabase

    Args:
        topic: ID del tema
        prompt: Consulta del usuario
        academy: ID de la academia
        model: Modelo de LLM a usar
        has4questions: Si tiene 4 preguntas
        num_of_q: Número de preguntas a generar

    Returns:
        Lista de preguntas generadas o diccionario con error
    """
    try:
        # Inicializar clientes
        client = OpenAIRepository()
        supabase_client = SupabaseRepository()
        rag = ProductionRAGService(embedding_provider="openai", model_name="text-embedding-3-large")

        # Buscar documentos similares usando RAG
        print(f"Buscando documentos similares para: '{prompt[:50]}...'")
        similar_documents = await rag.search_similar_documents(prompt, limit=5)

        if not similar_documents:
            print("⚠️ No se encontraron documentos similares, continuando sin contexto RAG")
            documents = ""
        else:
            print(f"✅ Encontrados {len(similar_documents)} documentos similares")
            # Extraer contenido de los documentos
            documents = [doc['content'] for doc in similar_documents]
            documents = ' '.join(documents)

            # Log para debugging (solo primeros 1000 caracteres)
            print(f"📄 Contexto RAG obtenido ({len(documents)} caracteres):")
            print(documents[:1000])
            if len(documents) > 1000:
                print("... [contenido truncado para el log]")

        # Generar preguntas con contexto RAG
        print(f"🤖 Generando {num_of_q} preguntas con modelo {model}")
        response = await client.generate_questions(
            topic=topic,
            academy=academy,
            has4questions=has4questions,
            prompt=prompt,
            num_of_q=num_of_q,
            model=model,
            context=documents
        )

        # Guardar preguntas en Supabase
        saved_count = 0
        for question in response:
            try:
                supabase_client.insert(
                    table="questions",
                    data=question.to_json_without_id()
                )
                saved_count += 1
            except Exception as save_error:
                print(f"❌ Error guardando pregunta: {save_error}")
                # Continuar con las demás preguntas
                continue

        print(f"💾 Guardadas {saved_count}/{len(response)} preguntas en base de datos")

        return response

    except Exception as e:
        print(f"❌ Error en get_questions: {e}")
        return {"error": str(e)}, 500