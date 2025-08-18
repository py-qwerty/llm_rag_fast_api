# utils/rag/vector_search.py
from typing import List, Dict, Any
from utils.repository.supabase_repository import SupabaseRepository

class VectorSearchService:
    """
    Servicio para búsqueda vectorial pura usando Supabase con pgvector
    """

    def __init__(self):
        self.supabase = SupabaseRepository()
        self.table_name = "law_items"  # Tabla de embeddings

    async def search_similar_vectors(
        self,
        embedding: List[float],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Busca los vectores más similares al embedding dado

        Args:
            embedding: Vector embedding de la consulta
            limit: Número máximo de resultados

        Returns:
            Lista de documentos con su contenido y score de similitud
        """
        try:

            print("embedding", embedding)
            # Llamar a la función RPC en Supabase
            result = self.supabase.client.schema("law_frame").rpc(

                "search_law_items",
                {
                    "p_query": embedding,
                    "p_limit_count": limit
                }
            ).execute()

            if result.data:
                documents = [
                    {
                        "id": item["id"],
                        "content": item["content"],
                        "similarity": item["similarity"]
                    }
                    for item in result.data
                ]
                return documents

            return []

        except Exception as e:
            print(f"❌ Error en búsqueda vectorial: {e}")
            return []
