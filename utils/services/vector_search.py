# utils/rag/vector_search.py
from typing import List, Dict, Any
import json
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
            # Verificar que el embedding no esté vacío
            if not embedding:
                print("❌ Embedding vacío")
                return []

            print(f"🔍 Buscando similitudes para embedding de dimensión: {len(embedding)}")
            
            # Convertir embedding a formato PostgreSQL array
            # Formato: [1.0,2.0,3.0] - sin espacios y con punto decimal
            embedding_array = [float(x) for x in embedding]  # Asegurar que son floats
            
            # Llamar a la función RPC en Supabase
            result = self.supabase.client.schema("law_frame").rpc(
                "search_law_items",
                {
                    "p_query": embedding_array,  # Enviar como array de floats
                    "p_limit_count": limit
                }
            ).execute()

            print(f"✅ Resultado de búsqueda recibido. Registros: {len(result.data) if result.data else 0}")
            
            # Debug: mostrar estructura del primer resultado
            if result.data and len(result.data) > 0:
                print(f"📋 Estructura del primer resultado: {list(result.data[0].keys())}")
                # print(f"📋 Primer resultado: {result.data[0]}")

            if result.data:
                documents = []
                for item in result.data:
                    # Adaptar a la estructura que retorna tu función SQL actual
                    doc = {
                        "id": item.get("id"),
                        "content": item.get("content", ""),
                        # Como tu función no retorna similarity, ponemos un valor por defecto
                        # o intentamos calcular basándonos en el orden (los primeros son más similares)
                        "similarity": item.get("similarity", 1.0)  # Por defecto alta similitud
                    }
                    
                    # Si hay otros campos que necesites, agrégalos aquí
                    if "title" in item:
                        doc["title"] = item["title"]
                    if "metadata" in item:
                        doc["metadata"] = item["metadata"]
                    
                    documents.append(doc)
                
                print(f"✅ Procesados {len(documents)} documentos similares")
                return documents

            print("⚠️ No se encontraron resultados")
            return []

        except Exception as e:
            print(f"❌ Error en búsqueda vectorial: {e}")
            print(f"❌ Tipo de error: {type(e).__name__}")
            
            # Información adicional para debug
            if hasattr(e, 'message'):
                print(f"❌ Mensaje del error: {e.message}")
            if hasattr(e, 'details'):
                print(f"❌ Detalles del error: {e.details}")
                
            return []

    def calculate_manual_similarity(self, query_embedding: List[float], doc_vector: List[float]) -> float:
        """
        Calcula similitud coseno manualmente si es necesario
        
        Args:
            query_embedding: Vector de la consulta
            doc_vector: Vector del documento
            
        Returns:
            Score de similitud entre 0 y 1
        """
        try:
            import numpy as np
            
            vec1 = np.array(query_embedding)
            vec2 = np.array(doc_vector)
            
            # Similitud coseno
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
            
        except Exception as e:
            print(f"Error calculando similitud: {e}")
            return 0.0