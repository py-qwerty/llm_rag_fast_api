# utils/rag/vector_search.py
from typing import List, Dict, Any, Optional
import json
import numpy as np
from utils.repository.supabase_repository import SupabaseRepository
from utils.models.law_item_response import LawItemWithLaw


class VectorSearchService:
    """
    Servicio para búsqueda vectorial optimizado usando funciones PostgreSQL
    """

    def __init__(self):
        self.supabase = SupabaseRepository()
        self.schema_name = "law_frame"

    async def search_similar_vectors(
            self,
            embedding: List[float],
            limit: int = 10,
            similarity_threshold: float = 0.5,
            include_deprecated: bool = False
    ) -> List[LawItemWithLaw]:
        """
        Busca vectores similares usando función PostgreSQL optimizada

        Args:
            embedding: Vector embedding de la consulta
            limit: Número máximo de resultados
            similarity_threshold: Umbral mínimo de similitud (0-1)
            include_deprecated: Si incluir elementos deprecados

        Returns:
            Lista de LawItemWithLaw con similitud calculada en PostgreSQL
        """
        try:
            # Verificar que el embedding no esté vacío
            if not embedding:
                print("❌ Embedding vacío")
                return []

            print(f"🔍 Buscando similitudes para embedding de dimensión: {len(embedding)}")

            # Convertir embedding a formato adecuado para PostgreSQL
            embedding_array = [float(x) for x in embedding]

            print(f"📡 Llamando a función PostgreSQL con parámetros:")
            print(f"   - Límite: {limit}")
            print(f"   - Umbral similitud: {similarity_threshold}")
            print(f"   - Incluir deprecados: {include_deprecated}")

            # Llamar a la función PostgreSQL usando RPC
            result = self.supabase.client.schema(self.schema_name).rpc(
                "search_similar_law_items",
                {
                    "p_query_embedding": embedding_array,
                    "p_limit_count": limit,
                    "p_similarity_threshold": similarity_threshold,
                    "p_include_deprecated": include_deprecated
                }
            ).execute()

            if not result.data:
                print("⚠️ No se encontraron resultados")
                return []

            print(f"✅ PostgreSQL retornó {len(result.data)} resultados")

            # Convertir resultados a modelos Pydantic
            law_items = []
            for item in result.data:
                try:
                    law_item = self.create_fused_model_from_pg_result(item)
                    law_items.append(law_item)
                except Exception as e:
                    print(f"⚠️ Error procesando resultado {item.get('item_id', 'unknown')}: {e}")
                    continue

            # Mostrar estadísticas
            if law_items:
                similarities = [item.similarity for item in law_items]
                print(
                    f"📊 Similitudes - Max: {max(similarities):.3f}, Min: {min(similarities):.3f}, Avg: {np.mean(similarities):.3f}")

            return law_items

        except Exception as e:
            print(f"❌ Error en búsqueda vectorial: {e}")
            print(f"❌ Tipo de error: {type(e).__name__}")

            # Información adicional para debug
            if hasattr(e, 'message'):
                print(f"❌ Mensaje del error: {e.message}")
            if hasattr(e, 'details'):
                print(f"❌ Detalles del error: {e.details}")

            return []

    def create_fused_model_from_pg_result(self, pg_result: Dict[str, Any]) -> LawItemWithLaw:
        """
        Crea un modelo LawItemWithLaw a partir del resultado de PostgreSQL

        Args:
            pg_result: Resultado directo de la función PostgreSQL

        Returns:
            Instancia de LawItemWithLaw
        """
        try:
            # Convertir vector si está presente (para uso posterior si es necesario)
            vector_data = pg_result.get('vector_data')
            if vector_data and isinstance(vector_data, str):
                try:
                    vector_data = json.loads(vector_data)
                except:
                    vector_data = None

            fused_model = LawItemWithLaw(
                # Datos de law_items
                item_id=pg_result.get('item_id'),
                law_id=pg_result.get('law_id', ''),
                second_id=pg_result.get('second_id', ''),
                item_title=pg_result.get('item_title'),
                content=pg_result.get('content'),
                url=pg_result.get('url'),
                num_questions=pg_result.get('num_questions'),
                max_number_questions=pg_result.get('max_number_questions'),
                chapter_info=pg_result.get('chapter_info'),
                vector=vector_data,
                embedding_model=pg_result.get('embedding_model'),
                item_updated_at=self.parse_datetime(pg_result.get('item_updated_at')),

                # Datos de laws
                law_unic_id=pg_result.get('law_unic_id'),
                law_identifier=pg_result.get('law_identifier', ''),
                law_title=pg_result.get('law_title'),
                last_update=self.parse_datetime(pg_result.get('last_update')),
                diario=pg_result.get('diario'),
                scope=pg_result.get('scope'),
                department=pg_result.get('department'),
                rank=pg_result.get('rank'),
                disposition_date=self.parse_datetime(pg_result.get('disposition_date')),
                publication_date=self.parse_datetime(pg_result.get('publication_date')),
                validity_date=self.parse_datetime(pg_result.get('validity_date')),
                expired_validity=pg_result.get('expired_validity'),
                url_eli=pg_result.get('url_eli'),
                url_html_consolidated=pg_result.get('url_html_consolidated'),

                # Similitud calculada por PostgreSQL
                similarity=float(pg_result.get('cosine_similarity', 0.0))
            )

            return fused_model

        except Exception as e:
            print(f"❌ Error creando modelo fusionado desde PG: {e}")
            # Retornar un modelo mínimo válido
            return LawItemWithLaw(
                item_id=pg_result.get('item_id', 0),
                law_id=pg_result.get('law_id', ''),
                second_id=pg_result.get('second_id', ''),
                law_identifier=pg_result.get('law_identifier', ''),
                similarity=float(pg_result.get('cosine_similarity', 0.0))
            )

    def parse_datetime(self, date_value):
        """
        Parsea diferentes formatos de fecha a datetime object
        """
        if not date_value:
            return None

        try:
            from datetime import datetime

            # Si ya es datetime, retornarlo
            if hasattr(date_value, 'year'):
                return date_value

            if isinstance(date_value, str):
                # Limpiar microsegundos si están presentes
                date_str = date_value.split('.')[0] if '.' in date_value else date_value

                # Intentar varios formatos
                formats = [
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d %H:%M:%S',
                    '%Y-%m-%d',
                    '%Y-%m-%dT%H:%M:%SZ'
                ]

                for fmt in formats:
                    try:
                        return datetime.strptime(date_str, fmt)
                    except ValueError:
                        continue

            return date_value

        except Exception as e:
            print(f"⚠️ Error parseando fecha {date_value}: {e}")
            return None

    async def get_vector_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del índice vectorial usando función PostgreSQL

        Returns:
            Diccionario con estadísticas del índice
        """
        try:
            result = self.supabase.client.schema(self.schema_name).rpc(
                "get_vector_index_stats"
            ).execute()

            if result.data and len(result.data) > 0:
                stats = result.data[0]
                return {
                    "total_vectors": stats.get('total_vectors', 0),
                    "avg_dimension": stats.get('avg_vector_dimension', 0),
                    "deprecated_count": stats.get('deprecated_count', 0),
                    "valid_laws_count": stats.get('valid_laws_count', 0),
                    "validity_percentage": (
                            stats.get('valid_laws_count', 0) / max(stats.get('total_vectors', 1), 1) * 100
                    )
                }
            return {}

        except Exception as e:
            print(f"❌ Error obteniendo estadísticas: {e}")
            return {}

    def calculate_manual_similarity(self, query_embedding: List[float], doc_vector: List[float]) -> float:
        """
        Método legacy - la similitud ahora se calcula en PostgreSQL
        Mantener para compatibilidad con código existente
        """
        print("⚠️ Usando calculate_manual_similarity - considera usar la función PostgreSQL")

        try:
            vec1 = np.array(query_embedding, dtype=np.float32)
            vec2 = np.array(doc_vector, dtype=np.float32)

            if len(vec1) != len(vec2):
                return 0.0

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        except Exception as e:
            print(f"❌ Error calculando similitud manual: {e}")
            return 0.0