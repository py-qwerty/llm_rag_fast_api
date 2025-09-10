"""
Buscador legal mínimo para agentes.
Expone una sola función: `search(query, limit=10, include_deprecated=False)`
Devuelve la lista de filas tal cual la retorna la RPC de PostgreSQL.
"""

import logging
from typing import List, Dict, Any

from utils.repository.supabase_repository import SupabaseRepository
from utils.models.law_item_response import LawItemWithLaw

logger = logging.getLogger(__name__)


class LegalSearchService:
    def __init__(self, schema_name: str = "law_frame", rpc_name: str = "search_legal_content"):
        self.supabase = SupabaseRepository()
        self.schema_name = schema_name
        self.rpc_name = rpc_name

        from utils.models.law_item_response import LawItemWithLaw

    def dict_to_law_item_with_law(self, data: dict) -> LawItemWithLaw:
        """
        Convierte un diccionario a LawItemWithLaw aplicando la permutación de campos.

        Args:
            data: Diccionario con los datos del resultado FTS/vectorial

        Returns:
            Instancia de LawItemWithLaw
        """
        # Aplicar permutación: law_id (int) -> law_identifier, law_identifier (str) -> law_id
        original_law_id = data.get('law_id')
        original_law_identifier = data.get('law_identifier')

        return LawItemWithLaw(
            # Campos básicos
            item_id=data.get('item_id', 0),
            law_id=str(original_law_identifier) if original_law_identifier else '',  # Permutado
            second_id=data.get('second_id', ''),

            # Campos opcionales de law_items
            item_title=data.get('item_title'),
            content=data.get('content'),
            url=data.get('url'),
            num_questions=data.get('num_questions'),
            max_number_questions=data.get('max_number_questions'),
            chapter_info=data.get('chapter_info'),
            vector=data.get('vector'),
            embedding_model=data.get('embedding_model'),
            item_updated_at=data.get('item_updated_at'),

            # Campos de laws
            law_unic_id=original_law_id,
            law_identifier=str(original_law_identifier),
            law_title=data.get('law_title'),
            last_update=data.get('last_update'),
            diario=data.get('diario'),
            scope=data.get('scope'),
            department=data.get('department'),
            rank=data.get('rank'),
            disposition_date=data.get('disposition_date'),
            publication_date=data.get('publication_date'),
            validity_date=data.get('validity_date'),
            expired_validity=data.get('expired_validity'),
            url_eli=data.get('url_eli'),
            url_html_consolidated=data.get('url_html_consolidated'),

            # Similitud
            similarity=float(data.get('search_rank', 0.0))
        )

    async def search(
        self,
        query: str,
        limit: int = 10,
        law_id: int = None,
        include_deprecated: bool = False
    ) -> List[LawItemWithLaw]:
        """Búsqueda plana en la función PostgreSQL."""
        if not query or not query.strip():
            return []

        try:
            result = (
                self.supabase.client
                .schema(self.schema_name)
                .rpc(
                    self.rpc_name,
                    {
                        "p_search_query": query,
                        "p_limit_count": limit,
                        "p_include_deprecated": include_deprecated,
                        "p_law_id":  law_id
                    },
                )
                .execute()
            )
            return [self.dict_to_law_item_with_law(item) for item in result.data] or []
        except Exception as e:
            logger.error(f"Error en búsqueda legal: {e}")
            return []
