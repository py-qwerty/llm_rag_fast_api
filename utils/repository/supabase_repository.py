import os
from typing import Optional, List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
from supabase import create_client, Client


class RandomLawItem(BaseModel):
    """
    Modelo para un item de ley aleatorio obtenido de la función get_random_law_items.
    """
    law_title: Optional[str] = Field(
        None,
        description="Nombre/título de la ley"
    )
    law_id: int = Field(
        description="ID único de la ley en la tabla laws"
    )
    law_identifier: str = Field(
        description="Identificador único de la ley"
    )
    law_item_id: int = Field(
        description="ID único del item de ley en la tabla law_items"
    )
    law_item_title: Optional[str] = Field(
        None,
        description="Título del artículo/item específico de la ley"
    )
    content: Optional[str] = Field(
        None,
        description="Contenido del artículo/item de la ley"
    )

    class Config:
        # Permite usar nombres de campos con snake_case
        allow_population_by_field_name = True
        # Ejemplo de datos para documentación
        schema_extra = {
            "example": {
                "law_title": "Ley Orgánica 3/2007, de 22 de marzo, para la igualdad efectiva de mujeres y hombres",
                "law_id": 1,
                "law_identifier": "BOE-A-2007-6115",
                "law_item_id": 12345,
                "law_item_title": "Artículo 1. Objeto de la Ley",
                "content": "Las mujeres y los hombres son iguales en dignidad humana..."
            }
        }


class RandomLawItemsResponse(BaseModel):
    """
    Modelo para la respuesta completa de items de ley aleatorios.
    """
    items: List[RandomLawItem] = Field(
        default_factory=list,
        description="Lista de items de ley aleatorios"
    )
    count: int = Field(
        description="Número total de items devueltos"
    )

    @property
    def total_items(self) -> int:
        """Alias para count."""
        return len(self.items)


class SupabaseRepository:
    def __init__(self):
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError("Faltan variables SUPABASE_URL o SUPABASE_KEY en el .env")

        self.client: Client = create_client(url, key)

    def select(self, table: str, filters: dict = None, order_by: str = None, order_dir: str = "asc", limit: int = None):
        """
        Selecciona registros de una tabla con filtros, ordenamiento y límite opcionales.

        :param table: nombre de la tabla
        :param filters: diccionario {columna: valor} para filtrar
        :param order_by: columna por la que ordenar
        :param order_dir: "asc" o "desc" (por defecto "asc")
        :param limit: número máximo de registros a devolver
        """
        query = self.client.table(table).select("*")

        # Aplicar filtros
        if filters:
            for col, val in filters.items():
                query = query.eq(col, val)

        # Aplicar ordenamiento
        if order_by:
            query = query.order(order_by, desc=(order_dir.lower() == "desc"))

        # Aplicar límite
        if limit is not None:
            query = query.limit(limit)

        # Ejecutar consulta
        return query.execute().data

    def insert(self, table: str, data: dict):
        return self.client.table(table).insert(data).execute().data

    def update(self, table: str, data: dict, filters: dict):
        query = self.client.table(table).update(data)
        for col, val in filters.items():
            query = query.eq(col, val)
        return query.execute().data

    def delete(self, table: str, filters: dict):
        query = self.client.table(table).delete()
        for col, val in filters.items():
            query = query.eq(col, val)
        return query.execute().data

    def parse_random_law_items(self, supabase_response: List[dict]) -> List[RandomLawItem]:
        """
        Convierte la respuesta de Supabase al modelo Pydantic.

        :param supabase_response: Lista de diccionarios desde Supabase
        :return: RandomLawItemsResponse con los datos parseados
        """
        items = [RandomLawItem(**item) for item in supabase_response]
        return items

    def get_random_laws(self, limit: int = 5) -> List[RandomLawItem]:
        """
        Obtiene un conjunto aleatorio de leyes.

        :param limit: número de leyes a obtener
        :return: RandomLawItemsResponse con los items de ley parseados y validados
        """
        response = self.client.schema('law_frame') \
            .rpc('get_random_law_items', {'p_limit': limit}).execute()

        return self.parse_random_law_items(response.data)