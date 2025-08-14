import os
from dotenv import load_dotenv
from supabase import create_client, Client

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
