from typing import List, Dict, Any, Optional
from utils.services.embedding_service import EmbeddingService
from utils.services.vector_search import VectorSearchService


class RAGRepository:
    """
    Servicio RAG simplificado usando únicamente OpenAI embeddings
    """

    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        Inicializa el servicio RAG con OpenAI embeddings

        Args:
            model_name: Modelo de OpenAI para embeddings
        """
        self.model_name = model_name
        self._embedding_service = None  # Lazy loading
        self._vector_search = None  # Lazy loading

    @property
    def embedding_service(self):
        if self._embedding_service is None:
            print(f"Initializing OpenAI EmbeddingService with model: {self.model_name}")
            self._embedding_service = EmbeddingService(model_name=self.model_name)
        return self._embedding_service

    @property
    def vector_search(self):
        if self._vector_search is None:
            print("Initializing VectorSearchService")
            self._vector_search = VectorSearchService()
        return self._vector_search

    async def search_similar_documents(
            self,
            query: str,
            limit: int = 10,
            min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Busca documentos similares usando OpenAI embeddings

        Args:
            query: Texto de la consulta
            limit: Número máximo de resultados
            min_similarity: Similitud mínima requerida

        Returns:
            Lista de documentos ordenados por similitud
        """
        try:
            print(f"Processing query: '{query[:100]}...'")

            # Generar embedding de la consulta usando OpenAI
            query_embedding = await self.embedding_service.generate_embedding(query)
            print(f"Generated OpenAI embedding. Dimensions: {len(query_embedding)}")

            # Buscar documentos similares
            similar_docs = await self.vector_search.search_similar_vectors(
                embedding=query_embedding,
                limit=limit
            )

            # Filtrar por similitud mínima
            if min_similarity > 0:
                similar_docs = [
                    doc for doc in similar_docs
                    if doc.get("similarity", 0) >= min_similarity
                ]
                print(f"Documents filtered by similarity >= {min_similarity}: {len(similar_docs)}")

            # Agregar metadatos
            for i, doc in enumerate(similar_docs):
                doc["rank"] = i + 1
                doc["query"] = query
                doc["embedding_model"] = self.model_name

            print(f"Found {len(similar_docs)} similar documents")
            return similar_docs

        except Exception as e:
            print(f"Error in RAG search: {e}")
            return []

    async def search_and_format_context(
            self,
            query: str,
            limit: int = 5,
            max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """
        Busca documentos y los formatea para contexto LLM

        Args:
            query: Consulta del usuario
            limit: Número de documentos a buscar
            max_context_length: Longitud máxima del contexto

        Returns:
            Dict con contexto formateado y metadatos
        """
        try:
            # Buscar documentos similares
            docs = await self.search_similar_documents(query, limit=limit)

            if not docs:
                return {
                    "context": "",
                    "sources": [],
                    "total_docs": 0,
                    "query": query,
                    "embedding_model": self.model_name
                }

            # Formatear contexto
            context_parts = []
            sources = []
            current_length = 0

            for doc in docs:
                content = doc.get("content", "")
                doc_id = doc.get("id", "unknown")
                similarity = doc.get("similarity", 0)

                # Verificar límite de contexto
                if current_length + len(content) > max_context_length:
                    if current_length == 0:  # Primer documento
                        remaining_space = max_context_length - 100
                        content = content[:remaining_space] + "..."
                    else:
                        break

                # Formatear contenido
                formatted_content = f"[Document {doc_id} - Similarity: {similarity:.3f}]\n{content}\n"
                context_parts.append(formatted_content)

                sources.append({
                    "id": doc_id,
                    "similarity": similarity,
                    "content_preview": content[:200] + "..." if len(content) > 200 else content
                })

                current_length += len(formatted_content)

            context = "\n".join(context_parts)

            return {
                "context": context,
                "sources": sources,
                "total_docs": len(docs),
                "used_docs": len(sources),
                "query": query,
                "context_length": len(context),
                "embedding_model": self.model_name
            }

        except Exception as e:
            print(f"Error formatting context: {e}")
            return {
                "context": "",
                "sources": [],
                "total_docs": 0,
                "query": query,
                "error": str(e),
                "embedding_model": self.model_name
            }

    def get_service_info(self) -> Dict[str, Any]:
        """
        Retorna información sobre la configuración del servicio
        """
        return {
            "embedding_provider": "openai",
            "embedding_model": self.model_name,
            "embedding_dimensions": 3072,
            "vector_search": "supabase" if hasattr(self.vector_search, 'supabase_client') else "unknown"
        }