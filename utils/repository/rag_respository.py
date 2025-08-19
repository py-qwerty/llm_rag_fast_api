# utils/rag/rag_service.py
from typing import List, Dict, Any, Optional

from utils.services.embedding_service import EmbeddingService
from utils.services.vector_search import VectorSearchService

class RAGRepository:
    """
    Servicio completo de RAG que combina embedding y búsqueda vectorial
    """
    
    def __init__(self, embedding_provider: str = "openai", model_name: Optional[str] = None):
        """
        Inicializa el servicio RAG
        
        Args:
            embedding_provider: "openai" o "sentence_transformers"
            model_name: Nombre específico del modelo
        """
        self.embedding_service = EmbeddingService(
            provider=embedding_provider, 
            model_name=model_name
        )
        self.vector_search = VectorSearchService()
        
    async def search_similar_documents(
        self, 
        query: str, 
        limit: int = 10,
        min_similarity: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Busca documentos similares a la consulta
        
        Args:
            query: Texto de la consulta
            limit: Número máximo de resultados
            min_similarity: Similitud mínima requerida
            
        Returns:
            Lista de documentos ordenados por similitud
        """
        try:
            print(f"🔍 Procesando consulta: '{query[:100]}...'")
            
            # 1. Generar embedding de la consulta
            query_embedding = await self.embedding_service.generate_embedding(query)
            print(f"✅ Embedding generado. Dimensión: {len(query_embedding)}")
            
            # 2. Buscar documentos similares
            similar_docs = await self.vector_search.search_similar_vectors(
                embedding=query_embedding,
                limit=limit
            )
            
            # 3. Filtrar por similitud mínima si se especifica
            if min_similarity > 0:
                similar_docs = [
                    doc for doc in similar_docs 
                    if doc.get("similarity", 0) >= min_similarity
                ]
                print(f"📊 Documentos filtrados por similitud >= {min_similarity}: {len(similar_docs)}")
            
            # 4. Agregar información adicional
            for i, doc in enumerate(similar_docs):
                doc["rank"] = i + 1
                doc["query"] = query
                
            print(f"✅ Encontrados {len(similar_docs)} documentos similares")
            return similar_docs
            
        except Exception as e:
            print(f"❌ Error en búsqueda RAG: {e}")
            return []
    
    async def search_and_format_context(
        self, 
        query: str, 
        limit: int = 5,
        max_context_length: int = 4000
    ) -> Dict[str, Any]:
        """
        Busca documentos y los formatea para usar como contexto en LLM
        
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
                    "query": query
                }
            
            # Formatear contexto
            context_parts = []
            sources = []
            current_length = 0
            
            for doc in docs:
                content = doc.get("content", "")
                doc_id = doc.get("id", "unknown")
                similarity = doc.get("similarity", 0)
                
                # Verificar si agregar este documento excede el límite
                if current_length + len(content) > max_context_length:
                    if current_length == 0:  # Si es el primer documento, incluir parte de él
                        remaining_space = max_context_length - 100  # Dejar espacio para metadatos
                        content = content[:remaining_space] + "..."
                    else:
                        break  # No incluir más documentos
                
                # Formatear el contenido del documento
                formatted_content = f"[Documento {doc_id} - Similitud: {similarity:.3f}]\n{content}\n"
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
                "context_length": len(context)
            }
            
        except Exception as e:
            print(f"❌ Error formateando contexto: {e}")
            return {
                "context": "",
                "sources": [],
                "total_docs": 0,
                "query": query,
                "error": str(e)
            }