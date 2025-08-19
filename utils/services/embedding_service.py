# utils/rag/embedding_service.py
import asyncio
import openai
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import os

class EmbeddingService:
    """
    Servicio para generar embeddings de texto usando diferentes proveedores
    """
    
    def __init__(self, provider: str = "openai", model_name: Optional[str] = None):
        """
        Inicializa el servicio de embeddings
        
        Args:
            provider: "openai", "huggingface", o "sentence_transformers"
            model_name: Nombre específico del modelo a usar
        """
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.model_name = model_name or "text-embedding-3-large"
            self.client = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
        elif self.provider == "sentence_transformers":
            self.model_name = model_name or "all-MiniLM-L6-v2"
            # Cargar modelo localmente
            self.model = SentenceTransformer(self.model_name)
            
        else:
            raise ValueError(f"Proveedor no soportado: {provider}")
    
    async def generate_embedding(self, text: str) -> List[float]:
        """
        Genera embedding para un texto dado
        
        Args:
            text: Texto a convertir en embedding
            
        Returns:
            Lista de floats representando el embedding
        """
        try:
            if self.provider == "openai":
                return await self._generate_openai_embedding(text)
            elif self.provider == "sentence_transformers":
                return await self._generate_sentence_transformer_embedding(text)
        except Exception as e:
            print(f"❌ Error generando embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos en batch
        
        Args:
            texts: Lista de textos a convertir en embeddings
            
        Returns:
            Lista de embeddings
        """
        try:
            if self.provider == "openai":
                return await self._generate_openai_embeddings_batch(texts)
            elif self.provider == "sentence_transformers":
                return await self._generate_sentence_transformer_embeddings_batch(texts)
        except Exception as e:
            print(f"❌ Error generando embeddings en batch: {e}")
            raise
    
    async def _generate_openai_embedding(self, text: str) -> List[float]:
        """Genera embedding usando OpenAI"""
        response = await self.client.embeddings.create(
            input=text,
            model=self.model_name
        )
        return response.data[0].embedding
    
    async def _generate_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings en batch usando OpenAI"""
        # OpenAI permite hasta 2048 textos por batch
        batch_size = 2048
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            response = await self.client.embeddings.create(
                input=batch_texts,
                model=self.model_name
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    async def _generate_sentence_transformer_embedding(self, text: str) -> List[float]:
        """Genera embedding usando Sentence Transformers"""
        # Ejecutar en un hilo separado para no bloquear el event loop
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, 
            lambda: self.model.encode([text])[0].tolist()
        )
        return embedding
    
    async def _generate_sentence_transformer_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Genera embeddings en batch usando Sentence Transformers"""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None,
            lambda: self.model.encode(texts).tolist()
        )
        return embeddings
    
    def get_embedding_dimension(self) -> int:
        """
        Retorna la dimensión del embedding según el modelo usado
        """
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
            "all-MiniLM-L6-v2": 384,
            "all-mpnet-base-v2": 768
        }
        
        return dimensions.get(self.model_name, 768)  # Default 768
    
    @staticmethod
    def calculate_cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calcula la similitud coseno entre dos embeddings
        """
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)