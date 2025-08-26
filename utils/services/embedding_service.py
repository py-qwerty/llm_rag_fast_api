import os
from typing import List
from openai import OpenAI
from dotenv import load_dotenv


class EmbeddingService:
    """
    Servicio de embeddings usando únicamente OpenAI
    """

    def __init__(self, model_name: str = "text-embedding-3-large"):
        # Only load .env if it exists (for local development)
        if os.path.exists('.env'):
            load_dotenv()

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY environment variable")

        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.dimensions = 3072  # text-embedding-3-large dimensions

        print(f"EmbeddingService initialized with OpenAI model: {self.model_name}")

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Genera embedding para un texto usando OpenAI

        Args:
            text: Texto de entrada

        Returns:
            Lista de floats representando el vector embedding
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            # Clean and truncate text if necessary
            cleaned_text = text.strip()

            # OpenAI has token limits, roughly 8191 tokens for text-embedding-3-large
            # Approximate: 1 token ≈ 4 characters, so limit to ~30k characters
            if len(cleaned_text) > 30000:
                cleaned_text = cleaned_text[:30000] + "..."
                print(f"Warning: Text truncated to 30k characters")

            response = self.client.embeddings.create(
                model=self.model_name,
                input=cleaned_text,
                encoding_format="float"
            )

            embedding = response.data[0].embedding

            if len(embedding) != self.dimensions:
                print(f"Warning: Expected {self.dimensions} dimensions, got {len(embedding)}")

            return embedding

        except Exception as e:
            print(f"Error generating embedding: {e}")
            raise

    async def generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Genera embeddings para múltiples textos de forma eficiente

        Args:
            texts: Lista de textos

        Returns:
            Lista de embeddings
        """
        if not texts:
            return []

        try:
            # Clean texts
            cleaned_texts = []
            for text in texts:
                if text and text.strip():
                    cleaned_text = text.strip()
                    if len(cleaned_text) > 30000:
                        cleaned_text = cleaned_text[:30000] + "..."
                    cleaned_texts.append(cleaned_text)

            if not cleaned_texts:
                return []

            # OpenAI allows batch requests
            response = self.client.embeddings.create(
                model=self.model_name,
                input=cleaned_texts,
                encoding_format="float"
            )

            embeddings = [item.embedding for item in response.data]

            print(f"Generated {len(embeddings)} embeddings in batch")
            return embeddings

        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            raise

    def get_model_info(self) -> dict:
        """
        Retorna información sobre el modelo actual
        """
        return {
            "provider": "openai",
            "model": self.model_name,
            "dimensions": self.dimensions,
            "max_tokens": 8191
        }