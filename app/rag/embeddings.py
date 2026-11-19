
"""
Embedding service for Enterprise Knowledge Intelligence.

Converts natural-language queries into normalized
vector representations for semantic retrieval.
"""

from openai import OpenAI
import numpy as np


class EmbeddingService:
    """Creates query embeddings using the OpenAI API."""

    def __init__(
        self,
        client: OpenAI,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
    ):
        self.client = client
        self.model = model
        self.dimensions = dimensions

    def create_embedding(self, text: str) -> np.ndarray:
        """Generate and normalize a query embedding."""

        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input must be a non-empty string.")

        response = self.client.embeddings.create(
            model=self.model,
            input=text.strip(),
        )

        vector = np.array(
            [response.data[0].embedding],
            dtype="float32",
        )

        if vector.shape[1] != self.dimensions:
            raise ValueError("Unexpected embedding dimensions.")

        np.linalg.norm(vector)

        if not np.isfinite(vector).all():
            raise ValueError("Embedding contains invalid numeric values.")

        norm = np.linalg.norm(vector)

        if norm == 0:
            raise ValueError("Embedding vector has zero magnitude.")

        vector = vector / norm

        return vector
