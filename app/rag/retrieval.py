
"""
Semantic retrieval module for Enterprise Knowledge Intelligence.

Loads the FAISS index and retrieves the most relevant
knowledge-base chunks for a user question.
"""

from typing import Any
import pickle

import faiss
import numpy as np


class EnterpriseRetriever:
    """Performs semantic search over the enterprise knowledge index."""

    def __init__(
        self,
        index_path: str,
        vectors_path: str,
        top_k: int = 3,
    ) -> None:

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        self.index_path = index_path
        self.vectors_path = vectors_path
        self.top_k = top_k

        # Load FAISS index
        self.index = faiss.read_index(self.index_path)

        # Load metadata associated with indexed vectors
        with open(self.vectors_path, "rb") as file:
            self.vectors = pickle.load(file)

        if self.index.ntotal != len(self.vectors):
            raise ValueError(
                "FAISS index and metadata count mismatch."
            )

    def retrieve(
        self,
        query_embedding: list[float],
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant knowledge chunks."""

        if not isinstance(query_embedding, list) or not query_embedding:
            raise ValueError("Query embedding cannot be empty.")

        query_vector = np.asarray(
            [query_embedding],
            dtype="float32",
        )

        if query_vector.shape[1] != self.index.d:
            raise ValueError(
                "Query embedding dimension mismatch."
            )

        if not np.isfinite(query_vector).all():
            raise ValueError(
                "Query embedding contains invalid values."
            )

        faiss.normalize_L2(query_vector)

        actual_k = min(self.top_k, self.index.ntotal)

        if actual_k == 0:
            return []

        similarities, indices = self.index.search(
            query_vector,
            actual_k,
        )

        results = []

        for similarity, index_position in zip(
            similarities[0],
            indices[0],
        ):
            if index_position < 0:
                continue

            record = self.vectors[index_position]

            results.append(
                {
                    "similarity": round(float(similarity), 4),
                    "chunk_id": record["chunk_id"],
                    "document_id": record["document_id"],
                    "document_title": record["document_title"],
                    "department": record.get("department"),
                    "text": record["text"],
                }
            )

        return results

    def retrieve_question(
        self,
        question: str,
        embedding_service: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate a query embedding and retrieve relevant chunks.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        query_vector = embedding_service.create_embedding(
            question.strip()
        )

        query_embedding = query_vector.flatten().tolist()

        return self.retrieve(query_embedding)
