
"""
Enterprise RAG pipeline.

Coordinates query embedding, semantic retrieval
and grounded answer generation.
"""

from typing import Any

from app.rag.embeddings import EmbeddingService
from app.rag.retrieval import EnterpriseRetriever
from app.rag.generation import AnswerGenerator


class EnterpriseRAGPipeline:
    """Coordinates the complete enterprise RAG workflow."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        retriever: EnterpriseRetriever,
        answer_generator: AnswerGenerator,
    ) -> None:

        self.embedding_service = embedding_service
        self.retriever = retriever
        self.answer_generator = answer_generator

    def answer_question(
        self,
        question: str,
    ) -> dict[str, Any]:
        """
        Process a question through the complete RAG pipeline.

        Workflow:
        1. Validate the question.
        2. Generate its embedding.
        3. Retrieve relevant knowledge chunks.
        4. Generate an answer grounded in retrieved context.
        5. Return the answer and source references.
        """

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Question cannot be empty.")

        normalized_question = question.strip()

        # Step 1: Generate query embedding and retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve_question(
            normalized_question,
            self.embedding_service,
        )

        # Step 2: Generate an answer using retrieved knowledge
        result = self.answer_generator.generate_answer(
            normalized_question,
            retrieved_chunks,
        )

        return {
            "question": normalized_question,
            "answer": result["answer"],
            "sources": result["sources"],
            "grounded": result["grounded"],
            "retrieved_chunks_count": len(retrieved_chunks),
        }
