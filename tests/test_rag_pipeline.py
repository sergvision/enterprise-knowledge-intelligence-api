
"""
Unit tests for the Enterprise RAG pipeline.

These tests use mock components to validate pipeline
orchestration without calling external APIs or FAISS.
"""

from unittest.mock import MagicMock

import pytest

from app.rag.pipeline import EnterpriseRAGPipeline


@pytest.fixture
def pipeline_components():
    """Create mocked RAG components."""

    embedding_service = MagicMock()
    retriever = MagicMock()
    answer_generator = MagicMock()

    pipeline = EnterpriseRAGPipeline(
        embedding_service=embedding_service,
        retriever=retriever,
        answer_generator=answer_generator,
    )

    return {
        "pipeline": pipeline,
        "embedding_service": embedding_service,
        "retriever": retriever,
        "answer_generator": answer_generator,
    }


def test_pipeline_returns_answer_and_sources(pipeline_components):
    """Verify successful question processing and response structure."""

    components = pipeline_components

    retrieved_chunks = [
        {
            "chunk_id": "CHUNK-001",
            "document_id": "DOC-001",
            "document_title": "Employee Handbook",
            "similarity": 0.91,
            "text": "Employees are entitled to annual leave.",
        }
    ]

    generated_result = {
        "answer": "Employees are entitled to annual leave.",
        "sources": [
            {
                "document_id": "DOC-001",
                "document_title": "Employee Handbook",
                "chunk_id": "CHUNK-001",
                "similarity": 0.91,
            }
        ],
        "grounded": True,
    }

    components["retriever"].retrieve_question.return_value = retrieved_chunks

    components["answer_generator"].generate_answer.return_value = (
        generated_result
    )

    result = components["pipeline"].answer_question(
        "What is the annual leave policy?"
    )

    assert result["question"] == "What is the annual leave policy?"
    assert result["answer"] == generated_result["answer"]
    assert result["sources"] == generated_result["sources"]
    assert result["grounded"] is True
    assert result["retrieved_chunks_count"] == 1

    components["retriever"].retrieve_question.assert_called_once_with(
        "What is the annual leave policy?",
        components["embedding_service"],
    )

    components["answer_generator"].generate_answer.assert_called_once_with(
        "What is the annual leave policy?",
        retrieved_chunks,
    )


@pytest.mark.parametrize(
    "question",
    [
        "",
        "   ",
        None,
        123,
    ],
)
def test_pipeline_rejects_invalid_questions(
    pipeline_components,
    question,
):
    """Verify that invalid questions are rejected."""

    with pytest.raises(ValueError, match="Question cannot be empty"):
        pipeline_components["pipeline"].answer_question(question)

    pipeline_components["retriever"].retrieve_question.assert_not_called()

    pipeline_components["answer_generator"].generate_answer.assert_not_called()


def test_pipeline_handles_no_retrieved_chunks(pipeline_components):
    """Verify response when the retriever finds no relevant chunks."""

    components = pipeline_components

    components["retriever"].retrieve_question.return_value = []

    components["answer_generator"].generate_answer.return_value = {
        "answer": (
            "I could not find relevant information "
            "in the available knowledge base."
        ),
        "sources": [],
        "grounded": False,
    }

    result = components["pipeline"].answer_question(
        "What is the company's travel policy?"
    )

    assert result["answer"].startswith("I could not find relevant information")
    assert result["sources"] == []
    assert result["grounded"] is False
    assert result["retrieved_chunks_count"] == 0


def test_pipeline_propagates_retrieval_errors(pipeline_components):
    """Verify that retrieval errors are not silently swallowed."""

    components = pipeline_components

    components["retriever"].retrieve_question.side_effect = RuntimeError(
        "Retrieval service unavailable"
    )

    with pytest.raises(
        RuntimeError,
        match="Retrieval service unavailable",
    ):
        components["pipeline"].answer_question(
            "What is the company policy?"
        )

    components["answer_generator"].generate_answer.assert_not_called()
