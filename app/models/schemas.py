"""
Pydantic schemas for Enterprise Knowledge Intelligence API.

Defines validated request and response contracts.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """Incoming enterprise knowledge question."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Natural-language question submitted by the user.",
    )


class SourceReference(BaseModel):
    """Reference to a knowledge-base document chunk."""

    document_id: str
    document_title: str
    chunk_id: str
    similarity: float = Field(ge=-1.0, le=1.0)


class AskResponse(BaseModel):
    """Successful answer returned by the question API."""

    status: Literal["SUCCESS"] = "SUCCESS"
    question: str
    answer: str
    source: SourceReference | None = None
    retrieved_source_count: int = Field(default=0, ge=0)


class ErrorResponse(BaseModel):
    """Standardized API error response."""

    status: Literal["ERROR"] = "ERROR"
    question: str | None = None
    error: str
