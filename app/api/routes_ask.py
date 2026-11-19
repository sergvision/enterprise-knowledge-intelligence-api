"""
Question and answer API routes.

Handles enterprise knowledge queries.
"""

from fastapi import APIRouter

router = APIRouter(
    prefix="/api/v1",
    tags=["Enterprise Knowledge"]
)
