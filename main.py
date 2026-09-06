import os
import pickle
import numpy as np
import faiss

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================

FAISS_PATH = "enterprise_index.faiss"
VECTORS_PATH = "enterprise_vectors.pkl"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY environment variable is not configured."
    )

client = OpenAI(api_key=OPENAI_API_KEY)


# ============================================================
# Load persistent knowledge base
# ============================================================

enterprise_index = faiss.read_index(FAISS_PATH)

with open(VECTORS_PATH, "rb") as f:
    enterprise_vectors = pickle.load(f)


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Enterprise Knowledge Intelligence API",
    version="1.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://healthsportvoyageai.com",
        "https://www.healthsportvoyageai.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Request model
# ============================================================

class AskRequest(BaseModel):
    question: str


# ============================================================
# Enterprise RAG
# ============================================================

def enterprise_rag_query(question, search_k=3):

    if not question or not isinstance(question, str):
        return {
            "status": "ERROR",
            "error": "Question must be a non-empty string."
        }

    question = question.strip()

    query_response = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    )

    query_vector = np.array(
        [query_response.data[0].embedding],
        dtype="float32"
    )

    faiss.normalize_L2(query_vector)

    actual_k = min(
        search_k,
        enterprise_index.ntotal
    )

    similarities, indices = enterprise_index.search(
        query_vector,
        actual_k
    )

    retrieved_sources = []

    for similarity, index_position in zip(
        similarities[0],
        indices[0]
    ):

        if index_position < 0:
            continue

        record = enterprise_vectors[index_position]

        retrieved_sources.append({
            "similarity": round(float(similarity), 4),
            "chunk_id": record["chunk_id"],
            "document_id": record["document_id"],
            "document_title": record["document_title"],
            "department": record["department"],
            "text": record["text"]
        })

    context_parts = []

    for rank, source in enumerate(
        retrieved_sources,
        start=1
    ):

        context_parts.append(
            f"""
Source {rank}:
Document ID: {source["document_id"]}
Document: {source["document_title"]}
Department: {source["department"]}
Chunk ID: {source["chunk_id"]}
Similarity: {source["similarity"]}

Text:
{source["text"]}
"""
        )

    context = "\n".join(context_parts)

    rag_prompt = f"""
You are a reliable Enterprise Knowledge Assistant.

Answer the user's question using ONLY the enterprise
knowledge provided below.

Rules:

1. Use only information contained in the provided context.
2. Do not invent or assume information.
3. If the answer is not contained in the context, say:
   "The available enterprise knowledge does not contain
   this information."
4. Keep the answer concise and professional.
5. Identify the relevant enterprise document when appropriate.

Enterprise Knowledge Context:

{context}

User Question:

{question}

Answer:
"""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a reliable enterprise "
                    "knowledge assistant."
                )
            },
            {
                "role": "user",
                "content": rag_prompt
            }
        ]
    )

    answer = response.choices[0].message.content.strip()

    primary_source = None

    if retrieved_sources:

        primary_source = {
            "document_id":
                retrieved_sources[0]["document_id"],

            "document_title":
                retrieved_sources[0]["document_title"],

            "chunk_id":
                retrieved_sources[0]["chunk_id"],

            "similarity":
                retrieved_sources[0]["similarity"]
        }

    return {
        "status": "SUCCESS",
        "question": question,
        "answer": answer,
        "primary_source": primary_source,
        "retrieved_sources": retrieved_sources,
        "retrieved_source_count": len(
            retrieved_sources
        )
    }


# ============================================================
# API response layer
# ============================================================

def build_api_response(question, rag_result):

    if not isinstance(rag_result, dict):

        return {
            "status": "ERROR",
            "error": "Invalid RAG result."
        }

    if rag_result.get("status") != "SUCCESS":

        return {
            "status": "ERROR",
            "question": question,
            "error": rag_result.get(
                "error",
                "Enterprise RAG request failed."
            )
        }

    primary_source = rag_result.get(
        "primary_source"
    )

    source = None

    if primary_source:

        source = {
            "document_id":
                primary_source.get("document_id"),

            "document_title":
                primary_source.get("document_title"),

            "chunk_id":
                primary_source.get("chunk_id"),

            "similarity":
                primary_source.get("similarity")
        }

    return {
        "status": "SUCCESS",
        "question": question,
        "answer": rag_result.get("answer"),
        "source": source,
        "retrieved_source_count":
            rag_result.get(
                "retrieved_source_count",
                0
            )
    }


# ============================================================
# Root endpoint
# ============================================================

@app.get("/")
def root():

    return {
        "status": "SUCCESS",
        "service": "Enterprise Knowledge Intelligence API",
        "version": "1.0"
    }


# ============================================================
# Health endpoint
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "HEALTHY",
        "service": "Enterprise Knowledge Intelligence API"
    }


# ============================================================
# Main RAG endpoint
# ============================================================

@app.post("/api/v1/ask")
def ask(request: AskRequest):

    question = request.question.strip()

    if not question:

        return {
            "status": "ERROR",
            "error": "Question cannot be empty."
        }

    try:

        rag_result = enterprise_rag_query(
            question,
            search_k=3
        )

        return build_api_response(
            question,
            rag_result
        )

    except Exception as e:

        return {
            "status": "ERROR",
            "question": question,
            "error": str(e)
        }
