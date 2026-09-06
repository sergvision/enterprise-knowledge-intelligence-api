# Enterprise Knowledge Intelligence API

FastAPI-based enterprise RAG API.

## Architecture

Client
→ FastAPI
→ FAISS semantic search
→ OpenAI embeddings
→ GPT generation
→ Source attribution

## Runtime

- FastAPI
- FAISS
- OpenAI API
- text-embedding-3-small
- gpt-5-mini

## Deployment

Target platform: Render

The OpenAI API key must be provided as an environment variable:

OPENAI_API_KEY

## API

GET /
GET /health
POST /api/v1/ask
