# AGENTS.md

## Project: Legal Contract Intelligence System (Corrective RAG)

## Setup

run:
pip install -r requirements.txt

## Start Server

run:
uvicorn app.main:app --reload

## Guidelines

- Use modular Python architecture
- Use FastAPI for API
- Use Pinecone for vector DB
- Use PostgreSQL for metadata
- Implement Corrective RAG loop:
  - retrieve
  - evaluate
  - refine
  - regenerate

## Testing

run:
pytest

## Notes

- Avoid deprecated LangChain APIs
- Ensure all outputs are grounded in retrieved clauses
