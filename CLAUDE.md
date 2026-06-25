# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Prerequisites

- **Ollama** must be running locally (`ollama serve`, default port 11434)
- Required models: `ollama pull embeddinggemma:latest llama3.2:latest`

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py

# Run all tests (unit only, skips integration tests that need Ollama)
pytest tests/ -v

# Run tests including integration (requires Ollama running)
pytest tests/ -v -m "integration or not integration"

# Run a single test file
pytest tests/test_document_loader.py -v

# Create .env from template before first run
cp .env.example .env
```

## Architecture

The app is a 100% local Agentic RAG system: users add documents → EmbeddingGemma produces vectors → LanceDB stores them → Llama 3.2 answers questions using retrieved context. The Agno framework orchestrates retrieval and generation.

### Layer stack

```
app.py                  Streamlit UI only — layout, session state, streaming render
  ├─ src/agent.py       AgentFactory: creates agno Agent with model + knowledge + instructions
  ├─ src/knowledge_base.py  KnowledgeBaseManager: coordinates loading → embedding → storage
  │     ├─ src/document_loader.py  Validates source type (URL vs local, supported formats)
  │     └─ src/vector_store.py    LanceDBStore wrapping agno's LanceDb (with optional Reranker)
  └─ src/config.py      All config centralized; reads from .env with sensible defaults
```

### Key abstractions (interview-ready design decisions)

- **BaseEmbedder** (`embedder.py`) — abstract interface; `OllamaEmbedder` is the current impl. Swap to `nomic-embed-text` or `bge-m3` by changing `EMBEDDING_MODEL` in `.env`.
- **BaseVectorStore** (`vector_store.py`) — abstract interface; `LanceDBStore` is the current impl. The abstraction exists to swap in ChromaDB/FAISS/Qdrant without changing upper layers.
- **AgentFactory** (`agent.py`) — classmethod pattern; `create()` builds the standard RAG agent, `create_with_custom_model()` enables A/B testing different LLMs.

### Reranker pipeline (optional, controlled by `RERANKER_ENABLED`)

When enabled, retrieval uses a two-stage approach: initial vector search fetches `RETRIEVAL_TOP_K` candidates (default 20), then a Cross-Encoder reranker (`sentence-transformers`) re-scores and keeps `RERANKER_TOP_N` (default 5). The reranker model is configured via `RERANKER_MODEL` in `.env`.

### Configuration

All tunable values live in `src/config.py` as class attributes on `Config`, each backed by an env var with a default. Copy `.env.example` → `.env` to override. The `config` singleton is importable from `src`.

### Tests

- Tests live in `tests/`, mirroring `src/` modules.
- Tests marked `@pytest.mark.integration` require Ollama running — CI should skip these by default.
- Unit tests (no marker) are pure logic tests and run always.
