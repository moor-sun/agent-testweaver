# TestWeaver - AI Agent for Automated Test Case Generation

TestWeaver is an AI-powered agent that generates tests with Retrieval-Augmented Generation (RAG) and Git-backed code access.

---

## Features

- FastAPI HTTP API with Swagger UI  
- LLM client for OpenAI-compatible endpoints (local or cloud)  
- Prompt-driven chat and test-generation flows  
- **RAG backed by Qdrant vector DB (Docker or embedded)**  
- RAG ingestion for PDFs (chunked) and Swagger specs  
- RAG document listing and deletion endpoints  
- Session-based short-term memory; Qdrant-backed long-term memory

---

## Project Structure

```
testweaver/
  data/
    qdrant_store/          # (optional) local Qdrant storage if using embedded mode
  testweaver/
    agent/                 # Agent orchestration
    api/                   # FastAPI routes
    llm/                   # LLM client
    mcp/                   # MCP Git client
    memory/                # Short- and long-term memory
    prompts/               # Prompt templates
    rag/                   # RAG loaders + index + Qdrant integration
    utils/                 # Logging/config helpers
    main.py                # CLI entrypoint target
  .env                     
  .env.example             
  pyproject.toml           
  README.md
```

---

## Prerequisites

- Python 3.10+  
- Poetry  
- An OpenAI-compatible LLM endpoint (local Ollama/LM Studio/etc. or cloud)  
- **Qdrant vector DB**, either via Docker or embedded

---

## Setup

### 1) Clone the repository

```
git clone <your-repo-url>
cd testweaver
```

### 2) Install dependencies

```
poetry install
```

### 3) Start Qdrant (Recommended: Docker)

```
docker run -p 6333:6333 -v qdrant_storage:/qdrant/storage qdrant/qdrant
```

- Qdrant API URL: `http://localhost:6333`
- Data persists inside Docker volume `qdrant_storage`

> TestWeaver uses **Qdrant HTTP mode** by default.

### 4) Configure environment variables

TestWeaver does not auto-load `.env`.  
Set them in your shell:

```
LLM_BASE_URL=<http://localhost:11434/v1 or https://api.openai.com/v1>
LLM_API_KEY=<token or blank if local>
LLM_MODEL_NAME=<model-id>
GIT_REPO_SVC_ACCOUNTING=<org/repo>
GIT_TOKEN=<token>

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=testweaver_memory

TESTWEAVER_HOST=0.0.0.0
TESTWEAVER_PORT=9090
```

PowerShell example:

```
$env:LLM_BASE_URL="http://localhost:11434/v1"
$env:LLM_MODEL_NAME="llama3.1"
$env:QDRANT_URL="http://localhost:6333"
$env:QDRANT_COLLECTION="testweaver_memory"
```

---

## Running the API

```
poetry run uvicorn testweaver.api.http_api:app --host 0.0.0.0 --port 9090
```

Swagger UI:  
`http://localhost:9090/docs`

---

## RAG & Vector Store (Qdrant)

### Long-Term Memory

TestWeaver now uses **Qdrant** as its vector store.

Each chunk stored includes:

- Numeric point ID (hash of `doc_id`)  
- Logical doc ID (`pdf:<file>:chunk:n`)  
- Payload containing:  
  - `text`  
  - `meta` (source filename, type, page, etc.)

### Ingestion

- `POST /ingest/pdf`  
  Takes a PDF → extracts text → chunks → stores chunks in Qdrant.

- `POST /ingest/swagger`  
  Stores summarized Swagger text blocks into Qdrant.

### Inspection

- `GET /rag/docs`  
  Lists documents stored in Qdrant.

- `GET /rag/chunks`  
  Shows chunk content previews (doc_id, meta, text sample).

### Deletion

- `DELETE /rag/docs/{doc_id}`  
  Deletes a **single** chunk mapped by `doc_id`.

*(You are **not** using the delete-by-source endpoint in this version.)*

---

## Prompts

- `testweaver/prompts/system_agent.md`  
- `testweaver/prompts/test_generation.md`  

Customize these to tune agent behavior & output quality.

---

## Notes

- `.env` is not auto-loaded; export environment variables manually or use:  
  ```
  python -m dotenv run -- uvicorn testweaver.api.http_api:app --host 0.0.0.0 --port 9090
  ```

- Works with:
  - Ollama (OpenAI-compatible server mode)
  - LM Studio
  - OpenAI API
  - Groq API (OpenAI format)

---

## Author

M.Tech AI/ML Dissertation Project  
**Sundaramoorthy N**  
BITS Pilani WILP Program
