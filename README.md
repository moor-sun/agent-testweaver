# TestWeaver – AI Agent for Automated Test Case Generation

TestWeaver is an AI-powered agent designed for automated test case generation and Quality Engineering support.
It works with local LLMs (via Ollama) and can optionally use OpenAI GPT models when cloud access is available.

This project is part of an M.Tech AI/ML dissertation focusing on applying AI to improve test productivity and coverage.

---

## Features

- Local/offline LLM support through Ollama
- API-driven usage through FastAPI (Swagger UI included)
- Structured test case generation using prompt templates
- Extensible agent design (future: Jira / Sonar / RAG support)
- Fully configurable via .env

---

## Project Structure

TESTWEAVER/
    data/                        # Future use: RAG knowledge base
    testweaver/                  # Main Python package
        agent/                   # Agent logic and orchestration
        api/                     # FastAPI routes (HTTP entrypoints)
        llm/                     # LLM client integration (Ollama / OpenAI)
        mcp/                     # Future multi-agent control
        memory/                  # Future session/history support
        prompts/                 # Prompt templates for behavior and output style
        rag/                     # Placeholder for doc retrieval logic
        utils/                   # Utility helpers
        __init__.py
        main.py                  # Local entrypoint (if needed)
        test_openai.py           # Simple validation script for OpenAI config
    .env                         # Environment settings (ignored in Git)
    .env.example                 # Template for environment variables
    .gitignore                   # Ignoring caches, venv, secrets, etc.
    poetry.lock
    pyproject.toml               # Python dependency + entrypoint
    README.md                    # This file

---

## Prerequisites

- Python 3.10+ (recommended to use Conda environment)
- Poetry package manager
- One of the following LLM setups:
  - Ollama (recommended) – free and offline
  - OpenAI API key – requires billing/quota

---

## Installation and Setup

1. Clone the repository:
    git clone <your-repo-url>
    cd testweaver

2. Activate Python environment:
    conda activate mtech
    OR:
    python -m venv .venv
    .\.venv\Scriptsctivate

3. Install dependencies:
    poetry install

---

## LLM Configuration

### Option A — Use Local LLM via Ollama (Recommended)

Install Ollama for Windows:  
https://ollama.com/download/windows

Pull a model:
    ollama pull llama3.1

Update .env:
    LLM_BASE_URL=http://localhost:11434/v1
    OPENAI_API_KEY=ollama
    LLM_MODEL_NAME=llama3.1

### Option B — Use OpenAI GPT Models (requires billing)

Update .env:
    LLM_BASE_URL=https://api.openai.com/v1
    OPENAI_API_KEY=sk-<your-key>
    LLM_MODEL_NAME=gpt-4o-mini

---

## Running the Application

Start FastAPI:
    poetry run uvicorn testweaver.api.http_api:app --reload

Open Swagger:
    http://localhost:8000/docs

---

## Sample Request (Swagger → POST /chat)

{
  "session_id": "demo",
  "message": "Generate test cases for login page",
  "query_for_rag": ""
}

---

## Customizing Test Output

Edit files under testweaver/prompts:

- system_agent.md
- test_generation.md

---

## Git Guidelines

Important ignored folders/files:
- .env
- __pycache__/
- *.pyc

Cleanup accidental tracked caches:
    git rm -r --cached __pycache__
    git commit -m "Remove cached files"

---

## Future Enhancements

- Jira enrichment agent
- SonarQube findings review
- Memory per session ID
- Retrieval-Augmented Generation (RAG)
- Model selection based on context

---

## Author

M.Tech AI/ML Dissertation Project  
Author: Sundaramoorthy N  
BITS Pilani WILP Program