# api/http_api.py
from fastapi.middleware.cors import CORSMiddleware

import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from ..memory.long_term import LongTermMemory
from ..memory.short_term import ShortTermMemory
from ..rag.index import RAGIndex
from ..rag.loaders.pdf_loader import load_pdf_as_chunks
from ..rag.loaders.swagger_loader import fetch_swagger_json, openapi_to_rag_chunks
from ..agent.core import TestWeaverAgent
from fastapi import HTTPException
from testweaver.utils import config as settings



app = FastAPI(
    title="TestWeaver Agent API",
    description="Agent for code-aware test case generation with RAG + MCP Git",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev mode – wide open; you can tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

lt_memory = LongTermMemory()
st_memory = ShortTermMemory()
rag_index = RAGIndex(lt_memory)

SVC_REPO = os.getenv("GIT_REPO_SVC_ACCOUNTING", "moor-sun/svc-accounting")

class ChatRequest(BaseModel):
    session_id: str
    message: str
    query_for_rag: str | None = None

class GenerateTestsRequest(BaseModel):
    session_id: str
    service_path: str
    extra_instructions: str | None = None

@app.post("/chat")
def chat(req: ChatRequest):
    agent = TestWeaverAgent(req.session_id, rag_index, st_memory, SVC_REPO)
    answer = agent.chat(req.message, query_for_rag=req.query_for_rag)
    return {"reply": answer}

@app.post("/generate-tests")
def generate_tests(req: GenerateTestsRequest):
    agent = TestWeaverAgent(req.session_id, rag_index, st_memory, SVC_REPO)
    code = agent.generate_tests_for_file(req.service_path, extra_instructions=req.extra_instructions or "")
    return {"test_code": code}

@app.post("/ingest/pdf")
async def ingest_pdf(session_id: str = Form(...), file: UploadFile = File(...)):
    os.makedirs(settings.DOC_STORE_PATH, exist_ok=True)
    temp_path = os.path.join(settings.DOC_STORE_PATH, file.filename)
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    chunks = load_pdf_as_chunks(temp_path, max_chars=1200, overlap_chars=200)

    for i, chunk in enumerate(chunks):
        doc_id = f"pdf:{file.filename}:chunk:{i}"
        rag_index.ingest_text(
            doc_id,
            chunk,
            meta={
                "type": "pdf",
                "session_id": session_id,
                "filename": file.filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
            },
        )

    return {"status": "ok", "chunks": len(chunks)}

@app.post("/ingest/swagger")
def ingest_swagger(url: str):
    openapi = fetch_swagger_json(url)

    chunks = openapi_to_rag_chunks(
        openapi,
        source_url=url,
        service_name="svc-accounting"
    )

    count = 0
    for ch in chunks:
        m = ch["meta"]
        if m["type"] == "operation":
            doc_id = f"swagger::op::{m['method']}::{m['path']}"
        else:
            doc_id = f"swagger::schema::{m['schema_name']}"

        rag_index.ingest_text(doc_id, ch["text"], meta=m)
        count += 1

    return {"ok": True, "chunks_ingested": count}


@app.get("/rag/docs")
def list_rag_docs(limit: int = 100):
    """
    List documents currently stored in long-term memory (Qdrant).

    `limit` controls how many docs we return from the first scroll page.
    """
    try:
        docs = lt_memory.list_documents(limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error listing RAG documents: {e}",
        )

    # Shape into a clean API response
    return {
        "limit": limit,
        "count": len(docs),
        "docs": docs,
    }

from fastapi import HTTPException

@app.get("/rag/chunks")
def list_chunks(limit: int = 20):
    """
    Shows actual stored chunks (doc_id, meta, and text preview).
    """
    try:
        points, _ = lt_memory.client.scroll(
            collection_name=lt_memory.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False
        )
    except Exception as e:
        raise HTTPException(500, f"Error reading chunks: {e}")

    out = []
    for p in points:
        payload = p.payload or {}
        doc_id = payload.get("doc_id", p.id)
        meta = payload.get("meta", {})
        text = payload.get("text", "")[:300]  # preview first 300 chars

        out.append({
            "qdrant_id": p.id,
            "doc_id": doc_id,
            "meta": meta,
            "text_preview": text
        })

    return {
        "count": len(out),
        "chunks": out
    }

@app.delete("/rag/docs/{doc_id}")
def delete_rag_doc(doc_id: str):
    """
    Delete a single RAG document by its logical doc_id.
    Example:
      DELETE /rag/docs/pdf:accounting_domain_business_details.pdf:chunk:0
    """
    ok = lt_memory.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    return {"deleted": True, "doc_id": doc_id}


@app.delete("/rag/docs")
def delete_rag_docs(doc_id: str | None = None):
    """
    Delete documents from RAG storage.

    - If `doc_id` is provided as a query parameter, delete that single document.
    - If `doc_id` is omitted or empty, delete ALL RAG content.

    Examples:
      DELETE /rag/docs?doc_id=pdf:foo.pdf:chunk:0
      DELETE /rag/docs  # deletes everything
    """
    try:
        ok = lt_memory.delete_document(doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting RAG documents: {e}")

    # When deleting a specific doc, preserve old behavior and return 404 if not found
    if doc_id:
        if not ok:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
        return {"deleted": True, "doc_id": doc_id}

    # Deleting all
    return {"deleted_all": True, "ok": bool(ok)}

