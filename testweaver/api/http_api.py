# api/http_api.py
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
from fastapi.responses import StreamingResponse

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

import inspect

def _call_rag_method(fn, query: str, top_k: int):
    """
    Call a rag method with whatever parameter names it expects.
    Supports common signatures like:
      fn(query, top_k=5)
      fn(query, k=5)
      fn(text=query, limit=5)
    """
    sig = inspect.signature(fn)
    kwargs = {}

    # best-effort mapping
    if "query" in sig.parameters:
        kwargs["query"] = query
    elif "text" in sig.parameters:
        kwargs["text"] = query
    elif len(sig.parameters) >= 1:
        # if it only takes positional query, pass it positionally
        return fn(query, **({ "top_k": top_k } if "top_k" in sig.parameters else {}))

    if "top_k" in sig.parameters:
        kwargs["top_k"] = top_k
    elif "k" in sig.parameters:
        kwargs["k"] = top_k
    elif "limit" in sig.parameters:
        kwargs["limit"] = top_k
    elif "n" in sig.parameters:
        kwargs["n"] = top_k

    return fn(**kwargs)


def get_rag_hits(query: str, top_k: int = 5):
    """
    Return retrieved chunks for UI display.
    This auto-detects the retrieval method present on RAGIndex.
    """
    if not query:
        return []

    # Try common method names on RAGIndex
    candidate_methods = [
        "search",
        "query",
        "similarity_search",
        "retrieve",
        "get_relevant",
        "find_similar",
    ]

    results = None
    for name in candidate_methods:
        fn = getattr(rag_index, name, None)
        if callable(fn):
            results = _call_rag_method(fn, query=query, top_k=top_k)
            break

    if results is None:
        # No supported retrieval method found
        return []

    # Normalize results into UI-friendly list of dicts
    hits = []
    for r in (results or []):
        # if result is not dict-like, string it
        if not isinstance(r, dict):
            hits.append({
                "doc_id": "",
                "score": None,
                "meta": {},
                "text_preview": str(r)[:400],
            })
            continue

        hits.append({
            "doc_id": r.get("doc_id") or r.get("id") or r.get("point_id") or "",
            "score": r.get("score") or r.get("distance"),
            "meta": r.get("meta") or r.get("payload", {}).get("meta", {}) or {},
            "text_preview": (r.get("text") or r.get("payload", {}).get("text", "") or "")[:400],
        })

    return hits


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

    rag_query = req.query_for_rag or req.message
    rag_hits = get_rag_hits(rag_query, top_k=5) if rag_query else []

    answer = agent.chat(req.message, query_for_rag=req.query_for_rag)

    return {
        "reply": answer,
        "rag_hits": rag_hits
    }


from fastapi import HTTPException

@app.post("/generate-tests")
def generate_tests(req: GenerateTestsRequest):
    try:
        agent = TestWeaverAgent(req.session_id, rag_index, st_memory, SVC_REPO)

        # ✅ Build a query to retrieve relevant chunks for test generation
        rag_query = f"{req.service_path}\n{req.extra_instructions or ''}".strip()
        rag_hits = get_rag_hits(rag_query, top_k=5) if rag_query else []

        result = agent.generate_tests_for_file(
            req.service_path,
            extra_instructions=req.extra_instructions or "",
            compile_after=True,
            max_attempts=3
        )

        # Ensure test_code is always a string (prevents [object Object])
        tc = result.get("test_code", "")
        if not isinstance(tc, str):
            result["test_code"] = str(tc)

        # ✅ Add rag_hits to keep response consistent with /chat
        result["rag_hits"] = rag_hits
        result["rag_query"] = rag_query  # optional but helpful for debugging/UI

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}")


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

@app.get("/generate-tests/stream")
def generate_tests_stream(service_path: str, extra_instructions: str = "", repo: str = "svc-accounting"):
    """
    Server-Sent Events stream. UI can listen and update status live.
    """

    async def event_generator():
        # Create agent (adapt constructor args to your setup)
        agent = TestWeaverAgent(
            session_id="ui-session",
            rag_index=rag_index,
            short_term=st_memory,
            repo=repo
        )

        # We replicate the bounded loop but emit events for UI
        max_attempts = 3

        # initial event
        yield f"data: {json.dumps({'stage':'start','message':'Starting test generation','max_attempts':max_attempts})}\n\n"
        await asyncio.sleep(0)

        # Call core method but “manual stream” progress:
        # easiest: run the new core method and just stream attempt_log at end (low effort)
        # better: copy the attempt loop here and emit after each stage.
        result = agent.generate_tests_for_file(
            service_path=service_path,
            extra_instructions=extra_instructions,
            compile_after=True,
            max_attempts=max_attempts
        )

        # final event
        yield f"data: {json.dumps({'stage':'done','result':result})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")