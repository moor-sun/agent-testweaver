# api/http_api.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from ..memory.long_term import LongTermMemory
from ..memory.short_term import ShortTermMemory
from ..rag.index import RAGIndex
from ..rag.loaders.pdf_loader import load_pdf_as_chunks
from ..rag.loaders.swagger_loader import fetch_swagger_json, summarise_swagger
from ..agent.core import TestWeaverAgent

app = FastAPI(
    title="TestWeaver Agent API",
    description="Agent for code-aware test case generation with RAG + MCP Git",
    version="0.1.0"
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
    text = summarise_swagger(openapi)
    doc_id = f"swagger:{url}"
    rag_index.ingest_text(doc_id, text, meta={"type": "swagger", "url": url})
    return {"status": "ok", "doc_id": doc_id}

@app.get("/rag/docs")
def list_rag_docs():
    docs = []
    for item in lt_memory.index:
        text = item.get("text", "")
        preview = text[:200] + ("..." if len(text) > 200 else "")
        docs.append(
            {
                "doc_id": item.get("doc_id"),
                "meta": item.get("meta", {}),
                "preview": preview,
                "length": len(text),
            }
        )
    return docs
