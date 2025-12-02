# rag/index.py
from typing import List
from ..memory.long_term import LongTermMemory

class RAGIndex:
    def __init__(self, store: LongTermMemory):
        self.store = store

    def ingest_text(self, doc_id: str, text: str, meta: dict):
        self.store.add_document(doc_id, text, meta)

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        results = self.store.search(query, top_k=top_k)
        context_chunks: List[str] = []
        for doc_id, text, meta in results:
            context_chunks.append(f"[DOC {doc_id} | {meta.get('type')}] {text}")
        return "\n\n".join(context_chunks)
