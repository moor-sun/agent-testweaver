# rag/index.py
from typing import List
from ..memory.long_term import LongTermMemory
from ..utils.logging import logger  # use your shared logger

class RAGIndex:
    def __init__(self, store: LongTermMemory):
        self.store = store

    def ingest_text(self, doc_id: str, text: str, meta: dict):
        logger.debug("RAG ingest: doc_id=%s meta=%s", doc_id, meta)
        self.store.add_document(doc_id, text, meta)

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        1. Search using the user query
        2. If no hits, fall back to a generic accounting-ish query
        3. Log how many hits we got and what chunks we’re using
        4. Return a concatenated context string
        """
        logger.debug("RAG: primary search for query %r (top_k=%d)", query, top_k)
        results = self.store.search(query, top_k=top_k)

        if not results:
            logger.debug(
                "RAG: no hits for query %r, falling back to generic query", query
            )
            fallback_query = "account transaction balance error"
            results = self.store.search(fallback_query, top_k=top_k)

        if not results:
            logger.debug("RAG: still no hits after fallback for query %r", query)
            return ""

        logger.debug(
            "RAG: %d hit(s) for query %r (top_k=%d)",
            len(results),
            query,
            top_k,
        )

        context_chunks: List[str] = []
        for doc_id, text, meta in results:
            preview = (text[:200] + "...") if len(text) > 200 else text

            logger.debug(
                "RAG chunk used | doc_id=%s | meta=%s | preview=%r",
                doc_id,
                meta,
                preview,
            )

            # Try to show something human-friendly in the prefix
            source = (
                meta.get("source")
                or meta.get("file_path")
                or meta.get("type")
                or "unknown"
            )

            context_chunks.append(f"[SOURCE {source} | DOC {doc_id}]\n{text}")

        context = "\n\n---\n\n".join(context_chunks)
        logger.debug(
            "RAG: built context with %d chunks (%d chars) for query %r",
            len(context_chunks),
            len(context),
            query,
        )
        return context

    def delete(self, doc_id: str) -> bool:
        """Delete a document from the store by id."""
        logger.debug("RAG delete requested for doc_id=%s", doc_id)
        return self.store.delete_document(doc_id)