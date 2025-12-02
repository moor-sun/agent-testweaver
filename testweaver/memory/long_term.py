# memory/long_term.py
from typing import List, Tuple
import pathlib
import json

class LongTermMemory:
    """
    Minimal long-term memory abstraction.
    You can replace the internals with a real vector DB (Chroma, Qdrant, pgvector).
    """

    def __init__(self, root: str = "./data/vector_store"):
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_file = self.root / "index.json"
        if self.index_file.exists():
            self.index = json.loads(self.index_file.read_text(encoding="utf-8"))
        else:
            self.index = []

    def add_document(self, doc_id: str, text: str, meta: dict) -> None:
        self.index.append({"doc_id": doc_id, "text": text, "meta": meta})
        self.index_file.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, dict]]:
        # naive keyword search placeholder; swap with embeddings later
        scored = []
        q = query.lower()
        for item in self.index:
            score = item["text"].lower().count(q)
            scored.append((score, item["doc_id"], item["text"], item["meta"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(doc_id, text, meta) for score, doc_id, text, meta in scored[:top_k] if score > 0]
