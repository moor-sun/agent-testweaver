# memory/long_term.py
from typing import List, Tuple, Dict, Any, Optional
import pathlib
import hashlib
import json
from collections.abc import Mapping, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from sentence_transformers import SentenceTransformer


class LongTermMemory:
    """
    Long-term memory backed by Qdrant vector DB.

    - Stores each document as a vector + payload
    - Uses SentenceTransformers for embeddings
    - Search is semantic (vector similarity)

    Requirements:
        pip install qdrant-client sentence-transformers
    """

    def __init__(
        self,
        collection_name: str = "testweaver_memory",
        qdrant_url: str = "http://localhost:6333",
        qdrant_api_key: Optional[str] = None,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        local_qdrant_path: Optional[str] = None,
    ):

        self.collection_name = collection_name

        # Embedding model
        self._embedder = SentenceTransformer(embedding_model_name)
        self.vector_dim = self._embedder.get_sentence_embedding_dimension()

        # Validate embedding dimension once at startup
        test_vec = self._embedder.encode("dim check")
        dim = len(test_vec)
        print("Embedder model:", embedding_model_name, "dim:", dim)

        if dim != 384:
            raise RuntimeError(
                f"Embedder dim is {dim}, but Qdrant collection expects 384. "
                f"Use all-MiniLM-L6-v2 (384) or recreate the collection to match."
            )
        
        # Qdrant client: embedded (file-based) or remote HTTP
        if local_qdrant_path:
            # Example: local_qdrant_path="./data/qdrant"
            local_path = pathlib.Path(local_qdrant_path)
            local_path.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(
                path=str(local_path),  # embedded Qdrant
            )
        else:
            # Remote Qdrant (Docker / k8s)
            self.client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,  # can be None if not secured
            )

        # Ensure collection exists
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _ensure_collection(self) -> None:
        """
        Create the collection if it does not exist.
        """
        collections = self.client.get_collections()
        existing = {c.name for c in collections.collections}

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.vector_dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )

    def _embed(self, text) -> list[float]:
        # ---- Normalize input to a single string ----
        if isinstance(text, Mapping):
            # swagger/openapi dict etc.
            text = json.dumps(text, ensure_ascii=False)
        elif isinstance(text, (list, tuple)):
            # list of strings -> join
            text = "\n".join(map(str, text))
        elif not isinstance(text, str):
            # any other type -> stringify
            text = str(text)

        vec = self._embedder.encode(text)

        # sentence-transformers returns numpy array for single string,
        # and numpy 2D array for list input. Convert to python.
        if hasattr(vec, "tolist"):
            vec = vec.tolist()

        # If we still somehow got 2D embeddings, fail loudly with a clear message
        if vec and isinstance(vec[0], list):
            raise ValueError(
                f"_embed received iterable input and produced 2D embeddings: "
                f"shape=({len(vec)}, {len(vec[0])}). Ensure you pass a single string to ingest_text()."
            )

        if len(vec) != self.vector_dim:
            raise ValueError(f"EMBED DIM mismatch: got {len(vec)} expected {self.vector_dim}")

        return vec



    def _make_point_id(self, doc_id: str) -> int:
        """
        Create a stable unsigned integer ID from a doc_id string,
        using a hash. Qdrant requires int or UUID for point IDs.
        """
        h = hashlib.md5(doc_id.encode("utf-8")).hexdigest()
        # Take first 16 hex chars -> convert to int
        return int(h[:16], 16)
    # ------------------------------------------------------------------
    # Public API – same method signatures as your original class
    # ------------------------------------------------------------------

    def add_document(self, doc_id: str, text: str, meta: dict) -> None:
        """
        Add or update a document in Qdrant.

        doc_id: logical ID for your doc (e.g. "pdf:...:chunk:0").
        Stored as payload; point ID is a numeric hash.
        """
        if meta is None:
            meta = {}

        vector = self._embed(text)

        # ---- Ensure vector is plain Python list[float] ----
        # (some embedders return numpy arrays)
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Embedder returned invalid vector for doc_id={doc_id}: {type(vector)}")

        # ---- Check expected dimension from Qdrant collection ----
        try:
            vcfg = self.client.get_collection(self.collection_name).config.params.vectors
            expected_dim = getattr(vcfg, "size", None)  # works for single-vector collections
        except Exception as e:
            raise RuntimeError(f"Failed to read Qdrant collection config for {self.collection_name}: {e}")

        if expected_dim is None:
            raise RuntimeError(
                f"Could not detect vector size for collection {self.collection_name}. "
                f"Vectors config was: {vcfg}"
            )

        if len(vector) != expected_dim:
            raise ValueError(
                f"Embedding dim mismatch for doc_id={doc_id}: got {len(vector)} expected {expected_dim}. "
                f"Fix by using the same embedding model everywhere OR recreate the Qdrant collection "
                f"with the correct size."
            )

        point_id = self._make_point_id(doc_id)

        point = qmodels.PointStruct(
            id=point_id,
            vector=vector,  # single vector (matches your config: size=384)
            payload={
                "doc_id": doc_id,
                "text": text,
                "meta": meta,
            },
        )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[point],
        )

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, dict]]:
        """
        Semantic search using vector similarity.

        Returns: List of (doc_id, text, meta) tuples.
        """
        if not query or not query.strip():
            return []

        query_vector = self._embed(query)

        hits = None

        # Preferred: new Query API (qdrant-client >= 1.10)
        if hasattr(self.client, "query_points"):
            resp = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,          # dense vector
                limit=top_k,
                with_payload=True,
            )
            hits = resp.points or []

        # Older helper API: search(...)
        elif hasattr(self.client, "search"):
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        # Very old helper API: search_points(...)
        elif hasattr(self.client, "search_points"):
            hits = self.client.search_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            )
        else:
            raise RuntimeError(
                "QdrantClient has no query/search methods. "
                "Please upgrade qdrant-client."
            )

        results: List[Tuple[str, str, dict]] = []
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            doc_id = payload.get("doc_id", str(getattr(hit, "id", "")))
            text = payload.get("text", "")
            meta = payload.get("meta", {})
            results.append((doc_id, text, meta))

        return results


    def delete_document(self, doc_id: Optional[str] = None) -> bool:
        """
        Delete documents from the Qdrant collection.

        If `doc_id` is provided (string), delete that single document.
        If `doc_id` is None or empty, delete ALL RAG content in the collection by
        scrolling through stored points and deleting them in batches.

        Returns True if the delete request was issued successfully, False on error.
        """
        from qdrant_client.http import models as qmodels

        try:
            if doc_id:
                # Single-document delete (stable numeric point id)
                point_id = self._make_point_id(doc_id)
                self.client.delete(
                    collection_name=self.collection_name,
                    points_selector=qmodels.PointIdsList(points=[point_id]),
                )
                return True

            # Bulk delete: no doc_id provided -> delete everything in the collection
            # We'll use `scroll` to enumerate point ids and delete in batches.
            batch_size = 500
            points_to_delete = []

            offset = 0
            while True:
                pts, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    with_payload=False,
                    with_vectors=False,
                )
                if not pts:
                    break
                ids = [getattr(p, "id", None) for p in pts]
                ids = [i for i in ids if i is not None]
                if ids:
                    # delete this batch
                    self.client.delete(
                        collection_name=self.collection_name,
                        points_selector=qmodels.PointIdsList(points=ids),
                    )
                # If fewer than batch_size returned, we're done
                if len(pts) < batch_size:
                    break

            return True

        except Exception:
            return False


    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        List up to `limit` documents from Qdrant using scroll.

        Returns a list of dicts:
        - qdrant_id: internal numeric ID
        - doc_id: logical string ID (payload["doc_id"])
        - meta: metadata dict from payload["meta"]
        """
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        docs: List[Dict[str, Any]] = []
        for pt in points:
            payload = getattr(pt, "payload", None) or {}
            qdrant_id = getattr(pt, "id", None)

            docs.append(
                {
                    "qdrant_id": qdrant_id,
                    "doc_id": payload.get("doc_id", str(qdrant_id)),
                    "meta": payload.get("meta", {}),
                }
            )

        return docs
