from qdrant_client import QdrantClient

COLLECTION = "testweaver_memory"   # <-- use the real name

client = QdrantClient(url="http://localhost:6333")

print("Collections:", [c.name for c in client.get_collections().collections])
print("Vector config:")
print(client.get_collection(COLLECTION).config.params.vectors)

# poetry run python testweaver/scripts/qdrant_debug.py
