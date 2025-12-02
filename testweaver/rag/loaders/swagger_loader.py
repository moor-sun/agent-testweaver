# rag/loaders/swagger_loader.py
import httpx
from typing import Dict

def fetch_swagger_json(url: str) -> Dict:
    """
    Fetches the OpenAPI/Swagger JSON from svc-accounting or any service.
    Example: http://localhost:8080/v3/api-docs
    """
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def summarise_swagger(openapi: Dict) -> str:
    """
    Minimal text summary for RAG: list paths, methods, schemas.
    The idea is to give LLM controller, DTO names, and payload shapes.
    """
    lines = ["OpenAPI Summary:"]
    for path, methods in openapi.get("paths", {}).items():
        for method, detail in methods.items():
            op_id = detail.get("operationId", "")
            summary = detail.get("summary", "")
            lines.append(f"{method.upper()} {path}  operationId={op_id}  summary={summary}")
    return "\n".join(lines)
