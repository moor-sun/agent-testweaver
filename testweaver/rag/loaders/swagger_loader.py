# rag/loaders/swagger_loader.py
import httpx
from typing import Any, Dict, List, Optional

def fetch_swagger_json(url: str) -> Dict[str, Any]:
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    return resp.json()

def _dedup(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out

def _collect_refs(obj: Any) -> List[str]:
    """Recursively collect all $ref values."""
    refs: List[str] = []
    if isinstance(obj, dict):
        ref = obj.get("$ref")
        if isinstance(ref, str):
            refs.append(ref)
        for v in obj.values():
            refs.extend(_collect_refs(v))
    elif isinstance(obj, list):
        for it in obj:
            refs.extend(_collect_refs(it))
    return _dedup(refs)

def _short_schema_signature(schema: Dict[str, Any]) -> str:
    """
    Compact signature for schemas: properties, types, enums, refs.
    Good for embeddings without bloating.
    """
    if not isinstance(schema, dict):
        return ""

    # $ref schema
    if "$ref" in schema and isinstance(schema["$ref"], str):
        return f"ref={schema['$ref']}"

    t = schema.get("type")
    fmt = schema.get("format")
    enum = schema.get("enum")
    items = schema.get("items")

    parts: List[str] = []
    if t:
        parts.append(f"type={t}{f'({fmt})' if fmt else ''}")
    if isinstance(enum, list) and enum:
        parts.append(f"enum={enum}")
    if isinstance(items, dict):
        parts.append(f"items[{_short_schema_signature(items)}]")

    # object props (only names + shallow type/ref to keep small)
    props = schema.get("properties")
    if isinstance(props, dict) and props:
        prop_bits = []
        for pname, pdef in props.items():
            if not isinstance(pdef, dict):
                continue
            if "$ref" in pdef:
                prop_bits.append(f"{pname}:ref({pdef['$ref']})")
            else:
                pt = pdef.get("type", "")
                pf = pdef.get("format", "")
                pen = pdef.get("enum")
                if isinstance(pen, list) and pen:
                    prop_bits.append(f"{pname}:{pt}{'('+pf+')' if pf else ''} enum={pen}")
                else:
                    prop_bits.append(f"{pname}:{pt}{'('+pf+')' if pf else ''}".strip(":"))
        parts.append("props={" + ", ".join(prop_bits) + "}")

    return "; ".join(parts)

def openapi_to_rag_chunks(
    openapi: Dict[str, Any],
    *,
    source_url: str = "",
    service_name: str = "svc-accounting",
    include_schemas: bool = True
) -> List[Dict[str, Any]]:
    """
    Best chunking for Swagger/OpenAPI RAG:
    - One chunk per operation (method+path)
    - One chunk per schema
    Returns: [{"text": str, "meta": dict}, ...]
    """
    chunks: List[Dict[str, Any]] = []

    # -------------------------
    # A) Operation chunks
    # -------------------------
    paths = openapi.get("paths") or {}
    for path, methods in paths.items():
        if not isinstance(methods, dict):
            continue

        for method, detail in methods.items():
            if not isinstance(detail, dict):
                continue

            method_up = str(method).upper()
            operation_id = detail.get("operationId", "") or ""
            tags = detail.get("tags", []) or []
            tags = tags if isinstance(tags, list) else [str(tags)]
            summary = detail.get("summary", "") or ""

            # parameters
            params = detail.get("parameters", []) or []
            param_lines: List[str] = []
            if isinstance(params, list):
                for p in params:
                    if not isinstance(p, dict):
                        continue
                    pname = p.get("name", "")
                    pin = p.get("in", "")
                    preq = bool(p.get("required", False))
                    pschema = p.get("schema", {}) or {}
                    p_sig = _short_schema_signature(pschema) if isinstance(pschema, dict) else ""
                    if pname:
                        param_lines.append(f"- {pname} in={pin} required={preq} {p_sig}".strip())

            # request body schemas
            req = detail.get("requestBody") or {}
            req_refs: List[str] = []
            req_lines: List[str] = []
            if isinstance(req, dict):
                content = req.get("content") or {}
                if isinstance(content, dict) and content:
                    req_lines.append("RequestBody:")
                    for ctype, cdef in content.items():
                        schema = (cdef.get("schema") or {}) if isinstance(cdef, dict) else {}
                        refs = _collect_refs(schema)
                        req_refs.extend(refs)
                        sig = _short_schema_signature(schema) if isinstance(schema, dict) else ""
                        req_lines.append(f"- {ctype} {sig}".strip())

            # response schemas
            resp = detail.get("responses") or {}
            resp_refs: List[str] = []
            resp_lines: List[str] = []
            if isinstance(resp, dict) and resp:
                resp_lines.append("Responses:")
                for code, rdef in resp.items():
                    if not isinstance(rdef, dict):
                        continue
                    content = rdef.get("content") or {}
                    if isinstance(content, dict) and content:
                        for ctype, cdef in content.items():
                            schema = (cdef.get("schema") or {}) if isinstance(cdef, dict) else {}
                            refs = _collect_refs(schema)
                            resp_refs.extend(refs)
                            sig = _short_schema_signature(schema) if isinstance(schema, dict) else ""
                            resp_lines.append(f"- {code} {ctype} {sig}".strip())
                    else:
                        resp_lines.append(f"- {code} (no content)")

            req_refs = _dedup(req_refs)
            resp_refs = _dedup(resp_refs)

            # IMPORTANT: put all key tokens into text for embedding match
            text_lines = [
                "OpenAPI Operation",
                f"service: {service_name}",
                f"method: {method_up}",
                f"path: {path}",
                f"operationId: {operation_id}",
                f"tags: {', '.join(tags)}" if tags else "tags: ",
                f"summary: {summary}" if summary else "summary: ",
            ]
            if param_lines:
                text_lines.append("Parameters:")
                text_lines.extend(param_lines)
            if req_lines:
                text_lines.extend(req_lines)
            if resp_lines:
                text_lines.extend(resp_lines)

            chunks.append({
                "text": "\n".join(text_lines),
                "meta": {
                    "type": "operation",
                    "service": service_name,
                    "source": source_url,
                    "method": method_up,
                    "path": path,
                    "operationId": operation_id,
                    "tags": tags,
                    "request_schema_refs": req_refs,
                    "response_schema_refs": resp_refs,
                }
            })

    # -------------------------
    # B) Schema chunks
    # -------------------------
    if include_schemas:
        schemas = (((openapi.get("components") or {}).get("schemas")) or {})
        if isinstance(schemas, dict):
            for name, sdef in schemas.items():
                sdef = sdef if isinstance(sdef, dict) else {}
                required = sdef.get("required", []) or []
                required = required if isinstance(required, list) else [str(required)]
                props = sdef.get("properties", {}) or {}

                # property lines (include enums + refs; keeps retrieval accurate)
                prop_lines: List[str] = []
                if isinstance(props, dict):
                    for pname, pdef in props.items():
                        if not isinstance(pdef, dict):
                            continue
                        if "$ref" in pdef:
                            prop_lines.append(f"- {pname}: ref({pdef['$ref']})")
                        else:
                            pt = pdef.get("type", "")
                            pf = pdef.get("format", "")
                            pen = pdef.get("enum")
                            if isinstance(pen, list) and pen:
                                prop_lines.append(f"- {pname}: {pt}{'('+pf+')' if pf else ''} enum={pen}")
                            else:
                                prop_lines.append(f"- {pname}: {pt}{'('+pf+')' if pf else ''}".strip(": "))

                refs = _collect_refs(sdef)
                ref_path = f"#/components/schemas/{name}"

                text_lines = [
                    "OpenAPI Schema",
                    f"service: {service_name}",
                    f"name: {name}",
                    f"ref: {ref_path}",
                    f"required: {required}",
                    "properties:",
                    *prop_lines
                ]

                chunks.append({
                    "text": "\n".join(text_lines),
                    "meta": {
                        "type": "schema",
                        "service": service_name,
                        "source": source_url,
                        "schema_name": name,
                        "ref": ref_path,
                        "schema_refs": refs,
                    }
                })

    return chunks
