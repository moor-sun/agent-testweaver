# mcp/git_client.py
import os
import base64
import httpx
from typing import List, Optional, Literal, Dict, Any

GIT_MCP_ENDPOINT = os.getenv("GIT_MCP_ENDPOINT", "http://localhost:9000/git-mcp")
GIT_TOKEN = os.getenv("GIT_TOKEN")

BuildTool = Literal["maven", "gradle"]
BuildGoal = Literal["test-compile", "test", "compile"]

class MCPGitClient:
    """
    Adapter over a Git MCP tool or HTTP service.
    For now, keep method signatures simple for LLM to reason on top.
    """

    def __init__(self, repo: str):
        self.repo = repo
        timeout = httpx.Timeout(
            connect=10.0,
            read=600.0,   # ✅ allow long reads (compile)
            write=30.0,
            pool=10.0
        )
        self.client = httpx.Client(
            base_url=GIT_MCP_ENDPOINT,
            headers={"Authorization": f"Bearer {GIT_TOKEN}"} if GIT_TOKEN else {},
            timeout=timeout
        )

    def get_file(self, path: str) -> str:
        resp = self.client.post("/file", json={"repo": self.repo, "path": path})
        resp.raise_for_status()
        data = resp.json()

        # Common cases: top-level 'content' possibly with 'encoding'
        if isinstance(data, dict):
            if "content" in data:
                content = data["content"]
                if data.get("encoding") == "base64":
                    try:
                        return base64.b64decode(content).decode("utf-8")
                    except Exception as e:
                        raise RuntimeError(f"failed to decode base64 content from git-mcp for '{path}': {e}")
                return content

            # Nested shapes: {"file": {"content":...}} or {"data": {...}}
            for k in ("file", "data", "result"):
                v = data.get(k)
                if isinstance(v, dict) and "content" in v:
                    content = v["content"]
                    if v.get("encoding") == "base64":
                        try:
                            return base64.b64decode(content).decode("utf-8")
                        except Exception as e:
                            raise RuntimeError(f"failed to decode base64 content from git-mcp: {e}")
                    return content

            # Sometimes the endpoint returns a list under 'files'
            files = data.get("files")
            if isinstance(files, list) and files:
                first = files[0]
                if isinstance(first, dict) and "content" in first:
                    content = first["content"]
                    if first.get("encoding") == "base64":
                        try:
                            return base64.b64decode(content).decode("utf-8")
                        except Exception as e:
                            raise RuntimeError(f"failed to decode base64 content from git-mcp for '{path}': {e}")
                    return content

        # Fall back behaviors:
        # - If the endpoint returned plain text body (not JSON), return it
        # - Otherwise raise a clear error including status and body for debugging
        text = resp.text
        if text and not text.isspace():
            return text

        # No usable content found
        body_preview = None
        try:
            body_preview = resp.content.decode('utf-8', errors='replace')
        except Exception:
            body_preview = str(resp.content)

        raise RuntimeError(
            f"git-mcp /file returned no 'content' for path '{path}'. "
            f"Status={resp.status_code} Body={body_preview} ParsedJSON={data}"
        )

    def list_java_files(self, base_path: str = "src/main/java") -> List[str]:
        resp = self.client.post("/list", json={"repo": self.repo, "base_path": base_path, "ext": ".java"})
        resp.raise_for_status()
        return resp.json()["files"]
    
    def get_pr_diff(self, pr_number: int) -> str:
        resp = self.client.post("/pr-diff", json={"repo": self.repo, "pr_number": pr_number})
        resp.raise_for_status()
        return resp.json()["diff"]

    def compile(
        self,
        tool: BuildTool = "maven",
        goal: BuildGoal = "test-compile",
        project_path: str = ".",
        timeout_seconds: int = 300,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "repo": self.repo,
            "tool": tool,
            "goal": goal,
            "project_path": project_path,
            "timeout_seconds": timeout_seconds,
            "extra_args": extra_args or []
        }

        # Ensure HTTP timeout is always longer than process timeout
        http_timeout = timeout_seconds + 60

        resp = self.client.post("/compile", json=payload, timeout=http_timeout)

        # 🚨 DO NOT raise_for_status here
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}

            # Standardize tool-failure wrapper (agent can detect and STOP retries)
            return {
                "ok": False,
                "http_status": resp.status_code,
                "error": body
            }

        data = resp.json()

        # Standardize compile success/failure shape even if server changes
        if isinstance(data, dict):
            data.setdefault("ok", False)
            data.setdefault("stdout", "")
            data.setdefault("stderr", "")
            data.setdefault("returncode", -1)

        return data
    
    def write_file(self, path: str, content: str, overwrite: bool = True) -> Dict[str, Any]:
        resp = self.client.post(
            "/write-file",
            json={"repo": self.repo, "path": path, "content": content, "overwrite": overwrite}
        )
        resp.raise_for_status()
        return resp.json()
