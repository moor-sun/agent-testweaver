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
    Adapter over a Git MCP HTTP service.
    """

    def __init__(self, repo: str):
        self.repo = repo
        timeout = httpx.Timeout(
            connect=10.0,
            read=600.0,   # long reads (compile)
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

        if isinstance(data, dict):
            if "content" in data:
                content = data["content"]
                if data.get("encoding") == "base64":
                    try:
                        return base64.b64decode(content).decode("utf-8")
                    except Exception as e:
                        raise RuntimeError(f"failed to decode base64 content for '{path}': {e}")
                return content

            for k in ("file", "data", "result"):
                v = data.get(k)
                if isinstance(v, dict) and "content" in v:
                    content = v["content"]
                    if v.get("encoding") == "base64":
                        try:
                            return base64.b64decode(content).decode("utf-8")
                        except Exception as e:
                            raise RuntimeError(f"failed to decode base64 content for '{path}': {e}")
                    return content

            files = data.get("files")
            if isinstance(files, list) and files:
                first = files[0]
                if isinstance(first, dict) and "content" in first:
                    content = first["content"]
                    if first.get("encoding") == "base64":
                        try:
                            return base64.b64decode(content).decode("utf-8")
                        except Exception as e:
                            raise RuntimeError(f"failed to decode base64 content for '{path}': {e}")
                    return content

        # fallback: return text if not json shape
        text = resp.text
        if text and not text.isspace():
            return text

        raise RuntimeError(f"git-mcp /file returned no usable content for '{path}'. status={resp.status_code}")

    def list_java_files(self, base_path: str = "src/main/java") -> List[str]:
        resp = self.client.post("/list", json={"repo": self.repo, "base_path": base_path, "ext": ".java"})
        resp.raise_for_status()
        return resp.json().get("files", [])

    def get_pr_diff(self, pr_number: int) -> str:
        resp = self.client.post("/pr-diff", json={"repo": self.repo, "pr_number": pr_number})
        resp.raise_for_status()
        return resp.json().get("diff", "")

    def compile(
        self,
        tool: BuildTool = "maven",
        goal: BuildGoal = "test-compile",
        project_path: str = ".",
        timeout_seconds: int = 600,
        extra_args: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "repo": self.repo,
            "tool": tool,
            "goal": goal,
            "project_path": project_path,
            "timeout_seconds": timeout_seconds,
            "extra_args": extra_args or [],
        }

        http_timeout = timeout_seconds + 60
        resp = self.client.post("/compile", json=payload, timeout=http_timeout)

        # do not raise_for_status; normalize error shape
        if resp.status_code >= 400:
            try:
                body = resp.json()
            except Exception:
                body = {"raw": resp.text}
            return {"ok": False, "http_status": resp.status_code, "error": body}

        data = resp.json()
        if isinstance(data, dict):
            data.setdefault("ok", False)
            data.setdefault("stdout", "")
            data.setdefault("stderr", "")
            data.setdefault("returncode", -1)
        return data

    def write_file(self, path: str, content: str, overwrite: bool = True) -> Dict[str, Any]:
        resp = self.client.post(
            "/write-file",
            json={"repo": self.repo, "path": path, "content": content, "overwrite": overwrite},
        )
        if resp.status_code >= 400:
            try:
                return {"ok": False, "http_status": resp.status_code, "error": resp.json()}
            except Exception:
                return {"ok": False, "http_status": resp.status_code, "error": {"raw": resp.text}}
        return resp.json()

    def open_pr_on_success(
        self,
        branch: str,
        base_branch: str,
        title: str,
        body: str,
        files: List[str],
        commit_message: str,
        labels: Optional[List[str]] = None,
        remote: str = "origin",   # ✅ add remote
    ) -> Dict[str, Any]:
        resp = self.client.post(
            "/open-pr",
            json={
                "repo": self.repo,
                "branch": branch,
                "base_branch": base_branch,
                "title": title,
                "body": body or "",
                "files": files or [],
                "commit_message": commit_message,
                "labels": labels or [],
                "remote": remote,     # ✅ send remote
            },
        )
        if resp.status_code >= 400:
            try:
                return {"ok": False, "http_status": resp.status_code, "error": resp.json()}
            except Exception:
                return {"ok": False, "http_status": resp.status_code, "error": {"raw": resp.text}}
        return resp.json()

    def push_branch_with_commit(
        self,
        branch: str,
        base_branch: str,
        commit_message: str,
        files: List[str],
        remote: str = "origin",
    ) -> Dict[str, Any]:
        resp = self.client.post(
            "/push-branch",
            json={
                "repo": self.repo,
                "branch": branch,
                "base_branch": base_branch,
                "commit_message": commit_message,
                "files": files or [],
                "remote": remote,
            },
        )
        if resp.status_code >= 400:
            try:
                return {"ok": False, "http_status": resp.status_code, "error": resp.json()}
            except Exception:
                return {"ok": False, "http_status": resp.status_code, "error": {"raw": resp.text}}
        return resp.json()
