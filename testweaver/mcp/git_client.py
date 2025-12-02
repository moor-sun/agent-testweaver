# mcp/git_client.py
import os
import httpx
from typing import List

GIT_MCP_ENDPOINT = os.getenv("GIT_MCP_ENDPOINT", "http://localhost:9000/git-mcp")
GIT_TOKEN = os.getenv("GIT_TOKEN")

class MCPGitClient:
    """
    Adapter over a Git MCP tool or HTTP service.
    For now, keep method signatures simple for LLM to reason on top.
    """

    def __init__(self, repo: str):
        self.repo = repo
        self.client = httpx.Client(
            base_url=GIT_MCP_ENDPOINT,
            headers={"Authorization": f"Bearer {GIT_TOKEN}"} if GIT_TOKEN else {}
        )

    def get_file(self, path: str) -> str:
        resp = self.client.post("/file", json={"repo": self.repo, "path": path})
        resp.raise_for_status()
        return resp.json()["content"]

    def list_java_files(self, base_path: str = "src/main/java") -> List[str]:
        resp = self.client.post("/list", json={"repo": self.repo, "base_path": base_path, "ext": ".java"})
        resp.raise_for_status()
        return resp.json()["files"]

    def get_pr_diff(self, pr_number: int) -> str:
        resp = self.client.post("/pr-diff", json={"repo": self.repo, "pr_number": pr_number})
        resp.raise_for_status()
        return resp.json()["diff"]
