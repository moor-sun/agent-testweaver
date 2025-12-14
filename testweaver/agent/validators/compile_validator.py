# agent/validators/compile_validator.py

from typing import Dict, Any
from mcp.git_client import MCPGitClient

class CompileValidator:
    """
    Validates generated test code by compiling it via MCP Git Server.
    """

    def __init__(self, repo_name: str):
        self.git = MCPGitClient(repo=repo_name)

    def validate(self) -> Dict[str, Any]:
        """
        Runs test compilation and returns structured result.
        """
        result = self.git.compile(
            tool="maven",
            goal="test-compile",
            project_path=".",
            timeout_seconds=300,
            extra_args=["-DskipTests=true"]
        )
        return result
