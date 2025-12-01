# agent/core.py
import pathlib
from typing import Optional

from ..llm.client import LLMClient
from ..memory.short_term import ShortTermMemory
from ..rag.index import RAGIndex
from ..mcp.git_client import MCPGitClient

class TestWeaverAgent:
    def __init__(self, session_id: str, rag_index: RAGIndex, short_term: ShortTermMemory, repo: str):
        self.session_id = session_id
        self.rag_index = rag_index
        self.short_term = short_term
        self.git = MCPGitClient(repo)
        self.llm = LLMClient()
        self.system_prompt = pathlib.Path("prompts/system_agent.md").read_text(encoding="utf-8")
        self.test_prompt = pathlib.Path("prompts/test_generation.md").read_text(encoding="utf-8")

    def _build_messages(self, user_message: str, task_context: Optional[str] = None):
        history = self.short_term.get_history(self.session_id)
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(history)  # prior turns
        if task_context:
            messages.append({"role": "user", "content": f"<context>\n{task_context}\n</context>"})
        messages.append({"role": "user", "content": user_message})
        return messages

    def chat(self, user_message: str, query_for_rag: Optional[str] = None) -> str:
        # retrieve RAG context if query provided
        task_context = ""
        if query_for_rag:
            task_context = self.rag_index.retrieve_context(query_for_rag, top_k=5)

        messages = self._build_messages(user_message, task_context=task_context)
        response = self.llm.chat(messages)
        self.short_term.append(self.session_id, "user", user_message)
        self.short_term.append(self.session_id, "assistant", response)
        return response

    def generate_tests_for_file(self, service_path: str, extra_instructions: str = "") -> str:
        java_source = self.git.get_file(service_path)

        rag_context = self.rag_index.retrieve_context("accounting service", top_k=5)

        user_msg = f"""
You must generate JUnit tests for the following Java file:

<source_path>{service_path}</source_path>

<source_code>
{java_source}
</source_code>

<context_from_docs>
{rag_context}
</context_from_docs>

Additional instructions:
{extra_instructions}

First, think if any business logic or requirements are unclear.
If unclear, ask clarifying questions instead of directly generating tests.
If clear, output ONLY a compilable Java test class.
"""

        messages = self._build_messages(user_msg, task_context=self.test_prompt)
        response = self.llm.chat(messages)
        self.short_term.append(self.session_id, "user", user_msg)
        self.short_term.append(self.session_id, "assistant", response)
        return response
