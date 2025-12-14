# agent/core.py
import pathlib
from typing import Optional, List, Dict, Any

from ..llm.client import LLMClient
from ..memory.short_term import ShortTermMemory
from ..rag.index import RAGIndex
from ..mcp.git_client import MCPGitClient
import re

class TestWeaverAgent:
    def __init__(self, session_id: str, rag_index: RAGIndex, short_term: ShortTermMemory, repo: str):
        self.session_id = session_id
        self.rag_index = rag_index
        self.short_term = short_term
        self.git = MCPGitClient(repo)
        self.llm = LLMClient()
        # Base directory where this file is located
        BASE_DIR = pathlib.Path(__file__).resolve().parent.parent  # one level up to testweaver/

        PROMPTS_DIR = BASE_DIR / "prompts"

        self.system_prompt = (PROMPTS_DIR / "system_agent.md").read_text(encoding="utf-8")
        self.test_prompt = (PROMPTS_DIR / "test_generation.md").read_text(encoding="utf-8")

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

    def generate_tests_for_file(
        self,
        service_path: str,
        extra_instructions: str = "",
        compile_after: bool = True,
        max_attempts: int = 3
    ) -> Dict[str, Any]:
        """
        Returns structured JSON so UI can show attempt logs easily.
        Retries only when Maven compilation fails (ok == False from compile result).
        Does NOT retry on tool/HTTP failures (because LLM can't fix env issues).
        """
        java_source = self.git.get_file(service_path)
       
        class_name = service_path.split("/")[-1].replace(".java", "")

        rag_query_parts = []

        if extra_instructions:
            rag_query_parts.append(extra_instructions)
            
        rag_query_parts.append(class_name)
        
        rag_query = " ".join(rag_query_parts).strip() or "test generation for service"
        
        rag_context = self.rag_index.retrieve_context(rag_query, top_k=5)
        
        user_msg = f"""
    You must generate JUnit tests for the following Java file:

    <source_path>{service_path}</source_path>

    <source_code>
    {java_source}
    </source_code>

    <context_from_docs>
    {rag_context}
    </context_from_docs>

    Additional instructions from the user:
    {extra_instructions}

    First, think if any business logic or requirements are unclear.
    If unclear, ask clarifying questions instead of directly generating tests.
    If clear, output ONLY a compilable Java test class.
    """

        base_messages = self._build_messages(user_msg, task_context=self.test_prompt)

        package_name = self._extract_package(java_source)
        test_path = self._guess_test_path(package_name, class_name)

        attempt_log: List[Dict[str, Any]] = []
        last_compile: Dict[str, Any] | None = None
        last_test_code = ""

        def _stderr_tail(comp: Dict[str, Any], n: int = 30) -> str:
            s = (comp.get("stderr") or "").strip()
            if not s:
                return ""
            lines = s.splitlines()
            return "\n".join(lines[-n:])

        for attempt in range(1, max_attempts + 1):
            messages = list(base_messages)

            # If we are retrying, append repair instruction using previous stderr tail
            if attempt > 1 and last_compile:
                diag = self._compile_diag(last_compile, n=80)
                tail = _stderr_tail(last_compile, n=80)
                messages.append({
                    "role": "user",
                    "content": f"""
            The generated test did NOT compile.

            Compiler output (tail):
            {diag or "[No compiler output captured. Check maven -q / logs / tool wrapper]"}

            Fix the test code so it compiles. Output ONLY the corrected Java test class.
            Do not change the production code. Do not include markdown fences.
            """
                })

                messages.append({
                    "role": "user",
                    "content": f"""
    The generated test did NOT compile.

    Compiler stderr (last lines):
    {tail}

    Fix the test code so it compiles. Output ONLY the corrected Java test class.
    Do not change the production code. Do not include markdown fences.
    """
                })

            # 1) Generate
            response = self.llm.chat(messages)
            raw = self._strip_code_fences(response)
            test_code = self._extract_java_class(raw)
            last_test_code = test_code

            # store memory trace
            self.short_term.append(
                self.session_id,
                "user",
                user_msg if attempt == 1 else f"[repair attempt {attempt}]"
            )
            self.short_term.append(self.session_id, "assistant", test_code)

            # 2) Write file
            print("🧾 Writing test file:", test_path)
            print("🧾 First 120 chars:\n", test_code[:120])
            print("🧾 Last 120 chars:\n", test_code[-120:])
       
            self.git.write_file(test_path, test_code, overwrite=True)

            # If compile disabled, return immediately
            if not compile_after:
                attempt_log.append({
                    "attempt": attempt,
                    "stage": "generated_only",
                    "ok": True
                })
                return {
                    "status": "GENERATED_ONLY",
                    "service_path": service_path,
                    "test_path": test_path,
                    "attempts_used": attempt,
                    "attempt_log": attempt_log,
                    "test_code": test_code
                }

            # 3) Compile
            last_compile = self.git.compile(
                tool="maven",
                goal="test-compile",
                project_path=".",
                timeout_seconds=300,
                extra_args=["-DskipTests=true"]
            )

            # ---- Handle TOOL/HTTP failures (do NOT retry) ----
            # If your git_client.compile returns wrapper like:
            # {"ok": False, "http_status": 500, "error": {...}}
            if last_compile.get("http_status"):
                attempt_log.append({
                    "attempt": attempt,
                    "stage": "compile",
                    "ok": False,
                    "error_type": "TOOL_FAILURE",
                    "http_status": last_compile.get("http_status"),
                    "error_summary": str(last_compile.get("error", ""))[:500],
                })
                return {
                    "status": "TOOL_FAILURE",
                    "service_path": service_path,
                    "test_path": test_path,
                    "attempts_used": attempt,
                    "attempt_log": attempt_log,
                    "test_code": test_code,
                    "compile": last_compile
                }

            # ---- Normal compile result ----
            ok = bool(last_compile.get("ok"))
            tail = _stderr_tail(last_compile, n=30)
            attempt_log.append({
                "attempt": attempt,
                "stage": "compile",
                "ok": ok,
                "returncode": last_compile.get("returncode"),
                "stderr_tail": tail,
                "duration_ms": last_compile.get("duration_ms"),
            })

            if ok:
                return {
                    "status": "SUCCESS",
                    "service_path": service_path,
                    "test_path": test_path,
                    "attempts_used": attempt,
                    "attempt_log": attempt_log,
                    "test_code": test_code,
                    "compile": last_compile
                }

            # If compile failed (ok == False), loop continues for retry
            # (only up to max_attempts)
            
        # All attempts failed
        return {
            "status": "COMPILATION_FAILED",
            "service_path": service_path,
            "test_path": test_path,
            "attempts_used": max_attempts,
            "attempt_log": attempt_log,
            "test_code": last_test_code,
            "compile": last_compile,
            "compile_stderr_tail": _stderr_tail(last_compile or {}, n=80)
        }

    def _extract_package(self, java_source: str) -> str:
        m = re.search(r"^\s*package\s+([\w\.]+)\s*;\s*$", java_source, re.MULTILINE)
        return m.group(1) if m else ""

    def _guess_test_path(self, package_name: str, class_name: str) -> str:
        pkg_path = package_name.replace(".", "/") if package_name else ""
        if pkg_path:
            return f"src/test/java/{pkg_path}/{class_name}Test.java"
        return f"src/test/java/{class_name}Test.java"

    def _strip_code_fences(self, text: str) -> str:
        """Remove markdown code fences or language hints from model output."""
        if not isinstance(text, str):
            return str(text)

        # common fenced code blocks ```java ... ``` or ``` ... ```
        # remove leading/trailing whitespace
        s = text.strip()

        # If the whole text is fenced, extract inner content
        m = re.search(r"^```(?:[a-zA-Z0-9_-]+)?\n([\s\S]*?)\n```$", s)
        if m:
            return m.group(1).strip()

        # Sometimes models return triple quotes or inline markers
        # Remove any leading 'java' markers on first line
        if s.startswith("java\n"):
            return s[len("java\n"):].strip()

        return s
    
    def _compile_diag(self, comp: Dict[str, Any], n: int = 60) -> str:
        stdout = (comp.get("stdout") or "").strip()
        stderr = (comp.get("stderr") or "").strip()

        merged = []
        if stdout:
            merged.append("=== STDOUT ===\n" + stdout)
        if stderr:
            merged.append("=== STDERR ===\n" + stderr)

        text = "\n\n".join(merged).strip()
        if not text:
            return ""

        lines = text.splitlines()
        return "\n".join(lines[-n:])

    def _extract_java_class(self, raw: str) -> str:
        if not raw:
            return ""

        t = raw.strip()

        # If fenced code exists, take the first fenced block
        m = re.search(r"```(?:java)?\s*(.*?)\s*```", t, flags=re.IGNORECASE | re.DOTALL)
        if m:
            t = m.group(1).strip()

        # Otherwise, cut everything before first "package"/"import"/"public class"/"class"
        starters = ["package ", "import ", "public class ", "class ", "@SpringBootTest", "@Test"]
        idxs = [t.find(s) for s in starters if t.find(s) != -1]
        if idxs:
            t = t[min(idxs):].lstrip()

        # Trim anything after last closing brace
        last = t.rfind("}")
        if last != -1:
            t = t[:last + 1].rstrip()

        return t
