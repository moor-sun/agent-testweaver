# agent/core.py
import pathlib
import os
import re
from typing import Optional, List, Dict, Any

from ..llm.client import LLMClient
from ..memory.short_term import ShortTermMemory
from ..rag.index import RAGIndex
from ..mcp.git_client import MCPGitClient


# --------------------------------------------------------------------------------------
# Maven compiler diagnostics extractor
# --------------------------------------------------------------------------------------

ERROR_PATTERNS = [
    r"\[ERROR\].*cannot find symbol",
    r"cannot find symbol",
    r"COMPILATION ERROR",
    r"Compilation failure",
    r"Failed to execute goal org\.apache\.maven\.plugins:maven-compiler-plugin",
    r"\[ERROR\].*\.java:\[\d+,\d+\]",
    r"\[ERROR\]\s+symbol:",
    r"\[ERROR\]\s+location:",
    r"\[ERROR\].*package .* does not exist",
    r"\[ERROR\].*is not public in",
    r"\[ERROR\].*incompatible types",
    r"\[ERROR\].*method .* cannot be applied to",
    r"\[ERROR\].*cannot access",
    r"\[ERROR\].*cannot be resolved",
    r"\[ERROR\].*class file for .* not found",
]


def extract_actionable_maven_error(maven_output: str, before: int = 60, after: int = 140) -> str:
    """
    Extracts the actionable compiler diagnostics from full Maven output.
    Prevents sending only stack traces to the LLM.
    """
    maven_output = (maven_output or "").strip()
    if not maven_output:
        return ""

    lines = maven_output.splitlines()
    regex = re.compile("|".join(ERROR_PATTERNS), re.IGNORECASE)

    hit_indices = [i for i, line in enumerate(lines) if regex.search(line)]
    if not hit_indices:
        return "\n".join(lines[-300:]).strip()

    start = max(0, hit_indices[0] - before)
    end = min(len(lines), hit_indices[-1] + after)
    return "\n".join(lines[start:end]).strip()


class TestWeaverAgent:
    def __init__(self, session_id: str, rag_index: RAGIndex, short_term: ShortTermMemory, repo: str):
        self.session_id = session_id
        self.rag_index = rag_index
        self.short_term = short_term
        self.git = MCPGitClient(repo)
        self.llm = LLMClient()

        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self._compiled_cache: Dict[str, str] = {}

        BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
        PROMPTS_DIR = BASE_DIR / "prompts"

        self.system_prompt = (PROMPTS_DIR / "system_agent.md").read_text(encoding="utf-8")
        self.test_prompt = (PROMPTS_DIR / "test_generation.md").read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str, query_for_rag: Optional[str] = None) -> str:
        """
        Simple conversational chat (used by /chat endpoint).
        Does NOT trigger test generation or compilation.
        """
        task_context = ""
        if query_for_rag:
            task_context = self.rag_index.retrieve_context(query_for_rag, top_k=5)

        messages = [{"role": "system", "content": self.system_prompt}]
        if task_context:
            messages.append({"role": "user", "content": task_context})
        messages.append({"role": "user", "content": user_message})

        response = self.llm.chat(messages, temperature=self.llm_temperature)

        # store short-term memory
        self.short_term.append(self.session_id, "user", user_message)
        self.short_term.append(self.session_id, "assistant", response)

        return response

    def generate_tests_for_file(
        self,
        service_path: str,
        extra_instructions: str = "",
        compile_after: bool = True,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:

        java_source = self.git.get_file(service_path)
        class_name = service_path.split("/")[-1].replace(".java", "")

        # RAG only on attempt 1 (keeps retries fast)
        rag_query = f"{class_name} {extra_instructions}".strip()
        rag_context = self.rag_index.retrieve_context(rag_query, top_k=5)

        user_msg = f"""
Generate JUnit 5 tests for this Java Spring Boot service.

<source_path>{service_path}</source_path>

<source_code>
{java_source}
</source_code>

<context>
{rag_context}
</context>

Additional instructions:
{extra_instructions}

Rules:
- Cover positive, negative, boundary cases
- Output ONLY Java code (no markdown, no explanation)
""".strip()

        base_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.test_prompt},
            {"role": "user", "content": user_msg},
        ]

        package_name = self._extract_package(java_source)
        test_path = self._guess_test_path(package_name, class_name)

        # fast path: already compiled once
        cached = self._compiled_cache.get(service_path)
        if cached and compile_after:
            return {
                "status": "SUCCESS",
                "service_path": service_path,
                "test_path": test_path,
                "attempts_used": 0,
                "attempt_log": [],
                "test_code": cached,
            }

        last_test_code = ""
        last_compile: Optional[Dict[str, Any]] = None
        attempt_log: List[Dict[str, Any]] = []

        for attempt in range(1, max_attempts + 1):

            # ---------------------------
            # Prompt selection
            # ---------------------------
            if attempt == 1:
                messages = list(base_messages)
            else:
                # IMPORTANT: send actionable compiler diagnostics (not stack trace tail)
                compiler_basis = self._compile_diag(last_compile or {}, n=260)

                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {
                        "role": "user",
                        "content": (
                            "Fix ONLY compilation errors; keep everything else exactly the same.\n"
                            "Do NOT rewrite the file.\n"
                            "Return ONLY the full corrected Java test class.\n"
                            "No markdown, no explanations."
                        ),
                    },
                    {"role": "user", "content": "BASE TEST FILE:\n" + (last_test_code or "")},
                    {"role": "user", "content": "COMPILER ERROR (actionable excerpt):\n" + (compiler_basis or "<EMPTY>")},
                ]

            # ---------------------------
            # LLM call
            # ---------------------------
            prev_test_before_llm = last_test_code or ""

            response = self.llm.chat(messages, temperature=self.llm_temperature)
            candidate = self._extract_java_class(self._strip_code_fences(response))

            if not self._is_valid_java_test_file(candidate, class_name):
                # Don’t overwrite the previous Java file with junk (XML/markdown/etc.)
                # Keep last_test_code and force a stricter retry prompt.
                candidate = prev_test_before_llm or last_test_code

            # ---------------------------
            # Guards only on repair attempts
            # ---------------------------
            if attempt > 1 and prev_test_before_llm:
                prev_norm = self._normalize_for_compare(prev_test_before_llm)
                cand_norm = self._normalize_for_compare(candidate)

                prev_tests = self._count_tests(prev_test_before_llm)
                cand_tests = self._count_tests(candidate)

                err_excerpt = self._compile_diag(last_compile or {}, n=260)

                # DEBUG
                print("\n" + "=" * 80)
                print("🧪 REPAIR DEBUG")
                print(f"attempt={attempt}")
                print(f"prev_tests={prev_tests} cand_tests={cand_tests}")
                print(f"changed={prev_norm != cand_norm}")
                print("-" * 80)

                print("▶ COMPILER ERROR EXCERPT (sent basis):")
                print(err_excerpt or "<EMPTY>")

                print("\n▶ PREV TEST (first 800 chars):")
                print((prev_test_before_llm or "")[:800])

                print("\n▶ CANDIDATE TEST (first 800 chars):")
                print((candidate or "")[:800])

                print("=" * 80 + "\n")

                # Guard A: wrong class name
                if not self._must_contain_class(candidate, class_name):
                    repair_msgs = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": (
                            f"WRONG CLASS. Keep the class as {class_name}Test.\n"
                            "Fix ONLY compilation errors with minimal edits.\n"
                            "Do NOT delete tests. Return ONLY full Java code.\n"
                            "No markdown, no explanations."
                        )},
                        {"role": "user", "content": "BASE TEST FILE:\n" + prev_test_before_llm},
                        {"role": "user", "content": "COMPILER ERROR (actionable excerpt):\n" + (err_excerpt or "<EMPTY>")},
                    ]
                    r2 = self.llm.chat(repair_msgs, temperature=self.llm_temperature)
                    candidate = self._extract_java_class(self._strip_code_fences(r2))
                    cand_norm = self._normalize_for_compare(candidate)
                    cand_tests = self._count_tests(candidate)

                # Guard B: removed tests
                if prev_tests and cand_tests < prev_tests:
                    repair_msgs = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": (
                            "YOU REMOVED TESTS. Preserve all existing @Test methods.\n"
                            "Fix ONLY compilation errors with minimal edits.\n"
                            "Return ONLY full Java code.\n"
                            "No markdown, no explanations.\n"
                            "MUST start with 'package '.\n"
                            "Do NOT output XML, pom.xml, dependencies, or explanations."
                        )},
                        {"role": "user", "content": "BASE TEST FILE:\n" + prev_test_before_llm},
                        {"role": "user", "content": "COMPILER ERROR (actionable excerpt):\n" + (err_excerpt or "<EMPTY>")},
                    ]
                    r3 = self.llm.chat(repair_msgs, temperature=self.llm_temperature)
                    candidate = self._extract_java_class(self._strip_code_fences(r3))
                    cand_norm = self._normalize_for_compare(candidate)

                # Guard C: no change
                if prev_norm and cand_norm == prev_norm:
                    repair_msgs = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": (
                            "NO-CHANGE. You returned the same file.\n"
                            "You MUST change the BASE TEST FILE to fix the compilation errors.\n"
                            "Fix ONLY compilation errors; keep everything else same.\n"
                            "Return ONLY full Java code.\n"
                            "No markdown, no explanations."
                        )},
                        {"role": "user", "content": "BASE TEST FILE:\n" + prev_test_before_llm},
                        {"role": "user", "content": "COMPILER ERROR (actionable excerpt):\n" + (err_excerpt or "<EMPTY>")},
                    ]
                    r4 = self.llm.chat(repair_msgs, temperature=self.llm_temperature)
                    candidate = self._extract_java_class(self._strip_code_fences(r4))

                test_code = candidate
            else:
                test_code = candidate

            last_test_code = test_code

            # ---------------------------
            # Write file
            # ---------------------------
            self.git.write_file(test_path, test_code, overwrite=True)

            if not compile_after:
                return self._success(service_path, test_path, test_code, attempt_log)

            # ---------------------------
            # Compile
            # ---------------------------
            last_compile = self.git.compile(
                tool="maven",
                goal="test-compile",
                project_path=".",
                timeout_seconds=300,
                extra_args=["-DskipTests=true"],
            )

            if not isinstance(last_compile, dict):
                last_compile = {"ok": False, "http_status": 500, "error": "compile() returned None (expected dict)"}

            # Tool-level failure (HTTP error): stop retries
            if last_compile.get("http_status"):
                attempt_log.append({
                    "attempt": attempt,
                    "stage": "compile",
                    "ok": False,
                    "http_status": last_compile.get("http_status"),
                    "error": str(last_compile.get("error", ""))[:500],
                })
                return self._tool_failure(service_path, test_path, test_code, attempt_log)

            ok = bool(last_compile.get("ok"))
            attempt_log.append({
                "attempt": attempt,
                "stage": "compile",
                "ok": ok,
                "returncode": last_compile.get("returncode"),
            })

            if ok:
                self._compiled_cache[service_path] = test_code
                return self._success(service_path, test_path, test_code, attempt_log, last_compile)

            # ---------------------------
            # Deterministic auto-fix (before next LLM attempt)
            # ---------------------------
            comp_text = self._compile_diag(last_compile, n=260)
            fixed = self._auto_fix_common_java_test_compile_errors(last_test_code, comp_text)

            if self._normalize_for_compare(fixed) != self._normalize_for_compare(last_test_code):
                last_test_code = fixed
                self.git.write_file(test_path, fixed, overwrite=True)

                last_compile = self.git.compile(
                    tool="maven",
                    goal="test-compile",
                    project_path=".",
                    timeout_seconds=300,
                    extra_args=["-DskipTests=true"],
                )

                if not isinstance(last_compile, dict):
                    last_compile = {"ok": False, "http_status": 500, "error": "compile() returned None (expected dict)"}

                if last_compile.get("http_status"):
                    attempt_log.append({
                        "attempt": attempt,
                        "stage": "compile",
                        "ok": False,
                        "http_status": last_compile.get("http_status"),
                        "error": str(last_compile.get("error", ""))[:500],
                    })
                    return self._tool_failure(service_path, test_path, last_test_code, attempt_log)

                attempt_log.append({
                    "attempt": attempt,
                    "stage": "compile_after_autofix",
                    "ok": bool(last_compile.get("ok")),
                    "returncode": last_compile.get("returncode"),
                })

                if last_compile.get("ok"):
                    self._compiled_cache[service_path] = fixed
                    return self._success(service_path, test_path, fixed, attempt_log, last_compile)

        return {
            "status": "COMPILATION_FAILED",
            "service_path": service_path,
            "test_path": test_path,
            "test_code": last_test_code,
            "attempt_log": attempt_log,
            "compile": last_compile,
        }

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _extract_package(self, java_source: str) -> str:
        m = re.search(r"^\s*package\s+([\w\.]+)\s*;", java_source, re.MULTILINE)
        return m.group(1) if m else ""

    def _guess_test_path(self, package_name: str, class_name: str) -> str:
        pkg = package_name.replace(".", "/") if package_name else ""
        return f"src/test/java/{pkg}/{class_name}Test.java" if pkg else f"src/test/java/{class_name}Test.java"

    def _strip_code_fences(self, text: str) -> str:
        s = (text or "").strip()
        m = re.search(r"```(?:java)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else s

    def _extract_java_class(self, raw: str) -> str:
        raw = (raw or "").strip()
        last = raw.rfind("}")
        return raw[: last + 1].strip() if last != -1 else raw

    def _must_contain_class(self, code: str, class_name: str) -> bool:
        code = code or ""
        return (f"class {class_name}Test" in code) or (f"{class_name}Test" in code)

    def _count_tests(self, code: str) -> int:
        return len(re.findall(r"(?m)^\s*@Test\b", code or ""))

    def _normalize_for_compare(self, s: str) -> str:
        if not s:
            return ""
        return "\n".join(line.rstrip() for line in s.strip().splitlines()).strip()

    def _merge_compile_streams(self, comp: Dict[str, Any]) -> str:
        """
        Merge stdout+stderr from MCP compile result.
        Your MCPGitClient.compile() guarantees these keys exist (defaulted).
        """
        stdout = (comp.get("stdout") or "").strip()
        stderr = (comp.get("stderr") or "").strip()
        merged = "\n".join([p for p in (stdout, stderr) if p]).strip()
        return merged

    def _compile_diag(self, comp: Dict[str, Any], n: int = 260) -> str:
        """
        Returns actionable diagnostics excerpt (not stack trace tail).
        """
        merged = self._merge_compile_streams(comp)
        if not merged:
            return ""
        actionable = extract_actionable_maven_error(merged)
        if not actionable:
            actionable = merged
        lines = actionable.splitlines()
        if len(lines) > n:
            return "\n".join(lines[-n:]).strip()
        return actionable.strip()

    def _trim_compile_error(self, comp: Optional[Dict[str, Any]], lines: int = 260) -> str:
        """
        Backwards-compatible (if you still call it somewhere else).
        """
        if not comp:
            return ""
        merged = self._merge_compile_streams(comp)
        if not merged:
            return ""
        actionable = extract_actionable_maven_error(merged) or merged
        lns = actionable.splitlines()
        if len(lns) > lines:
            return "\n".join(lns[-lines:]).strip()
        return actionable.strip()

    def _success(self, service_path, test_path, test_code, attempt_log, compile=None):
        return {
            "status": "SUCCESS",
            "service_path": service_path,
            "test_path": test_path,
            "test_code": test_code,
            "attempt_log": attempt_log,
            "compile": compile,
        }

    def _tool_failure(self, service_path, test_path, test_code, attempt_log):
        return {
            "status": "TOOL_FAILURE",
            "service_path": service_path,
            "test_path": test_path,
            "test_code": test_code,
            "attempt_log": attempt_log,
        }

    def _auto_fix_common_java_test_compile_errors(self, test_code: str, compile_text: str) -> str:
        """
        Deterministic fixes for common JUnit/Spring/Mockito test compile errors.
        Returns updated test_code (or original if no change).
        """
        if not test_code:
            return test_code

        compile_text = (compile_text or "")
        updated = test_code

        # ✅ Your exact error: Optional not found
        if "cannot find symbol" in compile_text and "Optional" in compile_text:
            if "Optional." in updated and "import java.util.Optional;" not in updated:
                updated = self._ensure_import(updated, "import java.util.Optional;")

        # fail(String) undefined
        if "fail(String)" in compile_text and "undefined" in compile_text:
            if "fail(" in updated and "import static org.junit.jupiter.api.Assertions.fail;" not in updated:
                updated = self._ensure_import(updated, "import static org.junit.jupiter.api.Assertions.fail;")

        # Assertions missing
        if "cannot find symbol" in compile_text and "Assertions" in compile_text:
            if "import static org.junit.jupiter.api.Assertions.*;" not in updated:
                updated = self._ensure_import(updated, "import static org.junit.jupiter.api.Assertions.*;")

        # SpringBootTest import
        if "@SpringBootTest" in updated and "SpringBootTest" in compile_text and "cannot find symbol" in compile_text:
            updated = self._ensure_import(updated, "import org.springframework.boot.test.context.SpringBootTest;")

        # Mockito annotation imports
        if "@Mock" in updated and "Mock" in compile_text and "cannot find symbol" in compile_text:
            updated = self._ensure_import(updated, "import org.mockito.Mock;")
        if "@InjectMocks" in updated and "InjectMocks" in compile_text and "cannot find symbol" in compile_text:
            updated = self._ensure_import(updated, "import org.mockito.InjectMocks;")

        # JUnit @Test import
        if "@Test" in updated and "Test" in compile_text and "cannot find symbol" in compile_text:
            updated = self._ensure_import(updated, "import org.junit.jupiter.api.Test;")

        return updated

    def _ensure_import(self, code: str, import_line: str) -> str:
        """
        Ensures an import exists. Inserts after the last import if present,
        else after package line, else at top.
        """
        if import_line in code:
            return code

        lines = code.splitlines()

        pkg_idx = -1
        last_import_idx = -1
        for i, ln in enumerate(lines):
            s = ln.strip()
            if s.startswith("package ") and s.endswith(";"):
                pkg_idx = i
            if s.startswith("import "):
                last_import_idx = i

        if last_import_idx != -1:
            insert_at = last_import_idx + 1
        elif pkg_idx != -1:
            insert_at = pkg_idx + 1
        else:
            insert_at = 0

        lines.insert(insert_at, import_line)
        return "\n".join(lines)

    def _is_valid_java_test_file(self, code: str, class_name: str) -> bool:
        s = (code or "").strip()
        if not s:
            return False
        # Must look like Java source, not XML/markdown
        if s.lstrip().startswith("<"):
            return False
        if "<dependencies>" in s or "<project" in s:
            return False
        # Must have package or imports or class
        if "class " not in s:
            return False
        # Must end with a closing brace (very common sanity check)
        if not s.rstrip().endswith("}"):
            return False
        # Must contain expected test class name
        if f"class {class_name}Test" not in s and f"{class_name}Test" not in s:
            return False
        return True
