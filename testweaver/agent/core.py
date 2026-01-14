# agent/core.py
import pathlib
import os
import re
from typing import Optional, List, Dict, Any, Tuple

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
    r"constructor .* cannot be applied to given types",
]


def extract_actionable_maven_error(maven_output: str, before: int = 60, after: int = 140) -> str:
    """
    Extract actionable compiler diagnostics from full Maven output.
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

        java_source = self.git.get_file(service_path) or ""
        class_name = service_path.split("/")[-1].replace(".java", "")

        # LIMIT JAVA SOURCE SIZE (prevents Ollama timeouts)
        java_source = java_source[:12000]

        # RAG only on attempt 1 (keeps retries fast)
        rag_query = f"{class_name} {extra_instructions}".strip()
        rag_context = self.rag_index.retrieve_context(rag_query, top_k=5) or ""
        rag_context = rag_context[:3000]

        # Pull related DTO/model sources to reduce constructor/getter hallucinations
        related_sources = self._collect_related_sources(service_path, java_source)
        related_api_summary = self._summarize_related_public_api(related_sources)

        related_sources_block = ""
        if related_sources:
            chunks = []
            for p, src in related_sources:
                src = (src or "")[:5000]
                chunks.append(f"\n<file path=\"{p}\">\n{src}\n</file>")
            related_sources_block = "\n<related_sources>\n" + "\n".join(chunks) + "\n</related_sources>"

        api_summary_block = ""
        if related_api_summary.strip():
            api_summary_block = "\n<related_public_api>\n" + related_api_summary.strip() + "\n</related_public_api>"

        user_msg = f"""
Generate JUnit 5 tests for this Java Spring Boot service.

<source_path>{service_path}</source_path>

<source_code>
{java_source}
</source_code>

<context>
{rag_context}
</context>
{related_sources_block}
{api_summary_block}

Additional instructions:
{extra_instructions}

Rules (MUST FOLLOW):
- Output ONLY Java code (no markdown, no explanation)
- Tests MUST compile against the provided source code and related sources above
- Do NOT invent constructors, getters, setters, enums, fields, or methods
- If a DTO/model has no visible all-args constructor, use no-arg constructor + setters (only if setters are visible)
- Do NOT use ResponseEntity unless the service method returns ResponseEntity in the source
- When unsure about a model API, assert only non-null and verify repository interactions
- Include required imports (JUnit5, Optional, BigDecimal, etc.)
- Prefer minimal/robust assertions over guessing domain fields
""".strip()

        base_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.test_prompt},
            {"role": "user", "content": user_msg},
        ]

        package_name = self._extract_package(java_source)
        test_path = self._guess_test_path(package_name, class_name)

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

            if attempt == 1:
                messages = list(base_messages)
            else:
                compiler_basis = self._compile_diag(last_compile or {}, n=160)

                # STRICT repair prompt: forbids creating extra files/types
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": (
                        "STRICT REPAIR MODE.\n"
                        "You are editing ONE file ONLY: the Java test class shown in BASE TEST FILE.\n"
                        "DO NOT output any other file, enum, interface, record, helper class, or explanation.\n"
                        "DO NOT propose creating new files.\n"
                        f"Output must be a single compilable Java file defining ONLY: public class {class_name}Test.\n"
                        "Fix ONLY the compilation errors from COMPILER ERROR.\n"
                        "Return ONLY the full corrected Java test class source code starting with 'package '."
                    )},
                    {"role": "user", "content": "BASE TEST FILE:\n" + (last_test_code or "")},
                    {"role": "user", "content": "COMPILER ERROR (actionable excerpt):\n" + (compiler_basis or "<EMPTY>")},
                ]

            prev_test_before_llm = last_test_code or ""
            response = self.llm.chat(messages, temperature=self.llm_temperature)

            candidate = self._extract_java_class(self._strip_code_fences(response))
            if attempt > 1 and prev_test_before_llm:
                candidate = self._rename_test_methods_to_match_base(candidate, prev_test_before_llm)

            # Base validation
            if not self._is_valid_java_test_file(candidate, class_name):
                candidate = prev_test_before_llm or last_test_code

            # STRICT validation for repair attempts: prevents enum-only output etc.
            if attempt > 1 and not self._is_only_test_class(candidate, class_name):
                # One immediate strict retry (optional but helps a lot)
                strict_msgs = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": (
                        "STRICT REPAIR MODE.\n"
                        "You are editing ONE file ONLY: the Java test class shown in BASE TEST FILE.\n"
                        "Fix ONLY the compilation errors from COMPILER ERROR.\n\n"
                        "HARD RULES:\n"
                        "- Do NOT rename any methods (especially @Test methods). Keep all method names EXACTLY the same.\n"
                        "- Do NOT rename the test class.\n"
                        "- Do NOT delete any @Test methods.\n"
                        "- Do NOT add new files, enums, interfaces, or helper classes.\n"
                        "- Do NOT change logic unless required to fix a compilation error.\n\n"
                        f"Return ONLY a single compilable Java file defining: public class {class_name}Test\n"
                        "Output must start with 'package '. No markdown, no explanations."
                    )},
                    {"role": "user", "content": "BASE TEST FILE:\n" + (prev_test_before_llm or "")},
                    {"role": "user", "content": "COMPILER ERROR:\n" + (self._compile_diag(last_compile or {}, n=160) or "<EMPTY>")},
                ]
                r = self.llm.chat(strict_msgs, temperature=self.llm_temperature)
                candidate2 = self._extract_java_class(self._strip_code_fences(r))
                if self._is_only_test_class(candidate2, class_name):
                    candidate = candidate2
                else:
                    candidate = prev_test_before_llm or last_test_code

            # Deterministic pre-sanitize even before compilation
            candidate = self._auto_fix_common_java_test_compile_errors(candidate, compile_text="")

            test_code = candidate
            last_test_code = test_code

            # Write file
            self.git.write_file(test_path, test_code, overwrite=True)

            if not compile_after:
                return self._success(service_path, test_path, test_code, attempt_log)

            # Compile
            last_compile = self.git.compile(
                tool="maven",
                goal="test-compile",
                project_path=".",
                timeout_seconds=600,
                extra_args=["-DskipTests=true"],
            )

            print("🔧 git.compile returned (summary):", {
                "ok": bool(last_compile.get("ok")) if isinstance(last_compile, dict) else False,
                "returncode": last_compile.get("returncode") if isinstance(last_compile, dict) else None,
                "http_status": last_compile.get("http_status") if isinstance(last_compile, dict) else None,
            })

            # If compile failed, print actionable tails to help debugging
            if isinstance(last_compile, dict) and not bool(last_compile.get("ok")):
                try:
                    print("🔧 compile stdout (tail):\n", (last_compile.get("stdout") or "")[-8000:])
                except Exception:
                    pass
                try:
                    print("🔧 compile stderr (tail):\n", (last_compile.get("stderr") or "")[-8000:])
                except Exception:
                    pass
                try:
                    print("🔧 compile error_summary (tail):\n", (last_compile.get("error_summary") or "")[-2000:])
                except Exception:
                    pass

            if not isinstance(last_compile, dict):
                last_compile = {"ok": False, "error": "compile() returned None (expected dict)"}

            # Tool failure only if HTTP status is 4xx/5xx
            http_status = last_compile.get("http_status")
            if http_status is not None and int(http_status) >= 400:
                attempt_log.append({
                    "attempt": attempt,
                    "stage": "compile",
                    "ok": False,
                    "http_status": http_status,
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
                print("🔔 compile succeeded — invoking generate_coverage_report() now...")
                coverage = self.generate_coverage_report()
                print("🔔 generate_coverage_report returned:", {"ok": bool(coverage.get("ok")) if isinstance(coverage, dict) else False, "returncode": coverage.get("returncode") if isinstance(coverage, dict) else None})
                return {
                    **self._success(service_path, test_path, test_code, attempt_log, last_compile),
                    "coverage": {
                        "ok": bool(coverage.get("ok")),
                        "report_html": "target/site/jacoco/index.html",
                        "report_xml": "target/site/jacoco/jacoco.xml",
                        "raw": coverage,
                    },
                }


            # Deterministic auto-fix (before next LLM attempt)
            comp_text = self._compile_diag(last_compile, n=220)
            fixed = self._auto_fix_common_java_test_compile_errors(last_test_code, comp_text)

            if self._normalize_for_compare(fixed) != self._normalize_for_compare(last_test_code):
                last_test_code = fixed
                self.git.write_file(test_path, fixed, overwrite=True)

                last_compile = self.git.compile(
                    tool="maven",
                    goal="test-compile",
                    project_path=".",
                    timeout_seconds=600,
                    extra_args=["-DskipTests=true"],
                )

                if not isinstance(last_compile, dict):
                    last_compile = {"ok": False, "error": "compile() returned None (expected dict)"}

                http_status = last_compile.get("http_status")
                if http_status is not None and int(http_status) >= 400:
                    attempt_log.append({
                        "attempt": attempt,
                        "stage": "compile_after_autofix",
                        "ok": False,
                        "http_status": http_status,
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
                    coverage = self.generate_coverage_report()
                    return {
                        **self._success(service_path, test_path, fixed, attempt_log, last_compile),
                        "coverage": {
                            "ok": bool(coverage.get("ok")),
                            "report_html": "target/site/jacoco/index.html",
                            "report_xml": "target/site/jacoco/jacoco.xml",
                            "raw": coverage,
                        },
                    }

        return {
            "status": "COMPILATION_FAILED",
            "service_path": service_path,
            "test_path": test_path,
            "test_code": last_test_code,
            "attempt_log": attempt_log,
            "compile": last_compile,
        }

    # ------------------------------------------------------------------
    # Related source loading (to improve generation itself)
    # ------------------------------------------------------------------

    def _collect_related_sources(self, service_path: str, service_source: str) -> List[Tuple[str, str]]:
        """
        Best-effort: load common DTO/model files that tests commonly depend on,
        so the LLM stops hallucinating constructors/getters.
        Also supports env var RELATED_SOURCES as comma-separated paths.
        """
        paths: List[str] = []

        # Allow explicit override
        env = (os.getenv("RELATED_SOURCES") or "").strip()
        if env:
            for p in env.split(","):
                p = p.strip()
                if p:
                    paths.append(p)

        # Heuristic imports from service source (dto/model lines)
        for m in re.finditer(r"^\s*import\s+([\w\.]+)\s*;\s*$", service_source or "", re.MULTILINE):
            fqcn = m.group(1)
            if ".dto." in fqcn or ".model." in fqcn:
                rel = "src/main/java/" + fqcn.replace(".", "/") + ".java"
                paths.append(rel)

        # De-dup while preserving order
        seen = set()
        uniq: List[str] = []
        for p in paths:
            if p not in seen:
                uniq.append(p)
                seen.add(p)

        out: List[Tuple[str, str]] = []
        for p in uniq[:10]:  # bounded
            try:
                src = self.git.get_file(p)
                if src:
                    out.append((p, src))
            except Exception:
                continue
        return out

    def _summarize_related_public_api(self, related: List[Tuple[str, str]]) -> str:
        """
        Lightweight API summary: constructors + public get*/set* methods.
        """
        lines: List[str] = []
        for path, src in related:
            cls = self._extract_primary_class_name(src) or path.split("/")[-1].replace(".java", "")
            constructors = self._extract_constructors(src, cls)
            methods = self._extract_public_get_set_methods(src)

            lines.append(f"{cls} ({path}):")
            if constructors:
                lines.append("  constructors: " + ", ".join(constructors))
            if methods:
                lines.append("  methods: " + ", ".join(methods[:25]))
            lines.append("")
        return "\n".join(lines).strip()

    def _extract_primary_class_name(self, src: str) -> str:
        m = re.search(r"\bclass\s+([A-Za-z_]\w*)\b", src or "")
        return m.group(1) if m else ""

    def _extract_constructors(self, src: str, class_name: str) -> List[str]:
        if not src or not class_name:
            return []
        pat = re.compile(rf"\b(public\s+)?{re.escape(class_name)}\s*\(([^)]*)\)", re.MULTILINE)
        ctors = []
        for m in pat.finditer(src):
            args = " ".join(m.group(2).split())
            if args.strip():
                ctors.append(f"{class_name}({args})")
            else:
                ctors.append(f"{class_name}()")
        seen = set()
        out = []
        for c in ctors:
            if c not in seen:
                out.append(c)
                seen.add(c)
        return out[:6]

    def _extract_public_get_set_methods(self, src: str) -> List[str]:
        if not src:
            return []
        pat = re.compile(r"\bpublic\s+([\w\<\>\[\]\.]+)\s+((get|set)[A-Z]\w*)\s*\(([^)]*)\)", re.MULTILINE)
        out = []
        for m in pat.finditer(src):
            rtype = m.group(1)
            name = m.group(2)
            args = " ".join(m.group(4).split())
            out.append(f"{rtype} {name}({args})")
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _extract_package(self, java_source: str) -> str:
        m = re.search(r"^\s*package\s+([\w\.]+)\s*;", java_source or "", re.MULTILINE)
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

    def _count_tests(self, code: str) -> int:
        return len(re.findall(r"(?m)^\s*@Test\b", code or ""))

    def _normalize_for_compare(self, s: str) -> str:
        if not s:
            return ""
        return "\n".join(line.rstrip() for line in s.strip().splitlines()).strip()

    def _merge_compile_streams(self, comp: Dict[str, Any]) -> str:
        stdout = (comp.get("stdout") or "").strip()
        stderr = (comp.get("stderr") or "").strip()
        return "\n".join([p for p in (stdout, stderr) if p]).strip()

    def _compile_diag(self, comp: Dict[str, Any], n: int = 260) -> str:
        merged = self._merge_compile_streams(comp)
        if not merged:
            return ""
        actionable = extract_actionable_maven_error(merged) or merged
        lines = actionable.splitlines()
        if len(lines) > n:
            return "\n".join(lines[-n:]).strip()
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

    # ------------------------------------------------------------------
    # Output validation (prevents "enum-only" etc.)
    # ------------------------------------------------------------------

    def _is_valid_java_test_file(self, code: str, class_name: str) -> bool:
        s = (code or "").strip()
        if not s:
            return False

        bad_starts = ("<", "mvn ", "gradle ", "./", "sh", "#!/bin", "```", "pom.xml")
        if s.lower().startswith(bad_starts):
            return False
        if "<dependencies>" in s or "<project" in s:
            return False

        if not s.startswith("package "):
            return False
        if "class " not in s:
            return False
        if not s.rstrip().endswith("}"):
            return False

        if f"class {class_name}Test" not in s and f"{class_name}Test" not in s:
            return False

        return True

    def _is_only_test_class(self, code: str, class_name: str) -> bool:
        """
        True only if the output is a single Java test file and does not define
        extra top-level enums/interfaces/records/classes besides <ClassName>Test.
        """
        s = (code or "").strip()
        if not s.startswith("package "):
            return False
        if f"class {class_name}Test" not in s and f"public class {class_name}Test" not in s:
            return False

        # Scrub comments/strings (best-effort) to reduce false positives
        scrubbed = re.sub(r"//.*?$|/\*.*?\*/|\".*?\"", "", s, flags=re.MULTILINE | re.DOTALL)

        # Any public class other than the test class is forbidden
        for m in re.finditer(r"(?m)^\s*public\s+class\s+([A-Za-z_]\w*)\b", scrubbed):
            if m.group(1) != f"{class_name}Test":
                return False

        # Forbid top-level public enum/interface/record
        if re.search(r"(?m)^\s*public\s+enum\s+\w+\b", scrubbed):
            return False
        if re.search(r"(?m)^\s*public\s+interface\s+\w+\b", scrubbed):
            return False
        if re.search(r"(?m)^\s*public\s+record\s+\w+\b", scrubbed):
            return False

        return True

    # ------------------------------------------------------------------
    # Deterministic fixes (generic-first)
    # ------------------------------------------------------------------

    def _auto_fix_common_java_test_compile_errors(self, test_code: str, compile_text: str) -> str:
        """
        Deterministic fixes for common JUnit/Mockito/Spring test compile errors.
        Keeps fixes generic across projects.

        Covers:
        - Missing imports for JUnit 5 annotations, Optional, BigDecimal
        - ResponseEntity mismatch pattern
        - "cannot find symbol TransactionType" fallback: replace TransactionType.X with null
        - "required: no arguments" constructors: rewrite new X(a,b,c) -> new X()
        - Remove assertions that call missing getters like getReference/getStatus/getMessage when compiler says so
        """
        if not test_code:
            return test_code

        compile_text = (compile_text or "")
        updated = test_code

        # ---- Ensure JUnit5 imports always (your later attempts lost them)
        if "@Test" in updated and "import org.junit.jupiter.api.Test;" not in updated:
            updated = self._ensure_import(updated, "import org.junit.jupiter.api.Test;")
        if "@BeforeEach" in updated and "import org.junit.jupiter.api.BeforeEach;" not in updated:
            updated = self._ensure_import(updated, "import org.junit.jupiter.api.BeforeEach;")
        if "Assertions." in updated or re.search(r"\bassert[A-Z]\w*\s*\(", updated):
            if "import static org.junit.jupiter.api.Assertions.*;" not in updated:
                updated = self._ensure_import(updated, "import static org.junit.jupiter.api.Assertions.*;")

        # ---- Optional / BigDecimal imports
        if ("Optional." in updated or "Optional<" in updated) and "import java.util.Optional;" not in updated:
            updated = self._ensure_import(updated, "import java.util.Optional;")
        if ("BigDecimal." in updated or "new BigDecimal" in updated) and "import java.math.BigDecimal;" not in updated:
            updated = self._ensure_import(updated, "import java.math.BigDecimal;")

        # ---- ResponseEntity mismatch pattern
        if ("incompatible types" in compile_text and "ResponseEntity" in compile_text) or ("ResponseEntity<" in updated and "ResponseEntity" in compile_text):
            updated = re.sub(
                r"ResponseEntity\s*<\s*([A-Za-z_]\w*)\s*>\s+(\w+)\s*=",
                r"\1 \2 =",
                updated
            )
            updated = updated.replace(".getBody()", "")
            updated = updated.replace("import org.springframework.http.ResponseEntity;\n", "")
            updated = updated.replace("import org.springframework.http.ResponseEntity;\r\n", "")

        # ---- If TransactionType is missing, don't invent it; use null to compile
        if ("TransactionType" in updated) and (("symbol:   class TransactionType" in compile_text) or ("cannot find symbol" in compile_text and "TransactionType" in compile_text)):
            updated = re.sub(r"\bTransactionType\s*\.\s*[A-Z_]+\b", "null", updated)
            # also remove any import line for it if present
            updated = re.sub(r"(?m)^\s*import\s+.*TransactionType\s*;\s*\r?\n", "", updated)

        # ---- Remove assertions calling getters that compiler says do not exist
        if "method getReference()" in compile_text or ("cannot find symbol" in compile_text and "getReference" in compile_text):
            updated = re.sub(r"(?m)^\s*assertEquals\([^;]*getReference\(\)[^;]*\);\s*\r?\n?", "", updated)
        if "method getStatus()" in compile_text or ("cannot find symbol" in compile_text and "getStatus" in compile_text):
            updated = re.sub(r"(?m)^\s*assertEquals\([^;]*getStatus\(\)[^;]*\);\s*\r?\n?", "", updated)
        if "method getMessage()" in compile_text or ("cannot find symbol" in compile_text and "getMessage" in compile_text):
            updated = re.sub(r"(?m)^\s*assertEquals\([^;]*getMessage\(\)[^;]*\);\s*\r?\n?", "", updated)

        # ---- Generic rewrite for "required: no arguments" constructors
        # Parse class names from compiler output: "constructor X ... required: no arguments"
        ctor_noarg_classes = self._classes_with_noarg_required(compile_text)
        for cls in ctor_noarg_classes:
            updated = self._rewrite_all_args_to_noarg(updated, cls)

        return updated

    def _classes_with_noarg_required(self, compile_text: str) -> List[str]:
        """
        From Maven output, extract class names where compiler says:
        "constructor X ... required: no arguments"
        """
        if not compile_text:
            return []
        out: List[str] = []
        for m in re.finditer(r"constructor\s+([A-Za-z_]\w*)\s+in\s+class\s+[\w\.]+\s+cannot\s+be\s+applied.*?required:\s+no arguments", compile_text, re.IGNORECASE | re.DOTALL):
            out.append(m.group(1))
        # also match shorter repeated form
        for m in re.finditer(r"constructor\s+([A-Za-z_]\w*)\s+cannot\s+be\s+applied.*?required:\s+no arguments", compile_text, re.IGNORECASE | re.DOTALL):
            out.append(m.group(1))
        # de-dup preserve order
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    def _rewrite_all_args_to_noarg(self, code: str, class_name: str) -> str:
        """
        Rewrite occurrences of:
          new ClassName(a,b,c)  -> new ClassName()
        This is generic and safe (compile-focused). It does NOT guess setters.
        """
        if not code or not class_name:
            return code
        pat = re.compile(rf"new\s+{re.escape(class_name)}\s*\(\s*[^)]*\)", re.MULTILINE)
        return pat.sub(f"new {class_name}()", code)

    # ------------------------------------------------------------------
    # Import helper
    # ------------------------------------------------------------------

    def _ensure_import(self, code: str, import_line: str) -> str:
        if import_line in code:
            return code

        lines = (code or "").splitlines()
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

    def _extract_test_method_names(self, code: str) -> List[str]:
        """
        Extract method names for @Test annotated methods (JUnit 5).
        Python re doesn't support \R, so we use (?:\r?\n).
        """
        if not code:
            return []

        # Match:
        # @Test
        # [optional other annotations]
        # public void methodName(...)
        pat = re.compile(
            r"@Test\s*(?:\r?\n)\s*"
            r"(?:@\w+(?:\([^)]*\))?\s*(?:\r?\n)\s*)*"
            r"(?:public|protected|private)?\s*"
            r"(?:static\s+)?"
            r"(?:void|[\w\<\>\[\]\.]+)\s+([A-Za-z_]\w*)\s*\(",
            re.MULTILINE
        )

        return [m.group(1) for m in pat.finditer(code)]


    def _rename_test_methods_to_match_base(self, candidate: str, base: str) -> str:
        """
        If the model renamed @Test methods, revert their names back to the base names.
        We do it positionally (1st @Test method maps to 1st @Test method, etc.)
        """
        base_names = self._extract_test_method_names(base)
        cand_names = self._extract_test_method_names(candidate)

        if not base_names or not cand_names:
            return candidate

        # Only enforce if counts match; if model deleted a test, another guard should catch it.
        if len(base_names) != len(cand_names):
            return candidate

        # If identical, nothing to do.
        if base_names == cand_names:
            return candidate

        updated = candidate
        for old, new in zip(cand_names, base_names):
            if old != new:
                # Replace method declaration name only (safer than global replace)
                updated = re.sub(
                    rf"(\b(?:public|protected|private)?\s*(?:void|[\w\<\>\[\]\.]+)\s+){re.escape(old)}(\s*\()",
                    rf"\1{new}\2",
                    updated,
                    count=1
                )
        return updated

    def generate_coverage_report(self) -> Dict[str, Any]:
        """
        Runs tests and generates JaCoCo report (requires JaCoCo plugin in project or available via goal).
        """
        # Run from the repo root (same location used by test compilation),
        # so jacoco:report executes against the correct project.
        extra = ["-Dmaven.test.failure.ignore=true", "jacoco:report"]
        print("🔍 generate_coverage_report: invoking git.compile with extra_args:", extra)
        res = self.git.compile(
            tool="maven",
            goal="test",
            project_path=".",
            timeout_seconds=1200,
            extra_args=extra,
        )
        try:
            ok = bool(res.get("ok"))
            rc = res.get("returncode")
        except Exception:
            ok = False
            rc = None
        print(f"🔍 generate_coverage_report: result ok={ok} returncode={rc}")
        print("🔍 generate_coverage_report: command:", res.get("command"))
        return res
