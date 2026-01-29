# agent/core.py
import pathlib
import os
import re
import json
import time
import uuid
import difflib
import html
import statistics
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
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
    r"The blank final field .* may not have been initialized",
    r"The final field .* cannot be assigned",
    r"cannot be resolved",
    r"may not have been initialized",
    r"cannot be assigned",
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


# --------------------------------------------------------------------------------------
# Evaluation Metrics (JSONL logger + lightweight evaluator)
# --------------------------------------------------------------------------------------

@dataclass
class EvalEvent:
    request_id: str
    session_id: str
    service_path: str
    test_path: str
    attempt: int
    stage: str  # gen | fix | compile | compile_after_autofix | tool_failure | success | fail
    ts_utc: str
    metrics: Dict[str, Any]
    compile_ok: Optional[bool] = None
    returncode: Optional[int] = None
    http_status: Optional[int] = None
    error_signatures: Optional[List[str]] = None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _safe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


class TestWeaverAgent:
    def __init__(self, session_id: str, rag_index: RAGIndex, short_term: ShortTermMemory, repo: str):
        self.session_id = session_id
        self.rag_index = rag_index
        self.short_term = short_term
        self.git = MCPGitClient(repo)
        self.repo_root = repo
        self.llm = LLMClient()

        self.llm_temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
        self._compiled_cache: Dict[str, str] = {}

        # --- Metrics config (dynamic, no hard-coded physical path) ---
        self.metrics_enabled = (os.getenv("METRICS_ENABLED", "true").strip().lower() == "true")
        self.auto_pr_on_success = (os.getenv("AUTO_PR_ON_SUCCESS", "false").strip().lower() == "true")

        BASE_DIR = pathlib.Path(__file__).resolve().parent.parent

        metrics_dir_env = (os.getenv("METRICS_DIR") or "").strip()
        metrics_subdir = (os.getenv("METRICS_SUBDIR") or "metrics").strip()

        if metrics_dir_env:
            p = pathlib.Path(metrics_dir_env)
            self.metrics_dir = str((BASE_DIR / p).resolve()) if not p.is_absolute() else str(p)
        else:
            self.metrics_dir = str((BASE_DIR / metrics_subdir).resolve())

        self.metrics_file = (os.getenv("METRICS_FILE") or "agent_metrics.jsonl").strip()

        PROMPTS_DIR = BASE_DIR / "prompts"
        self.system_prompt = (PROMPTS_DIR / "system_agent.md").read_text(encoding="utf-8")
        self.test_prompt = (PROMPTS_DIR / "test_generation.md").read_text(encoding="utf-8")

        # Aggregation window for CSR/avg/median metrics on report
        self.metrics_agg_window = _safe_int(os.getenv("METRICS_AGG_WINDOW", "50")) or 50

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    def _metrics_path(self) -> str:
        try:
            pathlib.Path(self.metrics_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(pathlib.Path(self.metrics_dir) / self.metrics_file)

    def _reports_dir(self) -> str:
        """
        Where HTML reports go. Dynamic, relative to project root.
        Default: <BASE_DIR>/metrics/reports
        """
        base_dir = pathlib.Path(__file__).resolve().parent.parent
        subdir = (os.getenv("METRICS_REPORTS_SUBDIR") or "metrics/reports").strip()
        out = (base_dir / subdir).resolve()
        try:
            out.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return str(out)

    # ------------------------------------------------------------------
    # JSONL logging
    # ------------------------------------------------------------------

    def _append_metrics_event(self, evt: EvalEvent) -> None:
        if not getattr(self, "metrics_enabled", True):
            return
        path = self._metrics_path()
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(evt.__dict__, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Basic quality metrics
    # ------------------------------------------------------------------

    def _count_tests(self, code: str) -> int:
        return len(re.findall(r"(?m)^\s*@Test\b", code or ""))

    def _count_assertions(self, code: str) -> int:
        if not code:
            return 0
        pats = [
            r"\bassertEquals\s*\(",
            r"\bassertNotEquals\s*\(",
            r"\bassertTrue\s*\(",
            r"\bassertFalse\s*\(",
            r"\bassertNull\s*\(",
            r"\bassertNotNull\s*\(",
            r"\bassertThrows\s*\(",
            r"\bassertDoesNotThrow\s*\(",
            r"\bassertAll\s*\(",
            r"\bfail\s*\(",
            r"\bAssertions\.\w+\s*\(",
        ]
        return sum(len(re.findall(p, code)) for p in pats)

    def _detect_leakage(self, code: str) -> bool:
        """
        Detects plain-text/markdown leakage: output should be compilable Java file.
        """
        if not code:
            return True

        s = code.strip()
        if not s.startswith("package "):
            return True

        bad_markers = ["```", "Explanation:", "NOTE:", "Sure,", "Here is", "Output:", "STRICT REPAIR MODE"]
        if any(m.lower() in s.lower() for m in bad_markers):
            return True

        if "class " not in s or not s.rstrip().endswith("}"):
            return True

        return False

    # ------------------------------------------------------------------
    # Compiler diagnostics normalization
    # ------------------------------------------------------------------

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

    def _error_signatures(self, compile_text: str, max_sigs: int = 12) -> List[str]:
        if not compile_text:
            return []
        text = compile_text
        text = re.sub(r"[A-Za-z]:[\\/][^\s]+", "<PATH>", text)
        text = re.sub(r"\.java:\[\d+,\d+\]", ".java:[X,Y]", text)
        text = re.sub(r"line\s+\d+", "line X", text, flags=re.IGNORECASE)

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        keep: List[str] = []
        key_re = re.compile(
            r"(cannot find symbol|package .* does not exist|incompatible types|cannot be applied to|Compilation failure|COMPILATION ERROR|Failed to execute goal)",
            re.IGNORECASE,
        )
        for ln in lines:
            if key_re.search(ln):
                keep.append(" ".join(ln.split()))

        seen = set()
        out: List[str] = []
        for k in keep:
            if k not in seen:
                out.append(k)
                seen.add(k)
            if len(out) >= max_sigs:
                break
        return out

    # ------------------------------------------------------------------
    # JaCoCo parsing (impressive metric)
    # ------------------------------------------------------------------

    def _parse_jacoco_xml(self, xml_rel_path: str = "target/site/jacoco/jacoco.xml") -> Dict[str, Any]:
        """
        Parse jacoco.xml counters to compute LINE and BRANCH coverage.
        Returns {} on failure.
        """
        try:
            p = pathlib.Path(xml_rel_path)
            if not p.exists():
                return {}
            tree = ET.parse(str(p))
            root = tree.getroot()

            totals: Dict[str, Dict[str, int]] = {}

            for counter in root.findall(".//counter"):
                ctype = counter.get("type")
                missed = _safe_int(counter.get("missed")) or 0
                covered = _safe_int(counter.get("covered")) or 0
                if ctype not in totals:
                    totals[ctype] = {"missed": 0, "covered": 0}
                totals[ctype]["missed"] += missed
                totals[ctype]["covered"] += covered

            def pct(covered: int, missed: int) -> float:
                denom = covered + missed
                if denom <= 0:
                    return 0.0
                return round((covered / denom) * 100.0, 2)

            line = totals.get("LINE", {"missed": 0, "covered": 0})
            branch = totals.get("BRANCH", {"missed": 0, "covered": 0})

            return {
                "line_covered": line["covered"],
                "line_missed": line["missed"],
                "line_pct": pct(line["covered"], line["missed"]),
                "branch_covered": branch["covered"],
                "branch_missed": branch["missed"],
                "branch_pct": pct(branch["covered"], branch["missed"]),
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Aggregate metrics across runs (CSR, avg/median attempts/time/coverage)
    # ------------------------------------------------------------------

    def _load_jsonl(self, path: str, limit: int = 2000) -> List[Dict[str, Any]]:
        """
        Loads up to last `limit` lines from JSONL.
        """
        try:
            p = pathlib.Path(path)
            if not p.exists():
                return []
            lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
            if len(lines) > limit:
                lines = lines[-limit:]
            out = []
            for ln in lines:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except Exception:
                    continue
            return out
        except Exception:
            return []

    def _aggregate_metrics_last_n_runs(self, n_runs: int = 50) -> Dict[str, Any]:
        """
        Builds aggregates from metrics JSONL grouped by request_id.
        Uses the final stage event per request_id (success/fail/tool_failure).
        """
        path = self._metrics_path()
        rows = self._load_jsonl(path, limit=max(2000, n_runs * 40))
        if not rows:
            return {}

        # Group by request_id
        by_req: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            rid = r.get("request_id")
            if not rid:
                continue
            by_req.setdefault(rid, []).append(r)

        # For each request_id, identify final event + attempts_used + total_seconds + coverage
        final_runs: List[Dict[str, Any]] = []
        for rid, evts in by_req.items():
            evts_sorted = sorted(evts, key=lambda e: e.get("ts_utc", ""))
            final = None
            for e in reversed(evts_sorted):
                if e.get("stage") in ("success", "fail", "tool_failure"):
                    final = e
                    break
            if not final:
                continue

            # attempts_used = max attempt number seen in that run
            attempts = []
            for e in evts_sorted:
                a = _safe_int(e.get("attempt"))
                if a is not None:
                    attempts.append(a)
            attempts_used = max(attempts) if attempts else None

            m = final.get("metrics") or {}
            total_seconds = _safe_float(m.get("total_seconds"))

            # coverage metrics might be in final.metrics if we logged them
            line_pct = _safe_float(m.get("line_pct"))
            branch_pct = _safe_float(m.get("branch_pct"))

            final_runs.append({
                "request_id": rid,
                "final_stage": final.get("stage"),
                "compile_ok": final.get("compile_ok"),
                "attempts_used": attempts_used,
                "total_seconds": total_seconds,
                "line_pct": line_pct,
                "branch_pct": branch_pct,
            })

        if not final_runs:
            return {}

        # Keep only last N runs by ordering request_id appearance in file
        # (approx: use the last event timestamp)
        def last_ts(rid: str) -> str:
            evts = by_req.get(rid, [])
            if not evts:
                return ""
            return max((e.get("ts_utc", "") for e in evts), default="")

        final_runs = sorted(final_runs, key=lambda x: last_ts(x["request_id"]))
        if len(final_runs) > n_runs:
            final_runs = final_runs[-n_runs:]

        total = len(final_runs)
        successes = [r for r in final_runs if r.get("final_stage") == "success" and bool(r.get("compile_ok"))]
        csr = round((len(successes) / total) * 100.0, 2) if total else 0.0

        def stats(vals: List[float]) -> Dict[str, Any]:
            vals = [v for v in vals if v is not None]
            if not vals:
                return {}
            vals_sorted = sorted(vals)

            def pctl(p: float) -> float:
                if not vals_sorted:
                    return 0.0
                k = int(round((p / 100.0) * (len(vals_sorted) - 1)))
                k = max(0, min(len(vals_sorted) - 1, k))
                return vals_sorted[k]

            return {
                "count": len(vals_sorted),
                "avg": round(sum(vals_sorted) / len(vals_sorted), 3),
                "median": round(statistics.median(vals_sorted), 3),
                "min": round(vals_sorted[0], 3),
                "p90": round(pctl(90), 3),
                "max": round(vals_sorted[-1], 3),
            }

        attempts_vals = [r.get("attempts_used") for r in final_runs if isinstance(r.get("attempts_used"), int)]
        time_vals = [r.get("total_seconds") for r in final_runs if isinstance(r.get("total_seconds"), (int, float))]
        line_vals = [r.get("line_pct") for r in final_runs if isinstance(r.get("line_pct"), (int, float))]
        branch_vals = [r.get("branch_pct") for r in final_runs if isinstance(r.get("branch_pct"), (int, float))]

        return {
            "window_runs": total,
            "csr_pct": csr,
            "attempts": stats([float(x) for x in attempts_vals]),
            "time_seconds": stats([float(x) for x in time_vals]),
            "line_coverage_pct": stats([float(x) for x in line_vals]),
            "branch_coverage_pct": stats([float(x) for x in branch_vals]),
        }

    # ------------------------------------------------------------------
    # Build per-event payload
    # ------------------------------------------------------------------

    def _build_metrics_payload(
        self,
        test_code: str,
        prev_test_code: str,
        compile_dict: Optional[Dict[str, Any]],
        rag_context: str,
    ) -> Dict[str, Any]:
        test_count = self._count_tests(test_code)
        assertion_count = self._count_assertions(test_code)
        leakage = self._detect_leakage(test_code)
        compile_diag = self._compile_diag(compile_dict or {}, n=260) if isinstance(compile_dict, dict) else ""

        rag_hit = bool((rag_context or "").strip())
        rag_chars = len(rag_context or "")

        payload: Dict[str, Any] = {
            "test_count": test_count,
            "assertion_count": assertion_count,
            "assertions_per_test": round((assertion_count / test_count), 3) if test_count else 0.0,
            "leakage": bool(leakage),
            "rag_hit": rag_hit,
            "rag_context_chars": rag_chars,
        }


        if compile_diag:
            payload["error_signatures"] = self._error_signatures(compile_diag)
            payload["compile_diag_chars"] = len(compile_diag)

        return payload

    # ------------------------------------------------------------------
    # HTML report (run + aggregate)
    # ------------------------------------------------------------------

    def _write_html_report_for_run(self, request_id: str, run_events: List[Dict[str, Any]]) -> str:
        if not run_events:
            return ""

        def esc(x):
            return html.escape(str(x)) if x is not None else ""

        run_events = sorted(run_events, key=lambda e: e.get("ts_utc", ""))

        first = run_events[0]
        session_id = first.get("session_id", "")
        service_path = first.get("service_path", "")
        test_path = first.get("test_path", "")

        final = None
        for e in reversed(run_events):
            if e.get("stage") in ("success", "fail", "tool_failure"):
                final = e
                break

        final_stage = (final or {}).get("stage", "UNKNOWN")
        compile_ok = (final or {}).get("compile_ok", None)
        metrics_final = (final or {}).get("metrics") or {}

        total_seconds = metrics_final.get("total_seconds")
        try:
            attempts_used = max(int(e.get("attempt") or 0) for e in run_events)
        except Exception:
            attempts_used = None

        # Outcome badge
        if final_stage == "success":
            outcome_badge = '<span class="badge good">✅ SUCCESS</span>'
        elif final_stage == "fail":
            outcome_badge = '<span class="badge bad">❌ FAILED</span>'
        else:
            outcome_badge = f'<span class="badge warn">⚠️ {esc(final_stage)}</span>'

        jacoco_cov = self._read_jacoco_coverage()
        cov_line = esc(jacoco_cov.get("line_pct")) if jacoco_cov.get("ok") else "N/A"
        cov_branch = esc(jacoco_cov.get("branch_pct")) if jacoco_cov.get("ok") else "N/A"
        cov_path = esc(jacoco_cov.get("path")) if jacoco_cov else ""
        cov_note = jacoco_cov.get("reason", "Parsed from jacoco.xml") if isinstance(jacoco_cov, dict) else ""

        # Aggregate metrics
        agg = self._aggregate_metrics_last_n_runs(self.metrics_agg_window) if self.metrics_enabled else {}

        # Table rows
        rows = []
        for e in run_events:
            m = e.get("metrics") or {}
            stage = (e.get("stage") or "").strip()
            rows.append(
                "<tr>"
                f"<td><span class='stage {esc(stage)}'>{esc(stage)}</span></td>"
                f"<td class='muted'>{esc(e.get('ts_utc'))}</td>"
                f"<td>{esc(e.get('attempt'))}</td>"
                f"<td>{esc(e.get('compile_ok'))}</td>"
                f"<td>{esc(e.get('returncode'))}</td>"
                f"<td>{esc(m.get('test_count'))}</td>"
                f"<td>{esc(m.get('assertion_count'))}</td>"
                f"<td>{esc(m.get('assertions_per_test'))}</td>"
                f"<td>{esc(m.get('leakage'))}</td>"
                f"<td>{esc(m.get('rag_hit'))}</td>"
                f"<td>{esc(m.get('rag_context_chars'))}</td>"
                "</tr>"
            )

        # Compile signature sections
        sig_sections = []
        for e in run_events:
            if e.get("stage") in ("compile", "compile_after_autofix"):
                sigs = (e.get("metrics") or {}).get("error_signatures") or []
                sig_html = (
                    "<ul>" + "".join(f"<li>{esc(s)}</li>" for s in sigs) + "</ul>"
                    if sigs else "<p class='small'><i>No error signatures.</i></p>"
                )
                sig_sections.append(
                    f"<h3>{esc(e.get('stage'))} (compile_ok={esc(e.get('compile_ok'))})</h3>{sig_html}"
                )

        html_text = f"""<!doctype html>
    <html>
    <head>
    <meta charset="utf-8"/>
    <title>TestWeaver Evaluation Report</title>
    <style>
    :root {{
    --bg:#0b1220; --panel:#0f1a2b; --panel2:#0c1626;
    --text:#e7edf7; --muted:#a8b3c7; --line:#22314a;
    --good:#22c55e; --bad:#ef4444; --warn:#f59e0b;
    }}
    body {{
    font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
    background: linear-gradient(180deg, var(--bg), #070b14 70%);
    color: var(--text); margin:0;
    }}
    .wrap {{ max-width:1100px; margin:auto; padding:28px 18px 60px; }}
    h1 {{ margin:0 0 6px; font-size:28px; }}
    .sub {{ color:var(--muted); margin-bottom:18px; }}
    .grid {{ display:grid; grid-template-columns:1.2fr .8fr; gap:14px; }}
    .card {{
    background:linear-gradient(180deg,var(--panel),var(--panel2));
    border:1px solid var(--line); border-radius:14px;
    padding:14px; box-shadow:0 10px 30px rgba(0,0,0,.35);
    }}
    .badge {{ padding:6px 10px; border-radius:999px; font-size:12px; }}
    .badge.good {{ background:rgba(34,197,94,.15); }}
    .badge.bad {{ background:rgba(239,68,68,.15); }}
    .badge.warn {{ background:rgba(245,158,11,.15); }}
    .kvs {{ display:grid; grid-template-columns:1fr 1fr; gap:10px; }}
    .kv {{ border:1px dashed rgba(255,255,255,.1); border-radius:12px; padding:10px; }}
    .k {{ color:var(--muted); font-size:12px; }}
    .v {{ font-size:13px; }}
    table {{
    width:100%; border-collapse:collapse; margin-top:14px;
    border:1px solid var(--line); border-radius:14px; overflow:hidden;
    }}
    th,td {{ padding:10px; font-size:13px; border-bottom:1px solid rgba(255,255,255,.08); }}
    th {{ color:var(--muted); text-transform:uppercase; font-size:11px; }}
    tr:nth-child(odd) {{ background:rgba(255,255,255,.02); }}
    .stage.gen {{ color:#60a5fa; }}
    .stage.fix {{ color:#a78bfa; }}
    .stage.compile {{ color:#93c5fd; }}
    .stage.compile_after_autofix {{ color:#34d399; }}
    .stage.success {{ color:var(--good); }}
    .stage.fail {{ color:var(--bad); }}
    .small {{ color:var(--muted); font-size:12px; }}
    </style>
    </head>

    <body>
    <div class="wrap">
    <h1>TestWeaver Evaluation Report</h1>
    <p class="sub">Compilation-aware test generation with RAG, bounded auto-repair, and JaCoCo evaluation</p>

    <div class="grid">
    <div class="card">
        <h2>Run Summary</h2>
        <div class="kvs">
        <div class="kv"><div class="k">Outcome</div><div class="v">{outcome_badge}</div></div>
        <div class="kv"><div class="k">Compilation OK</div><div class="v">{esc(compile_ok)}</div></div>
        <div class="kv"><div class="k">Attempts Used</div><div class="v">{esc(attempts_used)}</div></div>
        <div class="kv"><div class="k">Total Time (s)</div><div class="v">{esc(total_seconds)}</div></div>
        </div>
        <p class="small">request_id: <code>{esc(request_id)}</code></p>
        <p class="small">service: <code>{esc(service_path)}</code></p>
        <p class="small">test: <code>{esc(test_path)}</code></p>
    </div>

    <div class="card">
        <h2>Quality Signals</h2>
        <div class="kvs">
        <div class="kv"><div class="k">JaCoCo LINE %</div><div class="v">{cov_line}</div></div>
        <div class="kv"><div class="k">JaCoCo BRANCH %</div><div class="v">{cov_branch}</div></div>
        <div class="kv"><div class="k">jacoco.xml</div><div class="v"><code>{cov_path}</code></div></div>
        <div class="kv"><div class="k">Note</div><div class="v small">{esc(cov_note)}</div></div>
        </div>
    </div>
    </div>

    <div class="card">
    <h2>Stage Metrics (per attempt)</h2>
    <table>
    <thead>
    <tr>
    <th>Stage</th><th>Timestamp</th><th>Attempt</th><th>Compile OK</th><th>RC</th>
    <th>Tests</th><th>Assertions</th><th>A/Test</th><th>Leak</th><th>RAG</th><th>RAG chars</th>
    </tr>
    </thead>
    <tbody>
    {''.join(rows)}
    </tbody>
    </table>
    </div>

    <div class="card">
    <h2>Compilation Diagnostics</h2>
    {''.join(sig_sections) if sig_sections else "<p class='small'><i>No compile diagnostics.</i></p>"}
    </div>

    </div>
    </body>
    </html>
    """

        out_dir = pathlib.Path(self._reports_dir())
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"report_{request_id}.html"
        try:
            out_path.write_text(html_text, encoding="utf-8")
            return str(out_path)
        except Exception:
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(self, user_message: str, query_for_rag: Optional[str] = None) -> str:
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

        request_id = str(uuid.uuid4())
        t0 = time.time()
        run_events: List[Dict[str, Any]] = []

        service_path = self._norm_repo_path(service_path)

        java_source = self.git.get_file(service_path) or ""

        # Robust filename extraction even if service_path had "\" earlier
        class_name = pathlib.PurePosixPath(service_path).stem


        # LIMIT JAVA SOURCE SIZE (prevents Ollama timeouts)
        java_source = java_source[:12000]

        # RAG only on attempt 1
        rag_query = f"{class_name} {extra_instructions}".strip()
        rag_context = self.rag_index.retrieve_context(rag_query, top_k=5) or ""
        rag_context = rag_context[:3000]

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
            total_seconds = round(time.time() - t0, 3)
            evt = EvalEvent(
                request_id=request_id,
                session_id=self.session_id,
                service_path=service_path,
                test_path=test_path,
                attempt=0,
                stage="success",
                ts_utc=_utc_now_iso(),
                metrics={"cached": True, "total_seconds": total_seconds},
                compile_ok=True,
            )
            self._append_metrics_event(evt)
            run_events.append(evt.__dict__)
            report_path = self._write_html_report_for_run(request_id, run_events)
            return {
                "status": "SUCCESS",
                "service_path": service_path,
                "test_path": test_path,
                "attempts_used": 0,
                "attempt_log": [],
                "test_code": cached,
                "evaluation_report": {"ok": bool(report_path), "html": report_path},
            }

        last_test_code = ""
        baseline_test_code = ""
        last_compile: Optional[Dict[str, Any]] = None
        attempt_log: List[Dict[str, Any]] = []

        for attempt in range(1, max_attempts + 1):
            if attempt == 1:
                messages = list(base_messages)
            else:
                compiler_basis = self._compile_diag(last_compile or {}, n=160)
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": (
                        "STRICT REPAIR MODE.\n"
                        "You are editing ONE file ONLY: the Java test class shown in BASE TEST FILE.\n"
                        "DO NOT output any other file, enum, interface, record, helper class, or explanation.\n"
                        "Return ONLY the full corrected Java test class source code starting with 'package '.\n"
                        f"Output must define ONLY: public class {class_name}Test.\n"
                        "Fix ONLY the compilation errors from COMPILER ERROR."
                    )},
                    {"role": "user", "content": "BASE TEST FILE:\n" + (last_test_code or "")},
                    {"role": "user", "content": "COMPILER ERROR (actionable excerpt):\n" + (compiler_basis or "<EMPTY>")},
                ]

            prev_test_before_llm = last_test_code or ""
            response = self.llm.chat(messages, temperature=self.llm_temperature)

            candidate = self._extract_java_class(self._strip_code_fences(response))

            # FREEZE BASELINE BEFORE ANY AUTOFIX
            if attempt == 1 and not baseline_test_code:
                baseline_test_code = candidate

            if attempt > 1 and prev_test_before_llm:
                candidate = self._rename_test_methods_to_match_base(candidate, prev_test_before_llm)

            if not self._is_valid_java_test_file(candidate, class_name):
                candidate = prev_test_before_llm or last_test_code

            if attempt > 1 and not self._is_only_test_class(candidate, class_name):
                strict_msgs = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": (
                        "STRICT REPAIR MODE.\n"
                        "You are editing ONE file ONLY: the Java test class shown in BASE TEST FILE.\n"
                        "HARD RULES:\n"
                        "- Do NOT rename any @Test methods.\n"
                        "- Do NOT rename the test class.\n"
                        "- Do NOT delete any @Test methods.\n"
                        "- Do NOT add new files/types.\n"
                        "Return ONLY a single Java file starting with 'package '."
                    )},
                    {"role": "user", "content": "BASE TEST FILE:\n" + (prev_test_before_llm or "")},
                    {"role": "user", "content": "COMPILER ERROR:\n" + (self._compile_diag(last_compile or {}, n=160) or "<EMPTY>")},
                ]
                r = self.llm.chat(strict_msgs, temperature=self.llm_temperature)
                candidate2 = self._extract_java_class(self._strip_code_fences(r))
                candidate = candidate2 if self._is_only_test_class(candidate2, class_name) else (prev_test_before_llm or last_test_code)

            # Use last compile diag as hint if available; otherwise pass ""
            hint = self._compile_diag(last_compile or {}, n=200) if last_compile else ""
            candidate = self._auto_fix_common_java_test_compile_errors(candidate, compile_text=hint)
            test_code = candidate
            last_test_code = test_code

            if attempt == 1 and not baseline_test_code:
                baseline_test_code = test_code   # ✅ freeze baseline

            self.git.write_file(test_path, test_code, overwrite=True)

            # Metrics event: gen/fix
            try:
                payload = self._build_metrics_payload(
                    test_code,
                    baseline_test_code if baseline_test_code else prev_test_before_llm,
                    last_compile,
                    rag_context,
                )
                evt = EvalEvent(
                    request_id=request_id,
                    session_id=self.session_id,
                    service_path=service_path,
                    test_path=test_path,
                    attempt=attempt,
                    stage="gen" if attempt == 1 else "fix",
                    ts_utc=_utc_now_iso(),
                    metrics=payload,
                )
                self._append_metrics_event(evt)
                run_events.append(evt.__dict__)
            except Exception:
                pass

            if not compile_after:
                total_seconds = round(time.time() - t0, 3)
                evt = EvalEvent(
                    request_id=request_id,
                    session_id=self.session_id,
                    service_path=service_path,
                    test_path=test_path,
                    attempt=attempt,
                    stage="success",
                    ts_utc=_utc_now_iso(),
                    metrics={"compile_after": False, "total_seconds": total_seconds},
                    compile_ok=None,
                )
                self._append_metrics_event(evt)
                run_events.append(evt.__dict__)
                report_path = self._write_html_report_for_run(request_id, run_events)
                return {
                    "status": "SUCCESS",
                    "service_path": service_path,
                    "test_path": test_path,
                    "test_code": test_code,
                    "attempt_log": attempt_log,
                    "evaluation_report": {"ok": bool(report_path), "html": report_path},
                }

            # Compile
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
                attempt_log.append({"attempt": attempt, "stage": "compile", "ok": False, "http_status": http_status})

                diag = self._compile_diag(last_compile or {}, n=260)
                total_seconds = round(time.time() - t0, 3)
                evt = EvalEvent(
                    request_id=request_id,
                    session_id=self.session_id,
                    service_path=service_path,
                    test_path=test_path,
                    attempt=attempt,
                    stage="tool_failure",
                    ts_utc=_utc_now_iso(),
                    metrics={"total_seconds": total_seconds, "error_signatures": self._error_signatures(diag)},
                    compile_ok=False,
                    returncode=_safe_int(last_compile.get("returncode")),
                    http_status=_safe_int(last_compile.get("http_status")),
                    error_signatures=self._error_signatures(diag),
                )
                self._append_metrics_event(evt)
                run_events.append(evt.__dict__)

                report_path = self._write_html_report_for_run(request_id, run_events)
                return {
                    "status": "TOOL_FAILURE",
                    "service_path": service_path,
                    "test_path": test_path,
                    "test_code": test_code,
                    "attempt_log": attempt_log,
                    "evaluation_report": {"ok": bool(report_path), "html": report_path},
                }

            ok = bool(last_compile.get("ok"))
            attempt_log.append({"attempt": attempt, "stage": "compile", "ok": ok, "returncode": last_compile.get("returncode")})

            # Metrics event: compile
            try:
                compile_payload = self._build_metrics_payload(
                    last_test_code,
                    baseline_test_code if baseline_test_code else prev_test_before_llm,
                    last_compile,
                    rag_context
                )
                evt = EvalEvent(
                    request_id=request_id,
                    session_id=self.session_id,
                    service_path=service_path,
                    test_path=test_path,
                    attempt=attempt,
                    stage="compile",
                    ts_utc=_utc_now_iso(),
                    metrics=compile_payload,
                    compile_ok=ok,
                    returncode=_safe_int(last_compile.get("returncode")),
                    http_status=_safe_int(last_compile.get("http_status")),
                    error_signatures=compile_payload.get("error_signatures"),
                )
                self._append_metrics_event(evt)
                run_events.append(evt.__dict__)
            except Exception:
                pass

            if ok:
                self._compiled_cache[service_path] = test_code

                coverage = self.generate_coverage_report()
                jacoco_cov = self._read_jacoco_coverage()
                jacoco = self._parse_jacoco_xml("target/site/jacoco/jacoco.xml")

                total_seconds = round(time.time() - t0, 3)

                # SUCCESS event includes: time + coverage (LINE/BRANCH)
                evt = EvalEvent(
                    request_id=request_id,
                    session_id=self.session_id,
                    service_path=service_path,
                    test_path=test_path,
                    attempt=attempt,
                    stage="success",
                    ts_utc=_utc_now_iso(),
                    metrics={
                        "total_seconds": total_seconds,
                        **jacoco,
                    },
                    compile_ok=True,
                    returncode=_safe_int(last_compile.get("returncode")),
                    http_status=_safe_int(last_compile.get("http_status")),
                )
                self._append_metrics_event(evt)
                run_events.append(evt.__dict__)

                report_path = self._write_html_report_for_run(request_id, run_events)
                pr = self._maybe_open_pr(request_id, service_path, test_path, test_code)
                pr_or_branch = self._maybe_push_branch_on_success(request_id, service_path, test_path)

                return {
                    "status": "SUCCESS",
                    "service_path": service_path,
                    "test_path": test_path,
                    "test_code": test_code,
                    "attempt_log": attempt_log,
                    "compile": last_compile,
                    "pull_request": pr,
                    "pull_request": pr_or_branch,
                    "coverage": {
                        "ok": bool(coverage.get("ok")) if isinstance(coverage, dict) else False,
                        "report_html": "target/site/jacoco/index.html",
                        "report_xml": "target/site/jacoco/jacoco.xml",
                        "parsed": jacoco,
                        "raw": coverage,
                    },
                    "jacoco_coverage": jacoco_cov,
                    "evaluation_report": {"ok": bool(report_path), "html": report_path},
                }

            # Deterministic auto-fix path
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

                ok2 = bool(last_compile.get("ok"))
                attempt_log.append({"attempt": attempt, "stage": "compile_after_autofix", "ok": ok2, "returncode": last_compile.get("returncode")})

                # compile_after_autofix event
                try:
                    payload2 = self._build_metrics_payload(last_test_code, prev_test_before_llm, last_compile, rag_context)
                    evt2 = EvalEvent(
                        request_id=request_id,
                        session_id=self.session_id,
                        service_path=service_path,
                        test_path=test_path,
                        attempt=attempt,
                        stage="compile_after_autofix",
                        ts_utc=_utc_now_iso(),
                        metrics=payload2,
                        compile_ok=ok2,
                        returncode=_safe_int(last_compile.get("returncode")),
                        http_status=_safe_int(last_compile.get("http_status")),
                        error_signatures=payload2.get("error_signatures"),
                    )
                    self._append_metrics_event(evt2)
                    run_events.append(evt2.__dict__)
                except Exception:
                    pass

                if ok2:
                    self._compiled_cache[service_path] = fixed
                    coverage = self.generate_coverage_report()
                    jacoco_cov = self._read_jacoco_coverage()
                    jacoco = self._parse_jacoco_xml("target/site/jacoco/jacoco.xml")
                    total_seconds = round(time.time() - t0, 3)

                    evt = EvalEvent(
                        request_id=request_id,
                        session_id=self.session_id,
                        service_path=service_path,
                        test_path=test_path,
                        attempt=attempt,
                        stage="success",
                        ts_utc=_utc_now_iso(),
                        metrics={
                            "total_seconds": total_seconds,
                            "after_autofix": True,
                            **jacoco,
                        },
                        compile_ok=True,
                        returncode=_safe_int(last_compile.get("returncode")),
                        http_status=_safe_int(last_compile.get("http_status")),
                    )
                    self._append_metrics_event(evt)
                    run_events.append(evt.__dict__)

                    report_path = self._write_html_report_for_run(request_id, run_events)
                    pr = self._maybe_open_pr(request_id, service_path, test_path, fixed)
                    pr_or_branch = self._maybe_push_branch_on_success(request_id, service_path, test_path)

                    return {
                        "status": "SUCCESS",
                        "service_path": service_path,
                        "test_path": test_path,
                        "test_code": fixed,
                        "attempt_log": attempt_log,
                        "compile": last_compile,
                        "pull_request": pr,
                        "pull_request": pr_or_branch,
                        "coverage": {
                            "ok": bool(coverage.get("ok")) if isinstance(coverage, dict) else False,
                            "report_html": "target/site/jacoco/index.html",
                            "report_xml": "target/site/jacoco/jacoco.xml",
                            "parsed": jacoco,
                            "raw": coverage,
                        },
                        "jacoco_coverage": jacoco_cov,
                        "evaluation_report": {"ok": bool(report_path), "html": report_path},
                    }

        # FINAL FAIL
        diag = self._compile_diag(last_compile or {}, n=260)
        total_seconds = round(time.time() - t0, 3)
        evt = EvalEvent(
            request_id=request_id,
            session_id=self.session_id,
            service_path=service_path,
            test_path=test_path,
            attempt=max_attempts,
            stage="fail",
            ts_utc=_utc_now_iso(),
            metrics={"total_seconds": total_seconds, "final_error_signatures": self._error_signatures(diag)},
            compile_ok=False,
            returncode=_safe_int((last_compile or {}).get("returncode") if isinstance(last_compile, dict) else None),
            http_status=_safe_int((last_compile or {}).get("http_status") if isinstance(last_compile, dict) else None),
            error_signatures=self._error_signatures(diag),
        )
        self._append_metrics_event(evt)
        run_events.append(evt.__dict__)

        report_path = self._write_html_report_for_run(request_id, run_events)

        return {
            "status": "COMPILATION_FAILED",
            "service_path": service_path,
            "test_path": test_path,
            "test_code": last_test_code,
            "attempt_log": attempt_log,
            "compile": last_compile,
            "evaluation_report": {"ok": bool(report_path), "html": report_path},
        }

    # ------------------------------------------------------------------
    # Related source loading (to improve generation itself)
    # ------------------------------------------------------------------

    def _collect_related_sources(self, service_path: str, service_source: str) -> List[Tuple[str, str]]:
        paths: List[str] = []

        env = (os.getenv("RELATED_SOURCES") or "").strip()
        if env:
            for p in env.split(","):
                p = p.strip()
                if p:
                    paths.append(p)

        for m in re.finditer(r"^\s*import\s+([\w\.]+)\s*;\s*$", service_source or "", re.MULTILINE):
            fqcn = m.group(1)
            if ".dto." in fqcn or ".model." in fqcn:
                rel = "src/main/java/" + fqcn.replace(".", "/") + ".java"
                paths.append(rel)

        seen = set()
        uniq: List[str] = []
        for p in paths:
            if p not in seen:
                uniq.append(p)
                seen.add(p)

        out: List[Tuple[str, str]] = []
        for p in uniq[:10]:
            try:
                src = self.git.get_file(p)
                if src:
                    out.append((p, src))
            except Exception:
                continue
        return out

    def _summarize_related_public_api(self, related: List[Tuple[str, str]]) -> str:
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
            ctors.append(f"{class_name}({args})" if args.strip() else f"{class_name}()")
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

    def _norm_repo_path(self, p: str) -> str:
        """
        Normalize Windows/Unix path into repo-relative POSIX style.
        """
        p = (p or "").strip().replace("\\", "/")
        # remove drive letter if accidentally passed (e.g. D:/...)
        p = re.sub(r"^[A-Za-z]:/", "", p)
        return p.lstrip("/")

    def _maybe_push_branch_on_success(self, request_id: str, service_path: str, test_path: str) -> Dict[str, Any]:
        if (os.getenv("AUTO_PR_ON_SUCCESS", "false").strip().lower() != "true"):
            return {"ok": False, "skipped": True, "reason": "AUTO_PR_ON_SUCCESS=false"}

        base = (os.getenv("AUTO_PR_BASE_BRANCH") or "main").strip()
        prefix = (os.getenv("AUTO_PR_BRANCH_PREFIX") or "testweaver/").strip()
        remote = (os.getenv("GIT_REMOTE") or "origin").strip()

        class_name = service_path.split("/")[-1].replace(".java", "")
        branch = f"{prefix}{class_name}-tests-{request_id[:8]}"
        msg = f"Add {class_name} tests (TestWeaver {request_id[:8]})"

        if not hasattr(self.git, "push_branch_with_commit"):
            return {"ok": False, "skipped": True, "reason": "MCPGitClient.push_branch_with_commit not available"}

        return self.git.push_branch_with_commit(
            branch=branch,
            base_branch=base,
            commit_message=msg,
            files=["src/test/java"],
            remote=remote,
        )

    def _maybe_open_pr(self, request_id: str, service_path: str, test_path: str, test_code: str) -> Dict[str, Any]:
        if not getattr(self, "auto_pr_on_success", False):
            return {"ok": False, "skipped": True, "reason": "AUTO_PR_ON_SUCCESS=false"}

        base = (os.getenv("AUTO_PR_BASE_BRANCH") or "main").strip()
        prefix = (os.getenv("AUTO_PR_BRANCH_PREFIX") or "testweaver/").strip()
        labels = [x.strip() for x in (os.getenv("AUTO_PR_LABELS") or "").split(",") if x.strip()]

        class_name = service_path.split("/")[-1].replace(".java", "")
        branch = f"{prefix}{class_name}-tests-{request_id[:8]}"

        title = f"TestWeaver: {class_name} tests (compile OK)"
        body = "\n".join([
            f"- request_id: `{request_id}`",
            f"- service: `{service_path}`",
            f"- test: `{test_path}`",
            f"- status: compile OK",
        ])

        # open PR via Git MCP
        if not hasattr(self.git, "open_pr_on_success"):
            return {"ok": False, "skipped": True, "reason": "MCPGitClient.open_pr_on_success not available"}

        return self.git.open_pr_on_success(
            branch=branch,
            base_branch=base,
            title=title,
            body=body,
            files=["src/test/java"],  # only commit the generated test file
            commit_message=f"Add {class_name} JUnit tests ({request_id[:8]})",
            labels=labels,
        )

    def _extract_package(self, java_source: str) -> str:
        m = re.search(r"^\s*package\s+([\w\.]+)\s*;", java_source or "", re.MULTILINE)
        return m.group(1) if m else ""

    def _guess_test_path(self, package_name: str, class_name: str) -> str:
        class_name = (class_name or "").strip()
        class_name = re.sub(r"[^A-Za-z0-9_]", "", class_name)  # safety

        pkg = (package_name or "").strip().replace(".", "/")
        if pkg:
            return f"src/test/java/{pkg}/{class_name}Test.java"
        return f"src/test/java/{class_name}Test.java"

    def _strip_code_fences(self, text: str) -> str:
        s = (text or "").strip()
        m = re.search(r"```(?:java)?\s*(.*?)\s*```", s, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else s

    def _extract_java_class(self, raw: str) -> str:
        raw = (raw or "").strip()
        last = raw.rfind("}")
        return raw[: last + 1].strip() if last != -1 else raw

    def _normalize_for_compare(self, s: str) -> str:
        if not s:
            return ""
        return "\n".join(line.rstrip() for line in s.strip().splitlines()).strip()

    # ------------------------------------------------------------------
    # Output validation
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
        s = (code or "").strip()
        if not s.startswith("package "):
            return False
        if f"class {class_name}Test" not in s and f"public class {class_name}Test" not in s:
            return False

        scrubbed = re.sub(r"//.*?$|/\*.*?\*/|\".*?\"", "", s, flags=re.MULTILINE | re.DOTALL)

        for m in re.finditer(r"(?m)^\s*public\s+class\s+([A-Za-z_]\w*)\b", scrubbed):
            if m.group(1) != f"{class_name}Test":
                return False

        if re.search(r"(?m)^\s*public\s+enum\s+\w+\b", scrubbed):
            return False
        if re.search(r"(?m)^\s*public\s+interface\s+\w+\b", scrubbed):
            return False
        if re.search(r"(?m)^\s*public\s+record\s+\w+\b", scrubbed):
            return False

        return True

    # ------------------------------------------------------------------
    # Deterministic fixes (PATCHED)
    # ------------------------------------------------------------------

    def _auto_fix_common_java_test_compile_errors(self, test_code: str, compile_text: str) -> str:
        """
        Generic, repo-agnostic deterministic fixes for common Java *test* compilation issues.

        FIXED HERE:
        - If MockitoExtension is missing, we DO NOT re-inject it later (prevents endless loop)
        - De-duplicates stacked @Mock annotations robustly
        - Removes accidental Spring @Service leakage on test classes
        """
        if not test_code:
            return test_code

        compile_text = (compile_text or "")
        lc = compile_text.lower()
        updated = test_code

        # Detect missing MockitoExtension in classpath (JUnit5 Mockito extension not available)
        mockito_ext_missing = (
            "mockitoextension cannot be resolved" in lc
            or "class<mockitoextension>" in lc
            or ("org.mockito.junit.jupiter.mockitoextension" in lc and ("does not exist" in lc or "cannot access" in lc))
        )

        # Remove common Spring stereotype leakage from tests (harmless + avoids confusion)
        updated = re.sub(r"(?m)^\s*@Service\s*\r?\n", "", updated)
        updated = re.sub(r"(?m)^\s*import\s+org\.springframework\.stereotype\.Service\s*;\s*\r?\n", "", updated)

        # --- Strong dedupe: collapse any consecutive @Mock annotations into a single @Mock ---
        updated = re.sub(r"(?m)(^\s*@Mock\s*\r?\n){2,}", "@Mock\n", updated)
        updated = re.sub(
            r"(?m)(^\s*@Mock\s*\r?\n)+(?=\s*(?:private|protected|public)\s+)",
            "@Mock\n",
            updated,
        )

        # --- Fallback if MockitoExtension is missing in classpath ---
        if mockito_ext_missing:
            # Remove @ExtendWith(MockitoExtension.class)
            updated = re.sub(r"(?m)^\s*@ExtendWith\s*\(\s*MockitoExtension\.class\s*\)\s*\r?\n", "", updated)

            # Remove related imports
            updated = re.sub(r"(?m)^\s*import\s+org\.junit\.jupiter\.api\.extension\.ExtendWith\s*;\s*\r?\n", "", updated)
            updated = re.sub(r"(?m)^\s*import\s+org\.mockito\.junit\.jupiter\.MockitoExtension\s*;\s*\r?\n", "", updated)

            # Ensure MockitoAnnotations import
            updated = self._ensure_import(updated, "import org.mockito.MockitoAnnotations;")
            updated = self._ensure_import(updated, "import org.junit.jupiter.api.BeforeEach;")

            # Add AutoCloseable field to close mocks (optional but clean)
            if not re.search(r"\bAutoCloseable\s+mocks\b", updated):
                updated = self._ensure_import(updated, "import java.lang.AutoCloseable;")
                updated = re.sub(
                    r"(?m)^\s*public\s+class\s+\w+Test\s*\{\s*",
                    lambda m: m.group(0) + "\n    private AutoCloseable mocks;\n",
                    updated,
                    count=1
                )

            # Ensure @BeforeEach exists and calls openMocks
            if "@BeforeEach" in updated:
                if "MockitoAnnotations.openMocks(this)" not in updated:
                    updated = re.sub(
                        r"(?s)(@BeforeEach\s*\r?\n\s*(?:public|protected|private)?\s*void\s+\w+\s*\(\s*\)\s*\{\s*)",
                        r"\1\n        mocks = MockitoAnnotations.openMocks(this);\n",
                        updated,
                        count=1
                    )
            else:
                updated = re.sub(
                    r"(?m)^\s*public\s+class\s+\w+Test\s*\{\s*",
                    lambda m: m.group(0) + "\n    @BeforeEach\n    void setUp() {\n        mocks = MockitoAnnotations.openMocks(this);\n    }\n",
                    updated,
                    count=1
                )

            # Add @AfterEach to close mocks (optional, but avoids warnings/leaks)
            updated = self._ensure_import(updated, "import org.junit.jupiter.api.AfterEach;")
            if "@AfterEach" not in updated:
                updated = re.sub(
                    r"(?m)^\s*public\s+class\s+\w+Test\s*\{\s*",
                    lambda m: m.group(0) + "\n    @AfterEach\n    void tearDown() throws Exception {\n        if (mocks != null) mocks.close();\n    }\n",
                    updated,
                    count=1
                )

        # ----------------------------------------------------------------------------------
        # 0) Always-on "safe" structural fixes for test scaffolding (generic across repos)
        # ----------------------------------------------------------------------------------

        # 0.1 Remove 'final' from field declarations inside tests
        updated = re.sub(r"(?m)^(\s*(?:private|protected|public)\s+)final(\s+)", r"\1", updated)

        # 0.2 If "<something>Service." is referenced but the field doesn't exist, inject a field.
        refs = set(re.findall(r"\b([a-z_]\w*)\s*\.", updated))
        sut_var = ""
        for r in refs:
            if r.lower().endswith("service"):
                sut_var = r
                break

        m = re.search(r"(?m)^\s*public\s+class\s+([A-Za-z_]\w*)Test\s*\{", updated)
        sut_type = m.group(1) if m else ""

        if sut_var and sut_type and not re.search(rf"\b{re.escape(sut_type)}\s+{re.escape(sut_var)}\b", updated):
            updated = self._ensure_import(updated, "import org.mockito.InjectMocks;")

            # Prefer MockitoExtension when available; otherwise do NOT inject it.
            if not mockito_ext_missing:
                updated = self._ensure_import(updated, "import org.junit.jupiter.api.extension.ExtendWith;")
                updated = self._ensure_import(updated, "import org.mockito.junit.jupiter.MockitoExtension;")

                if "@ExtendWith(MockitoExtension.class)" not in updated:
                    updated = re.sub(
                        r"(?m)^\s*public\s+class\s+([A-Za-z_]\w*)Test\s*\{",
                        r"@ExtendWith(MockitoExtension.class)\npublic class \1Test {",
                        updated,
                        count=1,
                    )

                if "@InjectMocks" not in updated:
                    updated = re.sub(
                        r"(?m)^\s*@ExtendWith\(MockitoExtension\.class\)\s*\r?\n\s*public\s+class\s+\w+Test\s*\{\s*",
                        lambda mm: mm.group(0) + f"\n    @InjectMocks\n    private {sut_type} {sut_var};\n",
                        updated,
                        count=1,
                    )
            else:
                # No MockitoExtension: inject SUT field directly inside class
                if "@InjectMocks" not in updated:
                    updated = re.sub(
                        r"(?m)^\s*public\s+class\s+\w+Test\s*\{\s*",
                        lambda mm: mm.group(0) + f"\n    @InjectMocks\n    private {sut_type} {sut_var};\n",
                        updated,
                        count=1,
                    )

        # 0.3 If repo/dao fields exist, annotate them with @Mock (generic naming heuristics)
        updated = self._ensure_import(updated, "import org.mockito.Mock;")

        def _mock_field_repl(match: re.Match) -> str:
            vis = match.group(1)
            ftype = match.group(2)
            name = match.group(3)
            return f"    @Mock\n    {vis} {ftype} {name};"

        updated = re.sub(
            r"(?m)^(?!\s*@Mock\s*$)\s*(private|protected|public)\s+([A-Za-z_]\w*(?:Repository|Repo|Dao))\s+([A-Za-z_]\w*)\s*;\s*$",
            _mock_field_repl,
            updated,
        )

        # ----------------------------------------------------------------------------------
        # 1) Ensure JUnit 5 imports (compile-safe)
        # ----------------------------------------------------------------------------------
        if "@Test" in updated and "import org.junit.jupiter.api.Test;" not in updated:
            updated = self._ensure_import(updated, "import org.junit.jupiter.api.Test;")
        if "@BeforeEach" in updated and "import org.junit.jupiter.api.BeforeEach;" not in updated:
            updated = self._ensure_import(updated, "import org.junit.jupiter.api.BeforeEach;")
        if "@AfterEach" in updated and "import org.junit.jupiter.api.AfterEach;" not in updated:
            updated = self._ensure_import(updated, "import org.junit.jupiter.api.AfterEach;")

        if "Assertions." in updated or re.search(r"\bassert[A-Z]\w*\s*\(", updated):
            if "import static org.junit.jupiter.api.Assertions.*;" not in updated:
                updated = self._ensure_import(updated, "import static org.junit.jupiter.api.Assertions.*;")

        # ----------------------------------------------------------------------------------
        # 2) Common Java stdlib imports
        # ----------------------------------------------------------------------------------
        if ("Optional." in updated or "Optional<" in updated) and "import java.util.Optional;" not in updated:
            updated = self._ensure_import(updated, "import java.util.Optional;")
        if ("BigDecimal." in updated or "new BigDecimal" in updated) and "import java.math.BigDecimal;" not in updated:
            updated = self._ensure_import(updated, "import java.math.BigDecimal;")
        if ("List<" in updated or "ArrayList" in updated) and "import java.util.List;" not in updated:
            updated = self._ensure_import(updated, "import java.util.List;")

        # ----------------------------------------------------------------------------------
        # 3) Mockito static imports if Mockito usage is present
        # ----------------------------------------------------------------------------------
        if re.search(r"\bwhen\s*\(", updated) or re.search(r"\bverify\s*\(", updated) or "Mockito." in updated:
            updated = self._ensure_import(updated, "import static org.mockito.Mockito.*;")

        # ----------------------------------------------------------------------------------
        # 4) Compile-text-driven fixes
        # ----------------------------------------------------------------------------------
        lc = (compile_text or "").lower()

        # 4.1 ResponseEntity mismatch
        if ("incompatible types" in lc and "responseentity" in lc) or ("ResponseEntity<" in updated and "responseentity" in lc):
            updated = re.sub(
                r"ResponseEntity\s*<\s*([A-Za-z_]\w*)\s*>\s+(\w+)\s*=",
                r"\1 \2 =",
                updated,
            )
            updated = updated.replace(".getBody()", "")
            updated = re.sub(r"(?m)^\s*import\s+org\.springframework\.http\.ResponseEntity;\s*\r?\n", "", updated)

        # 4.2 TransactionType missing
        if ("TransactionType" in updated) and (
            ("symbol:   class transactiontype" in lc)
            or ("cannot find symbol" in lc and "transactiontype" in lc)
            or ("transactiontype cannot be resolved" in lc)
        ):
            updated = re.sub(r"\bTransactionType\s*\.\s*[A-Z_]+\b", "null", updated)
            updated = re.sub(r"(?m)^\s*import\s+.*TransactionType\s*;\s*\r?\n", "", updated)

        # 4.3 Remove assertions calling missing getters
        if "getreference" in lc and ("cannot find symbol" in lc or "method getreference" in lc):
            updated = re.sub(r"(?m)^\s*assertEquals\([^;]*getReference\(\)[^;]*\);\s*\r?\n?", "", updated)
        if "getstatus" in lc and ("cannot find symbol" in lc or "method getstatus" in lc):
            updated = re.sub(r"(?m)^\s*assertEquals\([^;]*getStatus\(\)[^;]*\);\s*\r?\n?", "", updated)
        if "getmessage" in lc and ("cannot find symbol" in lc or "method getmessage" in lc):
            updated = re.sub(r"(?m)^\s*assertEquals\([^;]*getMessage\(\)[^;]*\);\s*\r?\n?", "", updated)

        # 4.4 Rewrite all-args constructor to no-args if compiler says so
        ctor_noarg_classes = self._classes_with_noarg_required(compile_text)
        for cls in ctor_noarg_classes:
            updated = self._rewrite_all_args_to_noarg(updated, cls)

        # ----------------------------------------------------------------------------------
        # 5) Additional generic fixes for "cannot be resolved" WITHOUT re-injecting MockitoExtension when missing
        # ----------------------------------------------------------------------------------
        if "cannot be resolved" in lc:
            refs2 = set(re.findall(r"\b([a-z_]\w*)\s*\.", updated))
            sut_var2 = ""
            for r in refs2:
                if r.lower().endswith("service"):
                    sut_var2 = r
                    break
            m2 = re.search(r"(?m)^\s*public\s+class\s+([A-Za-z_]\w*)Test\s*\{", updated)
            sut_type2 = m2.group(1) if m2 else ""
            if sut_var2 and sut_type2 and not re.search(rf"\b{re.escape(sut_type2)}\s+{re.escape(sut_var2)}\b", updated):
                updated = self._ensure_import(updated, "import org.mockito.InjectMocks;")

                if not mockito_ext_missing:
                    updated = self._ensure_import(updated, "import org.junit.jupiter.api.extension.ExtendWith;")
                    updated = self._ensure_import(updated, "import org.mockito.junit.jupiter.MockitoExtension;")
                    if "@ExtendWith(MockitoExtension.class)" not in updated:
                        updated = re.sub(
                            r"(?m)^\s*public\s+class\s+([A-Za-z_]\w*)Test\s*\{",
                            r"@ExtendWith(MockitoExtension.class)\npublic class \1Test {",
                            updated,
                            count=1,
                        )
                    if "@InjectMocks" not in updated:
                        updated = re.sub(
                            r"(?m)^\s*@ExtendWith\(MockitoExtension\.class\)\s*\r?\n\s*public\s+class\s+\w+Test\s*\{\s*",
                            lambda mm: mm.group(0) + f"\n    @InjectMocks\n    private {sut_type2} {sut_var2};\n",
                            updated,
                            count=1,
                        )
                else:
                    if "@InjectMocks" not in updated:
                        updated = re.sub(
                            r"(?m)^\s*public\s+class\s+\w+Test\s*\{\s*",
                            lambda mm: mm.group(0) + f"\n    @InjectMocks\n    private {sut_type2} {sut_var2};\n",
                            updated,
                            count=1,
                        )

        # Final safety pass: collapse @Mock duplicates again
        updated = re.sub(r"(?m)(^\s*@Mock\s*\r?\n){2,}", "@Mock\n", updated)
        updated = re.sub(
            r"(?m)(^\s*@Mock\s*\r?\n)+(?=\s*(?:private|protected|public)\s+)",
            "@Mock\n",
            updated,
        )

        return updated

    def _classes_with_noarg_required(self, compile_text: str) -> List[str]:
        if not compile_text:
            return []
        out: List[str] = []
        for m in re.finditer(
            r"constructor\s+([A-Za-z_]\w*)\s+in\s+class\s+[\w\.]+\s+cannot\s+be\s+applied.*?required:\s+no arguments",
            compile_text, re.IGNORECASE | re.DOTALL
        ):
            out.append(m.group(1))
        for m in re.finditer(
            r"constructor\s+([A-Za-z_]\w*)\s+cannot\s+be\s+applied.*?required:\s+no arguments",
            compile_text, re.IGNORECASE | re.DOTALL
        ):
            out.append(m.group(1))
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    def _rewrite_all_args_to_noarg(self, code: str, class_name: str) -> str:
        if not code or not class_name:
            return code
        pat = re.compile(rf"new\s+{re.escape(class_name)}\s*\(\s*[^)]*\)", re.MULTILINE)
        return pat.sub(f"new {class_name}()", code)

    # ------------------------------------------------------------------
    # Import helper + rename helpers
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
        if not code:
            return []
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
        base_names = self._extract_test_method_names(base)
        cand_names = self._extract_test_method_names(candidate)
        if not base_names or not cand_names:
            return candidate
        if len(base_names) != len(cand_names):
            return candidate
        if base_names == cand_names:
            return candidate

        updated = candidate
        for old, new in zip(cand_names, base_names):
            if old != new:
                updated = re.sub(
                    rf"(\b(?:public|protected|private)?\s*(?:void|[\w\<\>\[\]\.]+)\s+){re.escape(old)}(\s*\()",
                    rf"\1{new}\2",
                    updated,
                    count=1
                )
        return updated

    # ------------------------------------------------------------------
    # Coverage (JaCoCo)
    # ------------------------------------------------------------------

    def generate_coverage_report(self) -> Dict[str, Any]:
        """
        Runs tests and generates JaCoCo report (requires JaCoCo plugin in project).
        """
        extra = ["-Dmaven.test.failure.ignore=true", "jacoco:report"]
        res = self.git.compile(
            tool="maven",
            goal="test",
            project_path=".",
            timeout_seconds=1200,
            extra_args=extra,
        )
        return res

    def _jacoco_xml_path(self) -> str:
        # Prefer explicit local path (JaCoCo output exists here)
        svc_root = (os.getenv("GIT_LOCAL_REPO") or "").strip()
        if svc_root:
            return str(pathlib.Path(svc_root) / "target" / "site" / "jacoco" / "jacoco.xml")

        # Fallback: try repo_root if it is a real filesystem path
        return str(pathlib.Path(self.repo_root) / "target" / "site" / "jacoco" / "jacoco.xml")

    def _read_jacoco_coverage(self) -> Dict[str, Any]:
        """
        Returns {"line_pct": float|None, "branch_pct": float|None, "path": str, "ok": bool}
        Parses <counter type="LINE|BRANCH" missed="" covered=""/> from jacoco.xml
        """
        p = pathlib.Path(self._jacoco_xml_path())
        if not p.exists():
            return {"ok": False, "path": str(p), "line_pct": None, "branch_pct": None, "reason": "jacoco.xml not found"}

        try:
            tree = ET.parse(str(p))
            root = tree.getroot()

            def pct(counter_type: str) -> Optional[float]:
                # Prefer top-level counters if present
                for c in root.findall("counter"):
                    if (c.get("type") or "").upper() == counter_type:
                        missed = int(c.get("missed") or "0")
                        covered = int(c.get("covered") or "0")
                        total = missed + covered
                        return round((covered * 100.0 / total), 2) if total else 0.0

                # Fallback: sum all counters of the same type (rare but safe)
                missed_sum = covered_sum = 0
                for c in root.iter("counter"):
                    if (c.get("type") or "").upper() == counter_type:
                        missed_sum += int(c.get("missed") or "0")
                        covered_sum += int(c.get("covered") or "0")
                total = missed_sum + covered_sum
                return round((covered_sum * 100.0 / total), 2) if total else None

            return {
                "ok": True,
                "path": str(p),
                "line_pct": pct("LINE"),
                "branch_pct": pct("BRANCH"),
            }
        except Exception as e:
            return {"ok": False, "path": str(p), "line_pct": None, "branch_pct": None, "reason": str(e)[:300]}

    # ------------------------------------------------------------------
    # Legacy helper (kept as-is; not used in main flow)
    # ------------------------------------------------------------------

    def _normalize_mockito_scaffold(self, test_code: str) -> str:
        if not test_code:
            return test_code
        s = test_code

        # 1) Remove 'final' from field declarations (safe + generic for tests)
        s = re.sub(r"(?m)^(\s*(?:private|protected|public)\s+)final(\s+)", r"\1", s)

        # 2) Detect class under test from filename convention: FooServiceTest -> FooService
        m = re.search(r"(?m)^\s*public\s+class\s+([A-Za-z_]\w*)Test\s*\{", s)
        sut_type = m.group(1) if m else ""

        # 3) If code references "<name>Service." but field not declared, inject field
        refs = set(re.findall(r"\b([a-z_]\w*)\s*\.", s))
        sut_var = ""
        for r in refs:
            if r.lower().endswith("service"):
                sut_var = r
                break

        if sut_var and not re.search(rf"\b{re.escape(sut_type)}\s+{re.escape(sut_var)}\b", s):
            # Ensure imports for Mockito extension + InjectMocks
            s = self._ensure_import(s, "import org.junit.jupiter.api.extension.ExtendWith;")
            s = self._ensure_import(s, "import org.mockito.InjectMocks;")
            s = self._ensure_import(s, "import org.mockito.junit.jupiter.MockitoExtension;")

            # Add @ExtendWith if missing
            if "@ExtendWith(MockitoExtension.class)" not in s:
                s = re.sub(
                    r"(?m)^\s*public\s+class\s+([A-Za-z_]\w*)Test\s*\{",
                    r"@ExtendWith(MockitoExtension.class)\npublic class \1Test {",
                    s,
                    count=1
                )

            # Add field near start of class body
            if sut_type:
                s = re.sub(
                    r"(?m)^\s*@ExtendWith\(MockitoExtension\.class\)\s*\r?\n\s*public\s+class\s+\w+Test\s*\{\s*",
                    lambda mm: mm.group(0) + f"\n    @InjectMocks\n    private {sut_type} {sut_var};\n",
                    s,
                    count=1
                )

        return s
