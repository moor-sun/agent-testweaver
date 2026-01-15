import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

def parse_iso(ts: str):
    return datetime.fromisoformat(ts)

def fmt_time(ts: str):
    dt = parse_iso(ts)
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def fmt_time_only(ts: str):
    dt = parse_iso(ts)
    return dt.strftime("%H:%M:%S")

def seconds_between(ts1: str, ts2: str):
    return (parse_iso(ts2) - parse_iso(ts1)).total_seconds()

def load_events(jsonl_path: str):
    events = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events

def group_by_request(events):
    by_req = defaultdict(list)
    for e in events:
        by_req[e.get("request_id", "UNKNOWN")].append(e)
    # sort each run by timestamp
    for rid in by_req:
        by_req[rid].sort(key=lambda x: x.get("ts_utc", ""))
    return by_req

def render_single_run_md(run_events):
    if not run_events:
        return "# Empty run\n"

    first = run_events[0]
    rid = first.get("request_id")
    session_id = first.get("session_id")
    service_path = first.get("service_path")
    test_path = first.get("test_path")

    # final status event
    final = None
    for e in reversed(run_events):
        if e.get("stage") in ("success", "fail", "tool_failure"):
            final = e
            break

    # derive attempt count
    attempts = sorted({e.get("attempt") for e in run_events if e.get("attempt") is not None})
    attempts_used = max(attempts) if attempts else 0

    # RAG info from first metrics record
    rag_hit = None
    rag_chars = None
    for e in run_events:
        m = e.get("metrics") or {}
        if "rag_hit" in m:
            rag_hit = m.get("rag_hit")
            rag_chars = m.get("rag_context_chars")
            break

    lines = []
    lines.append(f"# TestWeaver Evaluation Report (Single Run)")
    lines.append("")
    lines.append(f"**Run ID (request_id):** `{rid}`  ")
    lines.append(f"**Session:** `{session_id}`  ")
    lines.append(f"**Service Under Test:** `{service_path}`  ")
    lines.append(f"**Generated Test File:** `{test_path}`  ")
    lines.append(f"**RAG Used:** {'Yes' if rag_hit else 'No'} (context chars ≈ {rag_chars if rag_chars is not None else 'N/A'})")
    lines.append("")

    final_stage = final.get("stage") if final else "UNKNOWN"
    final_ok = final.get("compile_ok") if final else None
    total_seconds = None
    if final and isinstance(final.get("metrics"), dict):
        total_seconds = final["metrics"].get("total_seconds")

    lines.append("## Outcome Summary")
    if final_stage == "success":
        lines.append(f"- **Final Status:** ✅ SUCCESS")
    elif final_stage == "fail":
        lines.append(f"- **Final Status:** ❌ FAILED")
    else:
        lines.append(f"- **Final Status:** ⚠️ {final_stage}")

    lines.append(f"- **Attempts Used:** {attempts_used}")
    if total_seconds is not None:
        lines.append(f"- **Total Runtime (agent timer):** {total_seconds} seconds")
    lines.append("")

    lines.append("## Stage-by-stage evaluation")
    lines.append("")
    lines.append("| Stage | Time (UTC) | Compile OK | Tests | Assertions | Assertions/Test | Leakage | RAG Hit | Diff Added/Removed/Changed |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|")

    for e in run_events:
        stage = e.get("stage")
        ts = e.get("ts_utc")
        compile_ok = e.get("compile_ok")
        m = e.get("metrics") or {}

        test_count = m.get("test_count")
        assertion_count = m.get("assertion_count")
        apt = m.get("assertions_per_test")
        leakage = m.get("leakage")
        rag_hit_e = m.get("rag_hit")
        added = m.get("lines_added")
        removed = m.get("lines_removed")
        changed = m.get("lines_changed")

        def yn(v):
            if v is None:
                return "–"
            return "Yes" if v else "No"

        lines.append(
            f"| {stage} | {fmt_time_only(ts) if ts else '–'} | {yn(compile_ok)} | "
            f"{test_count if test_count is not None else '–'} | "
            f"{assertion_count if assertion_count is not None else '–'} | "
            f"{apt if apt is not None else '–'} | "
            f"{yn(leakage)} | {yn(rag_hit_e)} | "
            f"+{added if added is not None else '–'} / -{removed if removed is not None else '–'} / ~{changed if changed is not None else '–'} |"
        )

    # timing deltas between key stages
    lines.append("")
    lines.append("## Timing (between event timestamps)")
    for i in range(1, len(run_events)):
        s1 = run_events[i-1]["stage"]
        s2 = run_events[i]["stage"]
        t1 = run_events[i-1].get("ts_utc")
        t2 = run_events[i].get("ts_utc")
        if t1 and t2:
            lines.append(f"- {s1} → {s2}: **{seconds_between(t1, t2):.2f}s**")

    # compile signatures
    lines.append("")
    lines.append("## Compilation Diagnostics (normalized signatures)")
    for e in run_events:
        if e.get("stage") in ("compile", "compile_after_autofix"):
            sigs = (e.get("metrics") or {}).get("error_signatures") or []
            lines.append(f"### {e.get('stage')} (compile_ok={e.get('compile_ok')}, returncode={e.get('returncode')})")
            if not sigs:
                lines.append("- No error signatures (clean compile).")
            else:
                for i, s in enumerate(sigs, 1):
                    lines.append(f"{i}. {s}")
            lines.append("")

    return "\n".join(lines).strip() + "\n"

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="Path to agent_metrics.jsonl")
    ap.add_argument("--outdir", default="metrics_reports", help="Output folder for reports")
    ap.add_argument("--request_id", default=None, help="Generate only one request_id report")
    args = ap.parse_args()

    events = load_events(args.jsonl)
    runs = group_by_request(events)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.request_id:
        run_events = runs.get(args.request_id, [])
        md = render_single_run_md(run_events)
        (outdir / f"report_{args.request_id}.md").write_text(md, encoding="utf-8")
        print("Wrote:", outdir / f"report_{args.request_id}.md")
        return

    for rid, run_events in runs.items():
        md = render_single_run_md(run_events)
        (outdir / f"report_{rid}.md").write_text(md, encoding="utf-8")

    print(f"Wrote {len(runs)} report(s) to: {outdir}")

if __name__ == "__main__":
    main()
