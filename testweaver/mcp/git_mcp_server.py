# testweaver/mcp/git_mcp_server.py
"""
Git MCP Server (FastAPI)

Goals:
- Serve file read/list/write
- Run Maven/Gradle compile/test
- Create branch, add resolved paths (files/dirs/globs), commit, push
- open-pr = push-branch + GitHub PR creation (GitHub.com OR GitHub Enterprise)

Key fixes included:
✅ All endpoints are present under /git-mcp/*
✅ One single, consistent path resolver used by push-branch/open-pr
✅ Auto-abort pending merges (unmerged index) so checkout works
✅ Handles "nothing to commit" and non-fast-forward pushes
✅ Prevents NameError / UnboundLocalError by defining functions before use
✅ Adds "force_push_if_behind" option to avoid non-fast-forward failures
✅ CRITICAL FIX: Bulletproof backup/restore of generated files (no reliance on stash)
✅ CRITICAL: If remote branch already exists, base local branch on origin/<branch> to avoid wiping branch history
✅ GitHub PR creation for BOTH github.com and GitHub Enterprise (auto-detects API base; override with GITHUB_API_BASE)
✅ If PR creation fails, returns a clear error (raises 500) so it doesn’t silently “OK” without PR

ENV:
- GIT_LOCAL_REPO: repo root path (default points to svc-accounting)
- GIT_CMD: git executable (default: "git")
- MAVEN_CMD: mvn executable path (optional)
- GITHUB_TOKEN (preferred) or GIT_TOKEN (fallback)
- GITHUB_API_BASE (optional override for enterprise)
"""

import os
import re
import time
import shutil
import glob
import traceback
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, List, Tuple, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()  # loads .env from current working directory


# --------------------------------------------------------------------------------------
# App + config
# --------------------------------------------------------------------------------------

app = FastAPI(title="Git MCP Server")

# Repo root is a physical path on this machine (svc-accounting typically)
REPO_ROOT = Path(os.getenv("GIT_LOCAL_REPO", "D:/Sundar/MTech/Dissertation/svc-accounting")).resolve()
print("📁 MCP Git Server using repo root:", REPO_ROOT)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------

class FileRequest(BaseModel):
    repo: str
    path: str


class ListRequest(BaseModel):
    repo: str
    base_path: str
    ext: str


class PRDiffRequest(BaseModel):
    repo: str
    pr_number: int


BuildTool = Literal["maven", "gradle"]
BuildGoal = Literal["test-compile", "test", "compile"]


class CompileRequest(BaseModel):
    repo: str
    tool: BuildTool = Field(..., description="maven|gradle")
    goal: BuildGoal = Field("test-compile", description="test-compile|test|compile")
    project_path: str = Field(".", description="Relative path inside repo root")
    timeout_seconds: int = Field(300, ge=10, le=1800)
    extra_args: List[str] = Field(default_factory=list, description="Optional safe args (whitelisted)")


class OpenPRRequest(BaseModel):
    repo: str
    branch: str
    base_branch: str = "main"
    title: str = "testweaver: generated tests"
    body: str = ""
    files: List[str] = Field(default_factory=list)  # files/dirs/globs to add/commit
    commit_message: str = "testweaver: add generated tests"
    remote: str = "origin"
    force_push_if_behind: bool = True
    draft: bool = False  # optional


class PushBranchRequest(BaseModel):
    repo: str
    branch: str
    base_branch: str = "main"
    commit_message: str = "testweaver: add generated tests"
    files: List[str] = Field(default_factory=list)  # files/dirs/globs to add/commit
    remote: str = "origin"
    force_push_if_behind: bool = True


class WriteFileRequest(BaseModel):
    repo: str
    path: str
    content: str
    overwrite: bool = True


# --------------------------------------------------------------------------------------
# Error helper
# --------------------------------------------------------------------------------------

def _err500(where: str, e: Exception):
    tb = traceback.format_exc()
    print(f"❌ {where} crashed:\n{tb}")
    raise HTTPException(status_code=500, detail=f"{where}: {type(e).__name__}: {e}\n{tb[-2000:]}")


# --------------------------------------------------------------------------------------
# Git helpers
# --------------------------------------------------------------------------------------

def _run_git(args: List[str], cwd: Path, timeout: int = 120) -> Dict[str, Any]:
    git_cmd = os.getenv("GIT_CMD") or "git"
    cmd = [git_cmd] + args
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout, shell=False)
    return {
        "ok": p.returncode == 0,
        "returncode": p.returncode,
        "cmd": cmd,
        "stdout": p.stdout or "",
        "stderr": p.stderr or "",
    }


def _run_git_steps(cwd: Path, steps_plan: List[Tuple[str, List[str]]], timeout: int = 240) -> Dict[str, Any]:
    steps = []
    for name, args in steps_plan:
        r = _run_git(args, cwd, timeout=timeout if name in ("push", "fetch") else 60)
        steps.append({"step": name, **r})

        print(f"🧩 step={name} ok={r['ok']} rc={r['returncode']}")
        if not r["ok"]:
            print("CMD :", " ".join(r["cmd"]))
            print("STDOUT:\n", (r["stdout"] or "")[-1500:])
            print("STDERR:\n", (r["stderr"] or "")[-1500:])
            return {"ok": False, "failed_step": name, "steps": steps}
    return {"ok": True, "steps": steps}


def _git_current_branch(cwd: Path) -> str:
    r = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd)
    return (r.get("stdout") or "").strip() if r.get("ok") else ""


def _git_branch_exists(cwd: Path, branch: str) -> bool:
    r = _run_git(["rev-parse", "--verify", branch], cwd)
    return bool(r.get("ok"))


def _git_remote_branch_exists(cwd: Path, remote: str, branch: str) -> bool:
    r = _run_git(["show-ref", "--verify", f"refs/remotes/{remote}/{branch}"], cwd)
    return bool(r.get("ok"))


def _detect_default_branch(cwd: Path, remote: str = "origin") -> str:
    r = _run_git(["symbolic-ref", f"refs/remotes/{remote}/HEAD"], cwd)
    if r.get("ok"):
        ref = (r.get("stdout") or "").strip()
        if ref.startswith(f"refs/remotes/{remote}/"):
            return ref.split(f"refs/remotes/{remote}/", 1)[1].strip()
    return "main"


def _git_has_unmerged(cwd: Path) -> bool:
    r = _run_git(["status", "--porcelain"], cwd)
    if not r["ok"]:
        return False
    return any(line[:2] in {"UU", "AA", "DD", "AU", "UA", "DU", "UD"} for line in (r["stdout"] or "").splitlines())


def _git_abort_merge_if_any(cwd: Path):
    """
    Auto-abort pending merges so checkout works.
    """
    if _git_has_unmerged(cwd):
        print("⚠️ Repo has unmerged paths; aborting merge to proceed")
        _run_git(["merge", "--abort"], cwd, timeout=60)
        _run_git(["reset", "--hard"], cwd, timeout=60)


def _normalize_files(files: List[str]) -> List[str]:
    out = []
    for f in files or []:
        f2 = (f or "").strip()
        if not f2:
            continue
        f2 = f2.replace("\\", "/").lstrip("/")
        if ".." in f2.split("/"):
            raise HTTPException(status_code=400, detail=f"Invalid file path (path traversal): {f}")
        out.append(f2)
    return out


def _resolve_paths_for_git_add(repo_root: Path, paths: List[str]) -> List[str]:
    """
    Returns a list of repo-relative paths safe to pass to `git add`.

    Accepts:
      - exact file path
      - directory path  -> keep directory (git add dir is fine)
      - glob patterns (** supported) -> expands to matching files/dirs
      - bare filename -> searches repo (rglob)
    """
    out: List[str] = []
    for raw in (paths or []):
        p = (raw or "").strip().replace("\\", "/").lstrip("/")
        if not p:
            continue

        abs_p = (repo_root / p).resolve()

        if abs_p.exists():
            out.append(str(abs_p.relative_to(repo_root)).replace("\\", "/"))
            continue

        globbed = glob.glob(str(repo_root / p), recursive=True)
        for g in globbed:
            gp = Path(g)
            if gp.exists():
                out.append(str(gp.relative_to(repo_root)).replace("\\", "/"))

        name = Path(p).name
        if name and "." in name:
            for h in repo_root.rglob(name):
                if h.exists():
                    out.append(str(h.relative_to(repo_root)).replace("\\", "/"))

    # de-dupe preserve order
    seen = set()
    final = []
    for x in out:
        if x not in seen:
            seen.add(x)
            final.append(x)
    return final


def _is_nothing_to_commit(text: str) -> bool:
    t = (text or "").lower()
    return ("nothing to commit" in t) or ("working tree clean" in t)


def _detect_remote_url(cwd: Path, remote: str) -> str:
    r = _run_git(["remote", "get-url", remote], cwd, timeout=30)
    return (r.get("stdout") or "").strip() if r.get("ok") else ""


# --------------------------------------------------------------------------------------
# CRITICAL FIX: Bulletproof backup/restore (prevents generated files from disappearing)
# --------------------------------------------------------------------------------------

def _expand_to_existing_paths(repo_root: Path, inputs: List[str]) -> List[Path]:
    """
    Expand file/dir/glob inputs into absolute existing Paths inside repo.
    """
    out: List[Path] = []
    for raw in inputs or []:
        p = (raw or "").strip().replace("\\", "/").lstrip("/")
        if not p:
            continue

        # exact path
        abs_p = (repo_root / p).resolve()
        if abs_p.exists() and (abs_p == repo_root or repo_root in abs_p.parents):
            out.append(abs_p)
            continue

        # glob
        for g in glob.glob(str(repo_root / p), recursive=True):
            gp = Path(g).resolve()
            if gp.exists() and (gp == repo_root or repo_root in gp.parents):
                out.append(gp)

    # de-dupe preserve order
    seen = set()
    final: List[Path] = []
    for x in out:
        if x not in seen:
            seen.add(x)
            final.append(x)
    return final


def _backup_selected_paths(repo_root: Path, files: List[str]) -> str:
    """
    Copies the requested files/dirs/globs to a temp folder outside the repo.
    Returns the temp folder path ("" if nothing backed up).
    """
    targets = _expand_to_existing_paths(repo_root, files)
    if not targets:
        return ""

    tmpdir = tempfile.mkdtemp(prefix="testweaver-backup-")
    tmp_root = Path(tmpdir).resolve()

    for t in targets:
        rel = t.relative_to(repo_root)
        dst = (tmp_root / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)

        if t.is_dir():
            if dst.exists():
                shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(t, dst)
        else:
            shutil.copy2(t, dst)

    print(f"🧰 backup created: {tmp_root}")
    return str(tmp_root)


def _restore_selected_paths(repo_root: Path, backup_root: str):
    """
    Restores everything from backup_root back into repo_root (overwriting).
    """
    if not backup_root:
        return
    tmp_root = Path(backup_root).resolve()
    if not tmp_root.exists():
        return

    for src in tmp_root.rglob("*"):
        if src.is_dir():
            continue
        rel = src.relative_to(tmp_root)
        dst = (repo_root / rel).resolve()
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    shutil.rmtree(tmp_root, ignore_errors=True)
    print("🧰 backup restored and cleaned")


# --------------------------------------------------------------------------------------
# GitHub helpers (GitHub.com + GitHub Enterprise)
# --------------------------------------------------------------------------------------

def _parse_remote_host(remote_url: str) -> str:
    u = (remote_url or "").strip()
    if not u:
        return ""
    if u.startswith("git@") and ":" in u:
        return u.split("@", 1)[1].split(":", 1)[0].strip()
    if "://" in u:
        return u.split("://", 1)[1].split("/", 1)[0].strip()
    return ""


def _github_parse_owner_repo(remote_url: str) -> Tuple[str, str]:
    """
    Supports BOTH GitHub.com and GitHub Enterprise.

    Supports:
      https://host/owner/repo.git
      git@host:owner/repo.git
    """
    u = (remote_url or "").strip()
    if not u:
        raise ValueError("Empty remote_url")

    # Convert SSH to HTTPS-like for parsing
    if u.startswith("git@") and ":" in u:
        host = u.split("@", 1)[1].split(":", 1)[0]
        tail = u.split(":", 1)[1]
        u = f"https://{host}/{tail}"

    if "://" not in u:
        raise ValueError(f"Cannot parse URL: {remote_url}")

    host_and_path = u.split("://", 1)[1]
    if "/" not in host_and_path:
        raise ValueError(f"Cannot parse owner/repo from: {remote_url}")

    _, path = host_and_path.split("/", 1)
    path = path[:-4] if path.endswith(".git") else path
    parts = [p for p in path.split("/") if p]
    if len(parts) < 2:
        raise ValueError(f"Cannot parse owner/repo from: {remote_url}")
    return parts[0], parts[1]


def _github_api_base(remote_url: str) -> str:
    """
    GitHub.com -> https://api.github.com
    GitHub Enterprise -> https://<host>/api/v3

    Override with env:
      - GITHUB_API_BASE
    """
    override = (os.getenv("GITHUB_API_BASE") or "").strip().rstrip("/")
    if override:
        return override

    host = _parse_remote_host(remote_url).lower()
    if host in {"github.com", "www.github.com"}:
        return "https://api.github.com"
    if host:
        return f"https://{host}/api/v3"
    return "https://api.github.com"


# --------------------------------------------------------------------------------------
# Safe join (compile endpoint)
# --------------------------------------------------------------------------------------

def _safe_join(root: Path, rel: str) -> Path:
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail=f"Invalid project_path (path traversal): {rel}")
    return target


# --------------------------------------------------------------------------------------
# Build helpers (compile endpoint)
# --------------------------------------------------------------------------------------

SAFE_EXTRA_ARGS_PREFIXES = ("-D", "--no-daemon", "--stacktrace", "--info", "--debug")

def _validate_extra_args(args: List[str]) -> List[str]:
    for a in args:
        if a.startswith(SAFE_EXTRA_ARGS_PREFIXES):
            continue
        # allow plugin:goal like jacoco:report
        if re.match(r"^[\w\.-]+(:[\w\.-]+)+$", a):
            continue
        raise HTTPException(
            status_code=400,
            detail=f"Unsafe extra arg rejected: {a}. Allowed prefixes: {SAFE_EXTRA_ARGS_PREFIXES} or plugin:goal tokens",
        )
    return args


def _pick_gradle_executable(cwd: Path) -> List[str]:
    if (cwd / "gradlew").exists():
        return ["./gradlew"]
    if (cwd / "gradlew.bat").exists():
        return ["gradlew.bat"]
    gradle_path = shutil.which("gradle")
    if gradle_path:
        return [gradle_path]
    raise HTTPException(status_code=500, detail="Gradle not found (no gradlew/gradlew.bat and no system gradle).")


def _pick_maven_executable() -> List[str]:
    maven_cmd = os.getenv("MAVEN_CMD")
    if maven_cmd:
        p = Path(maven_cmd)
        if p.exists():
            return [str(p)]
        raise HTTPException(status_code=500, detail=f"MAVEN_CMD is set but file not found: {maven_cmd}")

    mvn_path = shutil.which("mvn")
    if mvn_path:
        return [mvn_path]

    raise HTTPException(status_code=500, detail="Maven not found in PATH. Set MAVEN_CMD or add mvn to PATH.")


def _build_command(tool: BuildTool, goal: BuildGoal, cwd: Path, extra_args: List[str]) -> List[str]:
    extra_args = _validate_extra_args(extra_args)

    if tool == "maven":
        if (cwd / "mvnw").exists():
            base = ["./mvnw"]
        elif (cwd / "mvnw.cmd").exists():
            base = ["mvnw.cmd"]
        else:
            base = _pick_maven_executable()
        common = ["-e"]
        return base + common + [goal] + extra_args

    if tool == "gradle":
        base = _pick_gradle_executable(cwd)
        mapping = {"test-compile": "testClasses", "test": "test", "compile": "classes"}
        return base + [mapping[goal]] + extra_args

    raise HTTPException(status_code=400, detail=f"Unsupported tool/goal: {tool}/{goal}")


def _tail(text: str, max_chars: int = 8000) -> str:
    return text[-max_chars:] if text else ""


def _normalize_build_output(stdout: str, stderr: str) -> Tuple[str, str, str]:
    stdout = stdout or ""
    stderr = stderr or ""
    combined = stderr.strip() if stderr.strip() else stdout.strip()
    summary = "\n".join(combined.splitlines()[-200:]) if combined else ""
    return _tail(stdout, 20000), _tail(stderr, 20000), _tail(summary, 8000)


def _run_command(cmd: List[str], cwd: Path, timeout_seconds: int) -> Dict[str, Any]:
    start = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_seconds, shell=False)
        duration_ms = int((time.time() - start) * 1000)
        stdout, stderr, summary = _normalize_build_output(p.stdout, p.stderr)
        return {
            "ok": (p.returncode == 0),
            "returncode": p.returncode,
            "command": cmd,
            "cwd": str(cwd),
            "duration_ms": duration_ms,
            "stdout": stdout,
            "stderr": stderr,
            "error_summary": "" if p.returncode == 0 else summary,
        }
    except subprocess.TimeoutExpired as e:
        duration_ms = int((time.time() - start) * 1000)
        stdout, stderr, summary = _normalize_build_output(e.stdout or "", e.stderr or "TIMEOUT")
        return {
            "ok": False,
            "returncode": 124,
            "command": cmd,
            "cwd": str(cwd),
            "duration_ms": duration_ms,
            "stdout": stdout,
            "stderr": stderr,
            "error_summary": summary or "TIMEOUT",
        }


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------

@app.get("/git-mcp/health")
def health():
    try:
        r = _run_git(["--version"], REPO_ROOT, timeout=20)
        return {"ok": True, "repo_root": str(REPO_ROOT), "git": r}
    except Exception as e:
        _err500("health", e)


@app.post("/git-mcp/file")
def get_file(req: FileRequest):
    try:
        file_path = (REPO_ROOT / req.path).resolve()
        if REPO_ROOT not in file_path.parents and file_path != REPO_ROOT:
            raise HTTPException(status_code=400, detail="Invalid path (path traversal)")
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
        return {"content": file_path.read_text(encoding="utf-8", errors="ignore")}
    except HTTPException:
        raise
    except Exception as e:
        _err500("file", e)


@app.post("/git-mcp/list")
def list_files(req: ListRequest):
    try:
        base = (REPO_ROOT / req.base_path).resolve()
        if REPO_ROOT not in base.parents and base != REPO_ROOT:
            raise HTTPException(status_code=400, detail="Invalid base_path (path traversal)")
        if not base.exists():
            return {"files": []}
        files = [str(p.relative_to(REPO_ROOT)).replace("\\", "/") for p in base.rglob(f"*{req.ext}")]
        return {"files": files}
    except HTTPException:
        raise
    except Exception as e:
        _err500("list", e)


@app.post("/git-mcp/pr-diff")
def get_pr_diff(req: PRDiffRequest):
    return {"diff": "// TODO: implement PR diff support"}


@app.post("/git-mcp/write-file")
def write_file(req: WriteFileRequest):
    """
    Creates parent folders automatically.
    """
    try:
        target = (REPO_ROOT / req.path).resolve()
        if REPO_ROOT not in target.parents and target != REPO_ROOT:
            raise HTTPException(status_code=400, detail="Invalid path (path traversal)")

        if target.exists() and not req.overwrite:
            raise HTTPException(status_code=409, detail=f"File already exists: {target}")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(req.content, encoding="utf-8")

        return {"ok": True, "path": str(target.relative_to(REPO_ROOT)).replace("\\", "/")}
    except HTTPException:
        raise
    except Exception as e:
        _err500("write-file", e)


@app.post("/git-mcp/compile")
def compile_project(req: CompileRequest):
    try:
        cwd = _safe_join(REPO_ROOT, req.project_path)

        print("🔧 Compile request")
        print("  tool:", req.tool)
        print("  goal:", req.goal)
        print("  cwd :", cwd)
        print("  extra_args:", req.extra_args)

        if not cwd.exists():
            raise HTTPException(status_code=404, detail=f"Project path not found: {cwd}")

        if req.tool == "maven" and not (cwd / "pom.xml").exists():
            raise HTTPException(status_code=400, detail=f"pom.xml not found in {cwd}")

        cmd = _build_command(req.tool, req.goal, cwd, req.extra_args)
        print("  cmd :", cmd)

        result = _run_command(cmd, cwd, req.timeout_seconds)
        print("  ✅ ok:", result.get("ok"), "returncode:", result.get("returncode"), "duration_ms:", result.get("duration_ms"))
        if not result.get("ok"):
            print("  ❌ error_summary tail:\n", (result.get("error_summary") or "")[-1000:])
        return result

    except HTTPException:
        raise
    except Exception as e:
        _err500("compile", e)


@app.post("/git-mcp/push-branch")
def push_branch(req: PushBranchRequest):
    """
    IMPORTANT:
    - Bulletproof backup of req.files BEFORE any checkout/reset (prevents file deletion).
    - If the remote branch already exists, base local branch on origin/<branch>
      to avoid wiping previously pushed commits.
    """
    try:
        cwd = REPO_ROOT
        if not cwd.exists():
            raise HTTPException(status_code=404, detail=f"Repo root not found: {cwd}")

        _git_abort_merge_if_any(cwd)

        # Resolve base branch
        base_branch = req.base_branch
        if not _git_branch_exists(cwd, base_branch) and (_git_remote_branch_exists(cwd, req.remote, base_branch) is False):
            detected = _detect_default_branch(cwd, req.remote)
            print(f"ℹ️ base_branch '{base_branch}' not found; using detected default '{detected}'")
            base_branch = detected

        combined_steps: List[Dict[str, Any]] = []

        # ✅ CRITICAL: backup the files/dirs/globs outside the repo BEFORE any checkout
        backup_root = _backup_selected_paths(REPO_ROOT, req.files)

        # Always fetch first so remote refs are up-to-date
        r1 = _run_git_steps(
            cwd,
            [
                ("remote_v", ["remote", "-v"]),
                ("fetch", ["fetch", req.remote]),
            ],
        )
        combined_steps.extend(r1.get("steps", []))
        if not r1.get("ok"):
            # best-effort cleanup of backup (don’t restore into unknown state)
            if backup_root:
                shutil.rmtree(Path(backup_root), ignore_errors=True)
            raise HTTPException(status_code=500, detail={"msg": "push-branch failed", **r1})

        # CRITICAL FIX: if remote branch exists, continue from origin/<branch>
        if _git_remote_branch_exists(cwd, req.remote, req.branch):
            rchk = _run_git_steps(
                cwd,
                [
                    ("checkout_branch_from_remote", ["checkout", "-B", req.branch, f"{req.remote}/{req.branch}"]),
                ],
            )
        else:
            rchk = _run_git_steps(
                cwd,
                [
                    ("checkout_base", ["checkout", base_branch]),
                    ("checkout_branch", ["checkout", "-B", req.branch]),
                ],
            )

        combined_steps.extend(rchk.get("steps", []))
        if not rchk.get("ok"):
            if backup_root:
                shutil.rmtree(Path(backup_root), ignore_errors=True)
            raise HTTPException(status_code=500, detail={"msg": "push-branch failed", **rchk})

        # ✅ Restore generated files AFTER checkout so they cannot be lost
        _restore_selected_paths(REPO_ROOT, backup_root)

        # Resolve + add AFTER restore (so files exist)
        files_in = _normalize_files(req.files)
        files = _resolve_paths_for_git_add(REPO_ROOT, files_in)

        if not files:
            raise HTTPException(
                status_code=400,
                detail={
                    "msg": "No matching files found to add/commit",
                    "given_files": req.files,
                    "normalized": files_in,
                    "tips": [
                        "Ensure write-file created the file(s) first",
                        "Pass a directory like 'src/test/java' to add all tests",
                        "Or use a glob like 'src/test/java/**/TransactionServiceTest*.java'",
                    ],
                },
            )

        r2 = _run_git_steps(
            cwd,
            [
                ("add", ["add"] + files),
                ("status", ["status", "--porcelain"]),
                ("commit", ["commit", "-m", req.commit_message]),
            ],
        )
        combined_steps.extend(r2.get("steps", []))

        # commit failure: nothing to commit => still push
        if not r2["ok"] and r2.get("failed_step") == "commit":
            last = r2["steps"][-1]
            text = ((last.get("stdout") or "") + "\n" + (last.get("stderr") or "")).strip()
            if _is_nothing_to_commit(text):
                r2 = {"ok": True, "steps": r2["steps"]}

        if not r2.get("ok"):
            raise HTTPException(status_code=500, detail={"msg": "push-branch failed", **r2})

        # Push (may need force)
        push_res = _run_git(["push", "-u", req.remote, req.branch], cwd, timeout=240)
        combined_steps.append({"step": "push", **push_res})
        print(f"🧩 step=push ok={push_res['ok']} rc={push_res['returncode']}")

        if not push_res["ok"]:
            err_text = (push_res.get("stderr") or "").lower()
            if req.force_push_if_behind and ("non-fast-forward" in err_text or "rejected" in err_text):
                push2 = _run_git(["push", "--force-with-lease", "-u", req.remote, req.branch], cwd, timeout=240)
                combined_steps.append({"step": "push_force_with_lease", **push2})
                print(f"🧩 step=push_force_with_lease ok={push2['ok']} rc={push2['returncode']}")
                if not push2["ok"]:
                    raise HTTPException(status_code=500, detail={"msg": "push failed (even after force-with-lease)", "steps": combined_steps})
            else:
                raise HTTPException(status_code=500, detail={"msg": "push failed", "steps": combined_steps})

        return {
            "ok": True,
            "branch": req.branch,
            "base_branch": base_branch,
            "remote": req.remote,
            "current_branch": _git_current_branch(cwd),
            "files": files,
            "steps": combined_steps,
        }

    except HTTPException:
        raise
    except Exception as e:
        _err500("push-branch", e)


@app.post("/git-mcp/open-pr")
def open_pr(req: OpenPRRequest):
    """
    open-pr does:
    1) push-branch using req.files + commit_message
    2) create GitHub PR (GitHub.com OR GitHub Enterprise)

    ENV:
      - GITHUB_TOKEN (preferred) or GIT_TOKEN (fallback)
      - optional: GITHUB_API_BASE (override API base for enterprise)
    """
    try:
        cwd = REPO_ROOT
        if not cwd.exists():
            raise HTTPException(status_code=404, detail=f"Repo root not found: {cwd}")

        pushed = push_branch(
            PushBranchRequest(
                repo=req.repo,
                branch=req.branch,
                base_branch=req.base_branch,
                commit_message=req.commit_message,
                files=req.files,
                remote=req.remote,
                force_push_if_behind=req.force_push_if_behind,
            )
        )

        remote_url = _detect_remote_url(cwd, req.remote)
        token = (os.getenv("GITHUB_TOKEN") or os.getenv("GIT_TOKEN") or "").strip()

        github_pr = {"created": False, "url": "", "number": "", "error": "", "not_configured": False}

        if not remote_url:
            github_pr["error"] = "Could not read git remote URL (remote get-url failed)."
            return {
                "ok": True,
                "branch": pushed.get("branch", req.branch),
                "base_branch": pushed.get("base_branch", req.base_branch),
                "remote": pushed.get("remote", req.remote),
                "files": pushed.get("files", []),
                "github_pr": github_pr,
                "note": "Branch pushed but PR not created (remote_url missing).",
                "current_branch": _git_current_branch(cwd),
                "push_steps": pushed.get("steps", []),
            }

        if not token:
            github_pr["not_configured"] = True
            github_pr["error"] = "No token found. Set GITHUB_TOKEN (preferred) or GIT_TOKEN in the server environment."
            return {
                "ok": True,
                "branch": pushed.get("branch", req.branch),
                "base_branch": pushed.get("base_branch", req.base_branch),
                "remote": pushed.get("remote", req.remote),
                "files": pushed.get("files", []),
                "github_pr": github_pr,
                "note": "Branch pushed but PR not created (token missing).",
                "current_branch": _git_current_branch(cwd),
                "push_steps": pushed.get("steps", []),
            }

        owner, repo = _github_parse_owner_repo(remote_url)
        api_base = _github_api_base(remote_url)

        base = pushed.get("base_branch") or req.base_branch
        head = req.branch  # same-repo branch

        print("🐙 GitHub PR attempt")
        print("  remote_url:", remote_url)
        print("  api_base  :", api_base)
        print("  owner/repo:", f"{owner}/{repo}")
        print("  head->base:", f"{head} -> {base}")

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        api_create = f"{api_base}/repos/{owner}/{repo}/pulls"
        payload = {
            "title": req.title,
            "head": head,
            "base": base,
            "body": req.body or "",
            "draft": bool(req.draft),
        }

        with httpx.Client(timeout=30.0) as c:
            resp = c.post(api_create, headers=headers, json=payload)

            if resp.status_code == 422:
                # PR may already exist -> fetch open PR for this head
                api_list = f"{api_base}/repos/{owner}/{repo}/pulls"
                r2 = c.get(api_list, headers=headers, params={"state": "open", "head": f"{owner}:{head}"})
                if r2.status_code < 400:
                    arr = r2.json() or []
                    if arr:
                        pr0 = arr[0]
                        github_pr = {
                            "created": True,
                            "url": pr0.get("html_url") or "",
                            "number": pr0.get("number") or "",
                            "error": "",
                            "not_configured": False,
                        }
                    else:
                        github_pr["error"] = resp.text[:1200]
                else:
                    github_pr["error"] = (r2.text or "")[:1200]
            elif resp.status_code >= 400:
                github_pr["error"] = resp.text[:1200]
            else:
                data = resp.json()
                github_pr = {
                    "created": True,
                    "url": data.get("html_url") or "",
                    "number": data.get("number") or "",
                    "error": "",
                    "not_configured": False,
                }

        # Fail loudly if PR creation failed (so it doesn't silently "OK")
        if not github_pr.get("created"):
            raise HTTPException(
                status_code=500,
                detail={
                    "msg": "PR creation failed",
                    "github_pr": github_pr,
                    "remote_url": remote_url,
                    "api_base": api_base,
                    "owner": owner,
                    "repo": repo,
                    "head": head,
                    "base": base,
                },
            )

        return {
            "ok": True,
            "branch": pushed.get("branch", req.branch),
            "base_branch": pushed.get("base_branch", req.base_branch),
            "remote": pushed.get("remote", req.remote),
            "files": pushed.get("files", []),
            "github_pr": github_pr,
            "note": "Branch pushed and PR created.",
            "current_branch": _git_current_branch(cwd),
            "push_steps": pushed.get("steps", []),
        }

    except HTTPException:
        raise
    except Exception as e:
        _err500("open-pr", e)
