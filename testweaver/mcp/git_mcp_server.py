# git_mcp_server.py
import os
import re
import time
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Literal, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Git MCP Server")

# Read repo root from env
REPO_ROOT = Path(os.getenv("GIT_LOCAL_REPO", "D:/Sundar/MTech/Dissertation/svc-accounting")).resolve()
print("📁 MCP Git Server using repo root:", REPO_ROOT)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Existing models ----------
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

# ---------- New compile models ----------
BuildTool = Literal["maven", "gradle"]
BuildGoal = Literal["test-compile", "test", "compile"]  # keep minimal

class CompileRequest(BaseModel):
    repo: str
    tool: BuildTool = Field(..., description="maven|gradle")
    goal: BuildGoal = Field("test-compile", description="test-compile|test|compile")
    project_path: str = Field(".", description="Relative path inside repo root")
    timeout_seconds: int = Field(300, ge=10, le=1800)
    extra_args: List[str] = Field(default_factory=list, description="Optional safe args (whitelisted)")

class CompileResponse(BaseModel):
    ok: bool
    returncode: int
    command: List[str]
    cwd: str
    duration_ms: int
    stdout: str
    stderr: str

# ---------- Helpers ----------
def _safe_join(root: Path, rel: str) -> Path:
    # Prevent path traversal
    target = (root / rel).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail=f"Invalid project_path (path traversal): {rel}")
    return target

def _pick_gradle_executable(cwd: Path) -> List[str]:
    # Prefer wrapper if present
    if (cwd / "gradlew").exists():
        return ["./gradlew"]
    if (cwd / "gradlew.bat").exists():
        return ["gradlew.bat"]
    # Fall back to system gradle (return absolute path if found)
    gradle_path = shutil.which("gradle")
    if gradle_path:
        return [gradle_path]
    raise HTTPException(status_code=500, detail="Gradle not found (no gradlew/gradlew.bat and no system gradle).")

def _pick_maven_executable() -> List[str]:
    # ✅ Prefer MAVEN_CMD if provided (Windows-safe)
    maven_cmd = os.getenv("MAVEN_CMD")
    if maven_cmd:
        p = Path(maven_cmd)
        if p.exists():
            return [str(p)]
        raise HTTPException(status_code=500, detail=f"MAVEN_CMD is set but file not found: {maven_cmd}")

    # fallback: system PATH (use absolute path when possible)
    mvn_path = shutil.which("mvn")
    if mvn_path:
        return [mvn_path]

    raise HTTPException(status_code=500, detail="Maven not found in PATH. Set MAVEN_CMD or add mvn to PATH.")

# Keep extra args safe. You can expand this later.
SAFE_EXTRA_ARGS_PREFIXES = (
    "-D",          # system properties (e.g., -DskipTests=true)
    "--no-daemon", # gradle
    "--stacktrace",
    "--info",
    "--debug",
)

def _validate_extra_args(args: List[str]) -> List[str]:
    for a in args:
        # Allow standard safe prefixes (system properties, logging flags, etc.)
        if a.startswith(SAFE_EXTRA_ARGS_PREFIXES):
            continue
        # Allow Maven/Gradle plugin:goal tokens like 'jacoco:report' or 'org.jacoco:jacoco-maven-plugin:report'
        if re.match(r"^[\w\.-]+(:[\w\.-]+)+$", a):
            continue
        raise HTTPException(
            status_code=400,
            detail=f"Unsafe extra arg rejected: {a}. Allowed prefixes: {SAFE_EXTRA_ARGS_PREFIXES} or plugin:goal tokens"
        )
    return args

def _build_command(tool: BuildTool, goal: BuildGoal, cwd: Path, extra_args: List[str]) -> List[str]:
    extra_args = _validate_extra_args(extra_args)

    if tool == "maven":
        # Use mvnw if present in project folder
        if (cwd / "mvnw").exists():
            base = ["./mvnw"]
        elif (cwd / "mvnw.cmd").exists():
            base = ["mvnw.cmd"]
        else:
            base = _pick_maven_executable()
        common = ["-e"]
        if goal == "test-compile":
            return base + common + ["test-compile"] + extra_args
        if goal == "test":
            return base + common + ["test"] + extra_args
        if goal == "compile":
            return base + common + ["compile"] + extra_args

    if tool == "gradle":
        base = _pick_gradle_executable(cwd)
        if goal == "test-compile":
            # Closest equivalent is testClasses
            return base + ["testClasses"] + extra_args
        if goal == "test":
            return base + ["test"] + extra_args
        if goal == "compile":
            # Java compile
            return base + ["classes"] + extra_args

    raise HTTPException(status_code=400, detail=f"Unsupported tool/goal: {tool}/{goal}")

def _run_command(cmd: List[str], cwd: Path, timeout_seconds: int):
    start = time.time()
    try:
        p = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            shell=False
        )
        duration_ms = int((time.time() - start) * 1000)

        stdout, stderr, summary = _normalize_maven_output(p.stdout, p.stderr)

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

        stdout, stderr, summary = _normalize_maven_output(e.stdout or "", e.stderr or "TIMEOUT")

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

def _tail(text: str, max_chars: int = 8000) -> str:
    if not text:
        return ""
    return text[-max_chars:]

def _normalize_maven_output(stdout: str, stderr: str) -> tuple[str, str, str]:
    """
    Maven often prints errors to stdout (especially with -q or certain plugins).
    We:
    - keep stdout/stderr
    - create a single combined error_summary (prefer stderr else stdout)
    """
    stdout = stdout or ""
    stderr = stderr or ""
    combined = stderr.strip() if stderr.strip() else stdout.strip()
    # tail only, so payload stays small
    summary = "\n".join(combined.splitlines()[-200:]) if combined else ""
    return _tail(stdout, 20000), _tail(stderr, 20000), _tail(summary, 8000)

# ---------- Existing endpoints ----------
@app.post("/git-mcp/file")
def get_file(req: FileRequest):
    file_path = REPO_ROOT / req.path
    if not file_path.exists():
        return {"error": f"File not found: {file_path}"}
    return {"content": file_path.read_text(encoding="utf-8")}

@app.post("/git-mcp/list")
def list_files(req: ListRequest):
    base = REPO_ROOT / req.base_path
    if not base.exists():
        return {"files": []}
    files = [str(p.relative_to(REPO_ROOT)) for p in base.rglob(f"*{req.ext}")]
    return {"files": files}

@app.post("/git-mcp/pr-diff")
def get_pr_diff(req: PRDiffRequest):
    return {"diff": "// TODO: implement PR diff support"}

# ---------- NEW endpoint ----------

@app.post("/git-mcp/compile")
def compile_project(req: CompileRequest):
    try:
        cwd = _safe_join(REPO_ROOT, req.project_path)

        print("🔧 Compile request")
        print("  tool:", req.tool)
        print("  goal:", req.goal)
        print("  cwd :", cwd)
        # Log extra args early to surface jacoco requests
        try:
            print("  extra_args:", req.extra_args)
        except Exception:
            pass

        if not cwd.exists():
            raise HTTPException(status_code=404, detail=f"Project path not found: {cwd}")

        if req.tool == "maven" and not (cwd / "pom.xml").exists():
            raise HTTPException(
                status_code=400,
                detail=f"pom.xml not found in {cwd}"
            )

        cmd = _build_command(req.tool, req.goal, cwd, req.extra_args)
        print("  cmd :", cmd)

        result = _run_command(cmd, cwd, req.timeout_seconds)
        print("  ✅ ok:", result.get("ok"), "returncode:", result.get("returncode"), "duration_ms:", result.get("duration_ms"))
        if not result.get("ok"):
            print("  ❌ error_summary tail:\n", result.get("error_summary", "")[-1000:])
        return result

    except HTTPException:
        raise

    except FileNotFoundError as e:
        print("❌ Build tool not found:", e)
        raise HTTPException(
            status_code=500,
            detail=f"Build tool not found: {e}"
        )

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print("❌ MCP compile crashed:\n", tb)
        raise HTTPException(
            status_code=500,
            detail=f"{type(e).__name__}: {e}"
        )



class WriteFileRequest(BaseModel):
    repo: str
    path: str
    content: str
    overwrite: bool = True


@app.post("/git-mcp/write-file")
def write_file(req: WriteFileRequest):
    target = (REPO_ROOT / req.path).resolve()

    # Security: prevent path traversal
    if REPO_ROOT not in target.parents and target != REPO_ROOT:
        raise HTTPException(status_code=400, detail="Invalid path (path traversal)")

    if target.exists() and not req.overwrite:
        raise HTTPException(status_code=409, detail=f"File already exists: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")

    return {
        "ok": True,
        "path": str(target.relative_to(REPO_ROOT))
    }