"""Start the same resumable CLI from the local dashboard, without a console window."""
import os
import subprocess
import sys
from pathlib import Path

from .pipeline import exclusive_lock


def is_running(artifacts):
    try:
        with exclusive_lock(Path(artifacts) / "pipeline.lock"):
            return False
    except RuntimeError:
        return True


def start_pipeline(root, source, artifacts, output, workers=6):
    root, artifacts = Path(root).resolve(), Path(artifacts).resolve()
    if is_running(artifacts):
        raise RuntimeError("The pipeline is already running.")
    artifacts.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-m", "reddit_insights.pipeline", "run", "--source", str(source),
           "--artifacts", str(artifacts), "--output", str(output), "--workers", str(workers)]
    with (artifacts / "pipeline.log").open("a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return proc.pid


def log_tail(artifacts):
    path = Path(artifacts) / "pipeline.log"
    if not path.exists():
        return "No dashboard-launched pipeline log yet. CLI output appears in its terminal."
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 6000))
        return f.read().decode("utf-8", errors="replace")
