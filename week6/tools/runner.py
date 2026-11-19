"""Starting the gateway and running ``run.sh`` once, shared by practice and grading."""

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
MAX_LOG_BYTES = 5_000_000


def start_gateway(port: int, rate_card, log_path, max_cost: float = 0.0, secrets=None, env=None, allow_login=False):
    """Start the gateway and wait until it answers. Returns the process."""
    command = [
        sys.executable, str(TOOLS / "gateway.py"),
        "--port", str(port), "--rate-card", str(rate_card),
        "--log", str(log_path), "--max-cost", str(max_cost),
    ]  # fmt: skip
    if secrets:
        command += ["--secrets", str(secrets)]
    if allow_login:
        command.append("--allow-login")
    errors_path = Path(f"{log_path}.gateway-errors")
    with open(errors_path, "wb") as errors:
        process = subprocess.Popen(command, env=env, stdout=subprocess.DEVNULL, stderr=errors)
    deadline = time.time() + 15
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"gateway exited: {errors_path.read_text(errors='replace')}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1).read()
            return process
        except OSError:
            time.sleep(0.1)
    process.kill()
    raise RuntimeError(f"gateway did not start on port {port}")


def stop_gateway(process) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _drain(stream, log_path: Path) -> None:
    """Copy output to the log, up to a cap, and keep reading so the child never blocks."""
    written = 0
    with open(log_path, "wb") as log:
        for chunk in iter(lambda: stream.read(65536), b""):
            if written < MAX_LOG_BYTES:
                log.write(chunk[: MAX_LOG_BYTES - written])
                written += len(chunk)


def run_entry(workspace: Path, task_name: str, env: dict, timeout: float, log_path: Path, user=None) -> dict:
    """Run ``./run.sh <task>`` in the workspace with a hard time limit.

    With ``user`` (grading, as root) the script runs as that unprivileged
    account with only ``env``. Without it (practice) it runs as you.
    """
    entry = workspace / "run.sh"
    if not entry.is_file():
        return {"exit_code": None, "timed_out": False, "seconds": 0.0, "error": "no run.sh"}
    entry.chmod(entry.stat().st_mode | 0o111)
    command = ["./run.sh", task_name]
    if user:
        assignments = [f"{name}={value}" for name, value in env.items()]
        command = ["runuser", "-u", user, "--", "env", "-i", *assignments, *command]
    started = time.time()
    try:
        process = subprocess.Popen(
            command, cwd=workspace, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True,
        )  # fmt: skip
    except OSError as exc:
        return {"exit_code": None, "timed_out": False, "seconds": 0.0, "error": f"could not start run.sh: {exc}"}
    pump = threading.Thread(target=_drain, args=(process.stdout, log_path), daemon=True)
    pump.start()
    timed_out = False
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
    # Stop anything the script left behind, finished or not.
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            continue
    if user:
        subprocess.run(["pkill", "-KILL", "-u", user], check=False)
    process.wait()
    pump.join(timeout=5)
    return {
        "exit_code": process.returncode,
        "timed_out": timed_out,
        "seconds": round(time.time() - started, 1),
        "error": None,
    }


def tail(path: Path, lines: int = 40, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_bytes()[-limit:].decode(errors="replace")
    return "\n".join(text.splitlines()[-lines:])
