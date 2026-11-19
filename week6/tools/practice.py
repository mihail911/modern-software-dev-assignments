#!/usr/bin/env python3
"""Practice run: what the grader does, on your machine, with the starter incident.

    python tools/practice.py            # one run
    python tools/practice.py --runs 3   # three, with the average

Each run copies this directory to a temporary workspace, starts the metering
gateway, runs ``./run.sh task.md`` there, then scores the result and prices the
tokens from the rate card. Your own ``app/`` is never modified. Logs for each
run are kept under ``runs/``.

Scoring works as it does in grading. The agent never sees what decides the
result: ``tools/starter/`` is left out of the workspace. When ``run.sh`` exits,
the agent's ``app/services/api``, ``router`` and ``worker`` are copied into a
clean copy of ``app/`` and played through the scenarios in
``tools/starter/scenarios.json``; ``out/impact.json`` is compared with
``tools/starter/truth.json``. Results are printed by category.

    repair   share of incident scenarios passed x share of the others passed
    impact   0.6 x F1 on the affected set + 0.4 x share of the figures exactly right

Credentials. If ``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY`` is set, the gateway
holds it and your harness gets a placeholder, exactly as in grading. If not,
the gateway passes your harness's own login through and only meters it; that
is what you want for a Claude subscription. Codex works the same way with a
ChatGPT login (``codex login``): practice copies ``~/.codex/auth.json`` into the
run's own Codex home and meters what goes through. Graded runs never use a login.

Needs Python 3, Node 24 or newer, and Go.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
WEEK = TOOLS.parent
SCORER = TOOLS / "starter"
sys.path.insert(0, str(TOOLS))
import costs  # noqa: E402
import harness  # noqa: E402
import runner  # noqa: E402

# Left out of the workspace. tools/starter decides the result, so the agent must not see it.
SKIP = shutil.ignore_patterns(
    ".git", "runs", "out", "__pycache__", ".pytest_cache", ".venv", "node_modules", ".build", "state", "starter"
)
SCORED = ("api", "router", "worker")
CODEX_LOGIN = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex") / "auth.json"
REAL_KEYS = {"anthropic": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"), "openai": ("OPENAI_API_KEY",)}


def warn_if_dirty() -> None:
    try:
        status = subprocess.run(
            ["git", "status", "--porcelain", "--", "app", "incident", "task.md"],
            cwd=WEEK, capture_output=True, text=True, timeout=10,
        ).stdout.strip()  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return
    if status:
        print(
            "warning: app/, incident/ or task.md has uncommitted changes. The grader replaces\n"
            "         all three with its own copies, and practice scores inside a copy of\n"
            "         your app/, so keep them clean.\n"
        )


def stop_leftovers(workspace: Path) -> None:
    """Services an agent started and left running would hold ports and state."""
    subprocess.run(["pkill", "-KILL", "-f", str(workspace)], check=False, capture_output=True)


def score(workspace: Path, clean: Path, temp: Path, out_dir: Path) -> dict:
    tree = temp / "scored"
    shutil.copytree(clean, tree, ignore=shutil.ignore_patterns(".build", "state", "__pycache__", "node_modules"))
    for service in SCORED:
        shutil.rmtree(tree / "services" / service)
        source = workspace / "app" / "services" / service
        if source.is_dir():
            shutil.copytree(source, tree / "services" / service, ignore=SKIP, symlinks=True)
    command = [sys.executable, str(SCORER / "score.py"), "--app", str(tree), "--verbose",
               "--scenarios", str(SCORER / "scenarios.json"), "--deployed", str(clean),
               "--impact", str(workspace / "out" / "impact.json"), "--truth", str(SCORER / "truth.json")]  # fmt: skip
    env = dict(os.environ, SWITCHYARD_REPLAY_BIN=str(clean / "bin"))
    done = subprocess.run(command, capture_output=True, text=True, timeout=900, env=env)
    (out_dir / "score.log").write_text(done.stdout + done.stderr)
    try:
        return json.loads(done.stdout)
    except ValueError:
        return {"categories": {}, "repair": 0.0, "impact": 0.0, "note": "the scorer failed; see score.log"}


def format_score(result: dict) -> str:
    lines = [f"repair {result.get('repair', 0):.2f}   impact {result.get('impact', 0):.2f}"
             f"  (precision {result.get('precision', 0):.2f}, recall {result.get('recall', 0):.2f},"
             f" figures {result.get('totals_right', 0)} of {result.get('totals', 0)} right)"]  # fmt: skip
    for name, entry in result.get("categories", {}).items():
        lines.append(f"{name:<12} {entry['passed']:>3} of {entry['total']}")
        for failed in entry["failed"][:8]:
            lines.append(f"  - {failed['name']}")
    if result.get("note"):
        lines.append(result["note"])
    return "\n".join(lines)


def one_run(index: int, args, stamp: str) -> dict:
    out_dir = WEEK / "runs" / f"{stamp}-{index}"
    out_dir.mkdir(parents=True)
    usage_log, agent_log = out_dir / "usage.jsonl", out_dir / "agent.log"

    with tempfile.TemporaryDirectory(prefix="cs146s-") as temp:
        temp = Path(temp)
        clean = temp / "clean-app"
        shutil.copytree(WEEK / "app", clean, ignore=SKIP)
        workspace = temp / "week6"
        shutil.copytree(WEEK, workspace, ignore=SKIP, symlinks=True)
        home = temp / "home"
        gateway_url = f"http://127.0.0.1:{args.port}"

        give_keys = {p: bool(os.environ.get(names[0])) for p, names in REAL_KEYS.items()}
        env = dict(os.environ)
        for provider, names in REAL_KEYS.items():
            if give_keys[provider]:
                for name in names:
                    env.pop(name, None)  # the gateway holds it; the harness must not
        env.update(harness.gateway_env(gateway_url, home, give_keys))
        codex_login = CODEX_LOGIN if not give_keys["openai"] and CODEX_LOGIN.is_file() else None
        harness.write_codex_config(home, gateway_url, login=codex_login)

        gateway = runner.start_gateway(args.port, WEEK / "rate_card.json", usage_log, max_cost=2.0,
                                       allow_login=codex_login is not None)  # fmt: skip
        print(f"run {index}: running ./run.sh task.md (up to {args.timeout / 60:.0f} min); nothing prints until it exits.\n"
              f"  each model request is appended to {usage_log.relative_to(WEEK)} as it happens;\n"
              f"  run.sh's own output lands in {agent_log.relative_to(WEEK)}", flush=True)  # fmt: skip
        try:
            result = runner.run_entry(workspace, "task.md", env, args.timeout, agent_log)
        finally:
            runner.stop_gateway(gateway)
            stop_leftovers(workspace)

        print(f"run {index}: run.sh exited, scoring...", flush=True)
        scored = score(workspace, clean, temp, out_dir)
        if args.keep:
            diff = subprocess.run(["diff", "-ruN", "-x", ".build", "-x", "state", "-x", "__pycache__",
                                   str(clean), str(workspace / "app")], capture_output=True, text=True)  # fmt: skip
            (out_dir / "app.diff").write_text(diff.stdout)
            if (workspace / "out").is_dir():
                shutil.copytree(workspace / "out", out_dir / "out")

    summary = costs.summarize(usage_log)
    passed = scored.get("repair") == 1.0 and scored.get("impact") == 1.0
    print(f"run {index}: {'PASS' if passed else 'FAIL'}")
    print("  " + format_score(scored).replace("\n", "\n  "))
    if result["error"]:
        print(f"  {result['error']}")
    elif result["timed_out"]:
        print(f"  run.sh hit the {args.timeout:.0f}s time limit and was stopped")
    else:
        print(f"  run.sh exited {result['exit_code']} after {result['seconds']}s")
    print("  " + costs.format_summary(summary).replace("\n", "\n  "))
    if summary["requests"] == 0:
        print("  no model requests reached the gateway: is your harness using the variables it sets?")
    print(f"  logs: {out_dir.relative_to(WEEK)}/")
    record = {"passed": passed, "score": scored, **result, **summary}
    (out_dir / "summary.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Run ./run.sh on the starter incident, score it and price it")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600, help="seconds allowed per run")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--keep", action="store_true", help="save each run's app diff and out/ under runs/")
    args = parser.parse_args()

    warn_if_dirty()
    stamp = time.strftime("%Y%m%d-%H%M%S")
    records = [one_run(i, args, stamp) for i in range(1, args.runs + 1)]
    if len(records) > 1:
        passing = [r for r in records if r["passed"]]
        print(f"\n{len(passing)} of {len(records)} runs passed in full")
        if passing:
            print(f"average cost of passing runs: ${sum(r['cost_usd'] for r in passing) / len(passing):.4f}")
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    sys.exit(main())
