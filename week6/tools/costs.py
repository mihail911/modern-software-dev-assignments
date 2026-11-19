"""Rate card lookups and cost arithmetic, shared by the gateway and the grader.

Cost is always tokens times the rate card. Dollar figures reported by a
harness or a provider are never used, so a model costs the same everywhere.
"""

import json
import re
from pathlib import Path

TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "cache_write_1h")

# Providers serve dated snapshots of a model under the same price.
_DATE_SUFFIX = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")


def load_rate_card(path) -> dict:
    card = json.loads(Path(path).read_text())
    for name, rates in card["models"].items():
        for field in ("provider", "input", "output", "cache_read", "cache_write"):
            if field not in rates:
                raise ValueError(f"rate card: {name} is missing {field!r}")
    return card


def resolve_model(card: dict, model) -> "str | None":
    """The rate-card name for a requested model id, or None if it is not allowed."""
    if not isinstance(model, str):
        return None
    models = card["models"]
    if model in models:
        return model
    undated = _DATE_SUFFIX.sub("", model)
    return undated if undated in models else None


def empty_usage() -> dict:
    return {field: 0 for field in TOKEN_FIELDS}


def cost_usd(card: dict, model: str, usage: dict) -> float:
    rates = card["models"][model]
    total = 0.0
    for field in TOKEN_FIELDS:
        # A model with no separate one-hour write price is charged the normal one.
        rate = rates.get(field, rates["cache_write"] if field == "cache_write_1h" else 0.0)
        total += usage.get(field, 0) * rate
    return total / 1_000_000


def summarize(log_path) -> dict:
    """Total a gateway usage log (one JSON object per line)."""
    summary = {"cost_usd": 0.0, "requests": 0, "rejected": [], "upstream_errors": 0, "by_model": {}}
    path = Path(log_path)
    if not path.exists():
        return summary
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        entry = json.loads(line)
        if entry.get("rejected"):
            summary["rejected"].append(entry["rejected"])
            continue
        if not entry.get("metered"):
            continue
        summary["requests"] += 1
        if entry.get("status", 200) >= 400:
            summary["upstream_errors"] += 1
        model = entry["model"]
        bucket = summary["by_model"].setdefault(
            model, {"requests": 0, "cost_usd": 0.0, **empty_usage()}
        )
        bucket["requests"] += 1
        bucket["cost_usd"] += entry.get("cost_usd", 0.0)
        for field in TOKEN_FIELDS:
            bucket[field] += entry.get("usage", {}).get(field, 0)
        summary["cost_usd"] += entry.get("cost_usd", 0.0)
    return summary


def format_summary(summary: dict) -> str:
    lines = [f"cost ${summary['cost_usd']:.4f}, model requests: {summary['requests']}"]
    for model, b in sorted(summary["by_model"].items()):
        lines.append(
            f"  {model}: ${b['cost_usd']:.4f}  in {b['input']:,}  out {b['output']:,}  "
            f"cache read {b['cache_read']:,}  cache write {b['cache_write'] + b['cache_write_1h']:,}"
        )
    if summary["rejected"]:
        reasons = sorted(set(summary["rejected"]))
        lines.append(f"  refused by the gateway ({len(summary['rejected'])}): " + "; ".join(reasons))
    if summary["upstream_errors"]:
        lines.append(f"  failed at the provider: {summary['upstream_errors']}")
    return "\n".join(lines)
