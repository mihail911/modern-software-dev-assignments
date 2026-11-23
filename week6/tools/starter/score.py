#!/usr/bin/env python3
"""Score a repaired app and an impact report, as the grader does for the starter incident.

    python3 tools/starter/score.py --app TREE --scenarios tools/starter/scenarios.json \
        --deployed CLEAN_APP [--impact out/impact.json --truth tools/starter/truth.json]

TREE is a complete app tree with the services to score. CLEAN_APP is the
release as deployed: some scenarios start on it and replace it with TREE while
they play. Set SWITCHYARD_REPLAY_BIN to CLEAN_APP/bin so the replay code comes
from the clean copy. tools/practice.py does all of this for you.

Left out, --scenarios is tools/starter/scenarios.json, and --deployed and the
replay code both come from week6/app, so keep that copy clean.
"""

import argparse
import concurrent.futures
import json
import os
import pathlib
import queue
import shutil
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# replaylib comes from a clean release tree: week6/app/bin, or SWITCHYARD_REPLAY_BIN.
sys.path.insert(0, os.environ.get("SWITCHYARD_REPLAY_BIN") or str(HERE.parents[1] / "app" / "bin"))
sys.path.insert(0, str(HERE.parent))

import replaylib as harness  # noqa: E402
import starter  # noqa: E402


def suite(name):
    """(scenarios, kind) for --scenarios: "starter", or a JSON file of starter scenarios."""
    if name == "starter":
        return starter.all_scenarios(), "starter"
    with open(name, encoding="utf-8") as f:
        return [tuple(cs) for cs in json.load(f)], "starter"

MAX_ATTEMPTS = 6


def invariants(result, allow=()):
    """Properties that hold after any scenario: what contracts/delivery.md states
    about deliveries, and that every request names a known delivery and its
    event and is signed with a secret of its endpoint. `allow` names properties
    a scenario leaves to other scenarios.
    """
    found = []
    candidate_from = result.get("candidate_from") or 0
    router = result["state"].get("router")
    if router is None:
        return ["router is not running at the end"]
    for service in ("api", "worker"):
        if result["state"].get(service) is None:
            found.append(f"{service} is not running at the end")
    deliveries = router["deliveries"]
    by_id = {d["id"]: d for d in deliveries}
    enabled = {e["id"]: bool(e["enabled"]) for e in router["endpoints"]}
    receipts = [r for r in result["received"] if r["processed"]]

    firsts = {}
    for d in deliveries:
        if d["resend_of"] is None:
            firsts.setdefault((d["event_id"], d["endpoint_id"]), []).append(d["id"])
    for pair, ids in firsts.items():
        if len(ids) > 1:
            found.append(f"{len(ids)} deliveries for one event and endpoint")

    answered_2xx = {r["headers"].get("webhook-id") for r in receipts if r["response"] and 200 <= r["response"] < 300}
    for d in deliveries:
        if d["state"] == "delivered" and d["id"] not in answered_2xx:
            found.append("a delivery is marked delivered with no 2xx response")
        if d["state"] == "dead" and d["attempts"] != MAX_ATTEMPTS:
            found.append(f"a delivery is dead after {d['attempts']} counted attempts")
        if d["state"] not in ("delivered", "dead") and enabled.get(d["endpoint_id"], True):
            found.append(f"a delivery is left {d['state']}")

    for r in receipts:
        delivery = by_id.get(r["headers"].get("webhook-id"))
        if delivery is None:
            found.append("a request carries an unknown delivery id")
            continue
        if r["headers"].get("x-switchyard-event") != delivery["event_id"]:
            found.append("a request names the wrong event")
        secrets = result["secrets"].get(delivery["endpoint_id"], [])
        if not any(harness.signed_with(r, secret) for secret in secrets):
            found.append("a request is not signed with any secret of its endpoint")
    return sorted(set(found))


def play(category, scenario, app, worker_bin, ports, earlier, kind="starter"):
    port_base = ports.get()
    state = tempfile.mkdtemp(prefix="score-")
    try:
        if kind == "starter":
            result = starter.play(harness, scenario, state, port_base, str(app), worker_bin, earlier)
            found = starter.problems(harness, result, scenario["expect"]) + starter.invariants(
                invariants, result, scenario
            )
    except Exception as problem:  # a scenario that cannot run has failed
        found = [f"could not run: {type(problem).__name__}: {str(problem)[:200]}"]
    finally:
        shutil.rmtree(state, ignore_errors=True)
        ports.put(port_base)
    return {"category": category, "name": scenario["name"], "problems": found}


def score_repair(app, jobs=4, port_base=7300, only=None, scenarios=str(HERE / "scenarios.json"), deployed=None):
    """deployed: for the starter, the release as deployed (a tree) that "start": "deployed" scenarios begin on."""
    app = pathlib.Path(app).resolve()
    listed, kind = suite(scenarios)
    build = tempfile.mkdtemp(prefix="score-build-")
    try:
        try:
            worker_bin = harness.build_worker(str(app), os.path.join(build, "worker"))
        except RuntimeError as problem:
            outcomes = [{"category": c, "name": s["name"], "problems": [str(problem)[:300]]} for c, s in listed]
            return summarise(outcomes)
        ports = queue.Queue()
        for i in range(jobs):
            ports.put(port_base + 10 * i)
        chosen = [(c, s) for c, s in listed if only is None or only in s["name"] or only == c]
        if kind == "starter":
            earlier = None
            if any(s.get("start") for _, s in chosen):
                if deployed is None:
                    deployed = str(HERE.parents[1] / "app")
                deployed = str(pathlib.Path(deployed).resolve())
                earlier = (deployed, harness.build_worker(deployed, os.path.join(build, "deployed-worker")))
        with concurrent.futures.ThreadPoolExecutor(jobs) as pool:
            outcomes = list(pool.map(lambda cs: play(cs[0], cs[1], app, worker_bin, ports, earlier, kind), chosen))
    finally:
        shutil.rmtree(build, ignore_errors=True)
    return summarise(outcomes)


def summarise(outcomes):
    categories = {}
    for outcome in outcomes:
        entry = categories.setdefault(outcome["category"], {"passed": 0, "total": 0, "failed": []})
        entry["total"] += 1
        if outcome["problems"]:
            entry["failed"].append({"name": outcome["name"], "problems": outcome["problems"]})
        else:
            entry["passed"] += 1

    def share(names):
        total = sum(categories.get(n, {}).get("total", 0) for n in names)
        passed = sum(categories.get(n, {}).get("passed", 0) for n in names)
        return passed / total if total else 0.0

    repair = share(["incident"]) * share(["regressions", "compat"])
    return {"categories": categories, "repair": round(repair, 4)}


def score_impact(report_path, truth_path):
    """0.6 x F1 on the affected set + 0.4 x share of the report's figures exactly right.

    The figures are the total for each currency and, when the truth has one, skipped_rows.
    """
    with open(truth_path, encoding="utf-8") as f:
        truth = json.load(f)
    blank = {"impact": 0.0, "precision": 0.0, "recall": 0.0, "totals_right": 0,
             "totals": len(truth["totals_minor"]) + ("skipped_rows" in truth)}  # fmt: skip
    try:
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f)
        claimed = {
            (item["event_id"], item["endpoint_id"], tuple(sorted(item["delivery_ids"])))
            for item in report["affected"]
        }
        totals = report.get("totals_minor", {})
        if not isinstance(totals, dict):
            totals = {}
    except (OSError, ValueError, KeyError, TypeError) as problem:
        return dict(blank, note=f"impact report unreadable: {type(problem).__name__}")
    actual = {(i["event_id"], i["endpoint_id"], tuple(sorted(i["delivery_ids"]))) for i in truth["affected"]}
    hit = len(claimed & actual)
    precision = hit / len(claimed) if claimed else 0.0
    recall = hit / len(actual) if actual else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    currencies = set(truth["totals_minor"]) | set(totals)
    right = sum(1 for c in currencies if totals.get(c) == truth["totals_minor"].get(c))
    figures = len(currencies)
    if "skipped_rows" in truth:
        figures += 1
        right += type(report.get("skipped_rows")) is int and report["skipped_rows"] == truth["skipped_rows"]
    totals_share = right / figures if figures else 0.0
    return {
        "impact": round(0.6 * f1 + 0.4 * totals_share, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "totals_right": right,
        "totals": figures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True)
    parser.add_argument("--impact")
    parser.add_argument("--truth")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--port-base", type=int, default=7300)
    parser.add_argument("--only", help="a category, or text in a scenario's name")
    parser.add_argument("--scenarios", default=str(HERE / "scenarios.json"), help="a JSON file of starter scenarios")
    parser.add_argument("--deployed", help="the release as deployed (default: week6/app)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    started = time.time()
    out = score_repair(args.app, args.jobs, args.port_base, args.only, args.scenarios, args.deployed)
    if args.impact and args.truth:
        out.update(score_impact(args.impact, args.truth))
    out["seconds"] = round(time.time() - started, 1)
    if not args.verbose:
        for entry in out["categories"].values():
            entry["failed"] = [f["name"] for f in entry["failed"]]
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
