"""
Start the services and drive them through a scenario.

Used by bin/up, bin/replay and bin/evaluate. A scenario is a JSON object:

    {"name": "...", "steps": [...], "expect": [...]}

Each step has "do" and usually "at", the number of seconds since the scenario's
start at which it happens. Before a step runs, every service's clock is set to
that time (contracts/harness.md).

    endpoint.create  id, merchant, [path], [secret], [event_types], [enabled]
    endpoint.update  id, and any of url/path, secret, event_types, enabled
    event            merchant, type, amount_minor, currency, [ref], [idempotency_key]
    resend           event (a ref), endpoint
    bus              message (posted to the router as is; "{receiver}" becomes the receiver's address), [ref]
    receiver.script  path, responses, [default]
    crash_at         service, point, [hits]
    tick             service
    restart          service
    stop             service
    drain            [limit] seconds of clock it may advance (default 8 days), [restart]
"""

import base64
import hashlib
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone

APP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = 1793577600  # 2026-11-02T00:00:00Z
OFFSET = {"api": 1, "router": 2, "worker": 3, "receiver": 4}
TICKED = ("api", "router", "worker")
DAY = 86400

sys.path.insert(0, os.path.join(APP, "services", "receiver"))
from standardwebhooks import Webhook  # noqa: E402


class ServiceDown(Exception):
    pass


def secret_for(name):
    return "whsec_" + base64.b64encode(hashlib.sha256(name.encode()).digest()[:24]).decode()


def build_worker(app=APP, out=None):
    """
    Compile the worker. Returns the path of the binary.
    """
    
    out = out or os.path.join(app, ".build", "worker")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    done = subprocess.run(
        ["go", "build", "-o", out, "."],
        cwd=os.path.join(app, "services", "worker"),
        capture_output=True,
        text=True,
    )
    
    if done.returncode != 0:
        raise RuntimeError("worker does not build:\n" + done.stdout + done.stderr)
    
    return out


class Deployment:
    def __init__(self, state_dir, port_base=7100, app=APP, worker_bin=None):
        self.app = app
        self.state = state_dir
        self.port_base = port_base
        self.worker_bin = worker_bin or build_worker(app)
        self.procs = {}
        self.conns = {}
        self.now = None
        self.crashes = []
        os.makedirs(os.path.join(state_dir, "logs"), exist_ok=True)

    def command(self, service):
        services = os.path.join(self.app, "services")
    
        if service == "api":
            return ["node", "--disable-warning=ExperimentalWarning", os.path.join(services, "api", "src", "main.ts")]
        elif service == "worker":
            return [self.worker_bin]
    
        return [sys.executable, os.path.join(services, service, "main.py")]

    def start(self, service):
        env = dict(os.environ, SWITCHYARD_STATE=self.state, SWITCHYARD_PORT_BASE=str(self.port_base))
    
        if self.now is not None:
            env["SWITCHYARD_CLOCK"] = str(self.now)
    
        errors = open(os.path.join(self.state, "logs", service + ".stderr"), "ab")
        self.procs[service] = subprocess.Popen(
            self.command(service), env=env, stdout=errors, stderr=errors, start_new_session=True
        )
        errors.close()
        
        deadline = time.time() + 15
        while time.time() < deadline:
            if self.procs[service].poll() is not None:
                raise RuntimeError(f"{service} exited on start; see {self.state}/logs/{service}.stderr")
            try:
                self.call(service, "GET", "/__admin/health")
                break
            except ServiceDown:
                time.sleep(0.02)
        else:
            raise RuntimeError(f"{service} did not become healthy")
        
        if self.now is not None:
            self.call(service, "POST", "/__admin/clock", {"now": self.now})

    def start_all(self):
        for service in ("receiver", "router", "worker", "api"):
            self.start(service)

    def running(self, service):
        proc = self.procs.get(service)
        return proc is not None and proc.poll() is None

    def stop(self, service):
        self.hang_up(service)
        proc = self.procs.get(service)
        
        if proc is None:
            return
        
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        
        proc.wait()

    def stop_all(self):
        for service in list(self.procs):
            self.stop(service)
        
        for service in list(self.conns):
            self.hang_up(service)

    def hang_up(self, service):
        conn = self.conns.pop(service, None)
        if conn is not None:
            conn.close()

    def call(self, service, method, path, body=None):
        """One request on a kept-open connection to the service.

        A kept connection can have been closed by the service while idle. In
        that case the request is sent once more on a new connection. If the
        service has stopped, the second try is refused and ServiceDown is raised.
        """
        data = None if body is None else json.dumps(body)
        headers = {"Content-Type": "application/json"} if data is not None else {}
        
        for fresh in (service not in self.conns, True):
            if service not in self.conns:
                self.conns[service] = http.client.HTTPConnection(
                    "127.0.0.1", self.port_base + OFFSET[service], timeout=30
                )
            try:
                conn = self.conns[service]
                conn.request(method, path, body=data, headers=headers)
                response = conn.getresponse()
                raw = response.read()
                return response.status, (json.loads(raw) if raw else None)
            except (OSError, http.client.HTTPException) as problem:
                self.hang_up(service)
                if fresh:
                    raise ServiceDown(f"{service}: {problem}") from None

    def set_clock(self, now):
        self.now = now
        for service in OFFSET:
            if self.running(service):
                try:
                    self.call(service, "POST", "/__admin/clock", {"now": now})
                except ServiceDown:
                    pass

    def tick(self, service):
        """Tick one service. Returns what it did; 0 if it is down."""
        if not self.running(service):
            return 0
        try:
            _, reply = self.call(service, "POST", "/__admin/tick", {})
        except ServiceDown:
            self.note_crash(service)
            return 0
        return int((reply or {}).get("did", 0))

    def note_crash(self, service):
        proc = self.procs.get(service)
        if proc is not None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                return
            self.crashes.append({"service": service, "now": self.now, "exit": proc.returncode})

    def drain(self, limit=8 * DAY, restart=False):
        deadline = self.now + limit
        for _ in range(100000):
            did = sum(self.tick(service) for service in TICKED)
            down = [s for s in TICKED if not self.running(s)]
            if down and restart:
                for service in down:
                    self.start(service)
                continue
            if did:
                continue
            if not self.running("router"):
                return
            wake = self.call("router", "GET", "/__admin/state")[1]["next_wake"]
            if wake is None or wake <= self.now or wake > deadline:
                return
            self.set_clock(wake)
        raise RuntimeError("drain did not settle")


class Run:
    """One scenario against one deployment."""

    def __init__(self, deployment, scenario):
        self.dep = deployment
        self.scenario = scenario
        self.epoch = scenario.get("epoch", EPOCH)
        self.refs = {}
        self.statuses = {}
        self.secrets = {}
        self.receiver_url = f"http://127.0.0.1:{deployment.port_base + OFFSET['receiver']}"

    def need(self, status, reply, step, allowed=(200, 201, 202)):
        if status not in allowed:
            raise RuntimeError(f"step {step['do']} got {status}: {reply}")

    def step(self, step):
        dep = self.dep
        if "at" in step:
            dep.set_clock(self.epoch + step["at"])
        do = step["do"]
        if do == "endpoint.create":
            secret = step.get("secret") or secret_for(step["id"])
            body = {
                "id": step["id"],
                "merchant_id": step["merchant"],
                "url": self.receiver_url + step.get("path", "/hooks/" + step["id"]),
                "secret": secret,
                "event_types": step.get("event_types", ["*"]),
                "enabled": step.get("enabled", True),
            }
            self.secrets[step["id"]] = [secret]
            self.need(*dep.call("api", "POST", "/v1/endpoints", body), step)
        elif do == "endpoint.update":
            body = {k: step[k] for k in ("secret", "event_types", "enabled", "url") if k in step}
            if "path" in step:
                body["url"] = self.receiver_url + step["path"]
            if "secret" in body:
                self.secrets.setdefault(step["id"], []).append(body["secret"])
            self.need(*dep.call("api", "PUT", "/v1/endpoints/" + step["id"], body), step)
        elif do == "event":
            body = {
                "merchant_id": step["merchant"],
                "type": step["type"],
                "data": {"amount_minor": step["amount_minor"], "currency": step["currency"]},
            }
            if "idempotency_key" in step:
                body["idempotency_key"] = step["idempotency_key"]
            status, reply = dep.call("api", "POST", "/v1/events", body)
            self.need(status, reply, step)
            if "ref" in step:
                self.refs[step["ref"]] = reply["id"]
        elif do == "resend":
            body = {"endpoint_id": step["endpoint"]}
            self.need(*dep.call("api", "POST", f"/v1/events/{self.refs[step['event']]}/resend", body), step)
        elif do == "bus":
            message = json.loads(json.dumps(step["message"]).replace("{receiver}", self.receiver_url))
            status, _ = dep.call("router", "POST", "/bus", message)
            if isinstance(message, dict):
                endpoint = message.get("endpoint") or (message.get("data") or {}).get("endpoint")
                if isinstance(endpoint, dict) and "secret" in endpoint:
                    self.secrets.setdefault(endpoint.get("id"), []).append(endpoint["secret"])
                event = message.get("event") or (message.get("data") or {}).get("event") or message
                if "ref" in step and isinstance(event, dict) and "id" in event:
                    self.refs[step["ref"]] = event["id"]
            if "ref" in step:
                self.statuses[step["ref"]] = status
        elif do == "receiver.script":
            body = {"path": step["path"], "responses": step.get("responses", []), "default": step.get("default", 200)}
            self.need(*dep.call("receiver", "POST", "/__admin/script", body), step)
        elif do == "crash_at":
            body = {"point": step["point"], "hits": step.get("hits", 1)}
            self.need(*dep.call(step["service"], "POST", "/__admin/crash_at", body), step)
        elif do == "tick":
            dep.tick(step["service"])
        elif do == "restart":
            dep.stop(step["service"])
            dep.start(step["service"])
        elif do == "stop":
            dep.stop(step["service"])
        elif do == "drain":
            dep.drain(step.get("limit", 8 * DAY), step.get("restart", False))
        else:
            raise RuntimeError(f"unknown step {do}")

    def result(self):
        dep = self.dep
        states = {}
        for service in ("api", "router", "worker", "receiver"):
            try:
                states[service] = dep.call(service, "GET", "/__admin/state")[1]
            except ServiceDown:
                states[service] = None
        received = (states.pop("receiver") or {}).get("received", [])
        return {
            "name": self.scenario.get("name"),
            "epoch": self.epoch,
            "refs": self.refs,
            "statuses": self.statuses,
            "secrets": self.secrets,
            "crashes": dep.crashes,
            "received": received,
            "state": states,
        }

    def play(self):
        for step in self.scenario["steps"]:
            self.step(step)
        return self.result()


def run_scenario(scenario, state_dir=None, port_base=7100, app=APP, worker_bin=None):
    """Play a scenario on a fresh deployment. Returns the result."""
    temporary = state_dir is None
    state_dir = state_dir or tempfile.mkdtemp(prefix="switchyard-")
    dep = Deployment(state_dir, port_base, app, worker_bin)
    try:
        dep.set_clock(scenario.get("epoch", EPOCH))
        dep.start_all()
        return Run(dep, scenario).play()
    finally:
        dep.stop_all()
        if temporary:
            shutil.rmtree(state_dir, ignore_errors=True)


# --- expectations ---------------------------------------------------------


def deliveries_for(result, where):
    event_id = result["refs"].get(where["event"], where["event"])
    router = result["state"].get("router") or {}
    return [
        d
        for d in router.get("deliveries", [])
        if d["event_id"] == event_id and d["endpoint_id"] == where["endpoint"]
    ]


def receipts_for(result, delivery_ids):
    return [
        r for r in result["received"] if r["processed"] and r["headers"].get("webhook-id") in delivery_ids
    ]


def signed_with(receipt, secret):
    """True if the receipt's signature is the Standard Webhooks signature under this secret."""
    headers = receipt["headers"]
    try:
        stamp = datetime.fromtimestamp(int(headers["webhook-timestamp"]), tz=timezone.utc)
        expected = Webhook(secret).sign(headers["webhook-id"], stamp, receipt["body"])
    except (TypeError, ValueError, KeyError):
        return False
    return expected in (headers.get("webhook-signature") or "").split(" ")


def check(result, expect):
    """Return a problem as text, or None if the expectation holds."""
    if "status" in expect:
        got = result["statuses"].get(expect["status"])
        return None if got == expect["is"] else f"bus message {expect['status']} got {got}, expected {expect['is']}"
    where = expect["delivery"]
    found = deliveries_for(result, where)
    label = f"{where['event']} -> {where['endpoint']}"
    if "count" in expect and len(found) != expect["count"]:
        return f"{label}: {len(found)} deliveries, expected {expect['count']}"
    firsts = [d for d in found if d["resend_of"] is None]
    if "state" in expect:
        if len(firsts) != 1:
            return f"{label}: {len(firsts)} first deliveries, expected 1"
        if firsts[0]["state"] != expect["state"]:
            return f"{label}: state {firsts[0]['state']}, expected {expect['state']}"
    if "counted" in expect and (len(firsts) != 1 or firsts[0]["attempts"] != expect["counted"]):
        return f"{label}: {[d['attempts'] for d in firsts]} attempts counted, expected {expect['counted']}"
    receipts = receipts_for(result, {d["id"] for d in found})
    accepted = [r for r in receipts if r["accepted"]]
    for key, rows in (("receipts", receipts), ("accepted", accepted)):
        if key in expect and len(rows) != expect[key]:
            return f"{label}: {len(rows)} {key}, expected {expect[key]}"
        if "min_" + key in expect and len(rows) < expect["min_" + key]:
            return f"{label}: {len(rows)} {key}, expected at least {expect['min_' + key]}"
        if "max_" + key in expect and len(rows) > expect["max_" + key]:
            return f"{label}: {len(rows)} {key}, expected at most {expect['max_' + key]}"
    if "gaps_at_least" in expect:
        times = [r["now"] for r in receipts]
        for i, least in enumerate(expect["gaps_at_least"]):
            if i + 1 >= len(times):
                return f"{label}: only {len(times)} receipts, expected a gap after number {i + 1}"
            if times[i + 1] - times[i] < least:
                return f"{label}: {times[i + 1] - times[i]} s after receipt {i + 1}, expected at least {least}"
    if "first_receipt_at_least" in expect:
        if not receipts:
            return f"{label}: nothing received"
        first = receipts[0]["now"] - result["epoch"]
        if first < expect["first_receipt_at_least"]:
            return f"{label}: first received at {first} s, expected no earlier than {expect['first_receipt_at_least']}"
    if "secrets" in expect:
        if len(receipts) != len(expect["secrets"]):
            return f"{label}: {len(receipts)} receipts, expected {len(expect['secrets'])}"
        for number, (receipt, secret) in enumerate(zip(receipts, expect["secrets"]), 1):
            if not signed_with(receipt, secret):
                return f"{label}: receipt {number} is not signed with the expected secret"
    if "first_receipt_at_most" in expect:
        if not receipts:
            return f"{label}: nothing received"
        first = receipts[0]["now"] - result["epoch"]
        if first > expect["first_receipt_at_most"]:
            return f"{label}: first received at {first} s, expected by {expect['first_receipt_at_most']}"
    if "path" in expect:
        paths = [r["path"] for r in receipts]
        if paths != expect["path"]:
            return f"{label}: received at {paths}, expected {expect['path']}"
    if "amount_minor" in expect:
        for r in receipts:
            data = json.loads(r["body"]).get("data", {})
            if data.get("amount_minor") != expect["amount_minor"] or data.get("currency") != expect.get(
                "currency", data.get("currency")
            ):
                return f"{label}: body carries {data.get('amount_minor')} {data.get('currency')}"
        if not receipts:
            return f"{label}: nothing received"
    return None


def problems(result, expects):
    return [p for p in (check(result, e) for e in expects) if p]
