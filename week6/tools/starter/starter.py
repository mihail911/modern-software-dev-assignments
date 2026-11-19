"""Scenarios for the starter incident (INC-2207, release 2.3.1).

Categories:

    incident     fails on the release as deployed: a merchant's retry after an api
                 restart is a second event, and what the deployed release created
                 that way is stopped once the release is replaced
    regressions  ordinary delivery, retries, configuration, restarts, idempotency keys
    compat       bus message versions 1 and 2

Two steps are added to replaylib's:

    pay       merchant, payment, key, [amount_minor], [currency], [type]: POST /v1/events
              with that idempotency key and payment id. A request that gets no reply
              (the api stopped at a crash point) is not an error.
    upgrade   api, router and worker are restarted from the tree under test. A scenario
              with "start": "deployed" runs the release as deployed until then.

Checks are on outcomes only. On top of each scenario's own expectations:

    - no payment reaches an endpoint as a second event once the tree under test runs
    - a delivery is not left queued or leased (any other final state is the release's call)
    - the contract invariants in score.py

    python3 scorer/starter.py --dump FILE     write the scenarios as JSON (what students get)
"""

import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
EPOCH = 1791763200  # 2026-10-12T00:00:00Z, three weeks before 2.4 exists
M = "m_acme"
SCENARIOS = []


def add(category, name, steps, expect, start=None):
    scenario = {"name": name, "epoch": EPOCH, "steps": steps, "expect": expect}
    if start:
        scenario["start"] = start
    SCENARIOS.append((category, scenario))


def ep(id_, at=0, merchant=M, **more):
    return {"at": at, "do": "endpoint.create", "id": id_, "merchant": merchant, **more}


def script(id_, responses=(), default=200, at=0):
    return {"at": at, "do": "receiver.script", "path": "/hooks/" + id_, "responses": list(responses), "default": default}


def pay(at, payment, key=None, merchant=M, amount=1250, currency="USD", **more):
    return {"at": at, "do": "pay", "merchant": merchant, "payment": payment, "key": key,
            "amount_minor": amount, "currency": currency, **more}  # fmt: skip


def crash(at, service, point, hits=1):
    return {"at": at, "do": "crash_at", "service": service, "point": point, "hits": hits}


def restart(at, service):
    return {"at": at, "do": "restart", "service": service}


def tick(at, service):
    return {"at": at, "do": "tick", "service": service}


def drain(at, **more):
    return {"at": at, "do": "drain", **more}


def upgrade(at):
    return {"at": at, "do": "upgrade"}


def once(payment, endpoint="ep_a"):
    """The payment reached the endpoint, as exactly one event."""
    return {"payment": payment, "endpoint": endpoint, "events": 1}


def retried_after_crash(at, payment, key, point=1, **more):
    """The api accepts the payment and stops before it answers; the merchant's client sends it again."""
    return [crash(at, "api", point), pay(at, payment, key, **more), restart(at + 2, "api"), pay(at + 3, payment, key, **more)]


DEPLOYED = "deployed"

# --- incident: the fix ------------------------------------------------------------

add("incident", "a payment sent again after the api stopped before answering is one event",
    [ep("ep_a"), *retried_after_crash(5, "pay_1", "k-1"), drain(10)],
    [once("pay_1")])

add("incident", "a payment sent again after an api restart is one event",
    [ep("ep_a"), pay(5, "pay_1", "k-1"), restart(20, "api"), pay(25, "pay_1", "k-1"), drain(30)],
    [once("pay_1")])

add("incident", "a payment sent again after the api stopped while publishing is one event",
    [ep("ep_a"), pay(5, "pay_1", "k-1"), crash(6, "api", 3), tick(6, "api"), restart(8, "api"),
     pay(9, "pay_1", "k-1"), drain(10)],
    [once("pay_1")])

add("incident", "a payment sent again two hours after an api restart is one event",
    [ep("ep_a"), pay(5, "pay_1", "k-1"), drain(6), restart(60, "api"), pay(7200, "pay_1", "k-1"), drain(7201)],
    [once("pay_1")])

# --- incident: what the deployed release did ------------------------------------

add("incident", "a repeat the deployed release queued for a retry is not sent once the release is replaced",
    [ep("ep_a"), script("ep_a", default=503), *retried_after_crash(5, "pay_1", "k-1"), drain(10, limit=200),
     upgrade(300), script("ep_a", default=200, at=301), drain(302)],
    [once("pay_1")], start=DEPLOYED)

add("incident", "a repeat whose original was delivered is never sent",
    [ep("ep_a"), ep("ep_b"), script("ep_b", default=503), *retried_after_crash(5, "pay_1", "k-1"),
     drain(10, limit=200), upgrade(300), script("ep_b", default=200, at=301), drain(302)],
    [{"payment": "pay_1", "endpoint": "ep_b", "events": 1}], start=DEPLOYED)

add("incident", "repeats of several payments are each stopped",
    [ep("ep_a"), script("ep_a", default=500),
     *retried_after_crash(5, "pay_1", "k-1"), *retried_after_crash(10, "pay_2", "k-2", amount=99, currency="JPY"),
     pay(15, "pay_3", "k-3"), drain(16, limit=300), upgrade(400), script("ep_a", default=200, at=401), drain(402)],
    [once("pay_1"), once("pay_2"), once("pay_3")], start=DEPLOYED)

add("incident", "a repeat that is being sent when the release is replaced is not retried",
    [ep("ep_a"), script("ep_a", [503, 503, 503]), *retried_after_crash(5, "pay_1", "k-1"),
     tick(10, "api"), crash(11, "worker", 4, hits=2), tick(11, "worker"), tick(11, "worker"), upgrade(12), drain(13)],
    [once("pay_1")], start=DEPLOYED)

add("incident", "starting the replaced release twice stops nothing more",
    [ep("ep_a"), script("ep_a", default=503), *retried_after_crash(5, "pay_1", "k-1"), drain(10, limit=200),
     upgrade(300), restart(310, "api"), restart(311, "router"), pay(320, "pay_2", "k-2"),
     script("ep_a", default=200, at=330), drain(331)],
    [once("pay_1"), once("pay_2")], start=DEPLOYED)

# --- regressions: what the deployed release got right ---------------------------

add("regressions", "the same key at two merchants is two events",
    [ep("ep_a"), ep("ep_b", merchant="m_bolt"), pay(5, "pay_1", "k-1"), pay(6, "pay_9", "k-1", merchant="m_bolt"),
     drain(7)],
    [once("pay_1"), once("pay_9", "ep_b")])

add("regressions", "events the deployed release took under the same key at two merchants are both delivered",
    [ep("ep_a"), ep("ep_b", merchant="m_bolt"), script("ep_a", default=503), script("ep_b", default=503),
     pay(5, "pay_1", "k-1"),
     pay(6, "pay_9", "k-1", merchant="m_bolt"), drain(7, limit=200), upgrade(300),
     script("ep_a", default=200, at=301), script("ep_b", default=200, at=301), drain(302)],
    [once("pay_1"), once("pay_9", "ep_b")], start=DEPLOYED)

add("regressions", "payments with the same amount and different keys are each delivered",
    [ep("ep_a"), script("ep_a", default=503), pay(5, "pay_1", "k-1"), pay(5, "pay_2", "k-2"),
     drain(6, limit=200), upgrade(300), script("ep_a", default=200, at=301), drain(302)],
    [once("pay_1"), once("pay_2")], start=DEPLOYED)

add("regressions", "events sent without a key are never stopped",
    [ep("ep_a"), script("ep_a", default=503), pay(5, "pay_1"), pay(5, "pay_1"), drain(6, limit=200),
     upgrade(300), script("ep_a", default=200, at=301), drain(302)],
    [{"payment": "pay_1", "endpoint": "ep_a", "events": 2}], start=DEPLOYED)

add("regressions", "a key sent again while the api runs is one event",
    [ep("ep_a"), pay(5, "pay_1", "k-1"), pay(6, "pay_1", "k-1"), drain(7)],
    [once("pay_1")])

add("regressions", "a payment the deployed release delivered once is not sent again",
    [ep("ep_a"), pay(5, "pay_1", "k-1"), drain(6), upgrade(100), drain(101)],
    [{"payment": "pay_1", "endpoint": "ep_a", "events": 1, "accepted": 1}], start=DEPLOYED)

add("regressions", "a request that got no answer is sent again under the same delivery after the release is replaced",
    [ep("ep_a"), script("ep_a", ["drop"], default=503), pay(5, "pay_1", "k-1"), drain(6, limit=200),
     upgrade(300), script("ep_a", default=200, at=301), drain(302)],
    [once("pay_1")], start=DEPLOYED)


# --- compat: older bus messages ------------------------------------------------


def bus(at, message, ref=None):
    step = {"at": at, "do": "bus", "message": message}
    if ref:
        step["ref"] = ref
    return step


def d(event, endpoint, **expect):
    return {"delivery": {"event": event, "endpoint": endpoint}, **expect}


add("compat", "a version-1 event is delivered once, with its amount in minor units",
    [ep("ep_a"), tick(1, "api"),
     bus(5, {"id": "evt_v1old000001", "merchant": M, "event_type": "payment.captured", "amount": "40.10",
             "currency": "EUR", "payment": "pay_v1old1", "ts": "2026-10-12T00:00:05Z"}, "e1"),
     bus(6, {"id": "evt_v1old000001", "merchant": M, "event_type": "payment.captured", "amount": "40.10",
             "currency": "EUR", "payment": "pay_v1old1", "ts": "2026-10-12T00:00:05Z"}),
     drain(7)],
    [d("e1", "ep_a", state="delivered", count=1, receipts=1, amount_minor=4010, currency="EUR")])

add("compat", "a version-2 endpoint and event are applied after the release is replaced",
    [{"at": 1, "do": "bus", "message": {"v": 2, "seq": 1, "kind": "endpoint", "endpoint": {
        "id": "ep_v2", "merchant": M, "url": "{receiver}/hooks/ep_v2", "secret": "whsec_dGVzdHRlc3R0ZXN0dGVzdA==",
        "types": ["*"], "active": True}}},
     upgrade(3),
     bus(5, {"v": 2, "seq": 2, "kind": "event", "event": {"id": "evt_v2old000001", "merchant": M,
             "type": "refund.created", "created": 1791763205, "payment_id": "pay_v2old1", "amount_minor": 77,
             "currency": "JPY"}}, "e1"),
     drain(6)],
    [d("e1", "ep_v2", state="delivered", count=1, receipts=1, amount_minor=77, currency="JPY")], start=DEPLOYED)




# --- playing ----------------------------------------------------------------------


def runner(harness):
    """The Run and Deployment classes, built on the replaylib the scorer was given."""

    class Upgrading(harness.Deployment):
        def __init__(self, state_dir, port_base, app, worker_bin, deployed, start):
            super().__init__(state_dir, port_base, app, worker_bin)
            self.candidate = (app, worker_bin)
            self.deployed = deployed
            self.release = "deployed" if start == DEPLOYED else "candidate"
            self.candidate_from = None if self.release == "deployed" else 0

        def command(self, service):
            app, worker_bin = self.candidate if self.release == "candidate" else self.deployed
            if service == "receiver":
                app = self.candidate[0]
            self.app, self.worker_bin = app, worker_bin
            try:
                return super().command(service)
            finally:
                self.app, self.worker_bin = self.candidate

        def upgrade(self):
            if self.release == "candidate":
                return
            for service in ("worker", "api", "router"):
                self.stop(service)
            self.release = "candidate"
            self.candidate_from = len(self.call("receiver", "GET", "/__admin/state")[1]["received"])
            for service in ("router", "worker", "api"):
                self.start(service)

    class Play(harness.Run):
        def __init__(self, dep, scenario):
            super().__init__(dep, scenario)
            self.payments = {}

        def step(self, step):
            if step["do"] == "upgrade":
                self.dep.set_clock(self.epoch + step["at"])
                return self.dep.upgrade()
            if step["do"] != "pay":
                return super().step(step)
            self.dep.set_clock(self.epoch + step["at"])
            body = {
                "merchant_id": step["merchant"],
                "type": step.get("type", "payment.captured"),
                "data": {"amount_minor": step["amount_minor"], "currency": step["currency"], "payment_id": step["payment"]},
            }
            if step.get("key"):
                body["idempotency_key"] = step["key"]
            try:
                status, reply = self.dep.call("api", "POST", "/v1/events", body)
            except harness.ServiceDown:
                self.dep.note_crash("api")
                return
            self.need(status, reply, step)

        def result(self):
            result = super().result()
            result["candidate_from"] = self.dep.candidate_from or 0
            return result

    return Upgrading, Play


def play(harness, scenario, state_dir, port_base, app, worker_bin, deployed):
    Upgrading, Play = runner(harness)
    dep = Upgrading(state_dir, port_base, app, worker_bin, deployed, scenario.get("start"))
    try:
        dep.set_clock(scenario.get("epoch", EPOCH))
        dep.start_all()
        return Play(dep, scenario).play()
    finally:
        dep.stop_all()


def receipts_by_payment(result):
    out = {}
    for r in result["received"]:
        if not r["processed"]:
            continue
        try:
            body = json.loads(r["body"])
        except ValueError:
            continue
        payment = (body.get("data") or {}).get("payment_id")
        endpoint = r["path"].rsplit("/", 1)[-1]
        out.setdefault((payment, endpoint), []).append((r, body.get("id")))
    return out


def problems(harness, result, expects):
    found = []
    payments = receipts_by_payment(result)
    for expect in expects:
        if "payment" not in expect:
            problem = harness.check(result, expect)
            if problem:
                found.append(problem)
            continue
        got = payments.get((expect["payment"], expect["endpoint"]), [])
        accepted = [(r, event) for r, event in got if r["accepted"]]
        label = f"{expect['payment']} -> {expect['endpoint']}"
        events = len({event for _, event in accepted})
        if "events" in expect and events != expect["events"]:
            found.append(f"{label}: delivered as {events} events, expected {expect['events']}")
        if "accepted" in expect and len(accepted) != expect["accepted"]:
            found.append(f"{label}: {len(accepted)} accepted requests, expected {expect['accepted']}")
    return found


def invariants(score_invariants, result, scenario):
    found = []
    unkeyed = set(scenario.get("unkeyed", ()))
    for problem in score_invariants(result, scenario.get("allow", ())):
        if problem.startswith("a delivery is left ") and problem.split()[-1] not in ("queued", "leased"):
            continue  # a release may end a stopped delivery in a state of its own
        if problem.startswith("a delivery is dead after "):
            continue  # or end it dead early; the scenarios check the retry schedule themselves
        found.append(problem)
    since = result.get("candidate_from") or 0
    for (payment, endpoint), got in receipts_by_payment(result).items():
        if payment is None:
            continue
        first = None
        for r, event in got:
            if not r["accepted"]:
                continue
            if first is None:
                first = event
            elif event != first and r["n"] > since and payment not in unkeyed:
                found.append("a payment reaches an endpoint as a second event")
                break
    return sorted(set(found))


def all_scenarios():
    return SCENARIOS


# Payments sent without a key may legitimately arrive as two events; scenarios say which.
for _, _scenario in SCENARIOS:
    _scenario["unkeyed"] = sorted({s["payment"] for s in _scenario["steps"] if s["do"] == "pay" and not s.get("key")})


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dump":
        pathlib.Path(sys.argv[2]).write_text(json.dumps(all_scenarios(), indent=1) + "\n", encoding="utf-8")
    else:
        print(__doc__)
