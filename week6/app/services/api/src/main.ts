// api: accepts events and manages endpoint configuration.
// Everything it tells the router goes through the outbox (store.ts) and is
// published on tick, in order.

import fs from "node:fs";
import http from "node:http";
import path from "node:path";

import { RecentKeys } from "./keys.ts";
import { type Endpoint, type EventRecord, Store } from "./store.ts";

const RELEASE = "2.3.1";
const EVENT_TYPES = new Set([
  "payment.captured",
  "payment.failed",
  "refund.created",
  "payout.paid",
]);
const CURRENCIES = new Set(["USD", "EUR", "GBP", "JPY", "KWD"]);

const stateDir = process.env.SWITCHYARD_STATE ?? "./state";
const portBase = Number(process.env.SWITCHYARD_PORT_BASE ?? "7100");
const routerUrl = `http://127.0.0.1:${portBase + 2}`;

const store = new Store(stateDir);
const keys = new RecentKeys();

// --- clock, crash points, log (contracts/harness.md) ----------------------

let clockNow: number | null = process.env.SWITCHYARD_CLOCK
  ? Number(process.env.SWITCHYARD_CLOCK)
  : null;
const now = (): number => clockNow ?? Math.floor(Date.now() / 1000);

let crashPoint: number | null = null;
let crashHits = 0;
function crash(point: number): void {
  if (crashPoint !== point) return;
  crashHits -= 1;
  if (crashHits <= 0) process.exit(86);
}

fs.mkdirSync(path.join(stateDir, "logs"), { recursive: true });
const logPath = path.join(stateDir, "logs", "api.ndjson");
function log(msg: string, fields: Record<string, unknown> = {}): void {
  const ts = new Date(now() * 1000).toISOString().replace(".000Z", "Z");
  fs.appendFileSync(
    logPath,
    JSON.stringify({ ts, svc: "api", msg, ...fields }) + "\n",
  );
}

// --- request handling -----------------------------------------------------

type Reply = [number, unknown];
type Body = Record<string, unknown>;

class Invalid extends Error {}

function text(body: Body, key: string): string {
  const value = body[key];
  if (typeof value !== "string" || value === "")
    throw new Invalid(`${key} is required`);
  return value;
}

function eventTypes(value: unknown): string[] {
  if (!Array.isArray(value) || value.length === 0)
    throw new Invalid("event_types must be a non-empty list");
  for (const item of value) {
    if (item !== "*" && !EVENT_TYPES.has(item))
      throw new Invalid(`unknown event type ${item}`);
  }
  return value as string[];
}

function createEndpoint(body: Body): Reply {
  const merchantId = text(body, "merchant_id");
  const url = text(body, "url");
  const types =
    body.event_types === undefined ? ["*"] : eventTypes(body.event_types);
  const endpoint = store.transaction((): Endpoint | null => {
    const id = typeof body.id === "string" ? body.id : store.nextId("ep");
    if (store.endpoint(id)) return null;
    const created: Endpoint = {
      id,
      merchant_id: merchantId,
      url,
      secret:
        typeof body.secret === "string" ? body.secret : store.nextId("whsec"),
      event_types: types,
      enabled: body.enabled !== false,
      version: 1,
    };
    store.saveEndpoint(created, now());
    return created;
  });
  if (!endpoint) return [409, { error: "endpoint exists" }];
  log("endpoint.created", {
    endpoint: endpoint.id,
    merchant: merchantId,
    version: 1,
    enabled: endpoint.enabled,
  });
  crash(2);
  return [201, endpoint];
}

function updateEndpoint(id: string, body: Body): Reply {
  const changes: Partial<Endpoint> = {};
  if (body.url !== undefined) changes.url = text(body, "url");
  if (body.secret !== undefined) changes.secret = text(body, "secret");
  if (body.event_types !== undefined)
    changes.event_types = eventTypes(body.event_types);
  if (body.enabled !== undefined) {
    if (typeof body.enabled !== "boolean")
      throw new Invalid("enabled must be true or false");
    changes.enabled = body.enabled;
  }
  const endpoint = store.transaction((): Endpoint | null => {
    const current = store.endpoint(id);
    if (!current) return null;
    const next: Endpoint = {
      ...current,
      ...changes,
      version: current.version + 1,
    };
    store.saveEndpoint(next, now());
    return next;
  });
  if (!endpoint) return [404, { error: "no such endpoint" }];
  log("endpoint.updated", {
    endpoint: id,
    version: endpoint.version,
    enabled: endpoint.enabled,
    changed: Object.keys(changes),
  });
  crash(2);
  return [200, endpoint];
}

function acceptEvent(body: Body): Reply {
  const merchantId = text(body, "merchant_id");
  const type = text(body, "type");
  if (!EVENT_TYPES.has(type)) throw new Invalid(`unknown event type ${type}`);
  const data = body.data as Body | undefined;
  if (typeof data !== "object" || data === null)
    throw new Invalid("data is required");
  const amount = data.amount_minor;
  if (typeof amount !== "number" || !Number.isInteger(amount) || amount < 0) {
    throw new Invalid(
      "data.amount_minor must be a non-negative integer in minor units",
    );
  }
  const currency = data.currency;
  if (typeof currency !== "string" || !CURRENCIES.has(currency))
    throw new Invalid("unknown currency");
  const key =
    typeof body.idempotency_key === "string" ? body.idempotency_key : null;

  const at = now();
  if (key !== null) {
    const earlier = keys.find(merchantId, key, at);
    const found = earlier ? store.event(earlier) : undefined;
    if (found) {
      log("event.repeated", {
        event: found.id,
        merchant: merchantId,
        idempotency_key: key,
      });
      return [200, found];
    }
  }
  const event = store.transaction((): EventRecord => {
    const created: EventRecord = {
      id: store.nextId("evt"),
      merchant_id: merchantId,
      type,
      created: at,
      data: {
        object: "payment",
        payment_id:
          typeof data.payment_id === "string"
            ? data.payment_id
            : store.nextId("pay"),
        amount_minor: amount,
        currency,
      },
    };
    store.saveEvent(created, key, at);
    return created;
  });
  if (key !== null) keys.remember(merchantId, key, event.id, at);
  log("event.accepted", {
    event: event.id,
    merchant: merchantId,
    type,
    amount_minor: amount,
    currency,
    idempotency_key: key,
  });
  crash(1);
  return [202, event];
}

function resend(eventId: string, body: Body): Reply {
  const endpointId = text(body, "endpoint_id");
  if (!store.event(eventId)) return [404, { error: "no such event" }];
  if (!store.endpoint(endpointId)) return [404, { error: "no such endpoint" }];
  store.transaction(() =>
    store.enqueue("resend.requested", now(), {
      event_id: eventId,
      endpoint_id: endpointId,
    }),
  );
  log("resend.requested", { event: eventId, endpoint: endpointId });
  return [202, { event_id: eventId, endpoint_id: endpointId }];
}

// Publish the outbox to the router, oldest first. A message the router fails
// on stays pending and is tried again on the next tick.
async function publish(): Promise<number> {
  let did = 0;
  for (const row of store.pending()) {
    const message = {
      v: 3,
      producer: "api",
      seq: row.seq,
      type: row.type,
      at: row.at,
      data: JSON.parse(row.data),
    };
    let status: number;
    try {
      const response = await fetch(`${routerUrl}/bus`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(message),
      });
      status = response.status;
      await response.arrayBuffer();
    } catch {
      log("bus.unreachable", { bus_seq: row.seq });
      break;
    }
    if (status === 422) {
      store.mark(row.seq, "rejected");
      log("bus.rejected", { bus_seq: row.seq, type: row.type });
      continue;
    }
    if (status !== 200) {
      log("bus.error", { bus_seq: row.seq, status });
      continue;
    }
    crash(3);
    store.mark(row.seq, "published");
    log("bus.published", { bus_seq: row.seq, type: row.type });
    crash(4);
    did += 1;
  }
  return did;
}

// For Support: the event a merchant's idempotency key was first accepted as.
function eventByKey(url: string): Reply {
  const query = new URL(url, "http://api").searchParams;
  const merchantId = query.get("merchant_id");
  const key = query.get("idempotency_key");
  if (!merchantId || !key)
    return [400, { error: "merchant_id and idempotency_key are required" }];
  const found = store.eventByKey(merchantId, key);
  return found ? [200, found] : [404, { error: "no such event" }];
}

async function route(method: string, url: string, body: Body): Promise<Reply> {
  if (method === "GET" && url === "/__admin/health")
    return [200, { service: "api", release: RELEASE }];
  if (method === "GET" && url === "/__admin/state")
    return [200, { service: "api", now: now(), ...store.summary() }];
  if (method === "POST" && url === "/__admin/clock") {
    clockNow = Number(body.now);
    return [200, { now: clockNow }];
  }
  if (method === "POST" && url === "/__admin/tick")
    return [200, { did: await publish() }];
  if (method === "POST" && url === "/__admin/crash_at") {
    crashPoint = Number(body.point);
    crashHits = Number(body.hits ?? 1);
    return [200, { armed: crashPoint }];
  }
  if (method === "POST" && url === "/v1/endpoints") return createEndpoint(body);
  if (method === "POST" && url === "/v1/events") return acceptEvent(body);

  const endpoint = /^\/v1\/endpoints\/([^/]+)$/.exec(url);
  if (endpoint && method === "PUT") return updateEndpoint(endpoint[1], body);
  if (endpoint && method === "GET") {
    const found = store.endpoint(endpoint[1]);
    return found ? [200, found] : [404, { error: "no such endpoint" }];
  }
  if (method === "GET" && url.startsWith("/v1/events?")) return eventByKey(url);
  const event = /^\/v1\/events\/([^/]+)$/.exec(url);
  if (event && method === "GET") {
    const found = store.event(event[1]);
    return found ? [200, found] : [404, { error: "no such event" }];
  }
  const again = /^\/v1\/events\/([^/]+)\/resend$/.exec(url);
  if (again && method === "POST") return resend(again[1], body);
  return [404, { error: "not found" }];
}

// One request at a time, so a tick never interleaves with an accept.
let queue: Promise<void> = Promise.resolve();

const server = http.createServer((request, response) => {
  const chunks: Buffer[] = [];
  request.on("data", (chunk: Buffer) => chunks.push(chunk));
  request.on("end", () => {
    queue = queue.then(async () => {
      let reply: Reply;
      try {
        const raw = Buffer.concat(chunks).toString("utf8");
        const body = raw ? JSON.parse(raw) : {};
        reply = await route(request.method ?? "GET", request.url ?? "/", body);
      } catch (problem) {
        if (problem instanceof Invalid || problem instanceof SyntaxError) {
          reply = [400, { error: problem.message }];
        } else {
          log("request.failed", { error: String(problem) });
          reply = [500, { error: "internal error" }];
        }
      }
      const data = JSON.stringify(reply[1]);
      response.writeHead(reply[0], {
        "content-type": "application/json",
        "content-length": Buffer.byteLength(data),
      });
      response.end(data);
    });
  });
});

server.listen(portBase + 1, "127.0.0.1", () =>
  log("api.started", { release: RELEASE }),
);
