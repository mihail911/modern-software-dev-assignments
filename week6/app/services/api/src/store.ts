// Storage for the api service: endpoints and their history, events, and the
// outbox of bus messages waiting to be published to the router.

import { createHash } from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { DatabaseSync } from "node:sqlite";

export interface Endpoint {
  id: string;
  merchant_id: string;
  url: string;
  secret: string;
  event_types: string[];
  enabled: boolean;
  version: number;
}

export interface EventRecord {
  id: string;
  merchant_id: string;
  type: string;
  created: number;
  data: Record<string, unknown>;
}

export interface OutboxRow {
  seq: number;
  type: string;
  at: number;
  data: string;
}

const SCHEMA = `
create table if not exists meta (key text primary key, value integer not null);
create table if not exists endpoints (
  id text primary key,
  merchant_id text not null,
  url text not null,
  secret text not null,
  event_types text not null,
  enabled integer not null,
  version integer not null,
  updated_at integer not null
);
create table if not exists endpoint_history (
  endpoint_id text not null,
  version integer not null,
  config text not null,
  at integer not null,
  primary key (endpoint_id, version)
);
create table if not exists events (
  id text primary key,
  merchant_id text not null,
  type text not null,
  created integer not null,
  data text not null,
  idempotency_key text
);
create table if not exists outbox (
  seq integer primary key autoincrement,
  type text not null,
  at integer not null,
  data text not null,
  status text not null default 'pending'
);
`;

export class Store {
  db: DatabaseSync;

  constructor(stateDir: string) {
    const dir = path.join(stateDir, "api");
    fs.mkdirSync(dir, { recursive: true });
    this.db = new DatabaseSync(path.join(dir, "api.db"));
    this.db.exec("pragma journal_mode = wal");
    this.db.exec("pragma synchronous = normal");
    this.db.exec(SCHEMA);
  }

  transaction<T>(work: () => T): T {
    this.db.exec("begin");
    try {
      const result = work();
      this.db.exec("commit");
      return result;
    } catch (problem) {
      this.db.exec("rollback");
      throw problem;
    }
  }

  // Opaque ids from a stored counter. Call inside a transaction.
  nextId(prefix: string): string {
    this.db
      .prepare(
        "insert into meta (key, value) values (?, 1) on conflict (key) do update set value = value + 1",
      )
      .run(prefix);
    const row = this.db
      .prepare("select value from meta where key = ?")
      .get(prefix) as { value: number };
    const digest = createHash("sha256")
      .update(`switchyard:${prefix}:${row.value}`)
      .digest();
    if (prefix === "whsec") {
      return `whsec_${digest.subarray(0, 24).toString("base64")}`;
    }
    return `${prefix}_${digest.toString("hex").slice(0, 14)}`;
  }

  endpoint(id: string): Endpoint | undefined {
    const row = this.db
      .prepare("select * from endpoints where id = ?")
      .get(id) as Record<string, unknown> | undefined;
    if (!row) return undefined;
    return {
      id: row.id as string,
      merchant_id: row.merchant_id as string,
      url: row.url as string,
      secret: row.secret as string,
      event_types: JSON.parse(row.event_types as string),
      enabled: row.enabled === 1,
      version: row.version as number,
    };
  }

  saveEndpoint(endpoint: Endpoint, now: number): void {
    this.db
      .prepare(
        `insert into endpoints (id, merchant_id, url, secret, event_types, enabled, version, updated_at)
         values (?, ?, ?, ?, ?, ?, ?, ?)
         on conflict (id) do update set url = excluded.url, secret = excluded.secret,
           event_types = excluded.event_types, enabled = excluded.enabled,
           version = excluded.version, updated_at = excluded.updated_at`,
      )
      .run(
        endpoint.id,
        endpoint.merchant_id,
        endpoint.url,
        endpoint.secret,
        JSON.stringify(endpoint.event_types),
        endpoint.enabled ? 1 : 0,
        endpoint.version,
        now,
      );
    this.db
      .prepare(
        "insert into endpoint_history (endpoint_id, version, config, at) values (?, ?, ?, ?)",
      )
      .run(endpoint.id, endpoint.version, JSON.stringify(endpoint), now);
    this.enqueue("endpoint.changed", now, { endpoint });
  }

  event(id: string): EventRecord | undefined {
    const row = this.db.prepare("select * from events where id = ?").get(id) as
      | Record<string, unknown>
      | undefined;
    return row ? this.toEvent(row) : undefined;
  }

  // The first event accepted under this merchant's key.
  eventByKey(merchantId: string, key: string): EventRecord | undefined {
    const row = this.db
      .prepare(
        "select * from events where merchant_id = ? and idempotency_key = ? order by rowid limit 1",
      )
      .get(merchantId, key) as Record<string, unknown> | undefined;
    return row ? this.toEvent(row) : undefined;
  }

  toEvent(row: Record<string, unknown>): EventRecord {
    return {
      id: row.id as string,
      merchant_id: row.merchant_id as string,
      type: row.type as string,
      created: row.created as number,
      data: JSON.parse(row.data as string),
    };
  }

  saveEvent(event: EventRecord, key: string | null, now: number): void {
    this.db
      .prepare(
        "insert into events (id, merchant_id, type, created, data, idempotency_key) values (?, ?, ?, ?, ?, ?)",
      )
      .run(
        event.id,
        event.merchant_id,
        event.type,
        event.created,
        JSON.stringify(event.data),
        key,
      );
    this.enqueue("event.accepted", now, { event });
  }

  enqueue(type: string, now: number, data: unknown): void {
    this.db
      .prepare("insert into outbox (type, at, data) values (?, ?, ?)")
      .run(type, now, JSON.stringify(data));
  }

  pending(): OutboxRow[] {
    return this.db
      .prepare(
        "select seq, type, at, data from outbox where status = 'pending' order by seq",
      )
      .all() as unknown as OutboxRow[];
  }

  mark(seq: number, status: "published" | "rejected"): void {
    this.db
      .prepare("update outbox set status = ? where seq = ?")
      .run(status, seq);
  }

  summary(): Record<string, unknown> {
    const count = (sql: string) =>
      (this.db.prepare(sql).get() as { n: number }).n;
    return {
      events: count("select count(*) as n from events"),
      endpoints: this.db
        .prepare(
          "select id, merchant_id, url, enabled, version from endpoints order by id",
        )
        .all(),
      outbox_pending: count(
        "select count(*) as n from outbox where status = 'pending'",
      ),
      outbox_rejected: count(
        "select count(*) as n from outbox where status = 'rejected'",
      ),
    };
  }
}
