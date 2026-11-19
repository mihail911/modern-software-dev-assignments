// Tests for the api's storage: ids, events and their idempotency keys,
// endpoint history, and the outbox.

import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";

import { type Endpoint, type EventRecord, Store } from "../src/store.ts";

function freshStore(): Store {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "api-test-"));
  const store = new Store(dir);
  // Every row the store holds is printed when a test ends, so a failure shows the data that caused it.
  test.after(() => {
    for (const table of [
      "meta",
      "endpoints",
      "endpoint_history",
      "events",
      "outbox",
    ]) {
      for (const row of store.db.prepare(`select * from ${table}`).all()) {
        console.log(`  ${table}: ${JSON.stringify(row)}`);
      }
    }
    store.db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });
  return store;
}

function event(store: Store, n: number, merchant = "m_acme"): EventRecord {
  return {
    id: store.nextId("evt"),
    merchant_id: merchant,
    type: "payment.captured",
    created: 1793577600 + n,
    data: {
      object: "payment",
      payment_id: `pay_${n}`,
      amount_minor: 100 + n,
      currency: "USD",
    },
  };
}

function endpoint(
  id: string,
  version = 1,
  url = "https://example.test/hooks",
): Endpoint {
  return {
    id,
    merchant_id: "m_acme",
    url,
    secret: "whsec_dGVzdA==",
    event_types: ["*"],
    enabled: true,
    version,
  };
}

test("ids have their prefix and do not repeat", () => {
  const store = freshStore();
  const seen = new Set<string>();
  store.transaction(() => {
    for (let n = 0; n < 300; n++) {
      const id = store.nextId("evt");
      console.log(`  id ${n + 1}: ${id}`);
      assert.match(id, /^evt_[0-9a-f]{14}$/);
      assert.ok(!seen.has(id), `${id} was issued twice`);
      seen.add(id);
    }
  });
});

test("each prefix has its own counter", () => {
  const store = freshStore();
  const first = store.transaction(() => [
    store.nextId("evt"),
    store.nextId("pay"),
    store.nextId("evt"),
  ]);
  assert.equal(new Set(first).size, 3);
  assert.match(first[1], /^pay_/);
});

test("a generated secret is whsec_ and base64", () => {
  const store = freshStore();
  const secret = store.transaction(() => store.nextId("whsec"));
  assert.match(secret, /^whsec_[A-Za-z0-9+/]{32}$/);
});

test("saving an event queues one bus message for it", () => {
  const store = freshStore();
  const saved = store.transaction(() => {
    const record = event(store, 1);
    store.saveEvent(record, null, 1793577601);
    return record;
  });
  const pending = store.pending();
  assert.equal(pending.length, 1);
  assert.equal(pending[0].type, "event.accepted");
  assert.deepEqual(JSON.parse(pending[0].data).event, saved);
  assert.deepEqual(store.event(saved.id), saved);
});

test("an event is found by its merchant's idempotency key only", () => {
  const store = freshStore();
  const saved = store.transaction(() => {
    const record = event(store, 1);
    store.saveEvent(record, "order-17", 1793577601);
    return record;
  });
  assert.deepEqual(store.eventByKey("m_acme", "order-17"), saved);
  assert.equal(store.eventByKey("m_bolt", "order-17"), undefined);
  assert.equal(store.eventByKey("m_acme", "order-18"), undefined);
});

test("a failed transaction leaves nothing behind", () => {
  const store = freshStore();
  assert.throws(() =>
    store.transaction(() => {
      store.saveEvent(event(store, 1), null, 1793577601);
      throw new Error("stop");
    }),
  );
  assert.equal(store.pending().length, 0);
  assert.equal(store.summary().events, 0);
});

test("every endpoint change is kept in its history and queued", () => {
  const store = freshStore();
  store.transaction(() => store.saveEndpoint(endpoint("ep_a", 1), 1793577600));
  store.transaction(() =>
    store.saveEndpoint(
      endpoint("ep_a", 2, "https://example.test/moved"),
      1793577700,
    ),
  );
  assert.equal(store.endpoint("ep_a")?.url, "https://example.test/moved");
  assert.equal(store.endpoint("ep_a")?.version, 2);
  const history = store.db
    .prepare(
      "select version from endpoint_history where endpoint_id = 'ep_a' order by version",
    )
    .all();
  assert.deepEqual(
    history.map((row) => row.version),
    [1, 2],
  );
  assert.deepEqual(
    store.pending().map((row) => row.type),
    ["endpoint.changed", "endpoint.changed"],
  );
});

test("the outbox is read oldest first and a published message leaves it", () => {
  const store = freshStore();
  for (let n = 0; n < 200; n++) {
    store.transaction(() =>
      store.saveEvent(event(store, n), null, 1793577600 + n),
    );
  }
  const pending = store.pending();
  assert.equal(pending.length, 200);
  for (let n = 1; n < pending.length; n++)
    assert.ok(pending[n].seq > pending[n - 1].seq);
  for (const row of pending.slice(0, 150)) store.mark(row.seq, "published");
  store.mark(pending[150].seq, "rejected");
  assert.equal(store.pending().length, 49);
  assert.equal(store.pending()[0].seq, pending[151].seq);
  assert.equal(store.summary().outbox_rejected, 1);
});
