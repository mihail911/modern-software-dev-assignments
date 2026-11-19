// Tests for idempotency keys (ADR-0013).

import assert from "node:assert/strict";
import { test } from "node:test";

import { KEY_WINDOW_SECONDS, RecentKeys } from "../src/keys.ts";

const T = 1790812800;

test("a key is found again by the merchant that sent it", () => {
  const keys = new RecentKeys();
  keys.remember("m_acme", "order-17", "evt_1", T);
  assert.equal(keys.find("m_acme", "order-17", T + 5), "evt_1");
  assert.equal(keys.find("m_bolt", "order-17", T + 5), undefined);
  assert.equal(keys.find("m_acme", "order-18", T + 5), undefined);
});

test("a key is kept for the whole window", () => {
  const keys = new RecentKeys();
  keys.remember("m_acme", "order-17", "evt_1", T);
  assert.equal(keys.find("m_acme", "order-17", T + KEY_WINDOW_SECONDS - 1), "evt_1");
});

test("a key is forgotten once its window has passed", () => {
  const keys = new RecentKeys();
  keys.remember("m_acme", "order-17", "evt_1", T);
  assert.equal(keys.find("m_acme", "order-17", T + KEY_WINDOW_SECONDS), undefined);
  assert.equal(keys.size, 0);
});

// Each api process answers from the keys it has seen (ADR-0013).
test("keys are kept per cache", () => {
  const one = new RecentKeys();
  one.remember("m_acme", "order-17", "evt_1", T);
  const other = new RecentKeys();
  assert.equal(other.find("m_acme", "order-17", T + 30), undefined);
  assert.equal(other.size, 0);
});
