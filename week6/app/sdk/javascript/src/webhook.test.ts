import * as assert from "node:assert/strict";
import { test } from "node:test";
import * as vm from "node:vm";
import * as base64 from "@stablelib/base64";
import * as ed25519 from "@stablelib/ed25519";
import * as utf8 from "@stablelib/utf8";
import * as sha256 from "fast-sha256";

import {
  Webhook,
  WebhookAsymmetric,
  WebhookAsymmetricSigner,
  WebhookVerificationError,
} from "./index";

const defaultMsgID = "msg_p5jXN8AQM9LWM0D4loKWxJek";
const defaultPayload = `{"test": 2432232314}`;
const defaultSecret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw";
const defaultSecretKey = base64.decode(defaultSecret.slice("whsec_".length));

const tolerance_in_ms = 5 * 60 * 1000;

class TestPayload {
  public id: string;
  public timestamp: number;
  public header: Record<string, string>;
  public secret: string;
  public payload: string;
  public signature: string;

  public constructor(timestamp = Date.now()) {
    this.id = defaultMsgID;
    this.timestamp = Math.floor(timestamp / 1000);

    this.payload = defaultPayload;
    this.secret = defaultSecret;

    const toSign = utf8.encode(`${this.id}.${this.timestamp}.${this.payload}`);
    this.signature = base64.encode(sha256.hmac(defaultSecretKey, toSign));

    this.header = {
      "webhook-id": this.id,
      "webhook-signature": `v1,${this.signature}`,
      "webhook-timestamp": this.timestamp.toString(),
    };
  }
}

test("empty key raises error", () => {
  assert.throws(() => {
    new Webhook("");
  }, Error);
  assert.throws(() => {
    new Webhook(undefined as any);
  }, Error);
  assert.throws(() => {
    new Webhook(null as any);
  }, Error);
  assert.throws(() => {
    new Webhook("whsec_");
  }, Error);
});

test("missing id raises error", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  delete testPayload.header["webhook-id"];

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);
});

test("missing timestamp raises error", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  delete testPayload.header["webhook-timestamp"];

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);
});

test("invalid timestamp throws error", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  testPayload.header["webhook-timestamp"] = "hello";

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);
});

test("missing signature raises error", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  delete testPayload.header["webhook-signature"];

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);
});

test("invalid signature throws error", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  testPayload.header["webhook-signature"] = "v1,dawfeoifkpqwoekfpqoekf";

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);
});

test("partial signature throws error", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  testPayload.header["webhook-signature"] = testPayload.header["webhook-signature"].slice(
    0,
    8
  );

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);

  testPayload.header["webhook-signature"] = "v1,";

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);
});

test("valid signature is valid and returns valid json", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");
  const testPayload = new TestPayload();

  const result = wh.verify(testPayload.payload, testPayload.header);
  assert.deepStrictEqual(result, { test: 2432232314 });
});

test("valid signature is valid without returning json", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");
  const testPayload = new TestPayload();

  const result = wh.verify(testPayload.payload, testPayload.header, { jsonParse: false });
  assert.strictEqual(result, undefined);
});

test("valid unbranded signature is valid and returns valid json", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  const unbrandedHeaders: Record<string, string> = {
    "webhook-id": testPayload.header["webhook-id"],
    "webhook-signature": testPayload.header["webhook-signature"],
    "webhook-timestamp": testPayload.header["webhook-timestamp"],
  };
  testPayload.header = unbrandedHeaders;

  wh.verify(testPayload.payload, testPayload.header);
});

test("old timestamp fails", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload(Date.now() - tolerance_in_ms - 1000);

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);
});

test("new timestamp fails", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload(Date.now() + tolerance_in_ms + 1000);

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);
});

test("multi sig payload is valid", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  const sigs = [
    "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
    "v2,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
    testPayload.header["webhook-signature"], // valid signature
    "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
  ];
  testPayload.header["webhook-signature"] = sigs.join(" ");

  wh.verify(testPayload.payload, testPayload.header);
});

test("malformed signatures are skipped", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  const sigs = [
    "v1",
    "v1,not-base64!",
    testPayload.header["webhook-signature"], // valid signature
  ];
  testPayload.header["webhook-signature"] = sigs.join(" ");

  wh.verify(testPayload.payload, testPayload.header);
});

test("valid signature with an unknown version fails", () => {
  const wh = new Webhook("MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const testPayload = new TestPayload();
  testPayload.header["webhook-signature"] = `v2,${testPayload.signature}`;

  assert.throws(() => {
    wh.verify(testPayload.payload, testPayload.header);
  }, WebhookVerificationError);
});

test("verification works with and without signature prefix", () => {
  const testPayload = new TestPayload();

  let wh = new Webhook(defaultSecret.slice("whsec_".length));
  wh.verify(testPayload.payload, testPayload.header);

  wh = new Webhook(defaultSecret);
  wh.verify(testPayload.payload, testPayload.header);
});

test("sign function works", () => {
  const key = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw";
  const msgId = "msg_p5jXN8AQM9LWM0D4loKWxJek";
  const timestamp = new Date(1614265330 * 1000);
  const payload = '{"test": 2432232314}';
  const expected = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=";

  const wh = new Webhook(key);

  const signature = wh.sign(msgId, timestamp, payload);
  assert.equal(signature, expected);
});

test("empty payload returns undefined", () => {
  const wh = new Webhook(defaultSecret);

  const msgId = defaultMsgID;
  const timestamp = Math.floor(Date.now() / 1000);
  const payload = "";

  const toSign = utf8.encode(`${msgId}.${timestamp}.${payload}`);
  const signature = base64.encode(sha256.hmac(defaultSecretKey, toSign));

  const header = {
    "webhook-id": msgId,
    "webhook-signature": `v1,${signature}`,
    "webhook-timestamp": timestamp.toString(),
  };

  const result = wh.verify(payload, header);
  assert.equal(result, undefined);
});

// Every byte value, so the payload is not valid UTF-8. The expected signature is the Go
// library's for the same inputs, which signs the payload bytes as given.
const binaryPayload = Uint8Array.from({ length: 256 }, (_, i) => i);
const binaryPayloadSignature = "v1,sEcZ042G8Crqgxbybb3y0eDxQR3TcfYsk11I1XULhSI=";
const signTimestamp = new Date(1614265330 * 1000);

// Headers for `payload` signed now. sign() itself is checked against the known-answer
// vectors above, so the verify tests can use it.
function signedHeader(wh: Webhook, payload: string | Uint8Array): Record<string, string> {
  const timestamp = new Date();
  return {
    "webhook-id": defaultMsgID,
    "webhook-signature": wh.sign(defaultMsgID, timestamp, payload),
    "webhook-timestamp": Math.floor(timestamp.getTime() / 1000).toString(),
  };
}

test("sign signs a binary payload byte for byte", () => {
  const wh = new Webhook(defaultSecret);

  assert.equal(
    wh.sign(defaultMsgID, signTimestamp, binaryPayload),
    binaryPayloadSignature
  );
  assert.equal(
    wh.sign(defaultMsgID, signTimestamp, Buffer.from(binaryPayload)),
    binaryPayloadSignature
  );
});

test("sign signs only the bytes of a Uint8Array view", () => {
  const wh = new Webhook(defaultSecret);
  const padded = new Uint8Array(binaryPayload.length + 4);
  padded.set(binaryPayload, 2);

  assert.equal(
    wh.sign(defaultMsgID, signTimestamp, padded.subarray(2, 2 + binaryPayload.length)),
    binaryPayloadSignature
  );
});

test("sign gives the same signature for a string and its UTF-8 bytes", () => {
  const wh = new Webhook(defaultSecret);
  const payload = '{"test": "ünïcödé ✓"}';

  assert.equal(
    wh.sign(defaultMsgID, signTimestamp, Buffer.from(payload)),
    wh.sign(defaultMsgID, signTimestamp, payload)
  );
  assert.equal(
    wh.sign(defaultMsgID, signTimestamp, new TextEncoder().encode(payload)),
    wh.sign(defaultMsgID, signTimestamp, payload)
  );
});

test("sign accepts a Uint8Array from another realm", () => {
  const wh = new Webhook(defaultSecret);
  const foreign = vm.runInNewContext(
    "new Uint8Array(256).map((_, i) => i)"
  ) as Uint8Array;
  assert.equal(foreign instanceof Uint8Array, false);

  assert.equal(wh.sign(defaultMsgID, signTimestamp, foreign), binaryPayloadSignature);
});

test("sign rejects a payload that is not a string or bytes", () => {
  const wh = new Webhook(defaultSecret);

  assert.throws(() => {
    wh.sign(defaultMsgID, signTimestamp, { test: 1 } as any);
  }, Error);
  assert.throws(() => {
    wh.sign(defaultMsgID, signTimestamp, Uint16Array.from([1, 2]) as any);
  }, Error);
  assert.throws(() => {
    wh.sign(defaultMsgID, signTimestamp, new DataView(binaryPayload.buffer) as any);
  }, Error);
});

test("binary payload is valid", () => {
  const wh = new Webhook(defaultSecret);
  const header = signedHeader(wh, binaryPayload);

  assert.equal(wh.verify(binaryPayload, header, { jsonParse: false }), undefined);
  assert.equal(
    wh.verify(Buffer.from(binaryPayload), header, { jsonParse: false }),
    undefined
  );
});

test("binary payload with a changed byte fails", () => {
  const wh = new Webhook(defaultSecret);
  const header = signedHeader(wh, binaryPayload);
  const changed = Uint8Array.from(binaryPayload);
  changed[255] = 0;

  assert.throws(() => {
    wh.verify(changed, header, { jsonParse: false });
  }, WebhookVerificationError);
});

test("JSON payload as Uint8Array is valid and returns valid json", () => {
  const wh = new Webhook(defaultSecret);
  const testPayload = new TestPayload();

  const json = wh.verify(
    new TextEncoder().encode(testPayload.payload),
    testPayload.header
  );
  assert.deepEqual(json, { test: 2432232314 });
});

test("JSON payload as Uint8Array with a UTF-8 BOM returns valid json", () => {
  const wh = new Webhook(defaultSecret);
  const json = new TextEncoder().encode('{"test": 2432232314}');
  const payload = new Uint8Array([0xef, 0xbb, 0xbf, ...json]);

  // The BOM is signed like any other bytes; only the JSON parse skips it.
  assert.deepEqual(wh.verify(payload, signedHeader(wh, payload)), {
    test: 2432232314,
  });
  assert.throws(() => {
    wh.verify(json, signedHeader(wh, payload));
  }, WebhookVerificationError);
});

test("empty Uint8Array payload returns undefined", () => {
  const wh = new Webhook(defaultSecret);
  const payload = new Uint8Array();

  assert.equal(wh.verify(payload, signedHeader(wh, payload)), undefined);
});

test("challenge response matches test vector", () => {
  const wh = new Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  const result = wh.endpointVerificationResponse(
    "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
  );
  assert.equal(result, "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=");
});

test("verify challenge response accepts test vector", () => {
  const wh = new Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  wh.verifyEndpointVerificationResponse(
    "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=",
    "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="
  );
});

test("verify challenge response rejects wrong challenge", () => {
  const wh = new Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  assert.throws(() => {
    wh.verifyEndpointVerificationResponse(
      "1njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=",
      "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="
    );
  }, WebhookVerificationError);
});

test("verify challenge response rejects lists", () => {
  const wh = new Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");

  assert.throws(() => {
    wh.verifyEndpointVerificationResponse(
      "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=",
      "v2,abc v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="
    );
  }, WebhookVerificationError);
});

test("challenge response depends on the secret", () => {
  const wh = new Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSx");

  const result = wh.endpointVerificationResponse(
    "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
  );
  assert.notEqual(result, "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=");
});

// Asymmetric signatures (v2a): test vectors from the spec

const asymSecretKey = "whsk_4uKPjji/dhB8w6GW3YYDnIMYvXcdHrvH6lidTr4Z+EY=";
const asymPublicKey = "whpk_uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w=";
const asymEndpointId = "ep_2fVZdC8Kx9TR4ZoCqHWt3NnLgRb";
const asymMsgId = "msg_2KWPBgLlAfxdpx2AI54pPJ85f4W";
const asymTimestamp = 1674087231;
const asymPayload = `{"type":"contact.created","timestamp":"2022-11-03T20:26:10.344522Z","data":{"id":"1f81eb52-5198-4599-803e-771906343485"}}`;
const asymDigest = "YXCi5iN+rjB+iEb6bvg1XbWbVBL+r2S+as3k3n1Ih/A=";
const asymSignature =
  "v2a,kG9XaxZjAEpKB8KeQQb5G1Xp73YH5Rf5gxoWe7cg1kYMiyspdtZXe9w+dAwKeOurAwxfqmlOG/QR4m9s37DzAg==";
const asymChallenge = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=";
const asymChallengeResponse = "v2a,t0RDFhzk7OtVzgvMsjuXnodSWuby2BVX+zPqw2/ksUk=";

const asymKeyPair = ed25519.generateKeyPairFromSeed(
  base64.decode(asymSecretKey.slice("whsk_".length))
);
const asymSigner = new WebhookAsymmetricSigner(asymSecretKey);
// Another key, e.g. the producer's newer key after a rotation
const otherSigner = new WebhookAsymmetricSigner(
  `whsk_${base64.encode(new Uint8Array(32).fill(7))}`
);
const otherPublicKey = otherSigner.publicKey();

function asymSign(
  secretKey: Uint8Array,
  endpointId: string,
  msgId: string,
  timestamp: number,
  payload: string | Uint8Array
): string {
  const prefix = utf8.encode(`${endpointId}.${msgId}.${timestamp}.`);
  const bytes = typeof payload === "string" ? utf8.encode(payload) : payload;
  const content = new Uint8Array(prefix.length + bytes.length);
  content.set(prefix);
  content.set(bytes, prefix.length);
  return `v2a,${base64.encode(ed25519.sign(secretKey, sha256.hash(content)))}`;
}

function asymHeaders(
  payload: string | Uint8Array = asymPayload,
  signer: WebhookAsymmetricSigner = asymSigner,
  endpointId: string = asymEndpointId,
  timestamp: number = Math.floor(Date.now() / 1000)
): Record<string, string> {
  return {
    "webhook-id": asymMsgId,
    "webhook-timestamp": timestamp.toString(),
    "webhook-signature": signer.sign(
      endpointId,
      asymMsgId,
      new Date(timestamp * 1000),
      payload
    ),
  };
}

test("asymmetric spec test vector matches the ed25519 primitive", () => {
  assert.equal(
    `whpk_${base64.encode(asymKeyPair.publicKey)}`,
    asymPublicKey,
    "public key"
  );
  const content = utf8.encode(
    `${asymEndpointId}.${asymMsgId}.${asymTimestamp}.${asymPayload}`
  );
  const digest = sha256.hash(content);
  assert.equal(base64.encode(digest), asymDigest);
  assert.ok(
    ed25519.verify(
      asymKeyPair.publicKey,
      digest,
      base64.decode(asymSignature.slice("v2a,".length))
    )
  );
  assert.equal(
    asymSign(
      asymKeyPair.secretKey,
      asymEndpointId,
      asymMsgId,
      asymTimestamp,
      asymPayload
    ),
    asymSignature
  );
});

test("asymmetric signer matches the spec test vector", () => {
  assert.equal(asymSigner.publicKey(), asymPublicKey);
  assert.equal(
    asymSigner.sign(
      asymEndpointId,
      asymMsgId,
      new Date(asymTimestamp * 1000),
      asymPayload
    ),
    asymSignature
  );
  assert.equal(
    asymSigner.sign(
      asymEndpointId,
      asymMsgId,
      new Date(asymTimestamp * 1000),
      utf8.encode(asymPayload)
    ),
    asymSignature
  );
});

test("asymmetric signedContentHash matches the spec test vector", () => {
  const digest = WebhookAsymmetric.signedContentHash(
    asymEndpointId,
    asymMsgId,
    new Date(asymTimestamp * 1000),
    asymPayload
  );
  assert.equal(digest.length, 32);
  assert.equal(base64.encode(digest), asymDigest);
});

test("asymmetric HSM flow produces the spec test vector", () => {
  const digest = WebhookAsymmetric.signedContentHash(
    asymEndpointId,
    asymMsgId,
    new Date(asymTimestamp * 1000),
    asymPayload
  );
  // Stand-in for the HSM's sign call
  const signature = ed25519.sign(asymKeyPair.secretKey, digest);
  assert.equal(WebhookAsymmetric.formatSignature(signature), asymSignature);
});

test("asymmetric formatSignature round-trips and rejects wrong lengths", () => {
  const signature = new Uint8Array(64).fill(0xab);
  const formatted = WebhookAsymmetric.formatSignature(signature);
  assert.ok(formatted.startsWith("v2a,"));
  assert.deepEqual(base64.decode(formatted.slice("v2a,".length)), signature);
  for (const length of [0, 32, 63, 65]) {
    assert.throws(() => WebhookAsymmetric.formatSignature(new Uint8Array(length)), Error);
  }
});

test("asymmetric formatPublicKey accepts raw and DER keys and rejects others", () => {
  const raw = base64.decode(asymPublicKey.slice("whpk_".length));
  assert.equal(raw.length, 32);
  assert.equal(WebhookAsymmetric.formatPublicKey(raw), asymPublicKey);
  const prefix = Uint8Array.of(
    0x30,
    0x2a,
    0x30,
    0x05,
    0x06,
    0x03,
    0x2b,
    0x65,
    0x70,
    0x03,
    0x21,
    0x00
  );
  const der = new Uint8Array([...prefix, ...raw]);
  assert.equal(WebhookAsymmetric.formatPublicKey(der), asymPublicKey);
  const badDer = der.slice();
  badDer[0] = 0x31;
  for (const key of [raw.subarray(0, 31), new Uint8Array([...raw, 0]), badDer]) {
    assert.throws(() => WebhookAsymmetric.formatPublicKey(key), Error);
  }
});

test("asymmetric signer constructor rejects invalid keys", () => {
  assert.throws(
    () => new WebhookAsymmetricSigner(asymSecretKey.slice("whsk_".length)),
    Error
  );
  assert.throws(() => new WebhookAsymmetricSigner(asymPublicKey), Error);
  assert.throws(() => new WebhookAsymmetricSigner("whsk_!!!"), Error);
  assert.throws(() => new WebhookAsymmetricSigner("whsk_"), Error);
  assert.throws(
    () => new WebhookAsymmetricSigner(`whsk_${base64.encode(new Uint8Array(31))}`),
    Error
  );
  assert.throws(
    () => new WebhookAsymmetricSigner(`whsk_${base64.encode(new Uint8Array(64))}`),
    Error
  );
  assert.throws(() => new WebhookAsymmetricSigner(undefined as any), Error);
});

test("asymmetric signing rejects invalid endpoint ids", () => {
  for (const endpointId of ["", "ep.1"]) {
    assert.throws(
      () => asymSigner.sign(endpointId, asymMsgId, new Date(), asymPayload),
      Error
    );
    assert.throws(
      () =>
        WebhookAsymmetric.signedContentHash(
          endpointId,
          asymMsgId,
          new Date(),
          asymPayload
        ),
      Error
    );
  }
});

test("asymmetric sign then verify round-trips", () => {
  const timestamp = new Date();
  const signature = asymSigner.sign(asymEndpointId, asymMsgId, timestamp, asymPayload);
  const wh = new WebhookAsymmetric(asymEndpointId, [asymSigner.publicKey()]);
  const headers = {
    "webhook-id": asymMsgId,
    "webhook-timestamp": Math.floor(timestamp.getTime() / 1000).toString(),
    "webhook-signature": signature,
  };
  assert.deepEqual(wh.verify(asymPayload, headers), JSON.parse(asymPayload));
});

test("asymmetric verify accepts the spec test vector", (t) => {
  t.mock.timers.enable({ apis: ["Date"], now: asymTimestamp * 1000 });
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  const headers = {
    "webhook-id": asymMsgId,
    "webhook-timestamp": asymTimestamp.toString(),
    "webhook-signature": `v1,K5oZfzN95Z9UVu1EsfQmfVNQhnkZ2pj9o9NDN/H/pI4= ${asymSignature}`,
  };
  assert.deepEqual(wh.verify(asymPayload, headers), JSON.parse(asymPayload));
});

test("asymmetric valid signature is valid and returns valid json", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  assert.deepEqual(wh.verify(asymPayload, asymHeaders()), JSON.parse(asymPayload));
  assert.equal(wh.verify(asymPayload, asymHeaders(), { jsonParse: false }), undefined);
});

test("asymmetric verify is case-insensitive for header names", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  const headers = asymHeaders();
  const upper: Record<string, string> = {};
  for (const [key, value] of Object.entries(headers)) {
    upper[key.toUpperCase()] = value;
  }
  assert.deepEqual(wh.verify(asymPayload, upper), JSON.parse(asymPayload));
});

test("asymmetric binary payload is valid", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  const payload = Uint8Array.from([0xff, 0xfe, 0x00, 0x80]);
  assert.equal(wh.verify(payload, asymHeaders(payload), { jsonParse: false }), undefined);
});

test("asymmetric valid signature after other signatures is valid", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  const headers = asymHeaders();
  headers["webhook-signature"] =
    `v1,K5oZfzN95Z9UVu1EsfQmfVNQhnkZ2pj9o9NDN/H/pI4= v1a,abc v2a,!!invalid v2a,AAAA ${headers["webhook-signature"]}`;
  assert.deepEqual(wh.verify(asymPayload, headers), JSON.parse(asymPayload));
});

test("asymmetric signature by the second key in the list is valid", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [otherPublicKey, asymPublicKey]);
  assert.deepEqual(wh.verify(asymPayload, asymHeaders()), JSON.parse(asymPayload));
  assert.deepEqual(
    wh.verify(asymPayload, asymHeaders(asymPayload, otherSigner)),
    JSON.parse(asymPayload)
  );
});

test("asymmetric signature by an untrusted key fails", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  assert.throws(() => {
    wh.verify(asymPayload, asymHeaders(asymPayload, otherSigner));
  }, WebhookVerificationError);
});

test("asymmetric wrong endpoint id fails", () => {
  const wh = new WebhookAsymmetric("ep_other", [asymPublicKey]);
  assert.throws(() => {
    wh.verify(asymPayload, asymHeaders());
  }, WebhookVerificationError);
});

test("asymmetric tampered payload fails", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  const headers = asymHeaders();
  assert.throws(() => {
    wh.verify(`${asymPayload} `, headers);
  }, WebhookVerificationError);
});

test("asymmetric v1 signature only fails", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  const headers = asymHeaders();
  headers["webhook-signature"] = headers["webhook-signature"].replace("v2a,", "v1,");
  assert.throws(() => {
    wh.verify(asymPayload, headers);
  }, WebhookVerificationError);
});

test("asymmetric missing headers raise error", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  for (const header of ["webhook-id", "webhook-timestamp", "webhook-signature"]) {
    const headers = asymHeaders();
    delete headers[header];
    assert.throws(
      () => {
        wh.verify(asymPayload, headers);
      },
      new WebhookVerificationError("Missing required headers"),
      header
    );
  }
});

test("asymmetric old timestamp fails", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  const timestamp = Math.floor((Date.now() - tolerance_in_ms) / 1000) - 5;
  assert.throws(() => {
    wh.verify(
      asymPayload,
      asymHeaders(asymPayload, asymSigner, asymEndpointId, timestamp)
    );
  }, new WebhookVerificationError("Message timestamp too old"));
});

test("asymmetric new timestamp fails", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey]);
  const timestamp = Math.floor((Date.now() + tolerance_in_ms) / 1000) + 5;
  assert.throws(() => {
    wh.verify(
      asymPayload,
      asymHeaders(asymPayload, asymSigner, asymEndpointId, timestamp)
    );
  }, new WebhookVerificationError("Message timestamp too new"));
});

test("asymmetric constructor rejects invalid arguments", () => {
  assert.throws(() => new WebhookAsymmetric(asymEndpointId, []), Error);
  assert.throws(() => new WebhookAsymmetric(asymEndpointId, undefined as any), Error);
  assert.throws(
    () => new WebhookAsymmetric(asymEndpointId, [asymPublicKey.slice("whpk_".length)]),
    Error
  );
  assert.throws(
    () =>
      new WebhookAsymmetric(asymEndpointId, [asymSecretKey.replace("whsk_", "whsec_")]),
    Error
  );
  assert.throws(() => new WebhookAsymmetric(asymEndpointId, ["whpk_"]), Error);
  assert.throws(
    () =>
      new WebhookAsymmetric(asymEndpointId, [
        `whpk_${base64.encode(new Uint8Array(31))}`,
      ]),
    Error
  );
  assert.throws(
    () =>
      new WebhookAsymmetric(asymEndpointId, [
        `whpk_${base64.encode(new Uint8Array(33))}`,
      ]),
    Error
  );
  assert.throws(() => new WebhookAsymmetric(asymEndpointId, ["whpk_!!!"]), Error);
  assert.throws(
    () => new WebhookAsymmetric(asymEndpointId, [asymPublicKey, "invalid"]),
    Error
  );
  assert.throws(() => new WebhookAsymmetric("", [asymPublicKey]), Error);
  assert.throws(() => new WebhookAsymmetric("ep.1", [asymPublicKey]), Error);
});

test("asymmetric challenge response matches test vector", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [asymPublicKey, otherPublicKey]);
  assert.equal(wh.endpointVerificationResponse(asymChallenge), asymChallengeResponse);
});

test("asymmetric challenge response uses the first key", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [otherPublicKey, asymPublicKey]);
  assert.notEqual(wh.endpointVerificationResponse(asymChallenge), asymChallengeResponse);
});

test("asymmetric challenge response depends on the endpoint id", () => {
  const wh = new WebhookAsymmetric("ep_other", [asymPublicKey]);
  assert.notEqual(wh.endpointVerificationResponse(asymChallenge), asymChallengeResponse);
});

test("asymmetric verify challenge response accepts any of the keys", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [otherPublicKey, asymPublicKey]);
  wh.verifyEndpointVerificationResponse(asymChallenge, asymChallengeResponse);
  wh.verifyEndpointVerificationResponse(
    asymChallenge,
    wh.endpointVerificationResponse(asymChallenge)
  );
});

test("asymmetric verify challenge response rejects wrong responses", () => {
  const wh = new WebhookAsymmetric(asymEndpointId, [otherPublicKey, asymPublicKey]);
  for (const header of [
    "v2a,p02It8u07wVKXUWJB4N8jl1LHoeidCN3wBffJ0vFVcQ=",
    asymChallengeResponse.replace("v2a,", "v1,"),
    `v1,abc ${asymChallengeResponse}`,
    "",
  ]) {
    assert.throws(() => {
      wh.verifyEndpointVerificationResponse(asymChallenge, header);
    }, WebhookVerificationError);
  }
  assert.throws(() => {
    wh.verifyEndpointVerificationResponse(
      "1njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=",
      asymChallengeResponse
    );
  }, WebhookVerificationError);
});
