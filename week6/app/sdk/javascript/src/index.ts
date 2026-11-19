import * as base64 from "@stablelib/base64";
import * as ed25519 from "@stablelib/ed25519";
import * as sha256 from "fast-sha256";
import { timingSafeEqual } from "./timing_safe_equal";

const WEBHOOK_TOLERANCE_IN_SECONDS = 5 * 60; // 5 minutes
// DER prefix of an ed25519 SubjectPublicKeyInfo, followed by the raw 32 byte key
const ED25519_SPKI_PREFIX = Uint8Array.of(
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

class ExtendableError extends Error {
  constructor(message: any) {
    super(message);
    Object.setPrototypeOf(this, ExtendableError.prototype);
    this.name = "ExtendableError";
    this.stack = new Error(message).stack;
  }
}

export class WebhookVerificationError extends ExtendableError {
  constructor(message: string) {
    super(message);
    Object.setPrototypeOf(this, WebhookVerificationError.prototype);
    this.name = "WebhookVerificationError";
  }
}

export interface WebhookUnbrandedRequiredHeaders {
  "webhook-id": string;
  "webhook-timestamp": string;
  "webhook-signature": string;
}

export interface WebhookOptions {
  format?: "raw";
}

export interface VerifyOptions {
  /** Whether to parse the payload as JSON on success (default: true) */
  jsonParse?: boolean;
}

/** The payload as signed: the body bytes exactly as received. A string is encoded as UTF-8. */
function payloadBytes(payload: string | Uint8Array): Uint8Array {
  if (typeof payload === "string") {
    return new TextEncoder().encode(payload);
  }
  // `ArrayBuffer.isView` rather than `instanceof Uint8Array`, so a Buffer or Uint8Array from
  // another realm (e.g. a test runner's VM context) is still accepted. Only byte views: a
  // DataView or a wider typed array (e.g. Uint16Array) is rejected rather than signed as its
  // platform-endian underlying bytes.
  if (ArrayBuffer.isView(payload) && payload.BYTES_PER_ELEMENT === 1) {
    return new Uint8Array(payload.buffer, payload.byteOffset, payload.byteLength);
  }
  throw new Error("Expected payload to be of type string, Buffer or Uint8Array.");
}

/** Normalize the headers (case-insensitive names), and check the timestamp is in tolerance. */
function parseHeaders(
  headers: WebhookUnbrandedRequiredHeaders | Record<string, string>
): { msgId: string; msgSignature: string; timestamp: Date } {
  const normalizedHeaders: Record<string, string> = {};
  for (const key of Object.keys(headers)) {
    normalizedHeaders[key.toLowerCase()] = (headers as Record<string, string>)[key];
  }

  const msgId = normalizedHeaders["webhook-id"];
  const msgSignature = normalizedHeaders["webhook-signature"];
  const msgTimestamp = normalizedHeaders["webhook-timestamp"];

  if (!msgSignature || !msgId || !msgTimestamp) {
    throw new WebhookVerificationError("Missing required headers");
  }

  return { msgId, msgSignature, timestamp: verifyTimestamp(msgTimestamp) };
}

function verifyTimestamp(timestampHeader: string): Date {
  const now = Math.floor(Date.now() / 1000);
  const timestamp = parseInt(timestampHeader, 10);
  if (Number.isNaN(timestamp)) {
    throw new WebhookVerificationError("Invalid Signature Headers");
  }

  if (now - timestamp > WEBHOOK_TOLERANCE_IN_SECONDS) {
    throw new WebhookVerificationError("Message timestamp too old");
  }
  if (timestamp > now + WEBHOOK_TOLERANCE_IN_SECONDS) {
    throw new WebhookVerificationError("Message timestamp too new");
  }
  return new Date(timestamp * 1000);
}

/** The return value of `verify` after successful verification. */
function parsePayload(
  payload: string | Uint8Array,
  bytes: Uint8Array,
  options?: VerifyOptions
): unknown {
  const jsonParse = options?.jsonParse ?? true;
  if (bytes.length === 0) {
    return undefined;
  }
  if (jsonParse) {
    return JSON.parse(
      typeof payload === "string" ? payload : new TextDecoder().decode(bytes)
    );
  } else {
    return undefined;
  }
}

function concatBytes(a: Uint8Array, b: Uint8Array): Uint8Array {
  const ret = new Uint8Array(a.length + b.length);
  ret.set(a);
  ret.set(b, a.length);
  return ret;
}

function validateEndpointId(endpointId: string): void {
  if (typeof endpointId !== "string" || endpointId.length === 0) {
    throw new Error("Endpoint ID can't be empty.");
  }
  if (endpointId.includes(".")) {
    throw new Error('Endpoint ID can\'t contain ".".');
  }
}

/** The response to an endpoint verification challenge, for the given endpoint key. */
function challengeResponse(
  endpointKey: Uint8Array,
  challenge: string,
  identifier: string
): string {
  const encoder = new TextEncoder();
  const verificationKey = sha256.hmac(endpointKey, encoder.encode("wev-v1"));
  const signature = base64.encode(
    sha256.hmac(verificationKey, encoder.encode(challenge))
  );
  return `${identifier},${signature}`;
}

export class Webhook {
  private static prefix = "whsec_";
  private readonly key: Uint8Array;

  constructor(secret: string | Uint8Array, options?: WebhookOptions) {
    if (options?.format === "raw") {
      if (secret instanceof Uint8Array) {
        this.key = secret;
      } else {
        this.key = Uint8Array.from(secret, (c) => c.charCodeAt(0));
      }
    } else {
      if (typeof secret !== "string") {
        throw new Error("Expected secret to be of type string");
      }
      if (secret.startsWith(Webhook.prefix)) {
        secret = secret.substring(Webhook.prefix.length);
      }
      this.key = base64.decode(secret);
    }
    if (this.key.length === 0) {
      throw new Error("Secret can't be empty.");
    }
  }

  /** Verify the given webhook headers against the body bytes (payload).
   *
   * Pass the raw request body exactly as received: a `Buffer` or `Uint8Array` is signed byte
   * for byte, so binary (non-UTF-8) bodies verify. A string is signed as its UTF-8 encoding.
   *
   * @returns After successful verification: if `options.jsonParse` is `true`, returns the
   *     JSON-parsed input data; if `options.jsonParse` is `false`, returns `undefined`
   * @throws `WebhookVerificationError` if one of the required headers is missing,
   *     invalid, too old or too new; or no matching signatures is found
   */
  public verify(
    payload: string | Uint8Array,
    headers: WebhookUnbrandedRequiredHeaders | Record<string, string>,
    options?: VerifyOptions
  ): unknown {
    const { msgId, msgSignature, timestamp } = parseHeaders(headers);

    const bytes = payloadBytes(payload);
    const encoder = new globalThis.TextEncoder();
    const expectedSignature = encoder.encode(this.sign(msgId, timestamp, bytes));

    const passedSignatures = msgSignature.split(" ");
    if (
      !passedSignatures.some((versionedSignature) =>
        timingSafeEqual(encoder.encode(versionedSignature), expectedSignature)
      )
    ) {
      throw new WebhookVerificationError("No matching signature found");
    }

    return parsePayload(payload, bytes, options);
  }

  /** Sign `payload` for the given message id and timestamp.
   *
   * A `Buffer` or `Uint8Array` payload is signed byte for byte, so binary (non-UTF-8) bodies
   * are signed as sent. A string is signed as its UTF-8 encoding.
   */
  public sign(msgId: string, timestamp: Date, payload: string | Uint8Array): string {
    const bytes = payloadBytes(payload);
    const timestampNumber = Math.floor(timestamp.getTime() / 1000);
    const prefix = new TextEncoder().encode(`${msgId}.${timestampNumber}.`);
    const toSign = concatBytes(prefix, bytes);
    const expectedSignature = base64.encode(sha256.hmac(this.key, toSign));
    return `v1,${expectedSignature}`;
  }

  /** Calculate the response to an endpoint verification challenge.
   *
   * @param challenge The value of the `webhook-challenge` request header
   * @returns The value for the `webhook-challenge-response` response header
   */
  public endpointVerificationResponse(challenge: string): string {
    return challengeResponse(this.key, challenge, "v1");
  }

  /** Verify the response to an endpoint verification challenge.
   *
   * @param challenge The challenge sent in the `webhook-challenge` request header
   * @param header The value of the `webhook-challenge-response` response header
   * @throws `WebhookVerificationError` if no matching signature is found
   */
  public verifyEndpointVerificationResponse(challenge: string, header: string): void {
    if (!header) {
      throw new WebhookVerificationError("Missing challenge response");
    }

    const encoder = new globalThis.TextEncoder();
    const expectedSignature = encoder.encode(
      this.endpointVerificationResponse(challenge)
    );
    if (!timingSafeEqual(encoder.encode(header), expectedSignature)) {
      throw new WebhookVerificationError("No matching signature found");
    }
  }
}

/** Verify webhooks signed with asymmetric (`v2a`, ed25519) signatures.
 *
 * Producers can sign using `WebhookAsymmetricSigner`, or with a KMS or an HSM using
 * `WebhookAsymmetric.signedContentHash` and `WebhookAsymmetric.formatSignature`.
 */
export class WebhookAsymmetric {
  private static prefix = "whpk_";
  private readonly publicKeys: Uint8Array[];
  private readonly endpointId: string;

  /**
   * @param endpointId The endpoint ID the producer shared with you together with the
   *     public key. Never take it from the request.
   * @param publicKeys The producer's trusted public keys (`whpk_...`). Webhooks signed with
   *     any of them are accepted, so keys can be rotated by adding the new key to the list.
   *     The first key must be the newest one: it's the one used for endpoint verification.
   */
  constructor(endpointId: string, publicKeys: string[]) {
    if (!Array.isArray(publicKeys) || publicKeys.length === 0) {
      throw new Error("Public keys can't be empty.");
    }
    this.publicKeys = publicKeys.map((publicKey) => {
      if (typeof publicKey !== "string") {
        throw new Error("Expected public key to be of type string");
      }
      if (!publicKey.startsWith(WebhookAsymmetric.prefix)) {
        throw new Error(`Public key must start with "${WebhookAsymmetric.prefix}".`);
      }
      const key = base64.decode(publicKey.substring(WebhookAsymmetric.prefix.length));
      if (key.length !== ed25519.PUBLIC_KEY_LENGTH) {
        throw new Error(
          `Public key must be ${ed25519.PUBLIC_KEY_LENGTH} bytes, got ${key.length}.`
        );
      }
      return key;
    });
    validateEndpointId(endpointId);
    this.endpointId = endpointId;
  }

  /** The SHA-256 hash (raw 32 bytes) of the content signed by `v2a` signatures
   * (`endpoint_id.msg_id.timestamp.payload`).
   *
   * For producers that sign with a KMS or an HSM: sign the returned hash using (pure)
   * ed25519, and pass the signature to `formatSignature`.
   *
   * A `Buffer` or `Uint8Array` payload is hashed byte for byte. A string is hashed as its
   * UTF-8 encoding.
   */
  public static signedContentHash(
    endpointId: string,
    msgId: string,
    timestamp: Date,
    payload: string | Uint8Array
  ): Uint8Array {
    validateEndpointId(endpointId);
    const bytes = payloadBytes(payload);
    const timestampNumber = Math.floor(timestamp.getTime() / 1000);
    const prefix = new TextEncoder().encode(`${endpointId}.${msgId}.${timestampNumber}.`);
    return sha256.hash(concatBytes(prefix, bytes));
  }

  /** Format a raw 64 byte ed25519 signature for the `webhook-signature` header
   * (`v2a,<base64 signature>`).
   */
  public static formatSignature(signature: Uint8Array): string {
    if (
      !(ArrayBuffer.isView(signature) && signature.BYTES_PER_ELEMENT === 1) ||
      signature.byteLength !== ed25519.SIGNATURE_LENGTH
    ) {
      throw new Error(`Signature must be ${ed25519.SIGNATURE_LENGTH} bytes.`);
    }
    return `v2a,${base64.encode(signature)}`;
  }

  /** Format an ed25519 public key as `whpk_<base64 public key>`, to share with consumers.
   *
   * Accepts either the raw 32 byte public key, or the DER encoded SubjectPublicKeyInfo of an
   * ed25519 public key (e.g. what AWS KMS `GetPublicKey` returns).
   */
  public static formatPublicKey(publicKey: Uint8Array): string {
    if (!(ArrayBuffer.isView(publicKey) && publicKey.BYTES_PER_ELEMENT === 1)) {
      throw new Error("Public key must be bytes.");
    }
    let raw = new Uint8Array(
      publicKey.buffer,
      publicKey.byteOffset,
      publicKey.byteLength
    );
    if (raw.byteLength === ED25519_SPKI_PREFIX.length + ed25519.PUBLIC_KEY_LENGTH) {
      if (!ED25519_SPKI_PREFIX.every((b, i) => raw[i] === b)) {
        throw new Error("Public key is not an ed25519 SubjectPublicKeyInfo.");
      }
      raw = raw.subarray(ED25519_SPKI_PREFIX.length);
    }
    if (raw.byteLength !== ed25519.PUBLIC_KEY_LENGTH) {
      throw new Error(`Public key must be ${ed25519.PUBLIC_KEY_LENGTH} bytes.`);
    }
    return `whpk_${base64.encode(raw)}`;
  }

  /** Verify the given webhook headers against the body bytes (payload).
   *
   * Pass the raw request body exactly as received: a `Buffer` or `Uint8Array` is verified
   * byte for byte, so binary (non-UTF-8) bodies verify. A string is verified as its UTF-8
   * encoding.
   *
   * @returns After successful verification: if `options.jsonParse` is `true`, returns the
   *     JSON-parsed input data; if `options.jsonParse` is `false`, returns `undefined`
   * @throws `WebhookVerificationError` if one of the required headers is missing,
   *     invalid, too old or too new; or no matching signatures is found
   */
  public verify(
    payload: string | Uint8Array,
    headers: WebhookUnbrandedRequiredHeaders | Record<string, string>,
    options?: VerifyOptions
  ): unknown {
    const { msgId, msgSignature, timestamp } = parseHeaders(headers);

    const bytes = payloadBytes(payload);
    const digest = WebhookAsymmetric.signedContentHash(
      this.endpointId,
      msgId,
      timestamp,
      bytes
    );

    const found = msgSignature.split(" ").some((versionedSignature) => {
      if (!versionedSignature.startsWith("v2a,")) {
        return false;
      }
      let signature: Uint8Array;
      try {
        signature = base64.decode(versionedSignature.substring("v2a,".length));
      } catch {
        return false;
      }
      if (signature.length !== ed25519.SIGNATURE_LENGTH) {
        return false;
      }
      return this.publicKeys.some((publicKey) =>
        ed25519.verify(publicKey, digest, signature)
      );
    });
    if (!found) {
      throw new WebhookVerificationError("No matching signature found");
    }

    return parsePayload(payload, bytes, options);
  }

  /** Calculate the response to an endpoint verification challenge, using the first (newest)
   * public key.
   *
   * @param challenge The value of the `webhook-challenge` request header
   * @returns The value for the `webhook-challenge-response` response header
   */
  public endpointVerificationResponse(challenge: string): string {
    return this.challengeResponse(this.publicKeys[0], challenge);
  }

  /** Verify the response to an endpoint verification challenge. The response is accepted if
   * it matches any of the public keys.
   *
   * @param challenge The challenge sent in the `webhook-challenge` request header
   * @param header The value of the `webhook-challenge-response` response header
   * @throws `WebhookVerificationError` if no matching signature is found
   */
  public verifyEndpointVerificationResponse(challenge: string, header: string): void {
    if (!header) {
      throw new WebhookVerificationError("Missing challenge response");
    }

    const encoder = new globalThis.TextEncoder();
    const passed = encoder.encode(header);
    for (const publicKey of this.publicKeys) {
      const expected = encoder.encode(this.challengeResponse(publicKey, challenge));
      if (timingSafeEqual(passed, expected)) {
        return;
      }
    }
    throw new WebhookVerificationError("No matching signature found");
  }

  private challengeResponse(publicKey: Uint8Array, challenge: string): string {
    const endpointKey = concatBytes(
      publicKey,
      new TextEncoder().encode(`.${this.endpointId}`)
    );
    return challengeResponse(endpointKey, challenge, "v2a");
  }
}

/** Sign webhooks with asymmetric (`v2a`, ed25519) signatures, using a private key you hold.
 *
 * It's recommended to hold the private key in a KMS or an HSM instead, and sign using
 * `WebhookAsymmetric.signedContentHash` and `WebhookAsymmetric.formatSignature`.
 *
 * A signer isn't tied to an endpoint: one key pair can serve all endpoints.
 */
export class WebhookAsymmetricSigner {
  private static prefix = "whsk_";
  private readonly secretKey: Uint8Array;
  private readonly publicKeyBytes: Uint8Array;

  /**
   * @param secretKey The secret key: `whsk_` followed by the base64 encoded 32 byte ed25519
   *     seed (private key).
   */
  constructor(secretKey: string) {
    if (typeof secretKey !== "string") {
      throw new Error("Expected secret key to be of type string");
    }
    if (!secretKey.startsWith(WebhookAsymmetricSigner.prefix)) {
      throw new Error(`Secret key must start with "${WebhookAsymmetricSigner.prefix}".`);
    }
    const seed = base64.decode(
      secretKey.substring(WebhookAsymmetricSigner.prefix.length)
    );
    if (seed.length !== ed25519.SEED_LENGTH) {
      throw new Error(
        `Secret key must be ${ed25519.SEED_LENGTH} bytes, got ${seed.length}.`
      );
    }
    const keyPair = ed25519.generateKeyPairFromSeed(seed);
    this.secretKey = keyPair.secretKey;
    this.publicKeyBytes = keyPair.publicKey;
  }

  /** Sign `payload` sent to the given endpoint, with the given message id and timestamp.
   *
   * A `Buffer` or `Uint8Array` payload is signed byte for byte. A string is signed as its
   * UTF-8 encoding.
   *
   * @returns The signature for the `webhook-signature` header (`v2a,<base64 signature>`)
   */
  public sign(
    endpointId: string,
    msgId: string,
    timestamp: Date,
    payload: string | Uint8Array
  ): string {
    const digest = WebhookAsymmetric.signedContentHash(
      endpointId,
      msgId,
      timestamp,
      payload
    );
    return WebhookAsymmetric.formatSignature(ed25519.sign(this.secretKey, digest));
  }

  /** The public key (`whpk_...`) matching the secret key, to share with consumers. */
  public publicKey(): string {
    return WebhookAsymmetric.formatPublicKey(this.publicKeyBytes);
  }
}
