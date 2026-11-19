# ADR-0010: Sign requests as Standard Webhooks describes

- Status: accepted
- Date: 2026-01-13
- Author: Ines

This records a decision taken in July 2025 and shipped in release 2.0. It was not written down then. ADR-0009 depends on it, so it is written down now.

## Context

Release 1 signed each request with `X-Switchyard-Signature: sha256=<hex>`, an HMAC-SHA256 of the body. The signature did not cover a timestamp, so a captured request could be replayed for ever, and it did not cover the delivery id, which travelled in a separate header. Every merchant wrote their own verification, and support spent a steady part of each week on hex-versus-base64 and on bodies that had been parsed and re-serialised before they were checked.

The Standard Webhooks specification defines headers, a signed payload and verification libraries in the languages our merchants use.

## Decision

From release 2.0 requests are signed as Standard Webhooks describes, with the symmetric `v1` scheme.

- Headers: `webhook-id`, `webhook-timestamp` (whole seconds) and `webhook-signature`.
- The signature is an HMAC-SHA256 over `{webhook-id}.{webhook-timestamp}.{body}`, keyed with the endpoint's secret, base64-encoded and prefixed `v1,`.
- A secret is `whsec_` followed by base64.
- The body that is signed is byte for byte the body that is sent.
- The old `X-Switchyard-Signature` header is no longer sent. There was no period in which both were.
- The same holds for a rotated secret: attempts made after the rotation use the new secret, and only it.

## Consequences

- Merchants verify with a published library; `sdk/` carries the ones we point them to. The worker and the test receiver use vendored copies.
- The timestamp is signed, so receivers can refuse old requests. A retry is signed again with the time of that attempt.
- `webhook-id` is signed and is the same on every attempt of a delivery. It is the value a merchant de-duplicates on.
- Rotation has no overlap. A merchant who rotates a secret has to deploy the new one on their side first, or accept failed attempts until they do. We decided this was simpler to explain than two signatures.
