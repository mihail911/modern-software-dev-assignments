Rust library for Standard Webhooks 

# Example

Verifying a webhook payload:

```rust
use standardwebhooks::Webhook;

let wh = Webhook::new(base64_secret);
wh.verify(webhook_payload, webhook_headers).expect("Webhook verification failed");
```

## Asymmetric signatures

Verifying a webhook payload signed with asymmetric (`v2a`) signatures, using the public keys and the endpoint ID shared by the producer:

```rust
use standardwebhooks::WebhookAsymmetric;

let wh = WebhookAsymmetric::new("ep_...", &["whpk_..."]).expect("Invalid public key or endpoint ID");
wh.verify(webhook_payload, webhook_headers).expect("Webhook verification failed");
```

A webhook is accepted if it was signed by any of the public keys in the list, so keys can be rotated by adding the new key to the list. Put the newest key first: it's the one used for responding to endpoint verification.

### Signing (asymmetric)

It's recommended to hold the private key in a KMS or an HSM. The library builds the hash to sign, you sign it with your HSM (pure ed25519), and the library formats the header value:

```rust
use standardwebhooks::WebhookAsymmetric;

let digest = WebhookAsymmetric::signed_content_hash("ep_...", msg_id, timestamp, payload)?;
let signature: Vec<u8> = hsm_ed25519_sign(&digest); // your HSM / KMS sign call
let header = WebhookAsymmetric::format_signature(&signature)?; // "v2a,..."

// The public key to give consumers. Accepts the raw 32-byte key or its DER SubjectPublicKeyInfo.
let public_key = WebhookAsymmetric::format_public_key(&hsm_ed25519_public_key())?; // "whpk_..."
```

Producers that hold the private key themselves can use `WebhookAsymmetricSigner`. One key pair serves all endpoints, so the endpoint ID is passed when signing:

```rust
use standardwebhooks::WebhookAsymmetricSigner;

let signer = WebhookAsymmetricSigner::new("whsk_...")?;
let header = signer.sign("ep_...", msg_id, timestamp, payload)?; // "v2a,..."
let public_key = signer.public_key(); // "whpk_...", share it with consumers
```

## Endpoint verification

Responding to [endpoint verification](../../spec/extensions/endpoint-verification.md):

```rust
use standardwebhooks::{HEADER_WEBHOOK_CHALLENGE, HEADER_WEBHOOK_CHALLENGE_RESPONSE};

if let Some(challenge) = request_headers.get(HEADER_WEBHOOK_CHALLENGE) {
    let challenge = challenge.to_str().expect("Invalid challenge header");
    response_headers.insert(
        HEADER_WEBHOOK_CHALLENGE_RESPONSE,
        wh.endpoint_verification_response(challenge).parse().unwrap(),
    );
    // Not a webhook delivery: respond (e.g. with 204 No Content) without processing the body
    return;
}
```

Producers can verify the `webhook-challenge-response` header with `wh.verify_endpoint_verification_response(challenge, header)`.

Both work the same way with `WebhookAsymmetric`.
