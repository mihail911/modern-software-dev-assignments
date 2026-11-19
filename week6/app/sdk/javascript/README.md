Typescript/Javascript library for Standard Webhooks 

# Example

Verifying a webhook payload:

```javascript
import { Webhook } from "standardwebhooks"

const wh = new Webhook(base64_secret);
wh.verify(webhook_payload, webhook_headers);
```

`webhook_payload` must be the raw request body exactly as received, before any JSON parsing.
Pass it as a `Buffer` or `Uint8Array` to verify the bytes as sent, including bodies that aren't
valid UTF-8. A string is verified as its UTF-8 encoding.

## Asymmetric signatures

Verifying a webhook signed with asymmetric (`v2a`) signatures, using the producer's public keys
and the endpoint ID it shared with you:

```javascript
import { WebhookAsymmetric } from "standardwebhooks"

const wh = new WebhookAsymmetric(endpoint_id, ["whpk_..."]);
wh.verify(webhook_payload, webhook_headers);
```

Webhooks signed with any of the public keys are accepted, so keys can be rotated by adding the
new key to the list. Put the newest key first: it's the one used for endpoint verification.

### Signing (asymmetric)

It's recommended to generate and hold the private key in a KMS or an HSM. Build the hash to sign,
sign it with your HSM (pure ed25519), and format the signature for the `webhook-signature` header:

```javascript
import { WebhookAsymmetric } from "standardwebhooks"

const digest = WebhookAsymmetric.signedContentHash(endpoint_id, msg_id, timestamp, payload);
const rawSignature = await yourHsm.signEd25519(digest); // 64 bytes
const signature = WebhookAsymmetric.formatSignature(rawSignature);

// The public key to share with consumers ("whpk_..."). Accepts the raw 32 byte key or its DER
// encoded SubjectPublicKeyInfo (e.g. what AWS KMS GetPublicKey returns).
const publicKey = WebhookAsymmetric.formatPublicKey(await yourHsm.getPublicKey());
```

If you hold the private key yourself (`whsk_...`), use `WebhookAsymmetricSigner`. One key pair
can serve all endpoints, so the endpoint ID is passed to each call:

```javascript
import { WebhookAsymmetricSigner } from "standardwebhooks"

const signer = new WebhookAsymmetricSigner("whsk_...");
const signature = signer.sign(endpoint_id, msg_id, timestamp, payload);
const publicKey = signer.publicKey(); // "whpk_...", share it with consumers
```

Responding to [endpoint verification](../../spec/extensions/endpoint-verification.md):

```javascript
const challenge = webhook_headers["webhook-challenge"];
if (challenge) {
  response_headers["webhook-challenge-response"] = wh.endpointVerificationResponse(challenge);
  // Not a webhook delivery: respond (e.g. with 204 No Content) without processing the body
  return;
}
```

Verifying the response as a producer (throws `WebhookVerificationError` if it doesn't match):

```javascript
wh.verifyEndpointVerificationResponse(challenge, response_headers["webhook-challenge-response"]);
```

# Development

## Requirements

 - node
 - npm

## Building the library
```sh
npm install
npm run build
```

## Contributing

Before opening a PR be sure to format your code!

```sh
npm run check:fix
```

## Running Tests

Simply run:

```sh
npm test
```
