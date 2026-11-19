Python library for Standard Webhooks

# Example

Verifying a webhook payload:

```python
from standardwebhooks.webhooks import Webhook

wh = Webhook(base64_secret)
wh.verify(webhook_payload, webhook_headers)
```

`webhook_payload` must be the raw request body exactly as received, before any JSON parsing.
Pass it as `bytes` to verify the bytes as sent, including bodies that aren't valid UTF-8. A
`str` is verified as its UTF-8 encoding.

## Asymmetric signatures

Verifying a webhook signed with asymmetric (`v2a`) signatures requires the `asymmetric` extra
(`pip install 'standardwebhooks[asymmetric]'`), and uses the producer's public keys and the
endpoint ID it shared with you:

```python
from standardwebhooks.webhooks import WebhookAsymmetric

wh = WebhookAsymmetric(endpoint_id, ["whpk_..."])
wh.verify(webhook_payload, webhook_headers)
```

Webhooks signed with any of the public keys are accepted, so keys can be rotated by adding the
new key to the list. Put the newest key first: it's the one used for endpoint verification.

### Signing (asymmetric)

It's recommended to generate and hold the private key in a KMS or an HSM. Build the hash to sign,
sign it with your HSM (pure ed25519), and format the signature for the `webhook-signature` header
(these two don't require the `asymmetric` extra):

```python
from standardwebhooks.webhooks import WebhookAsymmetric

digest = WebhookAsymmetric.signed_content_hash(endpoint_id, msg_id, timestamp, payload)
raw_signature = your_hsm.sign_ed25519(digest)  # 64 bytes
signature = WebhookAsymmetric.format_signature(raw_signature)

# The public key to share with consumers ("whpk_..."). Accepts the raw 32 byte key or its DER
# encoded SubjectPublicKeyInfo (e.g. what AWS KMS GetPublicKey returns).
public_key = WebhookAsymmetric.format_public_key(your_hsm.get_public_key())
```

If you hold the private key yourself (`whsk_...`), use `WebhookAsymmetricSigner`. One key pair
can serve all endpoints, so the endpoint ID is passed to each call:

```python
from standardwebhooks.webhooks import WebhookAsymmetricSigner

signer = WebhookAsymmetricSigner("whsk_...")
signature = signer.sign(endpoint_id, msg_id, timestamp, payload)
public_key = signer.public_key()  # "whpk_...", share it with consumers
```

Responding to [endpoint verification](../../spec/extensions/endpoint-verification.md):

```python
challenge = webhook_headers.get("webhook-challenge")
if challenge:
    response_headers["webhook-challenge-response"] = wh.endpoint_verification_response(challenge)
    # Not a webhook delivery: respond (e.g. with 204 No Content) without processing the body
    return
```

Verifying the response as a producer (raises `WebhookVerificationError` if it doesn't match):

```python
wh.verify_endpoint_verification_response(challenge, response_headers["webhook-challenge-response"])
```

# Development

## Requirements

 - python 3

## Installing dependencies

```sh
python -m venv .venv
pip install -r requirements.txt && pip install -r requirements-dev.txt
```

## Contributing

Before opening a PR be sure to format your code!

```sh
./scripts/format.sh
```

## Running Tests

Simply run:

```sh
pytest
```
