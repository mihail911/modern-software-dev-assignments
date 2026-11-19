Go library for Standard Webhooks

# Example

Verifying a webhook payload:

```go
import (
    standardwebhooks "github.com/standard-webhooks/standard-webhooks/libraries/go"
)

wh, err := standardwebhooks.NewWebhook(base64Secret)
err = wh.Verify(webhookPayload, webhookHeaders)
```

## Asymmetric signatures

Verifying a webhook payload signed with asymmetric (`v2a`) signatures, using the public keys and the endpoint ID shared by the producer:

```go
wh, err := standardwebhooks.NewWebhookAsymmetric("ep_...", []string{"whpk_..."})
err = wh.Verify(webhookPayload, webhookHeaders)
```

A webhook is accepted if it was signed by any of the public keys in the list, so keys can be rotated by adding the new key to the list. Put the newest key first: it's the one used for responding to endpoint verification.

### Signing (asymmetric)

It's recommended to generate and hold the private key in a KMS or an HSM. Build the hash to sign, sign it with your HSM (pure ed25519), and format the signature for the `webhook-signature` header:

```go
digest, err := standardwebhooks.AsymmetricSignedContentHash("ep_...", msgID, timestamp, payload)
rawSignature := yourHSM.SignEd25519(digest) // 64 bytes
signature, err := standardwebhooks.FormatAsymmetricSignature(rawSignature)

// The public key to share with consumers ("whpk_..."). Accepts the raw 32 byte key or its DER
// encoded SubjectPublicKeyInfo (e.g. what AWS KMS GetPublicKey returns).
publicKey, err := standardwebhooks.FormatAsymmetricPublicKey(yourHSM.PublicKey())
```

If you hold the private key yourself (`whsk_...`), use `WebhookAsymmetricSigner`. One key pair can serve all endpoints, so the endpoint ID is passed to each call:

```go
signer, err := standardwebhooks.NewWebhookAsymmetricSigner("whsk_...")
signature, err := signer.Sign("ep_...", msgID, timestamp, payload)
publicKey := signer.PublicKey() // "whpk_...", share it with consumers
```

## Endpoint verification

Responding to [endpoint verification](../../spec/extensions/endpoint-verification.md):

```go
if challenge := r.Header.Get(standardwebhooks.HeaderWebhookChallenge); challenge != "" {
    w.Header().Set(standardwebhooks.HeaderWebhookChallengeResponse, wh.EndpointVerificationResponse(challenge))
    // Not a webhook delivery: respond without processing the body
    w.WriteHeader(http.StatusNoContent)
    return
}
```

Producers can verify the `webhook-challenge-response` header with `wh.VerifyEndpointVerificationResponse(challenge, header)`.

Both work the same way with `WebhookAsymmetric`.

# Development

## Requirements

 - go >= 1.16

## Contributing

Before opening a PR be sure to format your code!

```sh
go fmt ./...
```

## Running Tests

Simply run:

```sh
go test ./...
```

## Publishing

Releases use go modules and are automatically published by tagging the release commit.
