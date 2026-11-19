C# library for Standard Webhooks

# Example

Verifying a webhook payload:

```cs
using StandardWebhooks;

var wh = new Webhook(base64Secret);
wh.Verify(webhookPayload, webhookHeaders);
```

Responding to [endpoint verification](../../spec/extensions/endpoint-verification.md):

```cs
var challenge = webhookHeaders.Get("webhook-challenge");
if (challenge != null)
{
    response.Headers["webhook-challenge-response"] = wh.EndpointVerificationResponse(challenge);
    // Not a webhook delivery: respond without processing the body
    response.StatusCode = 204;
    return;
}
```

Verifying a challenge response (for producers):

```cs
// Throws WebhookVerificationException if the response is invalid
wh.VerifyEndpointVerificationResponse(challenge, responseHeader);
```

## Asymmetric signatures

Verifying a webhook payload signed with an asymmetric (`v2a`) signature, using the endpoint's public keys and the endpoint ID assigned by the producer:

```cs
using StandardWebhooks;

var wh = new WebhookAsymmetric("ep_...", new[] { "whpk_..." });
wh.Verify(webhookPayload, webhookHeaders);
```

Keys can be rotated by adding the new key to the start of the list (newest first): a signature made by any of the keys is accepted, and the first key is used for endpoint verification responses. `WebhookAsymmetric` supports the same endpoint verification methods as `Webhook` above.

### Signing (asymmetric)

It's recommended to hold the private key in a KMS or an HSM. The library builds the hash to sign, you sign it with your HSM (pure ed25519), and the library formats the header value:

```cs
byte[] digest = WebhookAsymmetric.SignedContentHash("ep_...", msgId, timestamp, payload);
byte[] signature = YourHsm.SignEd25519(digest); // your HSM / KMS sign call
string header = WebhookAsymmetric.FormatSignature(signature); // "v2a,..."

// The public key to give consumers. Accepts the raw 32 byte key or its DER encoded SubjectPublicKeyInfo.
string publicKey = WebhookAsymmetric.FormatPublicKey(YourHsm.GetEd25519PublicKey()); // "whpk_..."
```

Producers that hold the private key themselves can use `WebhookAsymmetricSigner`. One key pair serves all endpoints, so the endpoint ID is passed when signing:

```cs
var signer = new WebhookAsymmetricSigner("whsk_...");
string header = signer.Sign("ep_...", msgId, timestamp, payload); // "v2a,..."
string publicKey = signer.PublicKey(); // "whpk_...", share it with consumers
```

# Development

## Requirements

 - Dotnet >=10.0

## Building the library
```sh
dotnet build
```

## Contributing

Before opening a PR be sure to format your code!

We use [dotnet-format](https://github.com/dotnet/format) for this project.

First install it then run:
```sh
dotnet-format
```

## Running Tests

Simply run:

```sh
dotnet test StandardWebhooks.Tests
```

