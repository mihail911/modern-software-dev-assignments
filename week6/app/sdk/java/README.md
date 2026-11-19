Java library for Standard Webhooks

# Example

Verifying a webhook payload:

```java
import com.standardwebhooks.Webhook;

Webhook webhook = new Webhook(base64Secret);
webhook.verify(webhookPayload, webhookHeaders);
```

Responding to [endpoint verification](../../spec/extensions/endpoint-verification.md):

```java
String challenge = request.getHeader("webhook-challenge");
if (challenge != null) {
    response.setHeader("webhook-challenge-response", webhook.endpointVerificationResponse(challenge));
    // Not a webhook delivery: respond without processing the body
    response.setStatus(204);
    return;
}
```

Verifying a challenge response (for producers):

```java
// Throws WebhookVerificationException if the response is invalid
webhook.verifyEndpointVerificationResponse(challenge, responseHeader);
```

## Asymmetric signatures

Verifying a webhook payload signed with an asymmetric (`v2a`) signature, using the endpoint's public keys and the endpoint ID assigned by the producer:

```java
import com.standardwebhooks.WebhookAsymmetric;

WebhookAsymmetric webhook = new WebhookAsymmetric("ep_...", Arrays.asList("whpk_..."));
webhook.verify(webhookPayload, webhookHeaders);
```

Keys can be rotated by adding the new key to the start of the list (newest first): a signature made by any of the keys is accepted, and the first key is used for endpoint verification responses. `WebhookAsymmetric` supports the same endpoint verification methods as `Webhook` above.

### Signing (asymmetric)

It's recommended to hold the private key in a KMS or an HSM. The library builds the hash to sign, you sign it with your HSM (pure ed25519), and the library formats the header value:

```java
byte[] digest = WebhookAsymmetric.signedContentHash("ep_...", msgId, timestamp, payload);
byte[] signature = yourHsm.signEd25519(digest); // your HSM / KMS sign call
String header = WebhookAsymmetric.formatSignature(signature); // "v2a,..."

// The public key to give consumers. Accepts the raw 32 byte key or its DER encoded SubjectPublicKeyInfo.
String publicKey = WebhookAsymmetric.formatPublicKey(yourHsm.getEd25519PublicKey()); // "whpk_..."
```

Producers that hold the private key themselves can use `WebhookAsymmetricSigner`. One key pair serves all endpoints, so the endpoint ID is passed when signing:

```java
import com.standardwebhooks.WebhookAsymmetricSigner;

WebhookAsymmetricSigner signer = new WebhookAsymmetricSigner("whsk_...");
String header = signer.sign("ep_...", msgId, timestamp, payload); // "v2a,..."
String publicKey = signer.publicKey(); // "whpk_...", share it with consumers
```

# Development

## Requirements

 - Java 1.8+
 - Gradle

## Building the library
```sh
./gradlew build
```

## Running Tests

Simply run:

```sh
./gradlew test
```

## Publishing to Maven

```sh
./gradlew publishToSonatype closeAndReleaseSonatypeStagingRepository
```

