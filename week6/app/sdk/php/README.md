PHP library for Standard Webhooks 

# Installation

# Example

Verifying a webhook payload:

```php
$wh = new \StandardWebhooks\Webhook($base64Secret);
$wh->verify($webhookPayload, $webhookHeaders);
```

Responding to [endpoint verification](../../spec/extensions/endpoint-verification.md):

```php
if (isset($webhookHeaders['webhook-challenge'])) {
    header('webhook-challenge-response: ' . $wh->endpointVerificationResponse($webhookHeaders['webhook-challenge']));
    // Not a webhook delivery: respond without processing the body
    http_response_code(204);
    exit;
}
```

Verifying a challenge response (for producers):

```php
// Throws WebhookVerificationException if the response is invalid
$wh->verifyEndpointVerificationResponse($challenge, $responseHeaders['webhook-challenge-response']);
```

## Asymmetric signatures

Webhooks signed with asymmetric (`v2a`) signatures are verified with `WebhookAsymmetric`, which takes the endpoint ID and the list of trusted public keys (`whpk_...`) the producer shared with you.
A webhook is accepted if it's signed by any of the keys, so keys can be rotated by adding the new key to the list.
Put the newest key first, as endpoint verification responses are calculated with the first key.

```php
$wh = new \StandardWebhooks\WebhookAsymmetric($endpointId, [$publicKey]);
$wh->verify($webhookPayload, $webhookHeaders);
```

`WebhookAsymmetric` also supports `endpointVerificationResponse` and `verifyEndpointVerificationResponse`.

### Signing (asymmetric)

It's recommended to generate and hold the private key in a KMS or an HSM, and never export it.
In that case, use `WebhookAsymmetric::signedContentHash` to get the 32-byte SHA-256 hash to sign, sign it (pure ed25519) with your HSM, and format the 64-byte signature with `WebhookAsymmetric::formatSignature`:

```php
$digest = \StandardWebhooks\WebhookAsymmetric::signedContentHash($endpointId, $msgId, $timestamp, $payload);
$signature = $myHsm->ed25519Sign($digest); // Your HSM / KMS sign call
$headers['webhook-signature'] = \StandardWebhooks\WebhookAsymmetric::formatSignature($signature);

// The `whpk_...` public key to share with consumers. Accepts the raw 32-byte key or its DER SubjectPublicKeyInfo.
$publicKey = \StandardWebhooks\WebhookAsymmetric::formatPublicKey($myHsm->ed25519PublicKey());
```

If you hold the private key yourself (`whsk_...`), use `WebhookAsymmetricSigner`.
One key pair can serve all of your endpoints, as the endpoint ID is part of the signed content:

```php
$signer = new \StandardWebhooks\WebhookAsymmetricSigner($secretKey);

// Share with consumers, together with the endpoint ID
$publicKey = $signer->publicKey(); // "whpk_..."

$headers['webhook-signature'] = $signer->sign($endpointId, $msgId, time(), $payload); // "v2a,..."
```

### Required Dependencies

Standard Webhooks PHP requires the following extensions in order to run:

- [`json`](https://secure.php.net/manual/en/book.json.php)
- [`sodium`](https://secure.php.net/manual/en/book.sodium.php) (for `WebhookAsymmetric` and `WebhookAsymmetricSigner`)

If you use Composer, these dependencies should be handled automatically. If you install manually, you'll want to make sure that these extensions are available.

# Development

## Requirements

 - PHP >= 5.6.0

## Installing dependencies
```sh
composer install
```

## Contributing

Before opening a PR be sure to format your code!

```sh
composer install
./vendor/bin/php-cs-fixer fix -v --using-cache=no .
```

## Running Tests

Simply run:

```sh
composer install
./vendor/bin/phpunit php/tests
```
