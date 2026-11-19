<?php

namespace StandardWebhooks;

/**
 * Signs webhooks with the asymmetric `v2a` signature scheme, for producers that hold the private
 * key themselves. It's recommended to hold the private key in a KMS or an HSM instead, and sign
 * using `WebhookAsymmetric::signedContentHash()` and `WebhookAsymmetric::formatSignature()`.
 *
 * A key pair isn't tied to an endpoint: one key pair can serve all endpoints, as the endpoint ID
 * is part of the signed content.
 */
class WebhookAsymmetricSigner
{
    private const SECRET_KEY_PREFIX = "whsk_";
    /** @var string libsodium (64 bytes) ed25519 secret key */
    private $secretKey;
    /** @var string raw (32 bytes) ed25519 public key */
    private $publicKey;

    /**
     * @param string $secretKey A `whsk_` prefixed, base64 encoded 32-byte ed25519 private key (seed)
     * @throws Exception\EmptyWebhookSecretException if the secret key is invalid
     */
    public function __construct($secretKey)
    {
        if (!is_string($secretKey) || strpos($secretKey, self::SECRET_KEY_PREFIX) !== 0) {
            throw new Exception\EmptyWebhookSecretException(
                "Secret key must start with \"" . self::SECRET_KEY_PREFIX . "\""
            );
        }
        $seed = base64_decode(substr($secretKey, strlen(self::SECRET_KEY_PREFIX)), true);
        if ($seed === false || strlen($seed) !== SODIUM_CRYPTO_SIGN_SEEDBYTES) {
            throw new Exception\EmptyWebhookSecretException("Invalid secret key");
        }

        $keypair = sodium_crypto_sign_seed_keypair($seed);
        $this->secretKey = sodium_crypto_sign_secretkey($keypair);
        $this->publicKey = sodium_crypto_sign_publickey($keypair);
    }

    /**
     * Sign a webhook. Takes the same arguments as `WebhookAsymmetric::signedContentHash()`.
     *
     * @param string $endpointId The ID of the endpoint the webhook is sent to
     * @param string $msgId The message ID (webhook-id header)
     * @param int|string $timestamp The timestamp in seconds since the epoch (webhook-timestamp header)
     * @param string $payload The exact payload, signed byte-for-byte
     * @return string The webhook-signature header value (`v2a,<base64 signature>`)
     * @throws Exception\EmptyWebhookSecretException if the endpoint ID is invalid
     * @throws Exception\WebhookSigningException if the timestamp is invalid
     */
    public function sign($endpointId, $msgId, $timestamp, $payload)
    {
        $digest = WebhookAsymmetric::signedContentHash($endpointId, $msgId, $timestamp, $payload);
        return WebhookAsymmetric::formatSignature(sodium_crypto_sign_detached($digest, $this->secretKey));
    }

    /**
     * @return string The `whpk_` prefixed public key to share with consumers
     */
    public function publicKey()
    {
        return WebhookAsymmetric::formatPublicKey($this->publicKey);
    }

    public function __debugInfo()
    {
        return ['publicKey' => $this->publicKey()];
    }
}
