<?php

namespace StandardWebhooks;

/**
 * Verifies webhooks signed with the asymmetric `v2a` signature scheme
 * (ed25519 over the SHA-256 hash of `endpoint_id.msg_id.timestamp.payload`).
 *
 * Producers signing with a KMS or an HSM can use `signedContentHash()` and `formatSignature()`.
 * Producers holding the private key themselves can use `WebhookAsymmetricSigner`.
 */
class WebhookAsymmetric
{
    private const PUBLIC_KEY_PREFIX = "whpk_";
    /** DER encoded SubjectPublicKeyInfo header for a raw ed25519 public key (hex) */
    private const ED25519_SPKI_PREFIX_HEX = "302a300506032b6570032100";
    private const SIGNATURE_IDENTIFIER = "v2a";
    private const TOLERANCE = 5 * 60;
    /** @var string[] raw (32 bytes) ed25519 public keys */
    private $publicKeys;
    /** @var string */
    private $endpointId;

    /**
     * @param string $endpointId The endpoint ID the producer shared with you together with the public
     *                           keys. Never take it from the request.
     * @param string[] $publicKeys A non-empty list of trusted `whpk_` prefixed public keys. A webhook
     *                             is accepted if it's signed by any of them, so keys can be rotated by
     *                             adding the new key to the list. The order only matters for endpoint
     *                             verification, which uses the first key: put the newest key first.
     * @throws Exception\EmptyWebhookSecretException if a public key or the endpoint ID is invalid
     */
    public function __construct($endpointId, $publicKeys)
    {
        if (!is_array($publicKeys) || count($publicKeys) === 0) {
            throw new Exception\EmptyWebhookSecretException("Missing public keys");
        }

        $this->publicKeys = [];
        foreach ($publicKeys as $publicKey) {
            $this->publicKeys[] = self::decodePublicKey($publicKey);
        }

        self::validateEndpointId($endpointId);
        $this->endpointId = $endpointId;
    }

    /**
     * Returns the raw 32-byte SHA-256 hash of the `v2a` signed content
     * (`endpoint_id.msg_id.timestamp.payload`). This is what gets signed with ed25519, e.g. by an
     * HSM or a KMS. Pass the signature to `formatSignature()` to get the webhook-signature header value.
     *
     * @param string $endpointId The ID of the endpoint the webhook is sent to
     * @param string $msgId The message ID (webhook-id header)
     * @param int|string $timestamp The timestamp in seconds since the epoch (webhook-timestamp header)
     * @param string $payload The exact payload, signed byte-for-byte
     * @return string The raw 32-byte hash
     * @throws Exception\EmptyWebhookSecretException if the endpoint ID is invalid
     * @throws Exception\WebhookSigningException if the timestamp is invalid
     */
    public static function signedContentHash($endpointId, $msgId, $timestamp, $payload)
    {
        self::validateEndpointId($endpointId);
        $timestamp = (string) $timestamp;
        if (!self::isPositiveInteger($timestamp)) {
            throw new Exception\WebhookSigningException("Invalid timestamp");
        }
        return hash('sha256', "{$endpointId}.{$msgId}.{$timestamp}.{$payload}", true);
    }

    /**
     * Formats a raw 64-byte ed25519 signature (of `signedContentHash()`) as a webhook-signature
     * header value: `v2a,<base64 signature>`.
     *
     * @param string $signature The raw 64-byte signature
     * @return string
     * @throws Exception\WebhookSigningException if the signature isn't 64 bytes
     */
    public static function formatSignature($signature)
    {
        if (!is_string($signature) || strlen($signature) !== SODIUM_CRYPTO_SIGN_BYTES) {
            throw new Exception\WebhookSigningException(
                "Signature must be " . SODIUM_CRYPTO_SIGN_BYTES . " bytes"
            );
        }
        return self::SIGNATURE_IDENTIFIER . "," . base64_encode($signature);
    }

    /**
     * Formats an ed25519 public key (e.g. from an HSM or a KMS) as the `whpk_` prefixed public key
     * to share with consumers.
     *
     * @param string $publicKey The raw 32-byte public key, or its 44-byte DER encoded
     *     SubjectPublicKeyInfo (e.g. what AWS KMS `GetPublicKey` returns)
     * @return string
     * @throws Exception\WebhookSigningException if the public key isn't in either format
     */
    public static function formatPublicKey($publicKey)
    {
        if (is_string($publicKey)) {
            $spkiPrefix = hex2bin(self::ED25519_SPKI_PREFIX_HEX);
            if (
                strlen($publicKey) === strlen($spkiPrefix) + SODIUM_CRYPTO_SIGN_PUBLICKEYBYTES
                && substr($publicKey, 0, strlen($spkiPrefix)) === $spkiPrefix
            ) {
                $publicKey = substr($publicKey, strlen($spkiPrefix));
            }
        }
        if (!is_string($publicKey) || strlen($publicKey) !== SODIUM_CRYPTO_SIGN_PUBLICKEYBYTES) {
            throw new Exception\WebhookSigningException(
                "Public key must be " . SODIUM_CRYPTO_SIGN_PUBLICKEYBYTES
                . " raw bytes or a DER encoded ed25519 SubjectPublicKeyInfo"
            );
        }
        return self::PUBLIC_KEY_PREFIX . base64_encode($publicKey);
    }

    public function verify($payload, $headers)
    {
        if (
            isset($headers['webhook-id'])
            && isset($headers['webhook-timestamp'])
            && isset($headers['webhook-signature'])
        ) {
            $msgId = $headers['webhook-id'];
            $msgTimestamp = $headers['webhook-timestamp'];
            $msgSignature = $headers['webhook-signature'];
        } else {
            throw new Exception\WebhookVerificationException("Missing required headers");
        }

        $timestamp = $this->verifyTimestamp($msgTimestamp);

        $digest = self::signedContentHash($this->endpointId, $msgId, $timestamp, $payload);

        $passedSignatures = explode(' ', $msgSignature);
        foreach ($passedSignatures as $versionedSignature) {
            $parts = explode(',', $versionedSignature, 2);
            if (count($parts) !== 2 || $parts[0] !== self::SIGNATURE_IDENTIFIER) {
                continue;
            }
            $signature = base64_decode($parts[1], true);
            if ($signature === false || strlen($signature) !== SODIUM_CRYPTO_SIGN_BYTES) {
                continue;
            }
            foreach ($this->publicKeys as $publicKey) {
                if (sodium_crypto_sign_verify_detached($signature, $digest, $publicKey)) {
                    return json_decode($payload, true);
                }
            }
        }
        throw new Exception\WebhookVerificationException("No matching signature found");
    }

    /**
     * Calculate the response to an endpoint verification challenge. Uses the first (newest) public key.
     *
     * @param string $challenge The value of the webhook-challenge request header
     * @return string The value for the webhook-challenge-response response header
     */
    public function endpointVerificationResponse($challenge)
    {
        return $this->calculateEndpointVerificationResponse($this->publicKeys[0], $challenge);
    }

    /**
     * Verify the response to an endpoint verification challenge. Succeeds if the response was
     * calculated with any of the public keys.
     *
     * @param string $challenge The challenge sent in the webhook-challenge request header
     * @param string $header The value of the webhook-challenge-response response header
     * @throws Exception\WebhookVerificationException if verification fails
     */
    public function verifyEndpointVerificationResponse($challenge, $header)
    {
        if (empty($challenge) || empty($header)) {
            throw new Exception\WebhookVerificationException("Missing challenge response");
        }

        foreach ($this->publicKeys as $publicKey) {
            $expectedSignature = $this->calculateEndpointVerificationResponse($publicKey, $challenge);
            if (hash_equals($expectedSignature, $header)) {
                return;
            }
        }
        throw new Exception\WebhookVerificationException("No matching signature found");
    }

    private function calculateEndpointVerificationResponse($publicKey, $challenge)
    {
        $endpointKey = $publicKey . "." . $this->endpointId;
        $verificationKey = hash_hmac('sha256', 'wev-v1', $endpointKey, true);
        $signature = base64_encode(hash_hmac('sha256', $challenge, $verificationKey, true));
        return self::SIGNATURE_IDENTIFIER . ",{$signature}";
    }

    private static function decodePublicKey($publicKey)
    {
        if (!is_string($publicKey) || strpos($publicKey, self::PUBLIC_KEY_PREFIX) !== 0) {
            throw new Exception\EmptyWebhookSecretException(
                "Public key must start with \"" . self::PUBLIC_KEY_PREFIX . "\""
            );
        }
        $rawKey = base64_decode(substr($publicKey, strlen(self::PUBLIC_KEY_PREFIX)), true);
        if ($rawKey === false || strlen($rawKey) !== SODIUM_CRYPTO_SIGN_PUBLICKEYBYTES) {
            throw new Exception\EmptyWebhookSecretException("Invalid public key");
        }
        return $rawKey;
    }

    private static function validateEndpointId($endpointId)
    {
        if (!is_string($endpointId) || $endpointId === "") {
            throw new Exception\EmptyWebhookSecretException("Missing endpoint ID");
        }
        if (strpos($endpointId, ".") !== false) {
            throw new Exception\EmptyWebhookSecretException("Endpoint ID must not contain a '.'");
        }
    }

    private static function isPositiveInteger($v)
    {
        return is_numeric($v) && !is_float($v + 0) && (int) $v == $v && (int) $v > 0;
    }

    private function verifyTimestamp($timestampHeader)
    {
        $now = time();
        $timestamp = intval($timestampHeader, 10);

        if ($timestamp < ($now - self::TOLERANCE)) {
            throw new Exception\WebhookVerificationException("Message timestamp too old");
        }
        if ($timestamp > ($now + self::TOLERANCE)) {
            throw new Exception\WebhookVerificationException("Message timestamp too new");
        }
        return $timestamp;
    }
}
