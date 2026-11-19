<?php

namespace StandardWebhooks;

class Webhook
{
    private const SECRET_PREFIX = "whsec_";
    private const TOLERANCE = 5 * 60;
    private $secret;

    public function __construct($secret)
    {
        if (substr($secret, 0, strlen(Webhook::SECRET_PREFIX)) === Webhook::SECRET_PREFIX) {
            $secret = substr($secret, strlen(Webhook::SECRET_PREFIX));
        }
        $this->secret = base64_decode($secret);
        if ($this->secret === "") {
            throw new Exception\EmptyWebhookSecretException("Missing webhook secret");
        }
    }

    public static function fromRaw($secret)
    {
        return new self(base64_encode($secret));
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

        $expectedSignature = $this->sign($msgId, $timestamp, $payload);

        $passedSignatures = explode(' ', $msgSignature);
        foreach ($passedSignatures as $versionedSignature) {
            if (hash_equals($expectedSignature, $versionedSignature)) {
                return json_decode($payload, true);
            }
        }
        throw new Exception\WebhookVerificationException("No matching signature found");
    }

    public function sign($msgId, $timestamp, $payload)
    {
        $timestamp = (string) $timestamp;
        $is_positive_integer = $this->isPositiveInteger($timestamp);
        if (!$is_positive_integer) {
            throw new Exception\WebhookSigningException("Invalid timestamp");
        }
        $toSign = "{$msgId}.{$timestamp}.{$payload}";
        $hex_hash = hash_hmac('sha256', $toSign, $this->secret);
        $signature = base64_encode(pack('H*', $hex_hash));
        return "v1,{$signature}";
    }

    /**
     * Calculate the response to an endpoint verification challenge.
     *
     * @param string $challenge The value of the webhook-challenge request header
     * @return string The value for the webhook-challenge-response response header
     */
    public function endpointVerificationResponse($challenge)
    {
        $verificationKey = hash_hmac('sha256', 'wev-v1', $this->secret, true);
        $signature = base64_encode(hash_hmac('sha256', $challenge, $verificationKey, true));
        return "v1,{$signature}";
    }

    /**
     * Verify the response to an endpoint verification challenge.
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

        $expectedSignature = $this->endpointVerificationResponse($challenge);

        if (!hash_equals($expectedSignature, $header)) {
            throw new Exception\WebhookVerificationException("No matching signature found");
        }
    }

    private function verifyTimestamp($timestampHeader)
    {
        $now = time();
        try {
            $timestamp = intval($timestampHeader, 10);
        } catch (\Exception $e) {
            throw new Exception\WebhookVerificationException("Invalid Signature Headers");
        }

        if ($timestamp < ($now - Webhook::TOLERANCE)) {
            throw new Exception\WebhookVerificationException("Message timestamp too old");
        }
        if ($timestamp > ($now + Webhook::TOLERANCE)) {
            throw new Exception\WebhookVerificationException("Message timestamp too new");
        }
        return $timestamp;
    }

    private function isPositiveInteger($v)
    {
        return is_numeric($v) && !is_float($v + 0) && (int) $v == $v && (int) $v > 0;
    }
}
