<?php

namespace StandardWebhooks;

final class WebhookAsymmetricTest extends \PHPUnit\Framework\TestCase
{
    private const TOLERANCE = 5 * 60;
    private const SECRET_KEY = "whsk_4uKPjji/dhB8w6GW3YYDnIMYvXcdHrvH6lidTr4Z+EY=";
    private const PUBLIC_KEY = "whpk_uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w=";
    private const ENDPOINT_ID = "ep_2fVZdC8Kx9TR4ZoCqHWt3NnLgRb";
    private const MSG_ID = "msg_p5jXN8AQM9LWM0D4loKWxJek";
    private const PAYLOAD = '{"test": 2432232315}';
    private const CHALLENGE = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=";
    private const CHALLENGE_RESPONSE = "v2a,t0RDFhzk7OtVzgvMsjuXnodSWuby2BVX+zPqw2/ksUk=";
    private const VECTOR_MSG_ID = "msg_2KWPBgLlAfxdpx2AI54pPJ85f4W";
    private const VECTOR_TIMESTAMP = 1674087231;
    private const VECTOR_PAYLOAD = '{"type":"contact.created","timestamp":"2022-11-03T20:26:10.344522Z","data":{"id":"1f81eb52-5198-4599-803e-771906343485"}}';
    private const VECTOR_CONTENT_HASH = "YXCi5iN+rjB+iEb6bvg1XbWbVBL+r2S+as3k3n1Ih/A=";
    private const VECTOR_SIGNATURE = "v2a,kG9XaxZjAEpKB8KeQQb5G1Xp73YH5Rf5gxoWe7cg1kYMiyspdtZXe9w+dAwKeOurAwxfqmlOG/QR4m9s37DzAg==";

    private static function specKeypair()
    {
        return sodium_crypto_sign_seed_keypair(base64_decode(substr(self::SECRET_KEY, strlen("whsk_"))));
    }

    private static function whpk($keypair)
    {
        return "whpk_" . base64_encode(sodium_crypto_sign_publickey($keypair));
    }

    private static function sign($keypair, $endpointId, $timestamp, $payload = self::PAYLOAD)
    {
        // A sodium secret key is the 32 byte seed followed by the public key
        $seed = substr(sodium_crypto_sign_secretkey($keypair), 0, SODIUM_CRYPTO_SIGN_SEEDBYTES);
        $signer = new \StandardWebhooks\WebhookAsymmetricSigner("whsk_" . base64_encode($seed));
        return $signer->sign($endpointId, self::MSG_ID, $timestamp, $payload);
    }

    private static function headers($signature, $timestamp)
    {
        return array(
            "webhook-id" => self::MSG_ID,
            "webhook-signature" => $signature,
            "webhook-timestamp" => strval($timestamp),
        );
    }

    private static function validHeaders()
    {
        $timestamp = time();
        return self::headers(self::sign(self::specKeypair(), self::ENDPOINT_ID, $timestamp), $timestamp);
    }

    private static function webhook()
    {
        return new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, [self::PUBLIC_KEY]);
    }

    public function testSpecKeypair()
    {
        $this->assertEquals(self::PUBLIC_KEY, self::whpk(self::specKeypair()));
    }

    public function testSpecSignatureVector()
    {
        // The vector's timestamp is outside of the tolerance, so check it against the primitive directly
        $content = 'ep_2fVZdC8Kx9TR4ZoCqHWt3NnLgRb.msg_2KWPBgLlAfxdpx2AI54pPJ85f4W.1674087231.{"type":"contact.created","timestamp":"2022-11-03T20:26:10.344522Z","data":{"id":"1f81eb52-5198-4599-803e-771906343485"}}';
        $digest = hash('sha256', $content, true);
        $this->assertEquals("YXCi5iN+rjB+iEb6bvg1XbWbVBL+r2S+as3k3n1Ih/A=", base64_encode($digest));

        $signature = base64_decode("kG9XaxZjAEpKB8KeQQb5G1Xp73YH5Rf5gxoWe7cg1kYMiyspdtZXe9w+dAwKeOurAwxfqmlOG/QR4m9s37DzAg==");
        $publicKey = base64_decode(substr(self::PUBLIC_KEY, strlen("whpk_")));
        $this->assertTrue(sodium_crypto_sign_verify_detached($signature, $digest, $publicKey));
        $this->assertEquals(
            $signature,
            sodium_crypto_sign_detached($digest, sodium_crypto_sign_secretkey(self::specKeypair()))
        );
    }

    public function testValidSignatureIsValidAndReturnsJson()
    {
        $json = self::webhook()->verify(self::PAYLOAD, self::validHeaders());

        $this->assertEquals(2432232315, $json['test'], "did not return expected json");
    }

    public function testValidSignatureAlongsideOtherSignaturesIsValid()
    {
        $headers = self::validHeaders();
        $headers['webhook-signature'] = implode(" ", [
            "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
            "v2a,not-base64!",
            "v2a,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
            "v2a",
            $headers['webhook-signature'],
        ]);

        $json = self::webhook()->verify(self::PAYLOAD, $headers);

        $this->assertEquals(2432232315, $json['test'], "did not return expected json");
    }

    public function testSignatureBySecondPublicKeyIsValid()
    {
        $keypair = sodium_crypto_sign_keypair();
        $timestamp = time();
        $headers = self::headers(self::sign($keypair, self::ENDPOINT_ID, $timestamp), $timestamp);

        $wh = new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, [self::PUBLIC_KEY, self::whpk($keypair)]);
        $json = $wh->verify(self::PAYLOAD, $headers);

        $this->assertEquals(2432232315, $json['test'], "did not return expected json");
    }

    public function testSignatureByUntrustedKeyThrowsException()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("No matching signature found");

        $timestamp = time();
        $headers = self::headers(self::sign(sodium_crypto_sign_keypair(), self::ENDPOINT_ID, $timestamp), $timestamp);
        self::webhook()->verify(self::PAYLOAD, $headers);
    }

    public function testSignatureForAnotherEndpointThrowsException()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("No matching signature found");

        $timestamp = time();
        $headers = self::headers(self::sign(self::specKeypair(), "ep_other", $timestamp), $timestamp);
        self::webhook()->verify(self::PAYLOAD, $headers);
    }

    public function testTamperedPayloadThrowsException()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("No matching signature found");

        self::webhook()->verify('{"test": 2432232316}', self::validHeaders());
    }

    public function testValidSignatureWithAnotherVersionThrowsException()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("No matching signature found");

        $headers = self::validHeaders();
        $headers['webhook-signature'] = preg_replace('/^v2a,/', 'v1a,', $headers['webhook-signature']);
        self::webhook()->verify(self::PAYLOAD, $headers);
    }

    public function testMissingIdThrowsException()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("Missing required headers");

        $headers = self::validHeaders();
        unset($headers['webhook-id']);
        self::webhook()->verify(self::PAYLOAD, $headers);
    }

    public function testMissingTimestampThrowsException()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("Missing required headers");

        $headers = self::validHeaders();
        unset($headers['webhook-timestamp']);
        self::webhook()->verify(self::PAYLOAD, $headers);
    }

    public function testMissingSignatureThrowsException()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("Missing required headers");

        $headers = self::validHeaders();
        unset($headers['webhook-signature']);
        self::webhook()->verify(self::PAYLOAD, $headers);
    }

    public function testOldTimestampThrowsException()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("Message timestamp too old");

        $timestamp = time() - self::TOLERANCE - 1;
        $headers = self::headers(self::sign(self::specKeypair(), self::ENDPOINT_ID, $timestamp), $timestamp);
        self::webhook()->verify(self::PAYLOAD, $headers);
    }

    public function testNewTimestampThrowsException()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("Message timestamp too new");

        $timestamp = time() + self::TOLERANCE + 1;
        $headers = self::headers(self::sign(self::specKeypair(), self::ENDPOINT_ID, $timestamp), $timestamp);
        self::webhook()->verify(self::PAYLOAD, $headers);
    }

    public function testRejectsEmptyPublicKeys()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, []);
    }

    public function testRejectsPublicKeyWithoutPrefix()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, [substr(self::PUBLIC_KEY, strlen("whpk_"))]);
    }

    public function testRejectsSecretKey()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, [self::SECRET_KEY]);
    }

    public function testRejectsShortPublicKey()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, ["whpk_" . base64_encode(str_repeat("a", 31))]);
    }

    public function testRejectsLongPublicKey()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, ["whpk_" . base64_encode(str_repeat("a", 33))]);
    }

    public function testRejectsInvalidBase64PublicKey()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, ["whpk_not-base64!"]);
    }

    public function testRejectsEmptyEndpointId()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetric("", [self::PUBLIC_KEY]);
    }

    public function testRejectsEndpointIdWithDot()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetric("ep_1.2", [self::PUBLIC_KEY]);
    }

    public function testChallengeResponse()
    {
        $this->assertEquals(self::CHALLENGE_RESPONSE, self::webhook()->endpointVerificationResponse(self::CHALLENGE));
    }

    public function testChallengeResponseUsesFirstPublicKey()
    {
        $otherKey = self::whpk(sodium_crypto_sign_keypair());

        $wh = new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, [self::PUBLIC_KEY, $otherKey]);
        $this->assertEquals(self::CHALLENGE_RESPONSE, $wh->endpointVerificationResponse(self::CHALLENGE));

        $wh = new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, [$otherKey, self::PUBLIC_KEY]);
        $this->assertNotEquals(self::CHALLENGE_RESPONSE, $wh->endpointVerificationResponse(self::CHALLENGE));
    }

    public function testVerifyChallengeResponseAcceptsAnyPublicKey()
    {
        $otherKey = self::whpk(sodium_crypto_sign_keypair());

        $wh = new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, [$otherKey, self::PUBLIC_KEY]);
        $wh->verifyEndpointVerificationResponse(self::CHALLENGE, self::CHALLENGE_RESPONSE);
        $this->addToAssertionCount(1);
    }

    public function testVerifyChallengeResponseRejectsWrongHeader()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("No matching signature found");

        self::webhook()->verifyEndpointVerificationResponse(self::CHALLENGE, "v2a,abc");
    }

    public function testVerifyChallengeResponseRejectsWrongVersion()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("No matching signature found");

        self::webhook()->verifyEndpointVerificationResponse(
            self::CHALLENGE,
            preg_replace('/^v2a,/', 'v1,', self::CHALLENGE_RESPONSE)
        );
    }

    public function testVerifyChallengeResponseRejectsWrongChallenge()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookVerificationException::class);
        $this->expectExceptionMessage("No matching signature found");

        self::webhook()->verifyEndpointVerificationResponse("wrong", self::CHALLENGE_RESPONSE);
    }

    public function testSignedContentHashVector()
    {
        $digest = \StandardWebhooks\WebhookAsymmetric::signedContentHash(
            self::ENDPOINT_ID,
            self::VECTOR_MSG_ID,
            self::VECTOR_TIMESTAMP,
            self::VECTOR_PAYLOAD
        );
        $this->assertEquals(32, strlen($digest));
        $this->assertEquals(self::VECTOR_CONTENT_HASH, base64_encode($digest));
    }

    public function testSignedContentHashRejectsEmptyEndpointId()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        \StandardWebhooks\WebhookAsymmetric::signedContentHash("", self::MSG_ID, time(), self::PAYLOAD);
    }

    public function testSignedContentHashRejectsEndpointIdWithDot()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        \StandardWebhooks\WebhookAsymmetric::signedContentHash("ep_1.2", self::MSG_ID, time(), self::PAYLOAD);
    }

    public function testSignedContentHashRejectsInvalidTimestamp()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookSigningException::class);
        \StandardWebhooks\WebhookAsymmetric::signedContentHash(self::ENDPOINT_ID, self::MSG_ID, "abc", self::PAYLOAD);
    }

    public function testFormatSignatureRoundTrip()
    {
        $signature = base64_decode(substr(self::VECTOR_SIGNATURE, strlen("v2a,")));
        $this->assertEquals(self::VECTOR_SIGNATURE, \StandardWebhooks\WebhookAsymmetric::formatSignature($signature));
    }

    public function testFormatSignatureRejectsShortSignature()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookSigningException::class);
        \StandardWebhooks\WebhookAsymmetric::formatSignature(str_repeat("a", 63));
    }

    public function testFormatSignatureRejectsLongSignature()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookSigningException::class);
        \StandardWebhooks\WebhookAsymmetric::formatSignature(str_repeat("a", 65));
    }

    public function testFormatPublicKeyRaw()
    {
        $publicKey = base64_decode("uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w=");
        $this->assertEquals(self::PUBLIC_KEY, \StandardWebhooks\WebhookAsymmetric::formatPublicKey($publicKey));
    }

    public function testFormatPublicKeyDer()
    {
        $publicKey = hex2bin("302a300506032b6570032100") . base64_decode("uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w=");
        $this->assertEquals(self::PUBLIC_KEY, \StandardWebhooks\WebhookAsymmetric::formatPublicKey($publicKey));
    }

    public function testFormatPublicKeyRejectsShortKey()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookSigningException::class);
        \StandardWebhooks\WebhookAsymmetric::formatPublicKey(str_repeat("a", 31));
    }

    public function testFormatPublicKeyRejectsLongKey()
    {
        $this->expectException(\StandardWebhooks\Exception\WebhookSigningException::class);
        \StandardWebhooks\WebhookAsymmetric::formatPublicKey(str_repeat("a", 33));
    }

    public function testFormatPublicKeyRejectsWrongDerPrefix()
    {
        $publicKey = hex2bin("312a300506032b6570032100") . base64_decode("uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w=");
        $this->expectException(\StandardWebhooks\Exception\WebhookSigningException::class);
        \StandardWebhooks\WebhookAsymmetric::formatPublicKey($publicKey);
    }

    public function testSignerPublicKeyVector()
    {
        $signer = new \StandardWebhooks\WebhookAsymmetricSigner(self::SECRET_KEY);
        $this->assertEquals(self::PUBLIC_KEY, $signer->publicKey());
    }

    public function testSignerSignatureVector()
    {
        $signer = new \StandardWebhooks\WebhookAsymmetricSigner(self::SECRET_KEY);
        $signature = $signer->sign(self::ENDPOINT_ID, self::VECTOR_MSG_ID, self::VECTOR_TIMESTAMP, self::VECTOR_PAYLOAD);
        $this->assertEquals(self::VECTOR_SIGNATURE, $signature);
    }

    public function testSignerSignatureVerifies()
    {
        $signer = new \StandardWebhooks\WebhookAsymmetricSigner(self::SECRET_KEY);
        $timestamp = time();
        $signature = $signer->sign(self::ENDPOINT_ID, self::MSG_ID, $timestamp, self::PAYLOAD);

        $wh = new \StandardWebhooks\WebhookAsymmetric(self::ENDPOINT_ID, [$signer->publicKey()]);
        $json = $wh->verify(self::PAYLOAD, self::headers($signature, $timestamp));

        $this->assertEquals(2432232315, $json['test'], "did not return expected json");
    }

    public function testSignerRejectsEmptyEndpointId()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        $signer = new \StandardWebhooks\WebhookAsymmetricSigner(self::SECRET_KEY);
        $signer->sign("", self::MSG_ID, time(), self::PAYLOAD);
    }

    public function testSignerRejectsEndpointIdWithDot()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        $signer = new \StandardWebhooks\WebhookAsymmetricSigner(self::SECRET_KEY);
        $signer->sign("ep_1.2", self::MSG_ID, time(), self::PAYLOAD);
    }

    public function testSignerRejectsSecretKeyWithoutPrefix()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetricSigner(substr(self::SECRET_KEY, strlen("whsk_")));
    }

    public function testSignerRejectsPublicKey()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetricSigner(self::PUBLIC_KEY);
    }

    public function testSignerRejectsInvalidBase64SecretKey()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetricSigner("whsk_not-base64!");
    }

    public function testSignerRejectsShortSecretKey()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetricSigner("whsk_" . base64_encode(str_repeat("a", 31)));
    }

    public function testSignerRejectsLongSecretKey()
    {
        $this->expectException(\StandardWebhooks\Exception\EmptyWebhookSecretException::class);
        new \StandardWebhooks\WebhookAsymmetricSigner("whsk_" . base64_encode(str_repeat("a", 33)));
    }
}
