using Xunit;

using System;
using System.Linq;
using System.Net;
using System.Security.Cryptography;
using System.Text;

using StandardWebhooks.Exceptions;

namespace StandardWebhooks.Tests
{
    public class WebhookAsymmetricTests
    {
        public const int TOLERANCE_IN_SECONDS = 5 * 60;

        // Test-only secret key (ed25519 seed) from the spec. Never use it for anything real.
        private const string SECRET_KEY = "whsk_4uKPjji/dhB8w6GW3YYDnIMYvXcdHrvH6lidTr4Z+EY=";
        private const string PUBLIC_KEY = "whpk_uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w=";
        private const string ENDPOINT_ID = "ep_2fVZdC8Kx9TR4ZoCqHWt3NnLgRb";
        private const string MSG_ID = "msg_2KWPBgLlAfxdpx2AI54pPJ85f4W";
        private const long TIMESTAMP = 1674087231;
        private const string PAYLOAD = "{\"type\":\"contact.created\",\"timestamp\":\"2022-11-03T20:26:10.344522Z\",\"data\":{\"id\":\"1f81eb52-5198-4599-803e-771906343485\"}}";
        private const string DIGEST = "YXCi5iN+rjB+iEb6bvg1XbWbVBL+r2S+as3k3n1Ih/A=";
        private const string SIGNATURE = "v2a,kG9XaxZjAEpKB8KeQQb5G1Xp73YH5Rf5gxoWe7cg1kYMiyspdtZXe9w+dAwKeOurAwxfqmlOG/QR4m9s37DzAg==";
        private const string CHALLENGE = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=";
        private const string CHALLENGE_RESPONSE = "v2a,t0RDFhzk7OtVzgvMsjuXnodSWuby2BVX+zPqw2/ksUk=";

        private static readonly WebhookAsymmetricSigner SecretKey = new WebhookAsymmetricSigner(SECRET_KEY);

        // A second, unrelated key pair, for key rotation tests
        private static readonly WebhookAsymmetricSigner OtherSecretKey =
            new WebhookAsymmetricSigner("whsk_" + Convert.ToBase64String(new byte[32]));
        private static readonly string OtherPublicKey = OtherSecretKey.PublicKey();

        private static string Sign(WebhookAsymmetricSigner key, string endpointId, string msgId, long timestamp, string payload)
        {
            return key.Sign(endpointId, msgId, DateTimeOffset.FromUnixTimeSeconds(timestamp), payload);
        }

        private static WebHeaderCollection Headers(string msgId, long timestamp, string signature)
        {
            var headers = new WebHeaderCollection();
            headers.Set("webhook-id", msgId);
            headers.Set("webhook-timestamp", timestamp.ToString());
            headers.Set("webhook-signature", signature);
            return headers;
        }

        private static long Now()
        {
            return DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        }

        private static WebhookAsymmetric Wh(params string[] publicKeys)
        {
            return new WebhookAsymmetric(ENDPOINT_ID, publicKeys);
        }

        [Fact]
        public void TestSpecVector()
        {
            var timestamp = DateTimeOffset.FromUnixTimeSeconds(TIMESTAMP);
            Assert.Equal(DIGEST, Convert.ToBase64String(WebhookAsymmetric.SignedContentHash(ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD)));
            Assert.Equal(PUBLIC_KEY, SecretKey.PublicKey());
            Assert.Equal(SIGNATURE, Sign(SecretKey, ENDPOINT_ID, MSG_ID, TIMESTAMP, PAYLOAD));

            // The vector's timestamp is outside the tolerance, so skip the timestamp check
            Wh(PUBLIC_KEY).VerifySignature(MSG_ID, DateTimeOffset.FromUnixTimeSeconds(TIMESTAMP), PAYLOAD, SIGNATURE);
            Assert.Throws<WebhookVerificationException>(() => Wh(PUBLIC_KEY).Verify(PAYLOAD, Headers(MSG_ID, TIMESTAMP, SIGNATURE)));
        }

        [Fact]
        public void TestValidSignatureIsValid()
        {
            var timestamp = Now();
            var signature = Sign(SecretKey, ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD);
            Wh(PUBLIC_KEY).Verify(PAYLOAD, Headers(MSG_ID, timestamp, signature));
        }

        [Fact]
        public void TestMultiSigPayloadIsValid()
        {
            var timestamp = Now();
            var signature = Sign(SecretKey, ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD);
            string[] sigs = new string[] {
                "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
                "v2a,invalid",
                "v1a,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
                signature,
            };
            Wh(PUBLIC_KEY).Verify(PAYLOAD, Headers(MSG_ID, timestamp, String.Join(" ", sigs)));
        }

        [Fact]
        public void TestSignatureBySecondKeyIsValid()
        {
            var timestamp = Now();
            var signature = Sign(SecretKey, ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD);
            Wh(OtherPublicKey, PUBLIC_KEY).Verify(PAYLOAD, Headers(MSG_ID, timestamp, signature));
        }

        [Fact]
        public void TestSignatureByUnknownKeyThrowsException()
        {
            var timestamp = Now();
            var signature = Sign(OtherSecretKey, ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD);
            Assert.Throws<WebhookVerificationException>(() => Wh(PUBLIC_KEY).Verify(PAYLOAD, Headers(MSG_ID, timestamp, signature)));
        }

        [Fact]
        public void TestWrongEndpointIdThrowsException()
        {
            var timestamp = Now();
            var signature = Sign(SecretKey, "ep_other", MSG_ID, timestamp, PAYLOAD);
            Assert.Throws<WebhookVerificationException>(() => Wh(PUBLIC_KEY).Verify(PAYLOAD, Headers(MSG_ID, timestamp, signature)));
        }

        [Fact]
        public void TestTamperedPayloadThrowsException()
        {
            var timestamp = Now();
            var signature = Sign(SecretKey, ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD);
            Assert.Throws<WebhookVerificationException>(() => Wh(PUBLIC_KEY).Verify(PAYLOAD + " ", Headers(MSG_ID, timestamp, signature)));
        }

        [Fact]
        public void TestCorrectSignatureWithDifferentVersionThrowsException()
        {
            var timestamp = Now();
            var signature = Sign(SecretKey, ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD);
            var header = "v1," + signature.Substring("v2a,".Length);
            Assert.Throws<WebhookVerificationException>(() => Wh(PUBLIC_KEY).Verify(PAYLOAD, Headers(MSG_ID, timestamp, header)));
        }

        [Theory]
        [InlineData("webhook-id")]
        [InlineData("webhook-timestamp")]
        [InlineData("webhook-signature")]
        public void TestMissingHeaderThrowsException(string header)
        {
            var timestamp = Now();
            var headers = Headers(MSG_ID, timestamp, Sign(SecretKey, ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD));
            headers.Remove(header);
            Assert.Throws<WebhookVerificationException>(() => Wh(PUBLIC_KEY).Verify(PAYLOAD, headers));
        }

        [Fact]
        public void TestOldTimestampThrowsException()
        {
            var timestamp = Now() - (TOLERANCE_IN_SECONDS + 1);
            var signature = Sign(SecretKey, ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD);
            Assert.Throws<WebhookVerificationException>(() => Wh(PUBLIC_KEY).Verify(PAYLOAD, Headers(MSG_ID, timestamp, signature)));
        }

        [Fact]
        public void TestNewTimestampThrowsException()
        {
            var timestamp = Now() + (TOLERANCE_IN_SECONDS + 1);
            var signature = Sign(SecretKey, ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD);
            Assert.Throws<WebhookVerificationException>(() => Wh(PUBLIC_KEY).Verify(PAYLOAD, Headers(MSG_ID, timestamp, signature)));
        }

        [Fact]
        public void TestConstructorErrors()
        {
            Assert.Throws<EmptyWebhookSecretException>(() => new WebhookAsymmetric(ENDPOINT_ID, new string[0]));
            Assert.Throws<ArgumentException>(() => Wh(PUBLIC_KEY.Substring("whpk_".Length)));
            Assert.Throws<ArgumentException>(() => Wh("whsec_uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w="));
            Assert.Throws<ArgumentException>(() => Wh("whpk_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"));
            Assert.Throws<ArgumentException>(() => Wh("whpk_not base64!"));
            Assert.Throws<ArgumentException>(() => new WebhookAsymmetric("", new[] { PUBLIC_KEY }));
            Assert.Throws<ArgumentException>(() => new WebhookAsymmetric("ep_1.2", new[] { PUBLIC_KEY }));
        }

        [Fact]
        public void TestSignerRoundTrip()
        {
            var signer = new WebhookAsymmetricSigner(SECRET_KEY);
            var timestamp = DateTimeOffset.UtcNow;
            var signature = signer.Sign(ENDPOINT_ID, MSG_ID, timestamp, PAYLOAD);
            Wh(signer.PublicKey()).Verify(PAYLOAD, Headers(MSG_ID, timestamp.ToUnixTimeSeconds(), signature));
        }

        [Fact]
        public void TestSignerConstructorErrors()
        {
            var unprefixed = SECRET_KEY.Substring("whsk_".Length);
            Assert.Throws<ArgumentException>(() => new WebhookAsymmetricSigner(null));
            Assert.Throws<ArgumentException>(() => new WebhookAsymmetricSigner(unprefixed));
            Assert.Throws<ArgumentException>(() => new WebhookAsymmetricSigner("whpk_" + unprefixed));
            Assert.Throws<ArgumentException>(() => new WebhookAsymmetricSigner(PUBLIC_KEY));
            Assert.Throws<ArgumentException>(() => new WebhookAsymmetricSigner("whsk_not base64!"));
            Assert.Throws<ArgumentException>(() => new WebhookAsymmetricSigner("whsk_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"));
            Assert.Throws<ArgumentException>(() => new WebhookAsymmetricSigner("whsk_"));
        }

        [Theory]
        [InlineData("")]
        [InlineData(null)]
        [InlineData("ep_1.2")]
        public void TestSigningWithInvalidEndpointIdThrowsException(string endpointId)
        {
            var timestamp = DateTimeOffset.UtcNow;
            Assert.Throws<ArgumentException>(() => WebhookAsymmetric.SignedContentHash(endpointId, MSG_ID, timestamp, PAYLOAD));
            Assert.Throws<ArgumentException>(() => SecretKey.Sign(endpointId, MSG_ID, timestamp, PAYLOAD));
        }

        [Fact]
        public void TestFormatSignature()
        {
            var signature = Convert.FromBase64String(SIGNATURE.Substring("v2a,".Length));
            Assert.Equal(SIGNATURE, WebhookAsymmetric.FormatSignature(signature));

            Assert.Throws<ArgumentException>(() => WebhookAsymmetric.FormatSignature(new byte[63]));
            Assert.Throws<ArgumentException>(() => WebhookAsymmetric.FormatSignature(new byte[65]));
            Assert.Throws<ArgumentException>(() => WebhookAsymmetric.FormatSignature(new byte[0]));
            Assert.Throws<ArgumentException>(() => WebhookAsymmetric.FormatSignature(null));
        }

        [Fact]
        public void TestFormatPublicKey()
        {
            var raw = Convert.FromBase64String(PUBLIC_KEY.Substring("whpk_".Length));
            Assert.Equal(32, raw.Length);
            Assert.Equal(PUBLIC_KEY, WebhookAsymmetric.FormatPublicKey(raw));

            var der = new byte[] { 0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00 }.Concat(raw).ToArray();
            Assert.Equal(PUBLIC_KEY, WebhookAsymmetric.FormatPublicKey(der));

            var wrongPrefix = (byte[])der.Clone();
            wrongPrefix[0] = 0x31;
            Assert.Throws<ArgumentException>(() => WebhookAsymmetric.FormatPublicKey(raw.Take(31).ToArray()));
            Assert.Throws<ArgumentException>(() => WebhookAsymmetric.FormatPublicKey(raw.Concat(new byte[] { 0 }).ToArray()));
            Assert.Throws<ArgumentException>(() => WebhookAsymmetric.FormatPublicKey(wrongPrefix));
            Assert.Throws<ArgumentException>(() => WebhookAsymmetric.FormatPublicKey(null));
        }

        [Fact]
        public void verifyChallengeResponseWorks()
        {
            Assert.Equal(CHALLENGE_RESPONSE, Wh(PUBLIC_KEY).EndpointVerificationResponse(CHALLENGE));
            // Uses the first (newest) key
            Assert.Equal(CHALLENGE_RESPONSE, Wh(PUBLIC_KEY, OtherPublicKey).EndpointVerificationResponse(CHALLENGE));
            Assert.NotEqual(CHALLENGE_RESPONSE, Wh(OtherPublicKey, PUBLIC_KEY).EndpointVerificationResponse(CHALLENGE));
        }

        [Fact]
        public void verifyChallengeResponseVerificationKey()
        {
            var endpointKey = new byte[32 + 1 + ENDPOINT_ID.Length];
            Convert.FromBase64String(PUBLIC_KEY.Substring("whpk_".Length)).CopyTo(endpointKey, 0);
            Encoding.UTF8.GetBytes("." + ENDPOINT_ID).CopyTo(endpointKey, 32);
            using (var hmac = new HMACSHA256(endpointKey))
            {
                var verificationKey = hmac.ComputeHash(Encoding.UTF8.GetBytes("wev-v1"));
                Assert.Equal("YYCPt0lD/zon3FFk7MHznJHji3Td5EEGcithf+hw924=", Convert.ToBase64String(verificationKey));
            }
        }

        [Fact]
        public void verifyChallengeResponseVerifiesWorks()
        {
            var wh = Wh(OtherPublicKey, PUBLIC_KEY);

            wh.VerifyEndpointVerificationResponse(CHALLENGE, CHALLENGE_RESPONSE);
            Assert.Throws<WebhookVerificationException>(() => wh.VerifyEndpointVerificationResponse(CHALLENGE, "v1," + CHALLENGE_RESPONSE.Substring("v2a,".Length)));
            Assert.Throws<WebhookVerificationException>(() => wh.VerifyEndpointVerificationResponse("wrong", CHALLENGE_RESPONSE));
            Assert.Throws<WebhookVerificationException>(() => wh.VerifyEndpointVerificationResponse(CHALLENGE, null));
            Assert.Throws<WebhookVerificationException>(() => Wh(OtherPublicKey).VerifyEndpointVerificationResponse(CHALLENGE, CHALLENGE_RESPONSE));
        }
    }
}
