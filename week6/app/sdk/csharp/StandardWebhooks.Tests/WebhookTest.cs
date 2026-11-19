using Xunit;

using System;
using System.Net;

using StandardWebhooks.Exceptions;

namespace StandardWebhooks.Tests
{

    class TestPayload
    {
        internal const string UNBRANDED_ID_HEADER_KEY = "webhook-id";
        internal const string UNBRANDED_SIGNATURE_HEADER_KEY = "webhook-signature";
        internal const string UNBRANDED_TIMESTAMP_HEADER_KEY = "webhook-timestamp";
        private const string DEFAULT_MSG_ID = "msg_p5jXN8AQM9LWM0D4loKWxJek";
        private const string DEFAULT_PAYLOAD = "{\"test\": 2432232314}";
        private const string DEFAULT_SECRET = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw";

        public string id;
        public DateTimeOffset timestamp;
        public WebHeaderCollection headers;

        public string secret;
        public string payload;
        public string signature;
        public TestPayload(DateTimeOffset timestamp)
        {
            id = DEFAULT_MSG_ID;
            this.timestamp = timestamp;

            payload = DEFAULT_PAYLOAD;
            secret = DEFAULT_SECRET;

            Webhook wh = new Webhook(secret);
            var signature = wh.Sign(id, this.timestamp, payload);

            headers = new WebHeaderCollection();
            headers.Set(UNBRANDED_ID_HEADER_KEY, id);
            headers.Set(UNBRANDED_SIGNATURE_HEADER_KEY, signature);
            headers.Set(UNBRANDED_TIMESTAMP_HEADER_KEY, timestamp.ToUnixTimeSeconds().ToString());
        }
    }

    public class WebhookTests
    {
        public const int TOLERANCE_IN_SECONDS = 5 * 60;

        [Fact]
        public void TestEmptySecretRasiesException()
        {
            Assert.Throws<EmptyWebhookSecretException>(() => new Webhook(""));
            Assert.Throws<EmptyWebhookSecretException>(() => new Webhook("whsec_"));
        }

        [Fact]
        public void TestMissingIdRasiesException()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);
            testPayload.headers.Remove(TestPayload.UNBRANDED_ID_HEADER_KEY);

            var wh = new Webhook(testPayload.secret);

            Assert.Throws<WebhookVerificationException>(() => wh.Verify(testPayload.payload, testPayload.headers));
        }

        [Fact]
        public void TestMissingTimestampThrowsException()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);
            testPayload.headers.Remove(TestPayload.UNBRANDED_TIMESTAMP_HEADER_KEY);

            var wh = new Webhook(testPayload.secret);

            Assert.Throws<WebhookVerificationException>(() => wh.Verify(testPayload.payload, testPayload.headers));
        }

        [Fact]
        public void TestMissingSignatureThrowsException()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);
            testPayload.headers.Remove(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY);

            var wh = new Webhook(testPayload.secret);

            Assert.Throws<WebhookVerificationException>(() => wh.Verify(testPayload.payload, testPayload.headers));
        }

        [Fact]
        public void TestInvalidSignatureThrowsException()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);
            testPayload.headers.Set(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY, "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLawdd");

            var wh = new Webhook(testPayload.secret);

            Assert.Throws<WebhookVerificationException>(() => wh.Verify(testPayload.payload, testPayload.headers));
        }

        [Fact]
        public void TestValidSignatureIsValid()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);
            var wh = new Webhook(testPayload.secret);

            wh.Verify(testPayload.payload, testPayload.headers);
        }

        [Fact]
        public void TestUnbrandedSignatureIsValid()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);
            WebHeaderCollection unbrandedHeaders = new WebHeaderCollection();
            unbrandedHeaders.Set("webhook-id", testPayload.headers.Get(TestPayload.UNBRANDED_ID_HEADER_KEY));
            unbrandedHeaders.Set("webhook-signature", testPayload.headers.Get(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY));
            unbrandedHeaders.Set("webhook-timestamp", testPayload.headers.Get(TestPayload.UNBRANDED_TIMESTAMP_HEADER_KEY));
            testPayload.headers = unbrandedHeaders;

            var wh = new Webhook(testPayload.secret);

            wh.Verify(testPayload.payload, testPayload.headers);
        }

        [Fact]
        public void TestOldTimestampThrowsException()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow.AddSeconds(-1 * (TOLERANCE_IN_SECONDS + 1)));

            var wh = new Webhook(testPayload.secret);

            Assert.Throws<WebhookVerificationException>(() => wh.Verify(testPayload.payload, testPayload.headers));
        }

        [Fact]
        public void TestNewTimestampThrowsException()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow.AddSeconds(TOLERANCE_IN_SECONDS + 1));

            var wh = new Webhook(testPayload.secret);

            Assert.Throws<WebhookVerificationException>(() => wh.Verify(testPayload.payload, testPayload.headers));
        }

        [Fact]
        public void TestMultiSigPayloadIsValid()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);

            string[] sigs = new string[] {
                "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
                "v2,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
                testPayload.headers.Get(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY), // valid signature
                "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
            };
            testPayload.headers.Set(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY, String.Join(" ", sigs));

            var wh = new Webhook(testPayload.secret);

            wh.Verify(testPayload.payload, testPayload.headers);
        }

        [Fact]
        public void TestEntryWithoutCommaFollowedByValidSignatureIsValid()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);
            var validSignature = testPayload.headers.Get(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY);
            testPayload.headers.Set(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY, "invalid " + validSignature);

            var wh = new Webhook(testPayload.secret);

            wh.Verify(testPayload.payload, testPayload.headers);
        }

        [Fact]
        public void TestMalformedEntryFollowedByValidSignatureIsValid()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);
            var validSignature = testPayload.headers.Get(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY);
            testPayload.headers.Set(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY, "v1, v1,,, " + validSignature);

            var wh = new Webhook(testPayload.secret);

            wh.Verify(testPayload.payload, testPayload.headers);
        }

        [Fact]
        public void TestCorrectSignatureWithDifferentVersionThrowsException()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);
            var validSignature = testPayload.headers.Get(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY);
            testPayload.headers.Set(TestPayload.UNBRANDED_SIGNATURE_HEADER_KEY, "v2," + validSignature.Split(',')[1]);

            var wh = new Webhook(testPayload.secret);

            Assert.Throws<WebhookVerificationException>(() => wh.Verify(testPayload.payload, testPayload.headers));
        }

        [Fact]
        public void TestSivnatureVerificationWorksWithoutPrefix()
        {
            var testPayload = new TestPayload(DateTimeOffset.UtcNow);

            var wh = new Webhook(testPayload.secret);
            wh.Verify(testPayload.payload, testPayload.headers);

            wh = new Webhook("whsec_" + testPayload.secret);
            wh.Verify(testPayload.payload, testPayload.headers);
        }

        [Fact]
        public void verifyWebhookSignWorks()
        {
            var key = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw";
            var msgId = "msg_p5jXN8AQM9LWM0D4loKWxJek";
            var timestamp = DateTimeOffset.FromUnixTimeSeconds(1614265330);
            var payload = "{\"test\": 2432232314}";
            var expected = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=";

            var wh = new Webhook(key);
            var signature = wh.Sign(msgId, timestamp, payload);
            Assert.Equal(signature, expected);
        }

        [Fact]
        public void verifyChallengeResponseWorks()
        {
            var wh = new Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");
            var response = wh.EndpointVerificationResponse("0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=");
            Assert.Equal("v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=", response);
        }

        [Fact]
        public void verifyChallengeResponseVerifiesWorks()
        {
            var wh = new Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");
            var challenge = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=";
            var header = "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=";

            wh.VerifyEndpointVerificationResponse(challenge, header);
            Assert.Throws<WebhookVerificationException>(() => wh.VerifyEndpointVerificationResponse(challenge, $"v2,abc {header}"));
            Assert.Throws<WebhookVerificationException>(() => wh.VerifyEndpointVerificationResponse("wrong", header));
        }
    }
}
