using StandardWebhooks.Exceptions;
using System;
using System.Net;
using System.Security.Cryptography;
using System.Text;

namespace StandardWebhooks
{
    public sealed class Webhook
    {

        internal static readonly UTF8Encoding SafeUTF8Encoding = new UTF8Encoding(false, true);
        internal const string UNBRANDED_ID_HEADER_KEY = "webhook-id";
        internal const string UNBRANDED_SIGNATURE_HEADER_KEY = "webhook-signature";
        internal const string UNBRANDED_TIMESTAMP_HEADER_KEY = "webhook-timestamp";

        private const int TOLERANCE_IN_SECONDS = 60 * 5;
        private static string prefix = "whsec_";
        private byte[] key;
        public Webhook(string key)
        {
            if (key.StartsWith(prefix))
            {
                key = key.Substring(prefix.Length);
            }

            this.key = Convert.FromBase64String(key);

            if (this.key == null || this.key.Length == 0)
            {
                throw new EmptyWebhookSecretException("Missing webhook secret");
            }
        }

        public Webhook(byte[] key)
        {
            this.key = key;
        }

        public void Verify(string payload, WebHeaderCollection headers)
        {
            string msgId = headers.Get(UNBRANDED_ID_HEADER_KEY);
            string msgSignature = headers.Get(UNBRANDED_SIGNATURE_HEADER_KEY);
            string msgTimestamp = headers.Get(UNBRANDED_TIMESTAMP_HEADER_KEY);
            if (String.IsNullOrEmpty(msgId) || String.IsNullOrEmpty(msgSignature) || String.IsNullOrEmpty(msgTimestamp))
            {
                throw new WebhookVerificationException("Missing Required Headers");
            }

            var timestamp = Webhook.VerifyTimestamp(msgTimestamp);

            var expectedSignature = this.Sign(msgId, timestamp, payload);

            var passedSignatures = msgSignature.Split(' ');
            foreach (string versionedSignature in passedSignatures)
            {
                if (Utils.SecureCompare(expectedSignature, versionedSignature))
                {
                    return;
                }
            }
            throw new WebhookVerificationException("No matching signature found");
        }

        internal static DateTimeOffset VerifyTimestamp(string timestampHeader)
        {
            DateTimeOffset timestamp;
            var now = DateTimeOffset.UtcNow;
            try
            {
                var timestampInt = long.Parse(timestampHeader);
                timestamp = DateTimeOffset.FromUnixTimeSeconds(timestampInt);
            }
            catch
            {
                throw new WebhookVerificationException("Invalid Signature Headers");
            }

            if (timestamp < (now.AddSeconds(-1 * TOLERANCE_IN_SECONDS)))
            {
                throw new WebhookVerificationException("Message timestamp too old");
            }
            if (timestamp > (now.AddSeconds(TOLERANCE_IN_SECONDS)))
            {
                throw new WebhookVerificationException("Message timestamp too new");
            }
            return timestamp;
        }

        public string Sign(string msgId, DateTimeOffset timestamp, string payload)
        {
            var toSign = $"{msgId}.{timestamp.ToUnixTimeSeconds().ToString()}.{payload}";
            var toSignBytes = SafeUTF8Encoding.GetBytes(toSign);
            using (var hmac = new HMACSHA256(this.key))
            {
                var hash = hmac.ComputeHash(toSignBytes);
                var signature = Convert.ToBase64String(hash);
                return $"v1,{signature}";
            }
        }

        /// <summary>
        /// Calculates the response to an endpoint verification challenge.
        /// </summary>
        /// <param name="challenge">The value of the webhook-challenge request header.</param>
        /// <returns>The value for the webhook-challenge-response response header.</returns>
        public string EndpointVerificationResponse(string challenge)
        {
            var signature = Convert.ToBase64String(EndpointVerificationSignature(this.key, challenge));
            return $"v1,{signature}";
        }

        internal static byte[] EndpointVerificationSignature(byte[] endpointKey, string challenge)
        {
            byte[] verificationKey;
            using (var hmac = new HMACSHA256(endpointKey))
            {
                verificationKey = hmac.ComputeHash(SafeUTF8Encoding.GetBytes("wev-v1"));
            }
            using (var hmac = new HMACSHA256(verificationKey))
            {
                return hmac.ComputeHash(SafeUTF8Encoding.GetBytes(challenge));
            }
        }

        /// <summary>
        /// Verifies the response to an endpoint verification challenge.
        /// </summary>
        /// <param name="challenge">The challenge sent in the webhook-challenge request header.</param>
        /// <param name="header">The value of the webhook-challenge-response response header.</param>
        public void VerifyEndpointVerificationResponse(string challenge, string header)
        {
            if (String.IsNullOrEmpty(challenge) || String.IsNullOrEmpty(header))
            {
                throw new WebhookVerificationException("Missing challenge response");
            }

            var expectedSignature = this.EndpointVerificationResponse(challenge);

            if (!Utils.SecureCompare(expectedSignature, header))
            {
                throw new WebhookVerificationException("No matching signature found");
            }
        }
    }
}
