using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Crypto.Signers;
using System;

namespace StandardWebhooks
{
    /// <summary>
    /// Signs webhooks with asymmetric (v2a) signatures, for producers that hold the private key themselves.
    /// It's recommended to hold the private key in a KMS or an HSM instead, and sign using
    /// <see cref="WebhookAsymmetric.SignedContentHash"/> and <see cref="WebhookAsymmetric.FormatSignature"/>.
    /// </summary>
    /// <remarks>
    /// The signer isn't tied to an endpoint: a single key pair can serve all endpoints, as the endpoint ID is part
    /// of the signed content.
    /// </remarks>
    public sealed class WebhookAsymmetricSigner
    {
        private const string SECRET_KEY_PREFIX = "whsk_";

        private readonly Ed25519PrivateKeyParameters secretKey;

        /// <summary>
        /// Creates a signer from a secret key.
        /// </summary>
        /// <param name="secretKey">
        /// <c>whsk_</c> followed by the base64 encoded 32 byte ed25519 private key (seed).
        /// </param>
        /// <exception cref="ArgumentException">The secret key is invalid.</exception>
        public WebhookAsymmetricSigner(string secretKey)
        {
            if (secretKey == null || !secretKey.StartsWith(SECRET_KEY_PREFIX, StringComparison.Ordinal))
            {
                throw new ArgumentException($"Webhook secret key must start with {SECRET_KEY_PREFIX}", nameof(secretKey));
            }

            byte[] key;
            try
            {
                key = Convert.FromBase64String(secretKey.Substring(SECRET_KEY_PREFIX.Length));
            }
            catch (FormatException e)
            {
                throw new ArgumentException("Webhook secret key must be base64 encoded", nameof(secretKey), e);
            }

            if (key.Length != Ed25519PrivateKeyParameters.KeySize)
            {
                throw new ArgumentException($"Webhook secret key must be {Ed25519PrivateKeyParameters.KeySize} bytes", nameof(secretKey));
            }
            this.secretKey = new Ed25519PrivateKeyParameters(key, 0);
        }

        /// <summary>
        /// Signs a webhook.
        /// </summary>
        /// <param name="endpointId">The ID of the endpoint the webhook is sent to.</param>
        /// <param name="msgId">The webhook's message ID.</param>
        /// <param name="timestamp">The webhook's timestamp.</param>
        /// <param name="payload">The webhook's payload.</param>
        /// <returns>The value for the webhook-signature header (<c>v2a,&lt;base64&gt;</c>).</returns>
        /// <exception cref="ArgumentException">The endpoint ID is invalid.</exception>
        public string Sign(string endpointId, string msgId, DateTimeOffset timestamp, string payload)
        {
            var digest = WebhookAsymmetric.SignedContentHash(endpointId, msgId, timestamp, payload);
            var signer = new Ed25519Signer();
            signer.Init(true, this.secretKey);
            signer.BlockUpdate(digest, 0, digest.Length);
            return WebhookAsymmetric.FormatSignature(signer.GenerateSignature());
        }

        /// <summary>
        /// The public key (<c>whpk_</c> prefixed), to share with consumers.
        /// </summary>
        public string PublicKey()
        {
            return WebhookAsymmetric.FormatPublicKey(this.secretKey.GeneratePublicKey().GetEncoded());
        }
    }
}
