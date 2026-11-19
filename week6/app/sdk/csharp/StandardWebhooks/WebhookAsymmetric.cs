using Org.BouncyCastle.Crypto.Parameters;
using Org.BouncyCastle.Crypto.Signers;
using StandardWebhooks.Exceptions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Security.Cryptography;

namespace StandardWebhooks
{
    /// <summary>
    /// Verifies asymmetric (v2a) webhook signatures.
    /// </summary>
    public sealed class WebhookAsymmetric
    {
        internal const string PUBLIC_KEY_PREFIX = "whpk_";
        private const string SIGNATURE_PREFIX = "v2a,";
        private const int SIGNATURE_LENGTH = 64;

        private readonly byte[][] publicKeys;
        private readonly Ed25519PublicKeyParameters[] publicKeyParameters;
        private readonly string endpointId;

        /// <summary>
        /// Creates a verifier for an endpoint's asymmetric signatures.
        /// </summary>
        /// <param name="endpointId">The endpoint ID assigned by the webhook producer.</param>
        /// <param name="publicKeys">
        /// The endpoint's public keys (<c>whpk_</c> prefixed), newest first. A signature made by any of them is
        /// accepted, so keys can be rotated by adding the new key to the start of the list. The first key is used
        /// for endpoint verification responses.
        /// </param>
        /// <exception cref="EmptyWebhookSecretException">No public keys were given.</exception>
        /// <exception cref="ArgumentException">A public key or the endpoint ID is invalid.</exception>
        public WebhookAsymmetric(string endpointId, IEnumerable<string> publicKeys)
        {
            var keys = publicKeys?.ToArray();
            if (keys == null || keys.Length == 0)
            {
                throw new EmptyWebhookSecretException("Missing webhook public keys");
            }
            ValidateEndpointId(endpointId);

            this.publicKeys = keys.Select(ParsePublicKey).ToArray();
            this.publicKeyParameters = this.publicKeys.Select(key => new Ed25519PublicKeyParameters(key, 0)).ToArray();
            this.endpointId = endpointId;
        }

        private static void ValidateEndpointId(string endpointId)
        {
            if (String.IsNullOrEmpty(endpointId))
            {
                throw new ArgumentException("Missing endpoint ID", nameof(endpointId));
            }
            if (endpointId.Contains("."))
            {
                throw new ArgumentException("Endpoint ID must not contain '.'", nameof(endpointId));
            }
        }

        /// <summary>
        /// Calculates the SHA-256 hash of the signed content (<c>endpoint_id.msg_id.timestamp.payload</c>).
        /// Producers that hold the private key in an HSM or a KMS sign this hash with pure ed25519, and pass the
        /// signature to <see cref="FormatSignature"/>.
        /// </summary>
        /// <param name="endpointId">The ID of the endpoint the webhook is sent to.</param>
        /// <param name="msgId">The webhook's message ID.</param>
        /// <param name="timestamp">The webhook's timestamp.</param>
        /// <param name="payload">The webhook's payload.</param>
        /// <returns>The raw 32 bytes of the hash.</returns>
        /// <exception cref="ArgumentException">The endpoint ID is invalid.</exception>
        public static byte[] SignedContentHash(string endpointId, string msgId, DateTimeOffset timestamp, string payload)
        {
            ValidateEndpointId(endpointId);

            var content = $"{endpointId}.{msgId}.{timestamp.ToUnixTimeSeconds().ToString()}.{payload}";
            using (var sha256 = SHA256.Create())
            {
                return sha256.ComputeHash(Webhook.SafeUTF8Encoding.GetBytes(content));
            }
        }

        /// <summary>
        /// Formats a raw ed25519 signature as a webhook-signature header value (<c>v2a,&lt;base64&gt;</c>).
        /// </summary>
        /// <param name="signature">The 64 byte ed25519 signature.</param>
        /// <exception cref="ArgumentException">The signature isn't 64 bytes.</exception>
        public static string FormatSignature(byte[] signature)
        {
            if (signature == null || signature.Length != SIGNATURE_LENGTH)
            {
                throw new ArgumentException($"Signature must be {SIGNATURE_LENGTH} bytes", nameof(signature));
            }
            return SIGNATURE_PREFIX + Convert.ToBase64String(signature);
        }

        // The DER prefix of an ed25519 SubjectPublicKeyInfo, followed by the 32 byte public key.
        private static readonly byte[] ED25519_SPKI_PREFIX = { 0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00 };

        /// <summary>
        /// Formats an ed25519 public key as a <c>whpk_</c> public key, to share with consumers. Useful for producers
        /// that hold the private key in an HSM or a KMS.
        /// </summary>
        /// <param name="publicKey">
        /// The raw 32 byte ed25519 public key, or the DER encoded SubjectPublicKeyInfo of an ed25519 public key (as
        /// returned by e.g. AWS KMS <c>GetPublicKey</c>).
        /// </param>
        /// <exception cref="ArgumentException">The public key isn't a raw or DER encoded ed25519 public key.</exception>
        public static string FormatPublicKey(byte[] publicKey)
        {
            var keySize = Ed25519PublicKeyParameters.KeySize;
            if (publicKey != null && publicKey.Length == keySize)
            {
                return PUBLIC_KEY_PREFIX + Convert.ToBase64String(publicKey);
            }
            if (publicKey != null && publicKey.Length == ED25519_SPKI_PREFIX.Length + keySize
                && publicKey.Take(ED25519_SPKI_PREFIX.Length).SequenceEqual(ED25519_SPKI_PREFIX))
            {
                return PUBLIC_KEY_PREFIX + Convert.ToBase64String(publicKey, ED25519_SPKI_PREFIX.Length, keySize);
            }
            throw new ArgumentException(
                $"Public key must be a {keySize} byte ed25519 public key or its DER encoded SubjectPublicKeyInfo",
                nameof(publicKey));
        }

        private static byte[] ParsePublicKey(string publicKey)
        {
            if (publicKey == null || !publicKey.StartsWith(PUBLIC_KEY_PREFIX, StringComparison.Ordinal))
            {
                throw new ArgumentException($"Webhook public key must start with {PUBLIC_KEY_PREFIX}", "publicKeys");
            }

            byte[] key;
            try
            {
                key = Convert.FromBase64String(publicKey.Substring(PUBLIC_KEY_PREFIX.Length));
            }
            catch (FormatException e)
            {
                throw new ArgumentException("Webhook public key must be base64 encoded", "publicKeys", e);
            }

            if (key.Length != Ed25519PublicKeyParameters.KeySize)
            {
                throw new ArgumentException($"Webhook public key must be {Ed25519PublicKeyParameters.KeySize} bytes", "publicKeys");
            }
            return key;
        }

        public void Verify(string payload, WebHeaderCollection headers)
        {
            string msgId = headers.Get(Webhook.UNBRANDED_ID_HEADER_KEY);
            string msgSignature = headers.Get(Webhook.UNBRANDED_SIGNATURE_HEADER_KEY);
            string msgTimestamp = headers.Get(Webhook.UNBRANDED_TIMESTAMP_HEADER_KEY);
            if (String.IsNullOrEmpty(msgId) || String.IsNullOrEmpty(msgSignature) || String.IsNullOrEmpty(msgTimestamp))
            {
                throw new WebhookVerificationException("Missing Required Headers");
            }

            var timestamp = Webhook.VerifyTimestamp(msgTimestamp);

            VerifySignature(msgId, timestamp, payload, msgSignature);
        }

        /// <summary>
        /// Verifies the signature header against the given message, without checking the timestamp.
        /// </summary>
        internal void VerifySignature(string msgId, DateTimeOffset timestamp, string payload, string msgSignature)
        {
            var digest = SignedContentHash(this.endpointId, msgId, timestamp, payload);

            foreach (string versionedSignature in msgSignature.Split(' '))
            {
                if (!versionedSignature.StartsWith(SIGNATURE_PREFIX, StringComparison.Ordinal))
                {
                    continue;
                }

                byte[] signature;
                try
                {
                    signature = Convert.FromBase64String(versionedSignature.Substring(SIGNATURE_PREFIX.Length));
                }
                catch (FormatException)
                {
                    continue;
                }
                if (signature.Length != SIGNATURE_LENGTH)
                {
                    continue;
                }

                foreach (var publicKey in this.publicKeyParameters)
                {
                    var verifier = new Ed25519Signer();
                    verifier.Init(false, publicKey);
                    verifier.BlockUpdate(digest, 0, digest.Length);
                    if (verifier.VerifySignature(signature))
                    {
                        return;
                    }
                }
            }
            throw new WebhookVerificationException("No matching signature found");
        }

        /// <summary>
        /// Calculates the response to an endpoint verification challenge, using the first (newest) public key.
        /// </summary>
        /// <param name="challenge">The value of the webhook-challenge request header.</param>
        /// <returns>The value for the webhook-challenge-response response header.</returns>
        public string EndpointVerificationResponse(string challenge)
        {
            return EndpointVerificationResponse(this.publicKeys[0], challenge);
        }

        /// <summary>
        /// Verifies the response to an endpoint verification challenge. The response is accepted if it was
        /// calculated with any of the public keys.
        /// </summary>
        /// <param name="challenge">The challenge sent in the webhook-challenge request header.</param>
        /// <param name="header">The value of the webhook-challenge-response response header.</param>
        public void VerifyEndpointVerificationResponse(string challenge, string header)
        {
            if (String.IsNullOrEmpty(challenge) || String.IsNullOrEmpty(header))
            {
                throw new WebhookVerificationException("Missing challenge response");
            }

            foreach (var publicKey in this.publicKeys)
            {
                var expectedSignature = EndpointVerificationResponse(publicKey, challenge);
                if (Utils.SecureCompare(expectedSignature, header))
                {
                    return;
                }
            }

            throw new WebhookVerificationException("No matching signature found");
        }

        private string EndpointVerificationResponse(byte[] publicKey, string challenge)
        {
            var endpointIdBytes = Webhook.SafeUTF8Encoding.GetBytes("." + this.endpointId);
            var endpointKey = new byte[publicKey.Length + endpointIdBytes.Length];
            Buffer.BlockCopy(publicKey, 0, endpointKey, 0, publicKey.Length);
            Buffer.BlockCopy(endpointIdBytes, 0, endpointKey, publicKey.Length, endpointIdBytes.Length);

            var signature = Convert.ToBase64String(Webhook.EndpointVerificationSignature(endpointKey, challenge));
            return $"{SIGNATURE_PREFIX}{signature}";
        }
    }
}
