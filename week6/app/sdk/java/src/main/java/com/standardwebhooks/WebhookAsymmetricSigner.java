package com.standardwebhooks;

import java.util.Base64;

import org.bouncycastle.crypto.params.Ed25519PrivateKeyParameters;
import org.bouncycastle.crypto.signers.Ed25519Signer;

import com.standardwebhooks.exceptions.WebhookSigningException;

/**
 * A class for signing webhooks with asymmetric (v2a) signatures, for producers that hold the private
 * key themselves. It's recommended to hold the private key in a KMS or an HSM instead, and sign using
 * {@link WebhookAsymmetric#signedContentHash(String, String, long, String)} and
 * {@link WebhookAsymmetric#formatSignature(byte[])}.
 *
 * <p>The signer isn't tied to an endpoint: a single key pair can serve all endpoints, as the endpoint
 * ID is part of the signed content.
 */
public final class WebhookAsymmetricSigner {
	public static final String SECRET_KEY_PREFIX = "whsk_";

	private final Ed25519PrivateKeyParameters secretKey;

	/**
	 * @param secretKey whsk_ followed by the base64 encoded 32 byte ed25519 private key (seed)
	 * @throws IllegalArgumentException if the secret key is invalid
	 */
	public WebhookAsymmetricSigner(final String secretKey) {
		if (secretKey == null || !secretKey.startsWith(SECRET_KEY_PREFIX)) {
			throw new IllegalArgumentException(String.format("Webhook secret key must start with %s", SECRET_KEY_PREFIX));
		}
		byte[] key = Base64.getDecoder().decode(secretKey.substring(SECRET_KEY_PREFIX.length()));
		if (key.length != Ed25519PrivateKeyParameters.KEY_SIZE) {
			throw new IllegalArgumentException(
				String.format("Webhook secret key must be %d bytes", Ed25519PrivateKeyParameters.KEY_SIZE));
		}
		this.secretKey = new Ed25519PrivateKeyParameters(key, 0);
	}

	/**
	 * Sign a webhook.
	 *
	 * @param endpointId The ID of the endpoint the webhook is sent to
	 * @param msgId The webhook's message ID
	 * @param timestamp The webhook's timestamp, in seconds since the epoch
	 * @param payload The webhook's payload
	 * @return The value for the webhook-signature header ({@code v2a,<base64>})
	 * @throws IllegalArgumentException if the endpoint ID is invalid
	 * @throws WebhookSigningException if the signature can't be calculated
	 */
	public String sign(final String endpointId, final String msgId, final long timestamp, final String payload)
			throws WebhookSigningException {
		byte[] digest = WebhookAsymmetric.signedContentHash(endpointId, msgId, timestamp, payload);
		Ed25519Signer signer = new Ed25519Signer();
		signer.init(true, this.secretKey);
		signer.update(digest, 0, digest.length);
		return WebhookAsymmetric.formatSignature(signer.generateSignature());
	}

	/**
	 * @return The public key (whpk_ prefixed), to share with consumers
	 */
	public String publicKey() {
		return WebhookAsymmetric.formatPublicKey(this.secretKey.generatePublicKey().getEncoded());
	}
}
