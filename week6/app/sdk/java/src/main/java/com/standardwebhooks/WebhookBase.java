package com.standardwebhooks;

import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

import com.standardwebhooks.exceptions.EmptyWebhookSecretException;
import com.standardwebhooks.exceptions.WebhookVerificationException;
import com.standardwebhooks.exceptions.WebhookSigningException;

/**
 * Base class containing all shared webhook verification logic.
 * This class is used by the all java verion implementations.
 * Package-private to prevent extension outside this package.
 */
abstract class WebhookBase {
	public static final String SECRET_PREFIX = "whsec_";
	public static final String UNBRANDED_MSG_ID_KEY = "webhook-id";
	public static final String UNBRANDED_MSG_SIGNATURE_KEY = "webhook-signature";
	public static final String UNBRANDED_MSG_TIMESTAMP_KEY = "webhook-timestamp";
	private static final String HMAC_SHA256 = "HmacSHA256";
	private static final int TOLERANCE_IN_SECONDS = 5 * 60; // 5 minutes
	private static final long SECOND_IN_MS = 1000L;

	protected final byte[] key;

	protected WebhookBase(final String secret)
			throws EmptyWebhookSecretException {
		String sec = secret;
		if (sec.startsWith(WebhookBase.SECRET_PREFIX)) {
			sec = sec.substring(WebhookBase.SECRET_PREFIX.length());
		}
		this.key = Base64.getDecoder().decode(sec);
		if (this.key.length == 0) {
			throw new EmptyWebhookSecretException("Webhook secret should not be empty");
		}
	}

	protected WebhookBase(final byte[] secret)
			throws EmptyWebhookSecretException {
		this.key = secret;
		if (this.key.length == 0) {
			throw new EmptyWebhookSecretException("Webhook secret should not be empty");
		}
	}

	/**
	 * Verify webhook signature using Map-based headers (Java 8 compatible).
	 *
	 * @param payload The webhook payload to verify
	 * @param headers Map of header names to list of values
	 * @throws WebhookVerificationException if verification fails
	 */
	public final void verify(final String payload, final Map<String, List<String>> headers) throws WebhookVerificationException {
		String msgId = getFirstHeader(headers, UNBRANDED_MSG_ID_KEY);
		String msgSignature = getFirstHeader(headers, UNBRANDED_MSG_SIGNATURE_KEY);
		String msgTimestamp = getFirstHeader(headers, UNBRANDED_MSG_TIMESTAMP_KEY);

		if (msgId == null || msgSignature == null || msgTimestamp == null) {
			throw new WebhookVerificationException("Missing required headers");
		}

		long timestamp = WebhookBase.verifyTimestamp(msgTimestamp);

		String expectedSignature;
		try {
			expectedSignature = this.sign(msgId, timestamp, payload);
		} catch (WebhookSigningException e) {
			throw new WebhookVerificationException("Failed to generate expected signature");
		}

		byte[] expected = expectedSignature.getBytes(StandardCharsets.UTF_8);
		for (String versionedSignature : msgSignature.split(" ")) {
			if (MessageDigest.isEqual(versionedSignature.getBytes(StandardCharsets.UTF_8), expected)) {
				return;
			}
		}
		throw new WebhookVerificationException("No matching signature found");
	}

	/**
	 * Case-insensitive header lookup helper. Shared with {@link WebhookAsymmetricBase}.
	 */
	static String getFirstHeader(Map<String, List<String>> headers, String name) {
		for (Map.Entry<String, List<String>> entry : headers.entrySet()) {
			if (entry.getKey().equalsIgnoreCase(name)) {
				List<String> values = entry.getValue();
				if (values != null && !values.isEmpty()) {
					return values.get(0);
				}
			}
		}
		return null;
	}

	public final String sign(final String msgId, final long timestamp, final String payload) throws WebhookSigningException {
		try {
			String toSign = String.format("%s.%s.%s", msgId, timestamp, payload);
			Mac sha512Hmac = Mac.getInstance(HMAC_SHA256);
			SecretKeySpec keySpec = new SecretKeySpec(this.key, HMAC_SHA256);
			sha512Hmac.init(keySpec);
			byte[] macData = sha512Hmac.doFinal(toSign.getBytes(StandardCharsets.UTF_8));
			String signature = Base64.getEncoder().encodeToString(macData);
			return String.format("v1,%s", signature);
		} catch (InvalidKeyException | NoSuchAlgorithmException e) {
			throw new WebhookSigningException(e.getMessage());
		}
	}

	/**
	 * Calculate the response to an endpoint verification challenge.
	 *
	 * @param challenge The value of the webhook-challenge request header
	 * @return The value for the webhook-challenge-response response header
	 * @throws WebhookSigningException if the response can't be calculated
	 */
	public final String endpointVerificationResponse(final String challenge) throws WebhookSigningException {
		String signature = Base64.getEncoder().encodeToString(WebhookBase.endpointVerificationSignature(this.key, challenge));
		return String.format("v1,%s", signature);
	}

	/**
	 * Calculate the raw endpoint verification signature of a challenge for an endpoint key.
	 * Shared with {@link WebhookAsymmetricBase}.
	 */
	static byte[] endpointVerificationSignature(final byte[] endpointKey, final String challenge) throws WebhookSigningException {
		try {
			Mac hmac = Mac.getInstance(HMAC_SHA256);
			hmac.init(new SecretKeySpec(endpointKey, HMAC_SHA256));
			byte[] verificationKey = hmac.doFinal("wev-v1".getBytes(StandardCharsets.UTF_8));
			hmac.init(new SecretKeySpec(verificationKey, HMAC_SHA256));
			return hmac.doFinal(challenge.getBytes(StandardCharsets.UTF_8));
		} catch (InvalidKeyException | NoSuchAlgorithmException e) {
			throw new WebhookSigningException(e.getMessage());
		}
	}

	/**
	 * Verify the response to an endpoint verification challenge.
	 *
	 * @param challenge The challenge sent in the webhook-challenge request header
	 * @param header The value of the webhook-challenge-response response header
	 * @throws WebhookVerificationException if verification fails
	 */
	public final void verifyEndpointVerificationResponse(final String challenge, final String header) throws WebhookVerificationException {
		if (challenge == null || header == null) {
			throw new WebhookVerificationException("Missing challenge response");
		}

		String expectedSignature;
		try {
			expectedSignature = this.endpointVerificationResponse(challenge);
		} catch (WebhookSigningException e) {
			throw new WebhookVerificationException("Failed to generate expected signature");
		}

		if (!MessageDigest.isEqual(header.getBytes(StandardCharsets.UTF_8), expectedSignature.getBytes(StandardCharsets.UTF_8))) {
			throw new WebhookVerificationException("No matching signature found");
		}
	}

	/**
	 * Parse the timestamp header and check it's within the tolerance. Shared with {@link WebhookAsymmetricBase}.
	 */
	static long verifyTimestamp(final String timestampHeader) throws WebhookVerificationException {
		long now = System.currentTimeMillis() / WebhookBase.SECOND_IN_MS;

		long timestamp;
		try {
			timestamp = Long.parseLong(timestampHeader);
		} catch (NumberFormatException e) {
			throw new WebhookVerificationException("Invalid Signature Headers");
		}

		if (timestamp < (now - TOLERANCE_IN_SECONDS)) {
			throw new WebhookVerificationException("Message timestamp too old");
		}
		if (timestamp > (now + TOLERANCE_IN_SECONDS)) {
			throw new WebhookVerificationException("Message timestamp too new");
		}
		return timestamp;
	}
}
