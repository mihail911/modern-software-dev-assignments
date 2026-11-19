package com.standardwebhooks;

import java.net.http.HttpHeaders;
import java.util.List;

import com.standardwebhooks.exceptions.WebhookVerificationException;
import com.standardwebhooks.exceptions.EmptyWebhookSecretException;

/**
 * A class for verifying asymmetric (v2a) webhook signatures.
 */
public final class WebhookAsymmetric extends WebhookAsymmetricBase {

	/**
	 * @param endpointId The endpoint ID assigned by the webhook producer
	 * @param publicKeys The endpoint's public keys (whpk_ prefixed), newest first. A signature made by
	 *                   any of them is accepted, so keys can be rotated by adding the new key to the
	 *                   start of the list. The first key is used for endpoint verification responses.
	 * @throws EmptyWebhookSecretException if no public keys are given
	 * @throws IllegalArgumentException if a public key or the endpoint ID is invalid
	 */
	public WebhookAsymmetric(final String endpointId, final List<String> publicKeys) throws EmptyWebhookSecretException {
		super(endpointId, publicKeys);
	}

	/**
	 * Verify webhook signature using HttpHeaders
	 *
	 * @param payload The webhook payload to verify
	 * @param headers HttpHeaders containing webhook headers
	 * @throws WebhookVerificationException if verification fails
	 */
	public void verify(final String payload, final HttpHeaders headers) throws WebhookVerificationException {
		// Convert HttpHeaders to Map for base class verify method
		verify(payload, headers.map());
	}
}
