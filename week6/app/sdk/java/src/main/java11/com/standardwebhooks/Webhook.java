package com.standardwebhooks;

import java.net.http.HttpHeaders;
import com.standardwebhooks.exceptions.WebhookVerificationException;
import com.standardwebhooks.exceptions.EmptyWebhookSecretException;

/**
 * A class for verifying and generating webhook signatures.
 */
public final class Webhook extends WebhookBase {

	public Webhook(final String secret) throws EmptyWebhookSecretException {
		super(secret);
	}

	public Webhook(final byte[] secret) throws EmptyWebhookSecretException {
		super(secret);
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
