package com.standardwebhooks;

import com.standardwebhooks.exceptions.EmptyWebhookSecretException;

/**
 *  A class for verifying and generating webhook signatures.
 */
public final class Webhook extends WebhookBase {

	public Webhook(final String secret) throws EmptyWebhookSecretException {
		super(secret);
	}

	public Webhook(final byte[] secret) throws EmptyWebhookSecretException {
		super(secret);
	}
}
