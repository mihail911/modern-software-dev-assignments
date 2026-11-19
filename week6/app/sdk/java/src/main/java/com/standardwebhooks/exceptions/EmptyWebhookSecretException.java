package com.standardwebhooks.exceptions;

public class EmptyWebhookSecretException extends Exception {

	public EmptyWebhookSecretException(final String message) {
		super(message);
	}
}
