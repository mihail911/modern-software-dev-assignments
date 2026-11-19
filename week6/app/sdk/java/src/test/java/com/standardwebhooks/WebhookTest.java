package com.standardwebhooks;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;

import com.standardwebhooks.exceptions.WebhookVerificationException;
import com.standardwebhooks.exceptions.WebhookSigningException;
import com.standardwebhooks.exceptions.EmptyWebhookSecretException;

import org.junit.Test;
import org.junit.function.ThrowingRunnable;

/**
 * Unit tests for WebhookBase core logic.
 * Tests all business logic using the Webhook class.
 */
public class WebhookTest {

	static int classMajorVersion(Class<?> clazz) throws IOException {
		String resource = "/" + clazz.getName().replace('.', '/') + ".class";
		try (InputStream in = clazz.getResourceAsStream(resource)) {
			if (in == null) {
				throw new IllegalStateException("Cannot find class resource: " + resource);
			}

			byte[] header = new byte[8];
			int read = in.read(header);
			if (read < 8) {
				throw new EOFException("Not enough bytes for class header");
			}

			// bytes 6–7 = major version (big-endian)
			return ((header[6] & 0xFF) << 8) | (header[7] & 0xFF);
		}
	}



	@Test
	public void verifyValidPayloadAndHeaders() throws WebhookVerificationException, EmptyWebhookSecretException {
		TestScenario scenario = TestScenario.valid();
		Webhook webhook = new Webhook(scenario.secret);

		webhook.verify(scenario.payload, scenario.headersAsMap());
	}

	@Test
	public void verifyValidPayloadWithMultipleSignaturesIsValid() throws WebhookVerificationException, EmptyWebhookSecretException {
		TestScenario scenario = TestScenario.valid().withMultipleSignatures();

		Webhook webhook = new Webhook(scenario.secret);
		webhook.verify(scenario.payload, scenario.headersAsMap());
	}

	@Test
	public void verifyEntryWithoutCommaFollowedByValidSignatureIsValid() throws WebhookVerificationException, EmptyWebhookSecretException {
		TestScenario scenario = TestScenario.valid();
		scenario.withSignatureHeader("invalid " + scenario.signatureHeader());

		Webhook webhook = new Webhook(scenario.secret);
		webhook.verify(scenario.payload, scenario.headersAsMap());
	}

	@Test
	public void verifyMalformedEntryFollowedByValidSignatureIsValid() throws WebhookVerificationException, EmptyWebhookSecretException {
		TestScenario scenario = TestScenario.valid();
		scenario.withSignatureHeader("v1, v1,,, " + scenario.signatureHeader());

		Webhook webhook = new Webhook(scenario.secret);
		webhook.verify(scenario.payload, scenario.headersAsMap());
	}

	@Test
	public void verifyCorrectSignatureWithDifferentVersionThrowsException() {
		TestScenario scenario = TestScenario.valid();
		scenario.withSignatureHeader("v2," + scenario.signature);
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifyCorrectSignatureWithTrailingDataThrowsException() {
		TestScenario scenario = TestScenario.valid();
		scenario.withSignatureHeader(scenario.signatureHeader() + ",extra");
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifyMissingIdThrowsException() {
		TestScenario scenario = TestScenario.valid().withMissingId();
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifyMissingTimestampThrowsException() {
		TestScenario scenario = TestScenario.valid().withMissingTimestamp();
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifyMissingSignatureThrowsException() {
		TestScenario scenario = TestScenario.valid().withMissingSignature();
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifySignatureWithDifferentVersionThrowsException() {
		TestScenario scenario = TestScenario.valid().withWrongVersion();
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifyMissingPartsInSignatureThrowsException() {
		TestScenario scenario = TestScenario.valid().withInvalidSignatureFormat();
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifySignatureMismatchThrowsException() {
		TestScenario scenario = TestScenario.valid().withInvalidSignatureValue();
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifyOldTimestampThrowsException() {
		TestScenario scenario = TestScenario.valid().withOldTimestamp();
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifyNewTimestampThrowsException() {
		TestScenario scenario = TestScenario.valid().withFutureTimestamp();
		assertThrows(WebhookVerificationException.class, verify(scenario));
	}

	@Test
	public void verifySecretWorksWithOrWithoutPrefix() throws WebhookVerificationException, EmptyWebhookSecretException {
		TestScenario scenario = TestScenario.valid();

		Webhook webhook = new Webhook(scenario.secret);
		webhook.verify(scenario.payload, scenario.headersAsMap());

		webhook = new Webhook(String.format("%s%s", WebhookBase.SECRET_PREFIX, scenario.secret));
		webhook.verify(scenario.payload, scenario.headersAsMap());
	}

	@Test
	public void verifyCaseInsensitiveHeaders() throws WebhookVerificationException, EmptyWebhookSecretException {
		TestScenario scenario = TestScenario.valid().withMixedCaseHeaders();

		Webhook webhook = new Webhook(scenario.secret);
		webhook.verify(scenario.payload, scenario.headersAsMap());
	}

	@Test
	public void verifyWebhookSignWorks() throws WebhookSigningException, EmptyWebhookSecretException {
		TestScenario scenario = TestScenario.validSigned();
		Webhook webhook = new Webhook(scenario.secret);
		String signature = webhook.sign(scenario.id, Long.parseLong(scenario.timestamp), scenario.payload);
		assertEquals(signature, scenario.signature);
	}

	@Test
	public void verifyChallengeResponseWorks() throws WebhookSigningException, EmptyWebhookSecretException {
		Webhook webhook = new Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");
		String response = webhook.endpointVerificationResponse("0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=");
		assertEquals("v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=", response);
	}

	@Test
	public void verifyChallengeResponseVerifiesWorks() throws WebhookVerificationException, EmptyWebhookSecretException {
		final Webhook webhook = new Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw");
		final String challenge = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=";
		final String header = "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=";

		webhook.verifyEndpointVerificationResponse(challenge, header);
		assertThrows(WebhookVerificationException.class, new ThrowingRunnable() {
			@Override
			public void run() throws WebhookVerificationException {
				webhook.verifyEndpointVerificationResponse(challenge, "v2,abc " + header);
			}
		});
		assertThrows(WebhookVerificationException.class, new ThrowingRunnable() {
			@Override
			public void run() throws WebhookVerificationException {
				webhook.verifyEndpointVerificationResponse("wrong", header);
			}
		});
	}

	private ThrowingRunnable verify(final TestScenario scenario) {
		return new ThrowingRunnable() {
			@Override
			public void run() throws WebhookVerificationException, EmptyWebhookSecretException {
				Webhook webhook = new Webhook(scenario.secret);
				webhook.verify(scenario.payload, scenario.headersAsMap());
			}
		};
	}
}
