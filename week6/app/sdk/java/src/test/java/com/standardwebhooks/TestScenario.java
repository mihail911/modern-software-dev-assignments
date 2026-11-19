package com.standardwebhooks;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;

/**
 * Shared test utility for creating test payloads with valid signatures.
 * Can be used by both Java 8 and Java 11 tests.
 *
 * Provides factory methods for common test scenarios that can be reused across
 * WebhookTest and integration tests.
 */
public class TestScenario {
	private static final String DEFAULT_MSG_ID = "msg_p5jXN8AQM9LWM0D4loKWxJek";
	private static final String DEFAULT_SECRET = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw";
	private static final String DEFAULT_PAYLOAD = "{\"test\": 2432232314}";
	private static final int SECOND_IN_MS = 1000;
	private static final int TOLERANCE_IN_MS = 5 * 60 * 1000;

	public String id;
	public String timestamp;
	public String payload;
	public String secret;
	public String signature;
	public HashMap<String, ArrayList<String>> headerMap;

	public TestScenario(final long timestampInMS) {
		this.id = DEFAULT_MSG_ID;
		this.timestamp = String.valueOf(timestampInMS / SECOND_IN_MS);
		this.payload = DEFAULT_PAYLOAD;
		this.secret = DEFAULT_SECRET;

		try {
			String toSign = String.format("%s.%s.%s", this.id, this.timestamp, this.payload);
			Mac sha512Hmac = Mac.getInstance("HmacSHA256");
			SecretKeySpec keySpec = new SecretKeySpec(Base64.getDecoder().decode(this.secret), "HmacSHA256");
			sha512Hmac.init(keySpec);
			byte[] macData = sha512Hmac.doFinal(toSign.getBytes(StandardCharsets.UTF_8));
			this.signature = Base64.getEncoder().encodeToString(macData);
		} catch (Exception e) {
			// pass
		}

		this.headerMap = new HashMap<>();
		headerMap.put("webhook-id", new ArrayList<>(Arrays.asList(this.id)));
		headerMap.put("webhook-timestamp", new ArrayList<>(Arrays.asList(this.timestamp)));
		headerMap.put("webhook-signature", new ArrayList<>(Arrays.asList(String.format("v1,%s", this.signature))));
	}

	public Map<String, List<String>> headersAsMap() {
		Map<String, List<String>> map = new HashMap<>();
		for (Map.Entry<String, ArrayList<String>> entry : this.headerMap.entrySet()) {
			map.put(entry.getKey(), entry.getValue());
		}
		return map;
	}

	public String signatureHeader() {
		return String.format("v1,%s", this.signature);
	}

	// Factory method and builder-style methods for common test scenarios

	/**
	 * Creates a valid test payload with current timestamp.
	 */
	public static TestScenario valid() {
		return new TestScenario(System.currentTimeMillis());
	}

	/**
	 * Modifies the timestamp to be too old (beyond tolerance).
	 */
	public TestScenario withOldTimestamp() {
		long timestampInMS = System.currentTimeMillis() - (TOLERANCE_IN_MS + SECOND_IN_MS);
		this.timestamp = String.valueOf(timestampInMS / SECOND_IN_MS);
		headerMap.put("webhook-timestamp", new ArrayList<>(Arrays.asList(this.timestamp)));
		return this;
	}

	/**
	 * Modifies the timestamp to be too new (beyond tolerance).
	 */
	public TestScenario withFutureTimestamp() {
		long timestampInMS = System.currentTimeMillis() + TOLERANCE_IN_MS + SECOND_IN_MS;
		this.timestamp = String.valueOf(timestampInMS / SECOND_IN_MS);
		headerMap.put("webhook-timestamp", new ArrayList<>(Arrays.asList(this.timestamp)));
		return this;
	}

	/**
	 * Modifies this payload to include multiple signatures (some invalid, one valid).
	 * Returns this for method chaining.
	 */
	public TestScenario withMultipleSignatures() {
		String multipleSignatures = String.join(" ",
			"v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
			"v2,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
			this.signatureHeader(), // valid signature
			"v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc="
		);
		this.headerMap.put("webhook-signature", new ArrayList<>(Arrays.asList(multipleSignatures)));
		return this;
	}

	/**
	 * Sets the webhook-signature header to the given value. Returns this for method chaining.
	 */
	public TestScenario withSignatureHeader(final String header) {
		this.headerMap.put("webhook-signature", new ArrayList<>(Arrays.asList(header)));
		return this;
	}

	/**
	 * Removes the webhook-id header. Returns this for method chaining.
	 */
	public TestScenario withMissingId() {
		this.headerMap.remove("webhook-id");
		return this;
	}

	/**
	 * Removes the webhook-timestamp header. Returns this for method chaining.
	 */
	public TestScenario withMissingTimestamp() {
		this.headerMap.remove("webhook-timestamp");
		return this;
	}

	/**
	 * Removes the webhook-signature header. Returns this for method chaining.
	 */
	public TestScenario withMissingSignature() {
		this.headerMap.remove("webhook-signature");
		return this;
	}

	/**
	 * Sets wrong signature version (v2 instead of v1). Returns this for method chaining.
	 */
	public TestScenario withWrongVersion() {
		this.headerMap.put("webhook-signature",
			new ArrayList<>(Arrays.asList("v2,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=")));
		return this;
	}

	/**
	 * Sets invalid signature format (missing version/signature separator). Returns this for method chaining.
	 */
	public TestScenario withInvalidSignatureFormat() {
		this.headerMap.put("webhook-signature",
			new ArrayList<>(Arrays.asList("invalid_signature")));
		return this;
	}

	/**
	 * Sets invalid signature value. Returns this for method chaining.
	 */
	public TestScenario withInvalidSignatureValue() {
		this.headerMap.put("webhook-signature",
			new ArrayList<>(Arrays.asList("v1,invalid_signature")));
		return this;
	}

	/**
	 * Converts headers to mixed case. Returns this for method chaining.
	 */
	public TestScenario withMixedCaseHeaders() {
		HashMap<String, ArrayList<String>> mixedCaseHeaders = new HashMap<>();
		mixedCaseHeaders.put("Webhook-Id", this.headerMap.get("webhook-id"));
		mixedCaseHeaders.put("WEBHOOK-TIMESTAMP", this.headerMap.get("webhook-timestamp"));
		mixedCaseHeaders.put("webhook-SIGNATURE", this.headerMap.get("webhook-signature"));
		this.headerMap = mixedCaseHeaders;
		return this;
	}

	/**
	 * Creates a valid test payload with hard-coded known values for testing signing
	 */
	public static TestScenario validSigned() {
		TestScenario scenario = TestScenario.valid();

		scenario.secret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw";
		scenario.id = "msg_p5jXN8AQM9LWM0D4loKWxJek";
		scenario.timestamp = "1614265330";
		scenario.payload = "{\"test\": 2432232314}";
		scenario.signature = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE=";

		scenario.headerMap = new HashMap<>();
		scenario.headerMap.put("webhook-id", new ArrayList<>(Arrays.asList(scenario.id)));
		scenario.headerMap.put("webhook-timestamp", new ArrayList<>(Arrays.asList(scenario.timestamp)));
		scenario.headerMap.put("webhook-signature", new ArrayList<>(Arrays.asList(String.format("v1,%s", scenario.signature))));

		return scenario;
	}
}
