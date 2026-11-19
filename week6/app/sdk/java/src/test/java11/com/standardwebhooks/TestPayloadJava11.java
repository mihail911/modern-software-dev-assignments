package com.standardwebhooks;

import java.net.http.HttpHeaders;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.BiPredicate;

/**
 * Java 11-specific test utilities.
 * Extends the base TestScenario with HttpHeaders support.
 *
 * Inherits all builder methods from TestScenario and adds HttpHeaders conversion.
 * Example: TestPayloadJava11.valid().withMissingId().headersAsHttpHeaders()
 */
public class TestPayloadJava11 extends TestScenario {

	public TestPayloadJava11(final long timestampInMS) {
		super(timestampInMS);
	}

	/**
	 * Creates a valid test payload with current timestamp.
	 * Convenience factory method that returns TestPayloadJava11 instead of TestPayload.
	 */
	public static TestPayloadJava11 valid() {
		return new TestPayloadJava11(System.currentTimeMillis());
	}

	public HttpHeaders headersAsHttpHeaders() {
		HashMap<String, List<String>> map = new HashMap<>();
		for (Map.Entry<String, java.util.ArrayList<String>> entry : this.headerMap.entrySet()) {
			map.put(entry.getKey(), entry.getValue());
		}

		return HttpHeaders.of(map, new BiPredicate<String, String>() {
			@Override
			public boolean test(String arg0, String arg1) {
				return true;
			}
		});
	}
}
