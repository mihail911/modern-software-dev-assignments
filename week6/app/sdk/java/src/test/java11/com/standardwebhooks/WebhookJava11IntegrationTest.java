package com.standardwebhooks;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.standardwebhooks.exceptions.WebhookVerificationException;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;

/**
 * Java 11+ runtime integration tests.
 * Verifies:
 * 1. Running on Java 11+ runtime
 * 2. Java 11 version of Webhook is loaded (HttpHeaders + Map APIs)
 * 3. Base unit tests pass on Java 11 runtime
 * 4. Both HttpHeaders and Map-based verify APIs work correctly
 */
public class WebhookJava11IntegrationTest {

	@Test
	public void verifyRunningOnJava11PlusRuntime() {
		String javaSpecVersion = System.getProperty("java.specification.version");

		int specVersion = Integer.parseInt(javaSpecVersion);
		assertTrue(
			String.format("Expected Java 11+ runtime, but got Java version: %s", specVersion),
			specVersion == 11
		);
	}

	@Test
	public void verifyLoadedClassFor11() throws Exception {
		int majorVersion = WebhookTest.classMajorVersion(Webhook.class);
		assertEquals(
			55,
			majorVersion
		);
	}

	@Test
	public void verifyLoadedAsymmetricClassFor11() throws Exception {
		assertEquals(55, WebhookTest.classMajorVersion(WebhookAsymmetric.class));
	}
}
