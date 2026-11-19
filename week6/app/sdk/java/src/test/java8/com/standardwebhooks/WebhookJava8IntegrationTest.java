package com.standardwebhooks;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import com.standardwebhooks.exceptions.WebhookVerificationException;

import java.beans.Transient;
import java.lang.reflect.Method;
import org.junit.Test;
import org.junit.runner.JUnitCore;
import org.junit.runner.Result;

/**
 * Java 8 runtime integration tests. s
 * 1. Running on Java 8 runtime
 * 2. Base unit tests pass on Java 8 runtime
 */
public class WebhookJava8IntegrationTest {

	@Test
	public void verifyRunningOnJava8Runtime() {
		String javaVersion = System.getProperty("java.version");
		String javaSpecVersion = System.getProperty("java.specification.version");

		assertTrue(
			String.format("Expected Java 8 runtime, but got Java version: %s (spec: %s)", javaVersion, javaSpecVersion),
			javaSpecVersion.equals("1.8") || javaSpecVersion.equals("8")
		); 
	}

	@Test
	public void verifyLoadedClassFor8() throws Exception {
		int majorVersion = WebhookTest.classMajorVersion(Webhook.class);
		assertEquals(
			// "Class bytecode in java 8 should be 52",
			52,
			majorVersion
		);
	}

	@Test
	public void verifyLoadedAsymmetricClassFor8() throws Exception {
		assertEquals(52, WebhookTest.classMajorVersion(WebhookAsymmetric.class));
	}
}
