/**
 * Module descriptor seen by Java 11+ runtimes (META-INF/versions/11).
 * Keep in sync with src/main/java9/module-info.java: a versioned descriptor may
 * only add non-transitive requires of java.* modules (so the requires of
 * org.bouncycastle.provider must be in both), and java.net.http (used by the
 * HttpHeaders overload of verify) does not exist before Java 11.
 */
module com.standardwebhooks {
	requires org.bouncycastle.provider;
	requires java.net.http;

	exports com.standardwebhooks;
	exports com.standardwebhooks.exceptions;
}
