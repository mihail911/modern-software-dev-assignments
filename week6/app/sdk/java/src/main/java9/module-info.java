/**
 * Module descriptor seen by Java 9 and 10 runtimes (META-INF/versions/9).
 * Java 11+ runtimes see src/main/java11/module-info.java instead, which
 * additionally reads java.net.http for the HttpHeaders overload of verify.
 */
module com.standardwebhooks {
	requires org.bouncycastle.provider;
	exports com.standardwebhooks;
	exports com.standardwebhooks.exceptions;
}
