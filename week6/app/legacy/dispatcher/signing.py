"""Request signing.

Every request to a merchant carries
    X-Switchyard-Signature: sha256=<hex HMAC-SHA256 of the raw body, keyed with the endpoint secret>
    X-Switchyard-Delivery: <delivery id>
"""

import hashlib
import hmac

SIGNATURE_HEADER = "X-Switchyard-Signature"
DELIVERY_HEADER = "X-Switchyard-Delivery"
PREFIX = "sha256="


def sign(secret, body):
    """The signature header value for a request body (bytes)."""
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return PREFIX + digest


def verify(secret, body, header):
    """True when the header value is the signature of body under secret."""
    if not isinstance(header, str) or not header.startswith(PREFIX):
        return False
    return hmac.compare_digest(sign(secret, body), header)
