import base64
import binascii
import hashlib
import hmac
import json
import typing as t
from datetime import datetime, timedelta, timezone
from math import floor

if t.TYPE_CHECKING:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

# DER prefix of an ed25519 SubjectPublicKeyInfo, followed by the raw 32 byte key
_ED25519_SPKI_PREFIX = bytes.fromhex("302a300506032b6570032100")


def hmac_data(key: bytes, data: bytes) -> bytes:
    return hmac.new(key, data, hashlib.sha256).digest()


def _payload_bytes(data: t.Union[bytes, str]) -> bytes:
    """The payload as signed: the body bytes exactly as received. A str is encoded as UTF-8."""
    if isinstance(data, str):
        return data.encode()
    # Only bytes-like objects: bytes() also turns an int or a list of ints into bytes, which
    # would sign unrelated bytes instead of failing. A memoryview must be over single bytes; one
    # over wider items (e.g. memoryview(array.array("H", ...))) is rejected rather than signed
    # as its platform-endian underlying bytes.
    if isinstance(data, (bytes, bytearray)) or (isinstance(data, memoryview) and data.itemsize == 1):
        return bytes(data)
    raise TypeError("Expected payload to be of type str or bytes")


class WebhookVerificationError(Exception):
    pass


def _parse_headers(headers: t.Dict[str, str]) -> t.Tuple[str, str, datetime]:
    """Normalize the headers (case-insensitive names), and check the timestamp is in tolerance.

    Returns the message id, the signature header and the timestamp.
    """
    headers = {k.lower(): v for (k, v) in headers.items()}
    msg_id = headers.get("webhook-id")
    msg_signature = headers.get("webhook-signature")
    msg_timestamp = headers.get("webhook-timestamp")
    if not (msg_id and msg_timestamp and msg_signature):
        raise WebhookVerificationError("Missing required headers")

    return msg_id, msg_signature, _verify_timestamp(msg_timestamp)


def _verify_timestamp(timestamp_header: str) -> datetime:
    webhook_tolerance = timedelta(minutes=5)
    now = datetime.now(tz=timezone.utc)
    try:
        timestamp = datetime.fromtimestamp(float(timestamp_header), tz=timezone.utc)
    except Exception:
        raise WebhookVerificationError("Invalid Signature Headers")

    if timestamp < (now - webhook_tolerance):
        raise WebhookVerificationError("Message timestamp too old")
    if timestamp > (now + webhook_tolerance):
        raise WebhookVerificationError("Message timestamp too new")
    return timestamp


def _timestamp_str(timestamp: datetime) -> str:
    """The integer timestamp, as signed. Naive datetimes are interpreted as UTC."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return str(floor(timestamp.timestamp()))


def _challenge_response(endpoint_key: bytes, challenge: str, identifier: str) -> str:
    """The response to an endpoint verification challenge, for the given endpoint key."""
    verification_key = hmac_data(endpoint_key, b"wev-v1")
    signature = hmac_data(verification_key, challenge.encode())
    return f"{identifier},{base64.b64encode(signature).decode('utf-8')}"


class EmptyWebhookSecretError(Exception):
    pass


class Webhook:
    _SECRET_PREFIX: str = "whsec_"
    _whsecret: bytes

    def __init__(self, whsecret: t.Union[str, bytes]):
        if isinstance(whsecret, str):
            if whsecret.startswith(self._SECRET_PREFIX):
                whsecret = whsecret[len(self._SECRET_PREFIX) :]
            # add padding in case whsecret is unpadded base64 (b64decode skips extra padding)
            whsecret = base64.b64decode(whsecret + "==")

        if not whsecret:
            raise EmptyWebhookSecretError("webhook secret may not be empty")

        if isinstance(whsecret, bytes):
            self._whsecret = whsecret
        else:
            raise RuntimeError("Invalid webhook secret")

    def verify(
        self,
        data: t.Union[bytes, str],
        headers: t.Dict[str, str],
        *,
        json_parse: bool = True,
    ) -> t.Any:
        """
        Verify the given webhook headers against the body bytes (data).

        Pass the raw request body exactly as received. bytes are verified byte for byte, so
        bodies that aren't valid UTF-8 verify. A str is verified as its UTF-8 encoding.

        Args:
            json_parse (bool): Whether to parse the data as JSON on success (default: True)

        Returns:
            After successful verification: if json_parse is True, returns the
            JSON-parsed input data; if it json_parse False, returns None.

        Raises:
            WebhookVerificationError if one of the required headers is missing,
            invalid, too old or too new; or no matching signature is found.
        """

        payload = _payload_bytes(data)
        msg_id, msg_signature, timestamp = _parse_headers(headers)

        expected_sig = self.sign(msg_id=msg_id, timestamp=timestamp, data=payload).encode()
        passed_sigs = msg_signature.split(" ")
        if not any(hmac.compare_digest(expected_sig, versioned_sig.encode()) for versioned_sig in passed_sigs):
            raise WebhookVerificationError("No matching signature found")

        if json_parse:
            return json.loads(payload)

    def sign(self, msg_id: str, timestamp: datetime, data: t.Union[bytes, str]) -> str:
        """
        Sign the given payload and return the versioned signature string.

        Args:
            msg_id: The unique message id (sent as the `webhook-id` header).
            timestamp: The message timestamp (sent as the `webhook-timestamp` header).
                Timezone-aware datetimes are converted to their POSIX timestamp, so any
                timezone can be used. Naive datetimes are interpreted as UTC.
            data: The raw payload. bytes are signed byte for byte, so bodies that aren't
                valid UTF-8 are signed as sent. A str is signed as its UTF-8 encoding.
        """
        to_sign = f"{msg_id}.{_timestamp_str(timestamp)}.".encode() + _payload_bytes(data)
        signature = hmac_data(self._whsecret, to_sign)
        return f"v1,{base64.b64encode(signature).decode('utf-8')}"

    def endpoint_verification_response(self, challenge: str) -> str:
        """
        Calculate the response to an endpoint verification challenge.

        Args:
            challenge (str): The value of the `webhook-challenge` request header

        Returns:
            The value for the `webhook-challenge-response` response header.
        """
        return _challenge_response(self._whsecret, challenge, "v1")

    def verify_endpoint_verification_response(self, challenge: str, header: str) -> None:
        """
        Verify the response to an endpoint verification challenge.

        Args:
            challenge (str): The challenge sent in the `webhook-challenge` request header
            header (str): The value of the `webhook-challenge-response` response header

        Raises:
            WebhookVerificationError if no matching signature is found.
        """
        expected_sig = self.endpoint_verification_response(challenge)
        if not hmac.compare_digest(expected_sig.encode(), header.encode()):
            raise WebhookVerificationError("No matching signature found")


def _import_ed25519(name: str = "WebhookAsymmetric") -> t.Any:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:
        raise ImportError(
            f"{name} requires the 'cryptography' package. Install it with: pip install 'standardwebhooks[asymmetric]'"
        ) from e
    return ed25519


def _validate_endpoint_id(endpoint_id: str) -> None:
    if not isinstance(endpoint_id, str) or not endpoint_id:
        raise RuntimeError("Endpoint ID may not be empty")
    if "." in endpoint_id:
        raise RuntimeError("Endpoint ID may not contain '.'")


class WebhookAsymmetric:
    """
    Verify webhooks signed with asymmetric (`v2a`, ed25519) signatures.

    Producers can sign using `WebhookAsymmetricSigner`, or with a KMS or an HSM using
    `WebhookAsymmetric.signed_content_hash` and `WebhookAsymmetric.format_signature`.

    Requires the `asymmetric` extra: `pip install 'standardwebhooks[asymmetric]'`.
    """

    _PUBLIC_KEY_PREFIX: str = "whpk_"
    _public_keys: t.List[bytes]
    _verifiers: t.List["Ed25519PublicKey"]
    _endpoint_id: str

    def __init__(self, endpoint_id: str, public_keys: t.List[str]):
        """
        Args:
            endpoint_id: The endpoint ID the producer shared with you together with the
                public key. Never take it from the request.
            public_keys: The producer's trusted public keys (`whpk_...`). Webhooks signed with
                any of them are accepted, so keys can be rotated by adding the new key to the
                list. The first key must be the newest one: it's the one used for endpoint
                verification.
        """
        ed25519 = _import_ed25519()

        if isinstance(public_keys, str) or not public_keys:
            raise RuntimeError("Public keys may not be empty")

        self._public_keys = []
        self._verifiers = []
        for public_key in public_keys:
            if not isinstance(public_key, str) or not public_key.startswith(self._PUBLIC_KEY_PREFIX):
                raise RuntimeError(f"Invalid public key: must start with '{self._PUBLIC_KEY_PREFIX}'")
            try:
                key = base64.b64decode(public_key[len(self._PUBLIC_KEY_PREFIX) :], validate=True)
            except binascii.Error:
                raise RuntimeError("Invalid public key: not valid base64")
            if len(key) != 32:
                raise RuntimeError(f"Invalid public key: must be 32 bytes, got {len(key)}")
            try:
                verifier = ed25519.Ed25519PublicKey.from_public_bytes(key)
            except ValueError:
                raise RuntimeError("Invalid public key: not a valid ed25519 public key")
            self._public_keys.append(key)
            self._verifiers.append(verifier)

        _validate_endpoint_id(endpoint_id)
        self._endpoint_id = endpoint_id

    @staticmethod
    def signed_content_hash(endpoint_id: str, msg_id: str, timestamp: datetime, data: t.Union[bytes, str]) -> bytes:
        """
        The SHA-256 hash (raw 32 bytes) of the content signed by `v2a` signatures
        (`endpoint_id.msg_id.timestamp.payload`).

        For producers that sign with a KMS or an HSM: sign the returned hash using (pure)
        ed25519, and pass the signature to `format_signature`.

        Doesn't require the `asymmetric` extra.

        Args:
            endpoint_id: The ID of the endpoint the webhook is sent to.
            msg_id: The unique message id (sent as the `webhook-id` header).
            timestamp: The message timestamp (sent as the `webhook-timestamp` header).
                Naive datetimes are interpreted as UTC.
            data: The raw payload. bytes are hashed byte for byte. A str is hashed as its
                UTF-8 encoding.
        """
        _validate_endpoint_id(endpoint_id)
        content = f"{endpoint_id}.{msg_id}.{_timestamp_str(timestamp)}.".encode() + _payload_bytes(data)
        return hashlib.sha256(content).digest()

    @staticmethod
    def format_signature(signature: bytes) -> str:
        """
        Format a raw 64 byte ed25519 signature for the `webhook-signature` header
        (`v2a,<base64 signature>`).

        Doesn't require the `asymmetric` extra.
        """
        if not isinstance(signature, (bytes, bytearray)) or len(signature) != 64:
            raise RuntimeError("Invalid signature: must be 64 bytes")
        return f"v2a,{base64.b64encode(signature).decode('utf-8')}"

    @staticmethod
    def format_public_key(public_key: bytes) -> str:
        """
        Format an ed25519 public key as `whpk_<base64 public key>`, to share with consumers.

        Accepts either the raw 32 byte public key, or the DER encoded SubjectPublicKeyInfo of an
        ed25519 public key (e.g. what AWS KMS `GetPublicKey` returns).

        Doesn't require the `asymmetric` extra.
        """
        if not isinstance(public_key, (bytes, bytearray)):
            raise RuntimeError("Invalid public key: must be bytes")
        raw = bytes(public_key)
        if len(raw) == len(_ED25519_SPKI_PREFIX) + 32:
            if not raw.startswith(_ED25519_SPKI_PREFIX):
                raise RuntimeError("Invalid public key: not an ed25519 SubjectPublicKeyInfo")
            raw = raw[len(_ED25519_SPKI_PREFIX) :]
        if len(raw) != 32:
            raise RuntimeError("Invalid public key: must be 32 bytes")
        return f"whpk_{base64.b64encode(raw).decode('utf-8')}"

    def verify(
        self,
        data: t.Union[bytes, str],
        headers: t.Dict[str, str],
        *,
        json_parse: bool = True,
    ) -> t.Any:
        """
        Verify the given webhook headers against the body bytes (data).

        Pass the raw request body exactly as received. bytes are verified byte for byte, so
        bodies that aren't valid UTF-8 verify. A str is verified as its UTF-8 encoding.

        Args:
            json_parse (bool): Whether to parse the data as JSON on success (default: True)

        Returns:
            After successful verification: if json_parse is True, returns the
            JSON-parsed input data; if it json_parse False, returns None.

        Raises:
            WebhookVerificationError if one of the required headers is missing,
            invalid, too old or too new; or no matching signature is found.
        """
        from cryptography.exceptions import InvalidSignature

        payload = _payload_bytes(data)
        msg_id, msg_signature, timestamp = _parse_headers(headers)

        digest = self.signed_content_hash(self._endpoint_id, msg_id, timestamp, payload)

        for versioned_sig in msg_signature.split(" "):
            version, _, encoded_sig = versioned_sig.partition(",")
            if version != "v2a":
                continue
            try:
                signature = base64.b64decode(encoded_sig, validate=True)
            except binascii.Error:
                continue
            if len(signature) != 64:
                continue
            for verifier in self._verifiers:
                try:
                    verifier.verify(signature, digest)
                except InvalidSignature:
                    continue
                if json_parse:
                    return json.loads(payload)
                return None

        raise WebhookVerificationError("No matching signature found")

    def endpoint_verification_response(self, challenge: str) -> str:
        """
        Calculate the response to an endpoint verification challenge, using the first
        (newest) public key.

        Args:
            challenge (str): The value of the `webhook-challenge` request header

        Returns:
            The value for the `webhook-challenge-response` response header.
        """
        return self.__challenge_response(self._public_keys[0], challenge)

    def verify_endpoint_verification_response(self, challenge: str, header: str) -> None:
        """
        Verify the response to an endpoint verification challenge. The response is accepted
        if it matches any of the public keys.

        Args:
            challenge (str): The challenge sent in the `webhook-challenge` request header
            header (str): The value of the `webhook-challenge-response` response header

        Raises:
            WebhookVerificationError if no matching signature is found.
        """
        passed = header.encode()
        for public_key in self._public_keys:
            expected_sig = self.__challenge_response(public_key, challenge)
            if hmac.compare_digest(expected_sig.encode(), passed):
                return
        raise WebhookVerificationError("No matching signature found")

    def __challenge_response(self, public_key: bytes, challenge: str) -> str:
        return _challenge_response(public_key + b"." + self._endpoint_id.encode(), challenge, "v2a")


class WebhookAsymmetricSigner:
    """
    Sign webhooks with asymmetric (`v2a`, ed25519) signatures, using a private key you hold.

    It's recommended to hold the private key in a KMS or an HSM instead, and sign using
    `WebhookAsymmetric.signed_content_hash` and `WebhookAsymmetric.format_signature`.

    A signer isn't tied to an endpoint: one key pair can serve all endpoints.

    Requires the `asymmetric` extra: `pip install 'standardwebhooks[asymmetric]'`.
    """

    _SECRET_KEY_PREFIX: str = "whsk_"
    _signer: "Ed25519PrivateKey"
    _public_key: bytes

    def __init__(self, secret_key: str):
        """
        Args:
            secret_key: The secret key: `whsk_` followed by the base64 encoded 32 byte
                ed25519 seed (private key).
        """
        ed25519 = _import_ed25519("WebhookAsymmetricSigner")
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

        if not isinstance(secret_key, str) or not secret_key.startswith(self._SECRET_KEY_PREFIX):
            raise RuntimeError(f"Invalid secret key: must start with '{self._SECRET_KEY_PREFIX}'")
        try:
            seed = base64.b64decode(secret_key[len(self._SECRET_KEY_PREFIX) :], validate=True)
        except binascii.Error:
            raise RuntimeError("Invalid secret key: not valid base64")
        if len(seed) != 32:
            raise RuntimeError(f"Invalid secret key: must be 32 bytes, got {len(seed)}")
        self._signer = ed25519.Ed25519PrivateKey.from_private_bytes(seed)
        self._public_key = self._signer.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)

    def sign(self, endpoint_id: str, msg_id: str, timestamp: datetime, data: t.Union[bytes, str]) -> str:
        """
        Sign the given payload sent to the given endpoint, and return the versioned signature
        string (`v2a,<base64 signature>`) for the `webhook-signature` header.

        Args:
            endpoint_id: The ID of the endpoint the webhook is sent to.
            msg_id: The unique message id (sent as the `webhook-id` header).
            timestamp: The message timestamp (sent as the `webhook-timestamp` header).
                Naive datetimes are interpreted as UTC.
            data: The raw payload. bytes are signed byte for byte. A str is signed as its
                UTF-8 encoding.
        """
        digest = WebhookAsymmetric.signed_content_hash(endpoint_id, msg_id, timestamp, data)
        return WebhookAsymmetric.format_signature(self._signer.sign(digest))

    def public_key(self) -> str:
        """The public key (`whpk_...`) matching the secret key, to share with consumers."""
        return WebhookAsymmetric.format_public_key(self._public_key)
