import array
import base64
import hashlib
import json
import typing as t
from datetime import datetime, timedelta, timezone, tzinfo
from math import floor

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from standardwebhooks import webhooks
from standardwebhooks.webhooks import (
    EmptyWebhookSecretError,
    Webhook,
    WebhookAsymmetric,
    WebhookAsymmetricSigner,
    WebhookVerificationError,
    hmac_data,
)

DEFAULT_MSG_ID = "msg_p5jXN8AQM9LWM0D4loKWxJek"
DEFAULT_PAYLOAD = '{"test": 2432232314}'
DEFAULT_SECRET = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"

TOLERANCE = timedelta(minutes=5)


class PayloadForTesting:
    id: str
    timestamp: str
    payload: str
    secret: str
    signature: str
    header: t.Dict[str, str]

    def __init__(self, timestamp: datetime = datetime.now(tz=timezone.utc), secret: str = DEFAULT_SECRET):
        ts = str(floor(timestamp.timestamp()))
        to_sign = f"{DEFAULT_MSG_ID}.{ts}.{DEFAULT_PAYLOAD}".encode()
        signature = base64.b64encode(hmac_data(base64.b64decode(secret), to_sign)).decode("utf-8")

        self.id = DEFAULT_MSG_ID
        self.timestamp = ts
        self.payload = DEFAULT_PAYLOAD
        self.secret = secret
        self.signature = signature
        self.header = {
            "webhook-id": DEFAULT_MSG_ID,
            "webhook-signature": "v1," + signature,
            "webhook-timestamp": self.timestamp,
        }


def test_empty_secret_raises_error() -> None:
    with pytest.raises(EmptyWebhookSecretError):
        Webhook("")
    with pytest.raises(EmptyWebhookSecretError):
        Webhook(b"")
    with pytest.raises(EmptyWebhookSecretError):
        Webhook("whsec_")


def test_missing_id_raises_error() -> None:
    test_payload = PayloadForTesting()
    del test_payload.header["webhook-id"]

    wh = Webhook(test_payload.secret)

    with pytest.raises(WebhookVerificationError):
        wh.verify(test_payload.payload, test_payload.header)


def test_timestamp_raises_error() -> None:
    test_payload = PayloadForTesting()
    del test_payload.header["webhook-timestamp"]

    wh = Webhook(test_payload.secret)

    with pytest.raises(WebhookVerificationError):
        wh.verify(test_payload.payload, test_payload.header)


def test_invalid_timestamp_raises_error() -> None:
    test_payload = PayloadForTesting()
    test_payload.header["webhook-timestamp"] = "hello"

    wh = Webhook(test_payload.secret)

    with pytest.raises(WebhookVerificationError):
        wh.verify(test_payload.payload, test_payload.header)


def test_missing_signature_raises_error() -> None:
    test_payload = PayloadForTesting()
    del test_payload.header["webhook-signature"]

    wh = Webhook(test_payload.secret)

    with pytest.raises(WebhookVerificationError):
        wh.verify(test_payload.payload, test_payload.header)


def test_invalid_signature_raises_error() -> None:
    test_payload = PayloadForTesting()
    test_payload.header["webhook-signature"] = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OA="

    wh = Webhook(test_payload.secret)

    with pytest.raises(WebhookVerificationError):
        wh.verify(test_payload.payload, test_payload.header)


def test_valid_signature_is_valid_and_returns_json() -> None:
    test_payload = PayloadForTesting()

    wh = Webhook(test_payload.secret)

    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314


def test_valid_signature_is_valid_without_returning_json() -> None:
    test_payload = PayloadForTesting()

    wh = Webhook(test_payload.secret)

    result = wh.verify(test_payload.payload, test_payload.header, json_parse=False)
    assert result is None


def test_valid_unbranded_signature_is_valid_and_returns_json() -> None:
    test_payload = PayloadForTesting()

    unbranded_headers = {
        "webhook-id": test_payload.header["webhook-id"],
        "webhook-signature": test_payload.header["webhook-signature"],
        "webhook-timestamp": test_payload.header["webhook-timestamp"],
    }
    test_payload.header = unbranded_headers

    wh = Webhook(test_payload.secret)

    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314


def test_old_timestamp_fails() -> None:
    test_payload = PayloadForTesting(datetime.now(tz=timezone.utc) - TOLERANCE - timedelta(seconds=1))

    wh = Webhook(test_payload.secret)

    with pytest.raises(WebhookVerificationError):
        wh.verify(test_payload.payload, test_payload.header)


def test_new_timestamp_fails() -> None:
    test_payload = PayloadForTesting(datetime.now(tz=timezone.utc) + TOLERANCE + timedelta(seconds=1))

    wh = Webhook(test_payload.secret)

    with pytest.raises(WebhookVerificationError):
        wh.verify(test_payload.payload, test_payload.header)


def test_multi_sig_payload_is_valid() -> None:
    test_payload = PayloadForTesting()
    sigs = [
        "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
        "v2,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
        test_payload.header["webhook-signature"],  # valid signature
        "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
    ]
    test_payload.header["webhook-signature"] = " ".join(sigs)

    wh = Webhook(test_payload.secret)

    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314


def test_malformed_signatures_are_skipped() -> None:
    test_payload = PayloadForTesting()
    sigs = [
        "v1",
        "v1,not-base64!",
        "v1,a,b",
        test_payload.header["webhook-signature"],  # valid signature
    ]
    test_payload.header["webhook-signature"] = " ".join(sigs)

    wh = Webhook(test_payload.secret)

    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314


def test_valid_signature_with_unknown_version_fails() -> None:
    test_payload = PayloadForTesting()
    test_payload.header["webhook-signature"] = "v2," + test_payload.signature

    wh = Webhook(test_payload.secret)

    with pytest.raises(WebhookVerificationError):
        wh.verify(test_payload.payload, test_payload.header)


def test_signature_verification_with_and_without_prefix() -> None:
    test_payload = PayloadForTesting()

    wh = Webhook(test_payload.secret)
    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314

    wh = Webhook("whsec_" + test_payload.secret)

    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314


def test_signature_verification_with_unpadded_secret() -> None:
    # needs no padding
    secret = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
    test_payload = PayloadForTesting(secret=secret)

    wh = Webhook(secret)
    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314

    # needs one padding
    secret = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaS"
    test_payload = PayloadForTesting(secret=secret + "=")

    wh = Webhook(secret)
    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314

    # needs two padding
    secret = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSwBr"
    test_payload = PayloadForTesting(secret=secret + "==")

    wh = Webhook(secret)
    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314

    # with prefix
    wh = Webhook("whsec_" + secret)

    json = wh.verify(test_payload.payload, test_payload.header)
    assert json["test"] == 2432232314


def test_sign_function() -> None:
    key = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
    msg_id = "msg_p5jXN8AQM9LWM0D4loKWxJek"
    timestamp = datetime.fromtimestamp(1614265330, tz=timezone.utc)
    payload = '{"test": 2432232314}'
    expected = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="

    wh = Webhook(key)
    signature = wh.sign(msg_id=msg_id, timestamp=timestamp, data=payload)
    assert signature == expected


def test_sign_function_with_non_utc_timezone_aware_timestamp() -> None:
    # The signature must depend on the instant, not on the timezone the datetime is expressed in.
    key = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
    msg_id = "msg_p5jXN8AQM9LWM0D4loKWxJek"
    payload = '{"test": 2432232314}'
    expected = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="

    wh = Webhook(key)
    utc_timestamp = datetime.fromtimestamp(1614265330, tz=timezone.utc)
    for offset_hours in (-5, 7, 13):
        local_timestamp = utc_timestamp.astimezone(timezone(timedelta(hours=offset_hours)))
        signature = wh.sign(msg_id=msg_id, timestamp=local_timestamp, data=payload)
        assert signature == expected


def test_sign_function_with_naive_timestamp_is_treated_as_utc() -> None:
    key = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
    msg_id = "msg_p5jXN8AQM9LWM0D4loKWxJek"
    payload = '{"test": 2432232314}'
    expected = "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="

    wh = Webhook(key)
    naive_utc_timestamp = datetime(2021, 2, 25, 15, 2, 10)  # 1614265330 as a naive UTC wall-clock time
    signature = wh.sign(msg_id=msg_id, timestamp=naive_utc_timestamp, data=payload)
    assert signature == expected


def test_verify_accepts_signature_made_with_non_utc_timezone_aware_timestamp() -> None:
    wh = Webhook(DEFAULT_SECRET)
    local_now = datetime.now(tz=timezone(timedelta(hours=7)))
    signature = wh.sign(msg_id=DEFAULT_MSG_ID, timestamp=local_now, data=DEFAULT_PAYLOAD)
    headers = {
        "webhook-id": DEFAULT_MSG_ID,
        "webhook-timestamp": str(floor(local_now.timestamp())),
        "webhook-signature": signature,
    }

    json = wh.verify(DEFAULT_PAYLOAD, headers)
    assert json["test"] == 2432232314


# Every byte value, so the payload is not valid UTF-8. The expected signature is the Go
# library's for the same inputs, which signs the payload bytes as given.
BINARY_PAYLOAD = bytes(range(256))
BINARY_PAYLOAD_SIGNATURE = "v1,sEcZ042G8Crqgxbybb3y0eDxQR3TcfYsk11I1XULhSI="
SIGN_TIMESTAMP = datetime.fromtimestamp(1614265330, tz=timezone.utc)


def signed_header(wh: Webhook, payload: bytes) -> t.Dict[str, str]:
    """Headers for payload signed now. sign() itself is checked against the known-answer vectors."""
    timestamp = datetime.now(tz=timezone.utc)
    return {
        "webhook-id": DEFAULT_MSG_ID,
        "webhook-signature": wh.sign(msg_id=DEFAULT_MSG_ID, timestamp=timestamp, data=payload),
        "webhook-timestamp": str(floor(timestamp.timestamp())),
    }


def test_sign_function_binary_payload() -> None:
    wh = Webhook(DEFAULT_SECRET)

    assert wh.sign(msg_id=DEFAULT_MSG_ID, timestamp=SIGN_TIMESTAMP, data=BINARY_PAYLOAD) == BINARY_PAYLOAD_SIGNATURE


def test_sign_function_same_for_str_and_its_utf8_bytes() -> None:
    wh = Webhook(DEFAULT_SECRET)
    payload = '{"test": "ünïcödé ✓"}'

    assert wh.sign(msg_id=DEFAULT_MSG_ID, timestamp=SIGN_TIMESTAMP, data=payload.encode()) == wh.sign(
        msg_id=DEFAULT_MSG_ID, timestamp=SIGN_TIMESTAMP, data=payload
    )


def test_binary_payload_is_valid() -> None:
    wh = Webhook(DEFAULT_SECRET)

    assert wh.verify(BINARY_PAYLOAD, signed_header(wh, BINARY_PAYLOAD), json_parse=False) is None


def test_binary_payload_with_changed_byte_fails() -> None:
    wh = Webhook(DEFAULT_SECRET)
    header = signed_header(wh, BINARY_PAYLOAD)
    changed = BINARY_PAYLOAD[:-1] + b"\x00"

    with pytest.raises(WebhookVerificationError):
        wh.verify(changed, header, json_parse=False)


def test_json_payload_as_bytes_returns_json() -> None:
    test_payload = PayloadForTesting()
    wh = Webhook(test_payload.secret)

    json = wh.verify(test_payload.payload.encode(), test_payload.header)
    assert json["test"] == 2432232314


def test_json_payload_as_bytes_with_utf8_bom_returns_json() -> None:
    json_bytes = b'{"test": 2432232314}'
    payload = b"\xef\xbb\xbf" + json_bytes
    wh = Webhook(DEFAULT_SECRET)

    # The BOM is signed like any other bytes; only the JSON parse skips it.
    assert wh.verify(payload, signed_header(wh, payload)) == {"test": 2432232314}
    with pytest.raises(WebhookVerificationError):
        wh.verify(json_bytes, signed_header(wh, payload))


def test_sign_function_bytes_like_payloads() -> None:
    wh = Webhook(DEFAULT_SECRET)

    for payload in (bytearray(BINARY_PAYLOAD), memoryview(BINARY_PAYLOAD)):
        signature = wh.sign(msg_id=DEFAULT_MSG_ID, timestamp=SIGN_TIMESTAMP, data=t.cast(t.Any, payload))
        assert signature == BINARY_PAYLOAD_SIGNATURE


def test_sign_function_rejects_non_bytes_payload() -> None:
    wh = Webhook(DEFAULT_SECRET)

    for payload in (5, [1, 2], {"test": 1}, None, memoryview(array.array("H", [1, 2]))):
        with pytest.raises(TypeError):
            wh.sign(msg_id=DEFAULT_MSG_ID, timestamp=SIGN_TIMESTAMP, data=t.cast(t.Any, payload))


def test_challenge_response() -> None:
    wh = Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    response = wh.endpoint_verification_response("0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=")
    assert response == "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="


def test_verify_challenge_response() -> None:
    wh = Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    wh.verify_endpoint_verification_response(
        "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=", "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="
    )


def test_verify_challenge_response_wrong_challenge() -> None:
    wh = Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    with pytest.raises(WebhookVerificationError):
        wh.verify_endpoint_verification_response(
            "1njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=", "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="
        )


def test_verify_challenge_response_rejects_lists() -> None:
    wh = Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
    with pytest.raises(WebhookVerificationError):
        wh.verify_endpoint_verification_response(
            "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=", "v2,abc v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="
        )


def test_challenge_response_depends_on_secret() -> None:
    wh = Webhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSx")
    response = wh.endpoint_verification_response("0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=")
    assert response != "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="


# Asymmetric signatures (v2a): test vectors from the spec

ASYM_SECRET_KEY = "whsk_4uKPjji/dhB8w6GW3YYDnIMYvXcdHrvH6lidTr4Z+EY="
ASYM_PUBLIC_KEY = "whpk_uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w="
ASYM_ENDPOINT_ID = "ep_2fVZdC8Kx9TR4ZoCqHWt3NnLgRb"
ASYM_MSG_ID = "msg_2KWPBgLlAfxdpx2AI54pPJ85f4W"
ASYM_TIMESTAMP = 1674087231
ASYM_PAYLOAD = '{"type":"contact.created","timestamp":"2022-11-03T20:26:10.344522Z","data":{"id":"1f81eb52-5198-4599-803e-771906343485"}}'
ASYM_DIGEST = "YXCi5iN+rjB+iEb6bvg1XbWbVBL+r2S+as3k3n1Ih/A="
ASYM_SIGNATURE = "v2a,kG9XaxZjAEpKB8KeQQb5G1Xp73YH5Rf5gxoWe7cg1kYMiyspdtZXe9w+dAwKeOurAwxfqmlOG/QR4m9s37DzAg=="
ASYM_CHALLENGE = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
ASYM_CHALLENGE_RESPONSE = "v2a,t0RDFhzk7OtVzgvMsjuXnodSWuby2BVX+zPqw2/ksUk="

ASYM_PRIVATE_KEY = Ed25519PrivateKey.from_private_bytes(base64.b64decode(ASYM_SECRET_KEY[len("whsk_") :]))
ASYM_SIGNER = WebhookAsymmetricSigner(ASYM_SECRET_KEY)
# Another key, e.g. the producer's newer key after a rotation
OTHER_SIGNER = WebhookAsymmetricSigner("whsk_" + base64.b64encode(bytes([7] * 32)).decode())
OTHER_PUBLIC_KEY = OTHER_SIGNER.public_key()


def asym_sign(
    private_key: Ed25519PrivateKey, endpoint_id: str, msg_id: str, timestamp: int, payload: t.Union[bytes, str]
) -> str:
    payload_bytes = payload.encode() if isinstance(payload, str) else payload
    content = f"{endpoint_id}.{msg_id}.{timestamp}.".encode() + payload_bytes
    signature = private_key.sign(hashlib.sha256(content).digest())
    return "v2a," + base64.b64encode(signature).decode()


def asym_headers(
    payload: t.Union[bytes, str] = ASYM_PAYLOAD,
    signer: WebhookAsymmetricSigner = ASYM_SIGNER,
    endpoint_id: str = ASYM_ENDPOINT_ID,
    timestamp: t.Optional[int] = None,
) -> t.Dict[str, str]:
    if timestamp is None:
        timestamp = floor(datetime.now(tz=timezone.utc).timestamp())
    return {
        "webhook-id": ASYM_MSG_ID,
        "webhook-timestamp": str(timestamp),
        "webhook-signature": signer.sign(
            endpoint_id, ASYM_MSG_ID, datetime.fromtimestamp(timestamp, tz=timezone.utc), payload
        ),
    }


def test_asymmetric_spec_test_vector_matches_ed25519_primitive() -> None:
    public_key = ASYM_PRIVATE_KEY.public_key()
    assert "whpk_" + base64.b64encode(public_key.public_bytes_raw()).decode() == ASYM_PUBLIC_KEY

    content = f"{ASYM_ENDPOINT_ID}.{ASYM_MSG_ID}.{ASYM_TIMESTAMP}.{ASYM_PAYLOAD}".encode()
    digest = hashlib.sha256(content).digest()
    assert base64.b64encode(digest).decode() == ASYM_DIGEST
    public_key.verify(base64.b64decode(ASYM_SIGNATURE[len("v2a,") :]), digest)
    assert asym_sign(ASYM_PRIVATE_KEY, ASYM_ENDPOINT_ID, ASYM_MSG_ID, ASYM_TIMESTAMP, ASYM_PAYLOAD) == ASYM_SIGNATURE


ASYM_DATETIME = datetime.fromtimestamp(ASYM_TIMESTAMP, tz=timezone.utc)


def test_asymmetric_signer_spec_test_vector() -> None:
    assert ASYM_SIGNER.public_key() == ASYM_PUBLIC_KEY
    assert ASYM_SIGNER.sign(ASYM_ENDPOINT_ID, ASYM_MSG_ID, ASYM_DATETIME, ASYM_PAYLOAD) == ASYM_SIGNATURE
    assert ASYM_SIGNER.sign(ASYM_ENDPOINT_ID, ASYM_MSG_ID, ASYM_DATETIME, ASYM_PAYLOAD.encode()) == ASYM_SIGNATURE


def test_asymmetric_signed_content_hash_spec_test_vector() -> None:
    digest = WebhookAsymmetric.signed_content_hash(ASYM_ENDPOINT_ID, ASYM_MSG_ID, ASYM_DATETIME, ASYM_PAYLOAD)
    assert len(digest) == 32
    assert base64.b64encode(digest).decode() == ASYM_DIGEST


def test_asymmetric_hsm_flow_spec_test_vector() -> None:
    digest = WebhookAsymmetric.signed_content_hash(ASYM_ENDPOINT_ID, ASYM_MSG_ID, ASYM_DATETIME, ASYM_PAYLOAD)
    # Stand-in for the HSM's sign call
    signature = ASYM_PRIVATE_KEY.sign(digest)
    assert WebhookAsymmetric.format_signature(signature) == ASYM_SIGNATURE


def test_asymmetric_format_signature_round_trip() -> None:
    signature = bytes([0xAB] * 64)
    formatted = WebhookAsymmetric.format_signature(signature)
    assert formatted.startswith("v2a,")
    assert base64.b64decode(formatted[len("v2a,") :]) == signature


@pytest.mark.parametrize("length", [0, 32, 63, 65])
def test_asymmetric_format_signature_rejects_wrong_length(length: int) -> None:
    with pytest.raises(RuntimeError):
        WebhookAsymmetric.format_signature(bytes(length))


def test_asymmetric_format_public_key() -> None:
    raw = base64.b64decode(ASYM_PUBLIC_KEY[len("whpk_") :])
    assert len(raw) == 32
    assert WebhookAsymmetric.format_public_key(raw) == ASYM_PUBLIC_KEY
    der = bytes.fromhex("302a300506032b6570032100") + raw
    assert WebhookAsymmetric.format_public_key(der) == ASYM_PUBLIC_KEY


@pytest.mark.parametrize(
    "public_key",
    [
        bytes(31),
        bytes(33),
        bytes.fromhex("312a300506032b6570032100") + bytes(32),
    ],
)
def test_asymmetric_format_public_key_rejects_invalid(public_key: bytes) -> None:
    with pytest.raises(RuntimeError):
        WebhookAsymmetric.format_public_key(public_key)


@pytest.mark.parametrize(
    "secret_key",
    [
        ASYM_SECRET_KEY[len("whsk_") :],
        ASYM_PUBLIC_KEY,
        "whsk_!!!",
        "whsk_",
        "whsk_" + base64.b64encode(bytes(31)).decode(),
        "whsk_" + base64.b64encode(bytes(64)).decode(),
    ],
)
def test_asymmetric_signer_constructor_rejects_invalid_keys(secret_key: str) -> None:
    with pytest.raises(RuntimeError):
        WebhookAsymmetricSigner(secret_key)


@pytest.mark.parametrize("endpoint_id", ["", "ep.1"])
def test_asymmetric_signing_rejects_invalid_endpoint_id(endpoint_id: str) -> None:
    now = datetime.now(tz=timezone.utc)
    with pytest.raises(RuntimeError):
        ASYM_SIGNER.sign(endpoint_id, ASYM_MSG_ID, now, ASYM_PAYLOAD)
    with pytest.raises(RuntimeError):
        WebhookAsymmetric.signed_content_hash(endpoint_id, ASYM_MSG_ID, now, ASYM_PAYLOAD)


def test_asymmetric_sign_then_verify() -> None:
    timestamp = datetime.now(tz=timezone.utc)
    signature = ASYM_SIGNER.sign(ASYM_ENDPOINT_ID, ASYM_MSG_ID, timestamp, ASYM_PAYLOAD)
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_SIGNER.public_key()])
    headers = {
        "webhook-id": ASYM_MSG_ID,
        "webhook-timestamp": str(floor(timestamp.timestamp())),
        "webhook-signature": signature,
    }
    assert wh.verify(ASYM_PAYLOAD, headers) == json.loads(ASYM_PAYLOAD)


def test_asymmetric_verify_accepts_spec_test_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: t.Optional[tzinfo] = None) -> "FrozenDatetime":
            return cls.fromtimestamp(ASYM_TIMESTAMP, tz=tz)

    monkeypatch.setattr(webhooks, "datetime", FrozenDatetime)

    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])
    headers = {
        "webhook-id": ASYM_MSG_ID,
        "webhook-timestamp": str(ASYM_TIMESTAMP),
        "webhook-signature": f"v1,K5oZfzN95Z9UVu1EsfQmfVNQhnkZ2pj9o9NDN/H/pI4= {ASYM_SIGNATURE}",
    }
    assert wh.verify(ASYM_PAYLOAD, headers) == json.loads(ASYM_PAYLOAD)


def test_asymmetric_valid_signature_is_valid_and_returns_json() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])

    assert wh.verify(ASYM_PAYLOAD, asym_headers()) == json.loads(ASYM_PAYLOAD)
    assert wh.verify(ASYM_PAYLOAD, asym_headers(), json_parse=False) is None


def test_asymmetric_verify_is_case_insensitive_for_header_names() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])
    headers = {k.upper(): v for (k, v) in asym_headers().items()}

    assert wh.verify(ASYM_PAYLOAD, headers) == json.loads(ASYM_PAYLOAD)


def test_asymmetric_binary_payload_is_valid() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])

    assert wh.verify(BINARY_PAYLOAD, asym_headers(BINARY_PAYLOAD), json_parse=False) is None


def test_asymmetric_valid_signature_after_other_signatures_is_valid() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])
    headers = asym_headers()
    headers["webhook-signature"] = (
        "v1,K5oZfzN95Z9UVu1EsfQmfVNQhnkZ2pj9o9NDN/H/pI4= v1a,abc v2a,!!invalid v2a,AAAA " + headers["webhook-signature"]
    )

    assert wh.verify(ASYM_PAYLOAD, headers) == json.loads(ASYM_PAYLOAD)


def test_asymmetric_signature_by_second_key_is_valid() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [OTHER_PUBLIC_KEY, ASYM_PUBLIC_KEY])

    assert wh.verify(ASYM_PAYLOAD, asym_headers()) == json.loads(ASYM_PAYLOAD)
    assert wh.verify(ASYM_PAYLOAD, asym_headers(signer=OTHER_SIGNER)) == json.loads(ASYM_PAYLOAD)


def test_asymmetric_signature_by_untrusted_key_fails() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])

    with pytest.raises(WebhookVerificationError, match="No matching signature found"):
        wh.verify(ASYM_PAYLOAD, asym_headers(signer=OTHER_SIGNER))


def test_asymmetric_wrong_endpoint_id_fails() -> None:
    wh = WebhookAsymmetric("ep_other", [ASYM_PUBLIC_KEY])

    with pytest.raises(WebhookVerificationError, match="No matching signature found"):
        wh.verify(ASYM_PAYLOAD, asym_headers())


def test_asymmetric_tampered_payload_fails() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])
    headers = asym_headers()

    with pytest.raises(WebhookVerificationError, match="No matching signature found"):
        wh.verify(ASYM_PAYLOAD + " ", headers)


def test_asymmetric_v1_signature_only_fails() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])
    headers = asym_headers()
    headers["webhook-signature"] = headers["webhook-signature"].replace("v2a,", "v1,")

    with pytest.raises(WebhookVerificationError, match="No matching signature found"):
        wh.verify(ASYM_PAYLOAD, headers)


@pytest.mark.parametrize("header", ["webhook-id", "webhook-timestamp", "webhook-signature"])
def test_asymmetric_missing_header_raises_error(header: str) -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])
    headers = asym_headers()
    del headers[header]

    with pytest.raises(WebhookVerificationError, match="Missing required headers"):
        wh.verify(ASYM_PAYLOAD, headers)


def test_asymmetric_old_timestamp_fails() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])
    timestamp = floor((datetime.now(tz=timezone.utc) - TOLERANCE).timestamp()) - 5

    with pytest.raises(WebhookVerificationError, match="Message timestamp too old"):
        wh.verify(ASYM_PAYLOAD, asym_headers(timestamp=timestamp))


def test_asymmetric_new_timestamp_fails() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY])
    timestamp = floor((datetime.now(tz=timezone.utc) + TOLERANCE).timestamp()) + 5

    with pytest.raises(WebhookVerificationError, match="Message timestamp too new"):
        wh.verify(ASYM_PAYLOAD, asym_headers(timestamp=timestamp))


@pytest.mark.parametrize(
    "public_keys,endpoint_id",
    [
        ([], ASYM_ENDPOINT_ID),
        (ASYM_PUBLIC_KEY, ASYM_ENDPOINT_ID),
        ([ASYM_PUBLIC_KEY[len("whpk_") :]], ASYM_ENDPOINT_ID),
        (["whsec_" + ASYM_PUBLIC_KEY[len("whpk_") :]], ASYM_ENDPOINT_ID),
        (["whpk_"], ASYM_ENDPOINT_ID),
        (["whpk_" + base64.b64encode(bytes(31)).decode()], ASYM_ENDPOINT_ID),
        (["whpk_" + base64.b64encode(bytes(33)).decode()], ASYM_ENDPOINT_ID),
        (["whpk_!!!"], ASYM_ENDPOINT_ID),
        ([ASYM_PUBLIC_KEY, "invalid"], ASYM_ENDPOINT_ID),
        ([ASYM_PUBLIC_KEY], ""),
        ([ASYM_PUBLIC_KEY], "ep.1"),
    ],
)
def test_asymmetric_constructor_rejects_invalid_arguments(public_keys: t.Any, endpoint_id: str) -> None:
    with pytest.raises(RuntimeError):
        WebhookAsymmetric(endpoint_id, public_keys)


def test_asymmetric_challenge_response() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [ASYM_PUBLIC_KEY, OTHER_PUBLIC_KEY])

    assert wh.endpoint_verification_response(ASYM_CHALLENGE) == ASYM_CHALLENGE_RESPONSE


def test_asymmetric_challenge_response_uses_first_key() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [OTHER_PUBLIC_KEY, ASYM_PUBLIC_KEY])

    assert wh.endpoint_verification_response(ASYM_CHALLENGE) != ASYM_CHALLENGE_RESPONSE


def test_asymmetric_challenge_response_depends_on_endpoint_id() -> None:
    wh = WebhookAsymmetric("ep_other", [ASYM_PUBLIC_KEY])

    assert wh.endpoint_verification_response(ASYM_CHALLENGE) != ASYM_CHALLENGE_RESPONSE


def test_asymmetric_verify_challenge_response_accepts_any_key() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [OTHER_PUBLIC_KEY, ASYM_PUBLIC_KEY])

    wh.verify_endpoint_verification_response(ASYM_CHALLENGE, ASYM_CHALLENGE_RESPONSE)
    wh.verify_endpoint_verification_response(ASYM_CHALLENGE, wh.endpoint_verification_response(ASYM_CHALLENGE))


@pytest.mark.parametrize(
    "header",
    [
        "v2a,p02It8u07wVKXUWJB4N8jl1LHoeidCN3wBffJ0vFVcQ=",
        ASYM_CHALLENGE_RESPONSE.replace("v2a,", "v1,"),
        "v1,abc " + ASYM_CHALLENGE_RESPONSE,
        "",
    ],
)
def test_asymmetric_verify_challenge_response_rejects_wrong_response(header: str) -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [OTHER_PUBLIC_KEY, ASYM_PUBLIC_KEY])

    with pytest.raises(WebhookVerificationError):
        wh.verify_endpoint_verification_response(ASYM_CHALLENGE, header)


def test_asymmetric_verify_challenge_response_wrong_challenge() -> None:
    wh = WebhookAsymmetric(ASYM_ENDPOINT_ID, [OTHER_PUBLIC_KEY, ASYM_PUBLIC_KEY])

    with pytest.raises(WebhookVerificationError):
        wh.verify_endpoint_verification_response(
            "1njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=", ASYM_CHALLENGE_RESPONSE
        )
