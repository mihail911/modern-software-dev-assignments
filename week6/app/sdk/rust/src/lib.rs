// SPDX-FileCopyrightText: © 2022 Svix Authors
// SPDX-License-Identifier: MIT

#![warn(clippy::all)]
#![forbid(unsafe_code)]

use base64::Engine;
use http::HeaderMap;
use std::fmt::{self, Debug};
use time::OffsetDateTime;

mod asymmetric;

pub use asymmetric::{WebhookAsymmetric, WebhookAsymmetricError, WebhookAsymmetricSigner};

#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum WebhookError {
    #[error("failed to parse timestamp")]
    InvalidTimestamp,

    #[error("invalid secret")]
    InvalidSecret(#[from] base64::DecodeError),

    #[error("secret may not be blank")]
    EmptySecret,

    #[error("invalid header {0}")]
    InvalidHeader(&'static str),

    #[error("signature timestamp too old")]
    TimestampTooOldError,

    #[error("signature timestamp too far in future")]
    FutureTimestampError,

    #[error("missing header {0}")]
    MissingHeader(&'static str),

    #[error("signature invalid")]
    InvalidSignature,

    #[error("payload invalid")]
    InvalidPayload,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Webhook {
    key: Box<[u8]>,
}

impl Debug for Webhook {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Webhook { key: [REDACTED] }")
    }
}

pub const HEADER_WEBHOOK_ID: &str = "webhook-id";
pub const HEADER_WEBHOOK_SIGNATURE: &str = "webhook-signature";
pub const HEADER_WEBHOOK_TIMESTAMP: &str = "webhook-timestamp";
pub const HEADER_WEBHOOK_CHALLENGE: &str = "webhook-challenge";
pub const HEADER_WEBHOOK_CHALLENGE_RESPONSE: &str = "webhook-challenge-response";

const PREFIX: &str = "whsec_";
const TOLERANCE_IN_SECONDS: i64 = 5 * 60;
const SIGNATURE_VERSION: &str = "v1";
const VERIFICATION_KEY_LABEL: &[u8] = b"wev-v1";

impl Webhook {
    pub fn new(secret: &str) -> Result<Self, WebhookError> {
        let secret = secret.strip_prefix(PREFIX).unwrap_or(secret);
        let key = base64::engine::general_purpose::STANDARD.decode(secret)?;

        if key.is_empty() {
            return Err(WebhookError::EmptySecret);
        }

        Ok(Webhook {
            key: key.into_boxed_slice(),
        })
    }

    pub fn from_bytes(secret: Vec<u8>) -> Result<Self, WebhookError> {
        if secret.is_empty() {
            return Err(WebhookError::EmptySecret);
        }

        Ok(Webhook {
            key: secret.into_boxed_slice(),
        })
    }

    /// Verify the webhook signature against payload and headers
    pub fn verify(&self, payload: &[u8], headers: &HeaderMap) -> Result<(), WebhookError> {
        let now = OffsetDateTime::now_utc().unix_timestamp();
        self.verify_with_timestamp(payload, headers, now)
    }

    /// Similar to [Self::verify] but uses a timestamp passed as a parameter instead of `now()`
    pub fn verify_with_timestamp(
        &self,
        payload: &[u8],
        headers: &HeaderMap,
        now: i64,
    ) -> Result<(), WebhookError> {
        let (msg_id, msg_signature, msg_ts) = parse_verify_headers(headers, now)?;

        let expected_signature = self.sign(msg_id, msg_ts, payload)?;

        msg_signature
            .split(' ')
            .any(|x| secure_compare(x, &expected_signature))
            .then_some(())
            .ok_or(WebhookError::InvalidSignature)
    }

    pub fn sign(
        &self,
        msg_id: &str,
        timestamp: i64,
        payload: &[u8],
    ) -> Result<String, WebhookError> {
        let payload = std::str::from_utf8(payload).map_err(|_| WebhookError::InvalidPayload)?;
        let to_sign = format!("{msg_id}.{timestamp}.{payload}",);
        let signed = hmac_sha256::HMAC::mac(to_sign.as_bytes(), &self.key);
        let encoded = base64::engine::general_purpose::STANDARD.encode(signed);

        Ok(format!("{SIGNATURE_VERSION},{encoded}"))
    }

    /// Calculate the response to an endpoint verification challenge
    ///
    /// Takes the value of the `webhook-challenge` header and returns the value for the
    /// `webhook-challenge-response` header.
    pub fn endpoint_verification_response(&self, challenge: &str) -> String {
        let encoded = sign_challenge(&self.key, challenge);

        format!("{SIGNATURE_VERSION},{encoded}")
    }

    /// Verify the value of the `webhook-challenge-response` header against the challenge sent
    pub fn verify_endpoint_verification_response(
        &self,
        challenge: &str,
        header: &str,
    ) -> Result<(), WebhookError> {
        secure_compare(header, &self.endpoint_verification_response(challenge))
            .then_some(())
            .ok_or(WebhookError::InvalidSignature)
    }
}

/// Extract the webhook id, signature and timestamp from the headers, and verify the timestamp
/// is within the tolerance of `now`
fn parse_verify_headers(headers: &HeaderMap, now: i64) -> Result<(&str, &str, i64), WebhookError> {
    let msg_id = get_header(headers, HEADER_WEBHOOK_ID, "id")?;
    let msg_signature = get_header(headers, HEADER_WEBHOOK_SIGNATURE, "signature")?;
    let msg_ts =
        get_header(headers, HEADER_WEBHOOK_TIMESTAMP, "timestamp").and_then(parse_timestamp)?;

    verify_timestamp(msg_ts, now)?;

    Ok((msg_id, msg_signature, msg_ts))
}

fn get_header<'a>(
    headers: &'a HeaderMap,
    unbranded_hdr: &'static str,
    err_name: &'static str,
) -> Result<&'a str, WebhookError> {
    headers
        .get(unbranded_hdr)
        .ok_or(WebhookError::MissingHeader(err_name))?
        .to_str()
        .map_err(|_| WebhookError::InvalidHeader(err_name))
}

fn parse_timestamp(hdr: &str) -> Result<i64, WebhookError> {
    str::parse::<i64>(hdr).map_err(|_| WebhookError::InvalidTimestamp)
}

fn verify_timestamp(ts: i64, now: i64) -> Result<(), WebhookError> {
    if now - ts > TOLERANCE_IN_SECONDS {
        Err(WebhookError::TimestampTooOldError)
    } else if ts > now + TOLERANCE_IN_SECONDS {
        Err(WebhookError::FutureTimestampError)
    } else {
        Ok(())
    }
}

fn secure_compare(a: &str, b: &str) -> bool {
    (a.len() == b.len())
        && (a
            .bytes()
            .zip(b.bytes())
            .fold(0, |acc, (x, y)| acc | (x ^ y))
            == 0)
}

/// Sign an endpoint verification challenge using a key derived from the endpoint key, and return
/// the base64 encoded signature
fn sign_challenge(endpoint_key: &[u8], challenge: &str) -> String {
    let verification_key = hmac_sha256::HMAC::mac(VERIFICATION_KEY_LABEL, endpoint_key);
    let signed = hmac_sha256::HMAC::mac(challenge.as_bytes(), verification_key);
    base64::engine::general_purpose::STANDARD.encode(signed)
}

#[cfg(test)]
mod tests {

    use super::*;
    use http::HeaderMap;

    fn get_unbranded_headers(msg_id: &str, signature: &str) -> HeaderMap {
        let mut headers = http::header::HeaderMap::new();
        headers.insert(HEADER_WEBHOOK_ID, msg_id.parse().unwrap());
        headers.insert(HEADER_WEBHOOK_SIGNATURE, signature.parse().unwrap());
        headers.insert(
            HEADER_WEBHOOK_TIMESTAMP,
            OffsetDateTime::now_utc()
                .unix_timestamp()
                .to_string()
                .parse()
                .unwrap(),
        );
        headers
    }

    #[test]
    fn test_sign() {
        let wh = Webhook::new("whsec_C2FVsBQIhrscChlQIMV+b5sSYspob7oD").unwrap();
        assert_eq!(
            "v1,tZ1I4/hDygAJgO5TYxiSd6Sd0kDW6hPenDe+bTa3Kkw=".to_owned(),
            wh.sign(
                "msg_27UH4WbU6Z5A5EzD8u03UvzRbpk",
                1649367553,
                br#"{"email":"test@example.com","username":"test_user"}"#
            )
            .unwrap()
        );
    }

    #[test]
    fn test_challenge_response() {
        let wh = Webhook::new("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw").unwrap();
        assert_eq!(
            "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=".to_owned(),
            wh.endpoint_verification_response("0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=")
        );
    }

    #[test]
    fn test_verify_challenge_response() {
        let wh = Webhook::new("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw").unwrap();
        let challenge = "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM=";
        let header = "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA=";

        wh.verify_endpoint_verification_response(challenge, header)
            .unwrap();
        assert_eq!(
            Err(WebhookError::InvalidSignature),
            wh.verify_endpoint_verification_response(challenge, &format!("v2,abc {header}"))
        );
        assert_eq!(
            Err(WebhookError::InvalidSignature),
            wh.verify_endpoint_verification_response("wrong-challenge", header)
        );
    }

    #[test]
    fn test_verify() {
        let secret = "whsec_C2FVsBQIhrscChlQIMV+b5sSYspob7oD".to_owned();
        let msg_id = "msg_27UH4WbU6Z5A5EzD8u03UvzRbpk";
        let payload = br#"{"email":"test@example.com","username":"test_user"}"#;
        let wh = Webhook::new(&secret).unwrap();

        let signature = wh
            .sign(msg_id, OffsetDateTime::now_utc().unix_timestamp(), payload)
            .unwrap();
        wh.verify(payload, &get_unbranded_headers(msg_id, &signature))
            .unwrap();
    }

    #[test]
    fn test_no_verify() {
        let secret = "whsec_C2FVsBQIhrscChlQIMV+b5sSYspob7oD".to_owned();
        let msg_id = "msg_27UH4WbU6Z5A5EzD8u03UvzRbpk";
        let payload = br#"{"email":"test@example.com","username":"test_user"}"#;
        let wh = Webhook::new(&secret).unwrap();

        let signature = "v1,R3PTzyfHASBKHH98a7yexTwaJ4yNIcGhFQc1yuN+BPU=".to_owned();
        let headers = get_unbranded_headers(msg_id, &signature);
        assert!(wh.verify(payload, &headers).is_err());
    }

    #[test]
    fn test_verify_partial_signature() {
        let secret = "whsec_C2FVsBQIhrscChlQIMV+b5sSYspob7oD".to_owned();
        let msg_id = "msg_27UH4WbU6Z5A5EzD8u03UvzRbpk";
        let payload = br#"{"email":"test@example.com","username":"test_user"}"#;
        let wh = Webhook::new(&secret).unwrap();

        let signature = wh
            .sign(msg_id, OffsetDateTime::now_utc().unix_timestamp(), payload)
            .unwrap();

        // Just `v1,`
        let mut headers = get_unbranded_headers(msg_id, &signature);
        let partial = format!(
            "{},",
            signature.split(',').collect::<Vec<&str>>().first().unwrap()
        );
        headers.insert(HEADER_WEBHOOK_SIGNATURE, partial.parse().unwrap());
        assert!(wh.verify(payload, &headers).is_err());

        // Non-empty but still partial signature (first few bytes)
        let mut headers = get_unbranded_headers(msg_id, &signature);
        let partial = &signature[0..8];
        headers.insert(HEADER_WEBHOOK_SIGNATURE, partial.parse().unwrap());
        assert!(wh.verify(payload, &headers).is_err());
    }

    #[test]
    fn test_verify_incorrect_timestamp() {
        let secret = "whsec_C2FVsBQIhrscChlQIMV+b5sSYspob7oD".to_owned();
        let msg_id = "msg_27UH4WbU6Z5A5EzD8u03UvzRbpk";
        let payload = br#"{"email":"test@example.com","username":"test_user"}"#;
        let wh = Webhook::new(&secret).unwrap();

        let signature = wh
            .sign(msg_id, OffsetDateTime::now_utc().unix_timestamp(), payload)
            .unwrap();

        let mut headers = get_unbranded_headers(msg_id, &signature);
        for ts in [
            OffsetDateTime::now_utc().unix_timestamp() - (super::TOLERANCE_IN_SECONDS + 1),
            OffsetDateTime::now_utc().unix_timestamp() + (super::TOLERANCE_IN_SECONDS + 1),
        ] {
            headers.insert(
                super::HEADER_WEBHOOK_TIMESTAMP,
                ts.to_string().parse().unwrap(),
            );

            assert!(wh.verify(payload, &headers,).is_err());
        }
    }

    #[test]
    fn test_verify_with_multiple_signatures() {
        let secret = "whsec_C2FVsBQIhrscChlQIMV+b5sSYspob7oD".to_owned();
        let msg_id = "msg_27UH4WbU6Z5A5EzD8u03UvzRbpk";
        let payload = br#"{"email":"test@example.com","username":"test_user"}"#;
        let wh = Webhook::new(&secret).unwrap();

        let signature = wh
            .sign(msg_id, OffsetDateTime::now_utc().unix_timestamp(), payload)
            .unwrap();

        let multi_sig = format!(
            "{} {} {} {}",
            "v1,tFtCZ5RDCPxzWQRWXWPgrCgE2frDBe9gjpbWQxnVfsQ=",
            "v1,Mm7xgUVICxZfQ3bgf0h0Dof65L/IFx+PnZvnDWPCX6Q=",
            signature,
            "v1,9DfC1c3eeOrXB6w/5dIDydLNQaEyww5KalE5jLBZucE=",
        );

        let headers = get_unbranded_headers(msg_id, &multi_sig);

        wh.verify(payload, &headers).unwrap();
    }

    #[test]
    fn test_no_verify_with_multiple_signatures() {
        let secret = "whsec_C2FVsBQIhrscChlQIMV+b5sSYspob7oD".to_owned();
        let msg_id = "msg_27UH4WbU6Z5A5EzD8u03UvzRbpk";
        let payload = br#"{"email":"test@example.com","username":"test_user"}"#;
        let wh = Webhook::new(&secret).unwrap();

        let missing_sig = format!(
            "{} {} {}",
            "v1,tFtCZ5RDCPxzWQRWXWPgrCgE2frDBe9gjpbWQxnVfsQ=",
            "v1,Mm7xgUVICxZfQ3bgf0h0Dof65L/IFx+PnZvnDWPCX6Q=",
            "v1,9DfC1c3eeOrXB6w/5dIDydLNQaEyww5KalE5jLBZucE=",
        );

        let headers = get_unbranded_headers(msg_id, &missing_sig);

        assert!(wh.verify(payload, &headers).is_err());
    }

    #[test]
    fn test_verify_with_malformed_signatures() {
        let secret = "whsec_C2FVsBQIhrscChlQIMV+b5sSYspob7oD".to_owned();
        let msg_id = "msg_27UH4WbU6Z5A5EzD8u03UvzRbpk";
        let payload = br#"{"email":"test@example.com","username":"test_user"}"#;
        let wh = Webhook::new(&secret).unwrap();

        let signature = wh
            .sign(msg_id, OffsetDateTime::now_utc().unix_timestamp(), payload)
            .unwrap();

        // An entry without a comma followed by a valid signature
        let headers = get_unbranded_headers(msg_id, &format!("v1 {signature}"));
        wh.verify(payload, &headers).unwrap();

        // Malformed entries followed by a valid signature
        let headers = get_unbranded_headers(msg_id, &format!(",,, v1,not,base64  {signature}"));
        wh.verify(payload, &headers).unwrap();

        // A valid signature with another version
        let headers = get_unbranded_headers(msg_id, &signature.replacen("v1,", "v2,", 1));
        assert_eq!(
            wh.verify(payload, &headers),
            Err(WebhookError::InvalidSignature)
        );
    }

    #[test]
    fn test_missing_headers() {
        let secret = "whsec_C2FVsBQIhrscChlQIMV+b5sSYspob7oD".to_owned();
        let msg_id = "msg_27UH4WbU6Z5A5EzD8u03UvzRbpk";
        let payload = br#"{"email":"test@example.com","username":"test_user"}"#;
        let wh = Webhook::new(&secret).unwrap();

        let signature = wh
            .sign(msg_id, OffsetDateTime::now_utc().unix_timestamp(), payload)
            .unwrap();
        for (mut hdr_map, hdrs) in [(
            get_unbranded_headers(msg_id, &signature),
            [
                HEADER_WEBHOOK_ID,
                HEADER_WEBHOOK_SIGNATURE,
                HEADER_WEBHOOK_TIMESTAMP,
            ],
        )] {
            for hdr in hdrs {
                hdr_map.remove(hdr);
                assert!(wh.verify(payload, &hdr_map).is_err());
            }
        }
    }
}
