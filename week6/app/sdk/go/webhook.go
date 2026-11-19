package standardwebhooks

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"
)

const (
	HeaderWebhookID        string = "webhook-id"
	HeaderWebhookSignature string = "webhook-signature"
	HeaderWebhookTimestamp string = "webhook-timestamp"

	HeaderWebhookChallenge         string = "webhook-challenge"
	HeaderWebhookChallengeResponse string = "webhook-challenge-response"

	webhookSecretPrefix  string = "whsec_"
	verificationKeyLabel string = "wev-v1"
)

var base64enc = base64.StdEncoding

var tolerance time.Duration = 5 * time.Minute

var (
	ErrRequiredHeaders     = errors.New("missing required headers")
	ErrInvalidHeaders      = errors.New("invalid signature headers")
	ErrNoMatchingSignature = errors.New("no matching signature found")
	ErrMessageTooOld       = errors.New("message timestamp too old")
	ErrMessageTooNew       = errors.New("message timestamp too new")
	ErrEmptySecret         = errors.New("webhook secret may not be empty")
)

type Webhook struct {
	key []byte
}

func NewWebhook(secret string) (*Webhook, error) {
	key, err := base64enc.DecodeString(strings.TrimPrefix(secret, webhookSecretPrefix))
	if err != nil {
		return nil, fmt.Errorf("unable to create webhook, err: %w", err)
	}
	if len(key) == 0 {
		return nil, ErrEmptySecret
	}
	return &Webhook{
		key: key,
	}, nil
}

func NewWebhookRaw(secret []byte) (*Webhook, error) {
	if len(secret) == 0 {
		return nil, ErrEmptySecret
	}
	return &Webhook{
		key: secret,
	}, nil
}

// Verify validates the payload against the webhook signature headers
// using the webhooks signing secret.
//
// Returns an error if the body or headers are missing/unreadable
// or if the signature doesn't match.
func (wh *Webhook) Verify(payload []byte, headers http.Header) error {
	return wh.verify(payload, headers, true)
}

// VerifyIgnoringTimestamp validates the payload against the webhook signature headers
// using the webhooks signing secret.
//
// Returns an error if the body or headers are missing/unreadable
// or if the signature doesn't match.
//
// WARNING: This function does not check the signature's timestamp.
// We recommend using the `Verify` function instead.
func (wh *Webhook) VerifyIgnoringTimestamp(payload []byte, headers http.Header) error {
	return wh.verify(payload, headers, false)
}

func (wh *Webhook) verify(payload []byte, headers http.Header, enforceTolerance bool) error {
	msgId, msgSignature, timestamp, err := parseVerifyHeaders(headers, enforceTolerance)
	if err != nil {
		return err
	}

	expectedSignature, err := wh.Sign(msgId, timestamp, payload)
	if err != nil {
		return fmt.Errorf("unable to verify payload, err: %w", err)
	}

	for _, versionedSignature := range strings.Split(msgSignature, " ") {
		if hmac.Equal([]byte(versionedSignature), []byte(expectedSignature)) {
			return nil
		}
	}

	return fmt.Errorf("unable to verify payload, err: %w", ErrNoMatchingSignature)
}

func (wh *Webhook) Sign(msgId string, timestamp time.Time, payload []byte) (string, error) {
	version, signature, err := wh.sign(msgId, timestamp, payload)
	return fmt.Sprintf("%s,%s", version, signature), err
}

func (wh *Webhook) sign(msgId string, timestamp time.Time, payload []byte) (version string, signature []byte, err error) {
	toSign := fmt.Sprintf("%s.%d.%s", msgId, timestamp.Unix(), payload)

	h := hmac.New(sha256.New, wh.key)
	h.Write([]byte(toSign))
	sig := make([]byte, base64enc.EncodedLen(h.Size()))
	base64enc.Encode(sig, h.Sum(nil))

	return "v1", sig, nil
}

// EndpointVerificationResponse calculates the response to an endpoint verification challenge.
//
// Takes the value of the `webhook-challenge` header and returns the value
// for the `webhook-challenge-response` header.
func (wh *Webhook) EndpointVerificationResponse(challenge string) string {
	return fmt.Sprintf("v1,%s", wh.signChallenge(challenge))
}

// VerifyEndpointVerificationResponse validates the value of the `webhook-challenge-response`
// header against the challenge that was sent.
//
// Returns an error if the signature doesn't match.
func (wh *Webhook) VerifyEndpointVerificationResponse(challenge string, header string) error {
	if hmac.Equal([]byte(header), []byte(wh.EndpointVerificationResponse(challenge))) {
		return nil
	}

	return fmt.Errorf("unable to verify challenge response, err: %w", ErrNoMatchingSignature)
}

func (wh *Webhook) signChallenge(challenge string) []byte {
	return signChallenge(wh.key, challenge)
}

// signChallenge signs an endpoint verification challenge using a key
// derived from the endpoint key, and returns the base64 encoded signature.
func signChallenge(endpointKey []byte, challenge string) []byte {
	verificationKey := hmac.New(sha256.New, endpointKey)
	verificationKey.Write([]byte(verificationKeyLabel))

	h := hmac.New(sha256.New, verificationKey.Sum(nil))
	h.Write([]byte(challenge))
	sig := make([]byte, base64enc.EncodedLen(h.Size()))
	base64enc.Encode(sig, h.Sum(nil))

	return sig
}

// parseVerifyHeaders extracts the webhook id, signature and timestamp from the
// headers, and verifies the timestamp is within the tolerance if enforceTolerance is set.
func parseVerifyHeaders(headers http.Header, enforceTolerance bool) (msgId string, msgSignature string, timestamp time.Time, err error) {
	msgId = headers.Get(HeaderWebhookID)
	msgSignature = headers.Get(HeaderWebhookSignature)
	msgTimestamp := headers.Get(HeaderWebhookTimestamp)
	if msgId == "" || msgSignature == "" || msgTimestamp == "" {
		return "", "", time.Time{}, fmt.Errorf("unable to verify payload, err: %w", ErrRequiredHeaders)
	}

	timestamp, err = parseTimestampHeader(msgTimestamp)
	if err != nil {
		return "", "", time.Time{}, fmt.Errorf("unable to verify payload, err: %w", err)
	}

	if enforceTolerance {
		if err := verifyTimestamp(timestamp); err != nil {
			return "", "", time.Time{}, fmt.Errorf("unable to verify payload, err: %w", err)
		}
	}

	return msgId, msgSignature, timestamp, nil
}

func parseTimestampHeader(timestampHeader string) (time.Time, error) {
	timeInt, err := strconv.ParseInt(timestampHeader, 10, 64)
	if err != nil {
		return time.Time{}, fmt.Errorf("unable to parse timestamp header, err: %w", errors.Join(err, ErrInvalidHeaders))
	}
	timestamp := time.Unix(timeInt, 0)
	return timestamp, nil
}

func verifyTimestamp(timestamp time.Time) error {
	now := time.Now()

	if now.Sub(timestamp) > tolerance {
		return ErrMessageTooOld
	}

	if timestamp.After(now.Add(tolerance)) {
		return ErrMessageTooNew
	}

	return nil
}
