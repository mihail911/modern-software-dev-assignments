package standardwebhooks

import (
	"bytes"
	"crypto/ed25519"
	"crypto/hmac"
	"crypto/sha256"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
)

const (
	webhookPublicKeyPrefix     string = "whpk_"
	webhookSecretKeyPrefix     string = "whsk_"
	asymmetricSignatureVersion string = "v2a"
)

var (
	ErrNoPublicKeys      = errors.New("at least one public key is required")
	ErrInvalidPublicKey  = errors.New("invalid public key")
	ErrInvalidEndpointID = errors.New("endpoint id may not be empty or contain '.'")
	ErrInvalidSecretKey  = errors.New("invalid secret key")
	ErrInvalidSignature  = errors.New("invalid signature")
)

// WebhookAsymmetric verifies webhooks signed with asymmetric (`v2a`) signatures.
//
// It only verifies signatures. Producers can sign webhooks using
// WebhookAsymmetricSigner, or with a private key held in a KMS or an HSM
// using AsymmetricSignedContentHash and FormatAsymmetricSignature.
type WebhookAsymmetric struct {
	publicKeys []ed25519.PublicKey
	endpointID string
}

// NewWebhookAsymmetric creates a verifier for asymmetric (`v2a`) signatures.
//
// endpointID is the ID of the endpoint, as shared by the producer. It may not
// be empty or contain a `.`.
//
// publicKeys is the list of trusted public keys, each serialized as `whpk_` followed
// by the base64 encoded ed25519 public key. A webhook is accepted if it was signed
// by any of them, so keys can be rotated by adding the new key to the list.
// The list should be ordered from newest to oldest: the first key is the one
// used for responding to endpoint verification challenges.
func NewWebhookAsymmetric(endpointID string, publicKeys []string) (*WebhookAsymmetric, error) {
	if len(publicKeys) == 0 {
		return nil, ErrNoPublicKeys
	}
	if err := validateEndpointID(endpointID); err != nil {
		return nil, err
	}

	keys := make([]ed25519.PublicKey, 0, len(publicKeys))
	for _, publicKey := range publicKeys {
		encoded, ok := strings.CutPrefix(publicKey, webhookPublicKeyPrefix)
		if !ok {
			return nil, fmt.Errorf("unable to create webhook, err: %w", ErrInvalidPublicKey)
		}
		key, err := base64enc.DecodeString(encoded)
		if err != nil {
			return nil, fmt.Errorf("unable to create webhook, err: %w", errors.Join(err, ErrInvalidPublicKey))
		}
		if len(key) != ed25519.PublicKeySize {
			return nil, fmt.Errorf("unable to create webhook, err: %w", ErrInvalidPublicKey)
		}
		keys = append(keys, ed25519.PublicKey(key))
	}

	return &WebhookAsymmetric{
		publicKeys: keys,
		endpointID: endpointID,
	}, nil
}

// Verify validates the payload against the webhook signature headers
// using the trusted public keys and the endpoint ID.
//
// Returns an error if the body or headers are missing/unreadable
// or if no signature matches.
func (wh *WebhookAsymmetric) Verify(payload []byte, headers http.Header) error {
	return wh.verify(payload, headers, true)
}

// VerifyIgnoringTimestamp validates the payload against the webhook signature headers
// using the trusted public keys and the endpoint ID.
//
// Returns an error if the body or headers are missing/unreadable
// or if no signature matches.
//
// WARNING: This function does not check the signature's timestamp.
// We recommend using the `Verify` function instead.
func (wh *WebhookAsymmetric) VerifyIgnoringTimestamp(payload []byte, headers http.Header) error {
	return wh.verify(payload, headers, false)
}

func (wh *WebhookAsymmetric) verify(payload []byte, headers http.Header, enforceTolerance bool) error {
	msgId, msgSignature, timestamp, err := parseVerifyHeaders(headers, enforceTolerance)
	if err != nil {
		return err
	}

	digest, err := AsymmetricSignedContentHash(wh.endpointID, msgId, timestamp, payload)
	if err != nil {
		return err
	}

	prefix := asymmetricSignatureVersion + ","
	for _, versionedSignature := range strings.Split(msgSignature, " ") {
		encoded, ok := strings.CutPrefix(versionedSignature, prefix)
		if !ok {
			continue
		}
		signature, err := base64enc.DecodeString(encoded)
		if err != nil || len(signature) != ed25519.SignatureSize {
			continue
		}
		for _, publicKey := range wh.publicKeys {
			if ed25519.Verify(publicKey, digest, signature) {
				return nil
			}
		}
	}

	return fmt.Errorf("unable to verify payload, err: %w", ErrNoMatchingSignature)
}

// EndpointVerificationResponse calculates the response to an endpoint verification challenge.
//
// Takes the value of the `webhook-challenge` header and returns the value
// for the `webhook-challenge-response` header. The response is calculated
// using the first (newest) public key.
func (wh *WebhookAsymmetric) EndpointVerificationResponse(challenge string) string {
	return wh.endpointVerificationResponse(wh.publicKeys[0], challenge)
}

// VerifyEndpointVerificationResponse validates the value of the `webhook-challenge-response`
// header against the challenge that was sent.
//
// The response is accepted if it was calculated using any of the public keys.
// Returns an error if the signature doesn't match.
func (wh *WebhookAsymmetric) VerifyEndpointVerificationResponse(challenge string, header string) error {
	for _, publicKey := range wh.publicKeys {
		if hmac.Equal([]byte(header), []byte(wh.endpointVerificationResponse(publicKey, challenge))) {
			return nil
		}
	}

	return fmt.Errorf("unable to verify challenge response, err: %w", ErrNoMatchingSignature)
}

func (wh *WebhookAsymmetric) endpointVerificationResponse(publicKey ed25519.PublicKey, challenge string) string {
	endpointKey := make([]byte, 0, len(publicKey)+1+len(wh.endpointID))
	endpointKey = append(endpointKey, publicKey...)
	endpointKey = append(endpointKey, '.')
	endpointKey = append(endpointKey, wh.endpointID...)

	return fmt.Sprintf("%s,%s", asymmetricSignatureVersion, signChallenge(endpointKey, challenge))
}

// WebhookAsymmetricSigner signs webhooks with asymmetric (`v2a`) signatures,
// using a private key held by the producer.
//
// It's recommended to hold the private key in a KMS or an HSM instead, and
// sign using AsymmetricSignedContentHash and FormatAsymmetricSignature.
//
// A signer isn't tied to an endpoint: one key pair can serve all endpoints.
type WebhookAsymmetricSigner struct {
	privateKey ed25519.PrivateKey
}

// NewWebhookAsymmetricSigner creates a signer for asymmetric (`v2a`) signatures.
//
// secretKey is serialized as `whsk_` followed by the base64 encoded 32 byte
// ed25519 seed (private key).
func NewWebhookAsymmetricSigner(secretKey string) (*WebhookAsymmetricSigner, error) {
	encoded, ok := strings.CutPrefix(secretKey, webhookSecretKeyPrefix)
	if !ok {
		return nil, fmt.Errorf("unable to create signer, err: %w", ErrInvalidSecretKey)
	}
	seed, err := base64enc.DecodeString(encoded)
	if err != nil {
		return nil, fmt.Errorf("unable to create signer, err: %w", errors.Join(err, ErrInvalidSecretKey))
	}
	if len(seed) != ed25519.SeedSize {
		return nil, fmt.Errorf("unable to create signer, err: %w", ErrInvalidSecretKey)
	}

	return &WebhookAsymmetricSigner{
		privateKey: ed25519.NewKeyFromSeed(seed),
	}, nil
}

// Sign calculates the `v2a` signature of a webhook sent to the endpoint
// with the given endpoint ID, and returns it formatted for the
// `webhook-signature` header (`v2a,<base64 signature>`).
func (s *WebhookAsymmetricSigner) Sign(endpointID string, msgID string, timestamp time.Time, payload []byte) (string, error) {
	digest, err := AsymmetricSignedContentHash(endpointID, msgID, timestamp, payload)
	if err != nil {
		return "", err
	}
	return FormatAsymmetricSignature(ed25519.Sign(s.privateKey, digest))
}

// PublicKey returns the public key matching the secret key, serialized as
// `whpk_` followed by the base64 encoded ed25519 public key.
// This is the key to share with consumers.
func (s *WebhookAsymmetricSigner) PublicKey() string {
	publicKey, _ := FormatAsymmetricPublicKey(s.privateKey.Public().(ed25519.PublicKey))
	return publicKey
}

// AsymmetricSignedContentHash returns the SHA-256 hash of the content signed
// by `v2a` signatures (`endpoint_id.msg_id.timestamp.payload`), as raw 32 bytes.
//
// This is useful for producers that sign with a KMS or an HSM: sign the
// returned hash using (pure) ed25519, and pass the signature to
// FormatAsymmetricSignature.
func AsymmetricSignedContentHash(endpointID string, msgID string, timestamp time.Time, payload []byte) ([]byte, error) {
	if err := validateEndpointID(endpointID); err != nil {
		return nil, err
	}

	h := sha256.New()
	fmt.Fprintf(h, "%s.%s.%d.", endpointID, msgID, timestamp.Unix())
	h.Write(payload)
	return h.Sum(nil), nil
}

// FormatAsymmetricSignature formats a raw 64 byte ed25519 signature for the
// `webhook-signature` header (`v2a,<base64 signature>`).
func FormatAsymmetricSignature(signature []byte) (string, error) {
	if len(signature) != ed25519.SignatureSize {
		return "", fmt.Errorf("unable to format signature, err: %w", ErrInvalidSignature)
	}
	return asymmetricSignatureVersion + "," + base64enc.EncodeToString(signature), nil
}

// ed25519SPKIPrefix is the DER prefix of an ed25519 SubjectPublicKeyInfo,
// followed by the raw 32 byte public key.
var ed25519SPKIPrefix = []byte{0x30, 0x2a, 0x30, 0x05, 0x06, 0x03, 0x2b, 0x65, 0x70, 0x03, 0x21, 0x00}

// FormatAsymmetricPublicKey formats an ed25519 public key as
// `whpk_<base64 public key>`, to share with consumers.
//
// publicKey is either the raw 32 byte public key, or the DER encoded
// SubjectPublicKeyInfo of an ed25519 public key (e.g. what AWS KMS
// GetPublicKey returns).
func FormatAsymmetricPublicKey(publicKey []byte) (string, error) {
	if len(publicKey) == len(ed25519SPKIPrefix)+ed25519.PublicKeySize {
		if !bytes.HasPrefix(publicKey, ed25519SPKIPrefix) {
			return "", fmt.Errorf("unable to format public key, not an ed25519 SubjectPublicKeyInfo, err: %w", ErrInvalidPublicKey)
		}
		publicKey = publicKey[len(ed25519SPKIPrefix):]
	}
	if len(publicKey) != ed25519.PublicKeySize {
		return "", fmt.Errorf("unable to format public key, err: %w", ErrInvalidPublicKey)
	}
	return webhookPublicKeyPrefix + base64enc.EncodeToString(publicKey), nil
}

func validateEndpointID(endpointID string) error {
	if endpointID == "" || strings.Contains(endpointID, ".") {
		return ErrInvalidEndpointID
	}
	return nil
}
