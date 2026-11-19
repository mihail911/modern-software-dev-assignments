package standardwebhooks_test

import (
	"bytes"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	standardwebhooks "github.com/standard-webhooks/standard-webhooks/libraries/go"
)

// Test-only keys from the spec. Never use them for anything else.
const (
	asymmetricSecretKey  = "whsk_4uKPjji/dhB8w6GW3YYDnIMYvXcdHrvH6lidTr4Z+EY="
	asymmetricPublicKey  = "whpk_uWKK13PfwqyOgrqT0OrWKcS2uVD7e89+oUKnkoPjx7w="
	asymmetricEndpointID = "ep_2fVZdC8Kx9TR4ZoCqHWt3NnLgRb"
	// An unrelated key, used for testing key lists
	asymmetricOtherSecretKey = "whsk_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSwMfKQ9r8GKYo="
)

func asymmetricSigner(t *testing.T, secretKey string) *standardwebhooks.WebhookAsymmetricSigner {
	t.Helper()
	signer, err := standardwebhooks.NewWebhookAsymmetricSigner(secretKey)
	if err != nil {
		t.Fatal(err)
	}
	return signer
}

func asymmetricSign(t *testing.T, signer *standardwebhooks.WebhookAsymmetricSigner, endpointID string, msgID string, timestamp time.Time, payload []byte) string {
	t.Helper()
	signature, err := signer.Sign(endpointID, msgID, timestamp, payload)
	if err != nil {
		t.Fatal(err)
	}
	return signature
}

func asymmetricHeaders(msgID string, signature string, timestamp time.Time) http.Header {
	header := http.Header{}
	header.Set(standardwebhooks.HeaderWebhookID, msgID)
	header.Set(standardwebhooks.HeaderWebhookSignature, signature)
	header.Set(standardwebhooks.HeaderWebhookTimestamp, fmt.Sprint(timestamp.Unix()))
	return header
}

func TestWebhookAsymmetricSpecVector(t *testing.T) {
	msgID := "msg_2KWPBgLlAfxdpx2AI54pPJ85f4W"
	timestamp := time.Unix(1674087231, 0)
	payload := []byte(`{"type":"contact.created","timestamp":"2022-11-03T20:26:10.344522Z","data":{"id":"1f81eb52-5198-4599-803e-771906343485"}}`)
	signature := "v2a,kG9XaxZjAEpKB8KeQQb5G1Xp73YH5Rf5gxoWe7cg1kYMiyspdtZXe9w+dAwKeOurAwxfqmlOG/QR4m9s37DzAg=="

	signer := asymmetricSigner(t, asymmetricSecretKey)
	if pk := signer.PublicKey(); pk != asymmetricPublicKey {
		t.Fatalf("public key %s != expected public key %s", pk, asymmetricPublicKey)
	}

	digest, err := standardwebhooks.AsymmetricSignedContentHash(asymmetricEndpointID, msgID, timestamp, payload)
	if err != nil {
		t.Fatal(err)
	}
	if d := base64.StdEncoding.EncodeToString(digest); d != "YXCi5iN+rjB+iEb6bvg1XbWbVBL+r2S+as3k3n1Ih/A=" {
		t.Fatalf("unexpected digest %s", d)
	}

	if s := asymmetricSign(t, signer, asymmetricEndpointID, msgID, timestamp, payload); s != signature {
		t.Fatalf("signature %s != expected signature %s", s, signature)
	}

	wh, err := standardwebhooks.NewWebhookAsymmetric(asymmetricEndpointID, []string{asymmetricPublicKey})
	if err != nil {
		t.Fatal(err)
	}

	header := asymmetricHeaders(msgID, signature, timestamp)
	if err := wh.VerifyIgnoringTimestamp(payload, header); err != nil {
		t.Fatalf("expected spec vector to verify, got %s", err)
	}
	if err := wh.Verify(payload, header); !errors.Is(err, standardwebhooks.ErrMessageTooOld) {
		t.Fatalf("expected %s, got %v", standardwebhooks.ErrMessageTooOld, err)
	}
}

func TestWebhookAsymmetric(t *testing.T) {
	privateKey := asymmetricSigner(t, asymmetricSecretKey)
	otherPublicKey := asymmetricSigner(t, asymmetricOtherSecretKey).PublicKey()

	testCases := []struct {
		name               string
		publicKeys         []string
		endpointID         string
		signingKey         *standardwebhooks.WebhookAsymmetricSigner
		timestamp          time.Time
		modify             func(header http.Header, payload *[]byte)
		noEnforceTimestamp bool
		expectedErr        error
	}{
		{
			name:       "valid signature is valid",
			signingKey: privateKey,
		},
		{
			name:       "v1 entry followed by a valid signature is valid",
			signingKey: privateKey,
			modify: func(header http.Header, payload *[]byte) {
				header.Set("webhook-signature", "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc= "+header.Get("webhook-signature"))
			},
		},
		{
			name:       "malformed entries followed by a valid signature is valid",
			signingKey: privateKey,
			modify: func(header http.Header, payload *[]byte) {
				header.Set("webhook-signature", ",,, v2a v2a,not-base64 v2a,AAAA v1a,abc  "+header.Get("webhook-signature"))
			},
		},
		{
			name:       "signature by the second key in the list is valid",
			publicKeys: []string{otherPublicKey, asymmetricPublicKey},
			signingKey: privateKey,
		},
		{
			name:        "signature by a key not in the list is invalid",
			publicKeys:  []string{otherPublicKey},
			signingKey:  privateKey,
			expectedErr: standardwebhooks.ErrNoMatchingSignature,
		},
		{
			name:        "wrong endpoint id is invalid",
			endpointID:  "ep_other",
			signingKey:  privateKey,
			expectedErr: standardwebhooks.ErrNoMatchingSignature,
		},
		{
			name:       "tampered payload is invalid",
			signingKey: privateKey,
			modify: func(header http.Header, payload *[]byte) {
				*payload = []byte(`{"test": 2432232315}`)
			},
			expectedErr: standardwebhooks.ErrNoMatchingSignature,
		},
		{
			name:       "valid signature with another version is invalid",
			signingKey: privateKey,
			modify: func(header http.Header, payload *[]byte) {
				header.Set("webhook-signature", strings.Replace(header.Get("webhook-signature"), "v2a,", "v1a,", 1))
			},
			expectedErr: standardwebhooks.ErrNoMatchingSignature,
		},
		{
			name:       "missing id returns error",
			signingKey: privateKey,
			modify: func(header http.Header, payload *[]byte) {
				header.Del("webhook-id")
			},
			expectedErr: standardwebhooks.ErrRequiredHeaders,
		},
		{
			name:       "missing timestamp returns error",
			signingKey: privateKey,
			modify: func(header http.Header, payload *[]byte) {
				header.Del("webhook-timestamp")
			},
			expectedErr: standardwebhooks.ErrRequiredHeaders,
		},
		{
			name:       "missing signature returns error",
			signingKey: privateKey,
			modify: func(header http.Header, payload *[]byte) {
				header.Del("webhook-signature")
			},
			expectedErr: standardwebhooks.ErrRequiredHeaders,
		},
		{
			name:        "old timestamp fails",
			signingKey:  privateKey,
			timestamp:   time.Now().Add(tolerance*-1 - time.Second),
			expectedErr: standardwebhooks.ErrMessageTooOld,
		},
		{
			name:        "new timestamp fails",
			signingKey:  privateKey,
			timestamp:   time.Now().Add(tolerance + time.Second),
			expectedErr: standardwebhooks.ErrMessageTooNew,
		},
		{
			name:               "old timestamp passes when ignoring tolerance",
			signingKey:         privateKey,
			timestamp:          time.Now().Add(tolerance*-1 - time.Second),
			noEnforceTimestamp: true,
		},
	}

	for _, tc := range testCases {
		publicKeys := tc.publicKeys
		if publicKeys == nil {
			publicKeys = []string{asymmetricPublicKey}
		}
		endpointID := tc.endpointID
		if endpointID == "" {
			endpointID = asymmetricEndpointID
		}
		timestamp := tc.timestamp
		if timestamp.IsZero() {
			timestamp = time.Now()
		}

		payload := defaultPayload
		header := asymmetricHeaders(defaultMsgID, asymmetricSign(t, tc.signingKey, asymmetricEndpointID, defaultMsgID, timestamp, payload), timestamp)
		if tc.modify != nil {
			tc.modify(header, &payload)
		}

		wh, err := standardwebhooks.NewWebhookAsymmetric(endpointID, publicKeys)
		if err != nil {
			t.Errorf("%s: %s", tc.name, err)
			continue
		}
		if tc.noEnforceTimestamp {
			err = wh.VerifyIgnoringTimestamp(payload, header)
		} else {
			err = wh.Verify(payload, header)
		}
		if tc.expectedErr == nil && err != nil {
			t.Errorf("%s: failed with err %s but shouldn't have", tc.name, err)
		} else if tc.expectedErr != nil && !errors.Is(err, tc.expectedErr) {
			t.Errorf("%s: expected %s, got %v", tc.name, tc.expectedErr, err)
		}
	}
}

func TestWebhookAsymmetricConstructorErrors(t *testing.T) {
	testCases := []struct {
		name        string
		publicKeys  []string
		endpointID  string
		expectedErr error
	}{
		{"no public keys", []string{}, asymmetricEndpointID, standardwebhooks.ErrNoPublicKeys},
		{"nil public keys", nil, asymmetricEndpointID, standardwebhooks.ErrNoPublicKeys},
		{"missing whpk_ prefix", []string{strings.TrimPrefix(asymmetricPublicKey, "whpk_")}, asymmetricEndpointID, standardwebhooks.ErrInvalidPublicKey},
		{"whsec_ prefix", []string{"whsec_" + strings.TrimPrefix(asymmetricPublicKey, "whpk_")}, asymmetricEndpointID, standardwebhooks.ErrInvalidPublicKey},
		{"invalid base64", []string{"whpk_not base64!"}, asymmetricEndpointID, standardwebhooks.ErrInvalidPublicKey},
		{"key too short", []string{"whpk_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"}, asymmetricEndpointID, standardwebhooks.ErrInvalidPublicKey},
		{"empty key", []string{"whpk_"}, asymmetricEndpointID, standardwebhooks.ErrInvalidPublicKey},
		{"one invalid key in the list", []string{asymmetricPublicKey, "whpk_"}, asymmetricEndpointID, standardwebhooks.ErrInvalidPublicKey},
		{"empty endpoint id", []string{asymmetricPublicKey}, "", standardwebhooks.ErrInvalidEndpointID},
		{"endpoint id with a dot", []string{asymmetricPublicKey}, "ep_abc.def", standardwebhooks.ErrInvalidEndpointID},
	}

	for _, tc := range testCases {
		wh, err := standardwebhooks.NewWebhookAsymmetric(tc.endpointID, tc.publicKeys)
		if wh != nil || !errors.Is(err, tc.expectedErr) {
			t.Errorf("%s: expected %s, got %v", tc.name, tc.expectedErr, err)
		}
	}
}

func TestWebhookAsymmetricChallengeResponse(t *testing.T) {
	challenge := "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
	expected := "v2a,t0RDFhzk7OtVzgvMsjuXnodSWuby2BVX+zPqw2/ksUk="
	otherPublicKey := asymmetricSigner(t, asymmetricOtherSecretKey).PublicKey()

	wh, err := standardwebhooks.NewWebhookAsymmetric(asymmetricEndpointID, []string{asymmetricPublicKey, otherPublicKey})
	if err != nil {
		t.Fatal(err)
	}

	response := wh.EndpointVerificationResponse(challenge)
	if response != expected {
		t.Fatalf("challenge response %s != expected challenge response %s", response, expected)
	}
}

func TestWebhookAsymmetricVerifyChallengeResponse(t *testing.T) {
	challenge := "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
	header := "v2a,t0RDFhzk7OtVzgvMsjuXnodSWuby2BVX+zPqw2/ksUk="
	otherPublicKey := asymmetricSigner(t, asymmetricOtherSecretKey).PublicKey()

	// The matching key is the second one in the list
	wh, err := standardwebhooks.NewWebhookAsymmetric(asymmetricEndpointID, []string{otherPublicKey, asymmetricPublicKey})
	if err != nil {
		t.Fatal(err)
	}

	if err := wh.VerifyEndpointVerificationResponse(challenge, header); err != nil {
		t.Fatalf("expected challenge response to be valid, got %s", err)
	}

	if err := wh.VerifyEndpointVerificationResponse(challenge, "v2,abc "+header); !errors.Is(err, standardwebhooks.ErrNoMatchingSignature) {
		t.Fatalf("expected %s for a list of signatures, got %v", standardwebhooks.ErrNoMatchingSignature, err)
	}

	if err := wh.VerifyEndpointVerificationResponse("wrong-challenge", header); !errors.Is(err, standardwebhooks.ErrNoMatchingSignature) {
		t.Fatalf("expected %s, got %v", standardwebhooks.ErrNoMatchingSignature, err)
	}

	wh, err = standardwebhooks.NewWebhookAsymmetric("ep_other", []string{asymmetricPublicKey})
	if err != nil {
		t.Fatal(err)
	}
	if err := wh.VerifyEndpointVerificationResponse(challenge, header); !errors.Is(err, standardwebhooks.ErrNoMatchingSignature) {
		t.Fatalf("expected %s for another endpoint id, got %v", standardwebhooks.ErrNoMatchingSignature, err)
	}
}

func TestWebhookAsymmetricSignerConstructorErrors(t *testing.T) {
	testCases := []struct {
		name      string
		secretKey string
	}{
		{"missing whsk_ prefix", strings.TrimPrefix(asymmetricSecretKey, "whsk_")},
		{"whpk_ key instead of whsk_", asymmetricPublicKey},
		{"invalid base64", "whsk_not base64!"},
		{"key too short", "whsk_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"},
		{"key too long", "whsk_" + base64.StdEncoding.EncodeToString(make([]byte, 64))},
		{"empty key", "whsk_"},
	}

	for _, tc := range testCases {
		signer, err := standardwebhooks.NewWebhookAsymmetricSigner(tc.secretKey)
		if signer != nil || !errors.Is(err, standardwebhooks.ErrInvalidSecretKey) {
			t.Errorf("%s: expected %s, got %v", tc.name, standardwebhooks.ErrInvalidSecretKey, err)
		}
	}
}

func TestWebhookAsymmetricSignInvalidEndpointID(t *testing.T) {
	signer := asymmetricSigner(t, asymmetricSecretKey)
	for _, endpointID := range []string{"", "ep_abc.def"} {
		if _, err := signer.Sign(endpointID, defaultMsgID, time.Now(), defaultPayload); !errors.Is(err, standardwebhooks.ErrInvalidEndpointID) {
			t.Errorf("Sign(%q): expected %s, got %v", endpointID, standardwebhooks.ErrInvalidEndpointID, err)
		}
		if _, err := standardwebhooks.AsymmetricSignedContentHash(endpointID, defaultMsgID, time.Now(), defaultPayload); !errors.Is(err, standardwebhooks.ErrInvalidEndpointID) {
			t.Errorf("AsymmetricSignedContentHash(%q): expected %s, got %v", endpointID, standardwebhooks.ErrInvalidEndpointID, err)
		}
	}
}

func TestFormatAsymmetricSignature(t *testing.T) {
	signature := bytes.Repeat([]byte{0xab}, ed25519.SignatureSize)
	formatted, err := standardwebhooks.FormatAsymmetricSignature(signature)
	if err != nil {
		t.Fatal(err)
	}
	encoded, ok := strings.CutPrefix(formatted, "v2a,")
	if !ok {
		t.Fatalf("expected v2a prefix, got %s", formatted)
	}
	decoded, err := base64.StdEncoding.DecodeString(encoded)
	if err != nil || !bytes.Equal(decoded, signature) {
		t.Fatalf("formatted signature %s doesn't round-trip", formatted)
	}

	for _, size := range []int{0, 32, 63, 65} {
		if _, err := standardwebhooks.FormatAsymmetricSignature(make([]byte, size)); !errors.Is(err, standardwebhooks.ErrInvalidSignature) {
			t.Errorf("size %d: expected %s, got %v", size, standardwebhooks.ErrInvalidSignature, err)
		}
	}
}

func TestFormatAsymmetricPublicKey(t *testing.T) {
	raw, err := base64.StdEncoding.DecodeString(strings.TrimPrefix(asymmetricPublicKey, "whpk_"))
	if err != nil || len(raw) != ed25519.PublicKeySize {
		t.Fatalf("bad test key: %v", err)
	}
	prefix, _ := hex.DecodeString("302a300506032b6570032100")
	der := append(append([]byte{}, prefix...), raw...)

	for name, key := range map[string][]byte{"raw": raw, "DER": der} {
		formatted, err := standardwebhooks.FormatAsymmetricPublicKey(key)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if formatted != asymmetricPublicKey {
			t.Errorf("%s: expected %s, got %s", name, asymmetricPublicKey, formatted)
		}
	}

	badDER := append([]byte{}, der...)
	badDER[0] = 0x31
	for name, key := range map[string][]byte{"31 bytes": raw[:31], "33 bytes": append(append([]byte{}, raw...), 0), "wrong DER prefix": badDER} {
		if _, err := standardwebhooks.FormatAsymmetricPublicKey(key); !errors.Is(err, standardwebhooks.ErrInvalidPublicKey) {
			t.Errorf("%s: expected %s, got %v", name, standardwebhooks.ErrInvalidPublicKey, err)
		}
	}
}

func TestWebhookAsymmetricSignAndVerify(t *testing.T) {
	signer := asymmetricSigner(t, asymmetricSecretKey)
	timestamp := time.Now()

	// HSM flow: sign the hash externally, then format
	digest, err := standardwebhooks.AsymmetricSignedContentHash(asymmetricEndpointID, defaultMsgID, timestamp, defaultPayload)
	if err != nil {
		t.Fatal(err)
	}
	seed, _ := base64.StdEncoding.DecodeString(strings.TrimPrefix(asymmetricSecretKey, "whsk_"))
	hsmSignature, err := standardwebhooks.FormatAsymmetricSignature(ed25519.Sign(ed25519.NewKeyFromSeed(seed), digest))
	if err != nil {
		t.Fatal(err)
	}

	signature := asymmetricSign(t, signer, asymmetricEndpointID, defaultMsgID, timestamp, defaultPayload)
	if signature != hsmSignature {
		t.Fatalf("signer signature %s != HSM flow signature %s", signature, hsmSignature)
	}

	wh, err := standardwebhooks.NewWebhookAsymmetric(asymmetricEndpointID, []string{signer.PublicKey()})
	if err != nil {
		t.Fatal(err)
	}
	if err := wh.Verify(defaultPayload, asymmetricHeaders(defaultMsgID, signature, timestamp)); err != nil {
		t.Fatalf("expected signature to verify, got %s", err)
	}
}
