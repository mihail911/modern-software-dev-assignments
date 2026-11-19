package standardwebhooks_test

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"

	standardwebhooks "github.com/standard-webhooks/standard-webhooks/libraries/go"
)

var defaultMsgID = "msg_p5jXN8AQM9LWM0D4loKWxJek"
var defaultPayload = []byte(`{"test": 2432232314}`)
var defaultSecret = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
var tolerance time.Duration = 5 * time.Minute

type testPayload struct {
	id        string
	timestamp time.Time
	header    http.Header
	secret    string
	payload   []byte
	signature string
}

func newTestPayload(timestamp time.Time) *testPayload {
	tp := &testPayload{}
	tp.id = defaultMsgID
	tp.timestamp = timestamp

	tp.payload = defaultPayload
	tp.secret = defaultSecret

	wh, _ := standardwebhooks.NewWebhook(tp.secret)
	tp.signature, _ = wh.Sign(tp.id, tp.timestamp, tp.payload)

	tp.header = http.Header{}
	tp.header.Set(standardwebhooks.HeaderWebhookID, tp.id)
	tp.header.Set(standardwebhooks.HeaderWebhookSignature, tp.signature)
	tp.header.Set(standardwebhooks.HeaderWebhookTimestamp, fmt.Sprint(tp.timestamp.Unix()))

	return tp
}

func TestWebhook(t *testing.T) {

	testCases := []struct {
		name               string
		testPayload        *testPayload
		modifyPayload      func(*testPayload)
		noEnforceTimestamp bool
		expectedErr        bool
	}{
		{
			name:        "valid signature is valid",
			testPayload: newTestPayload(time.Now()),
			expectedErr: false,
		},
		{
			name:        "missing id returns error",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Del("webhook-id")
			},
			expectedErr: true,
		},
		{
			name:        "missing timestamp returns error",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Del("webhook-timestamp")
			},
			expectedErr: true,
		},
		{
			name:        "missing signature returns error",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Del("webhook-signature")
			},
			expectedErr: true,
		},
		{
			name:        "invalid signature is invalid",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Set("webhook-signature", "v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=")
			},
			expectedErr: true,
		},
		{
			name:        "partial signature is invalid",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Set("webhook-signature", "v1,")
			},
			expectedErr: true,
		},
		{
			name:        "old timestamp fails",
			testPayload: newTestPayload(time.Now().Add(tolerance * -1)),
			expectedErr: true,
		},
		{
			name:        "new timestamp fails",
			testPayload: newTestPayload(time.Now().Add(tolerance + time.Second)),
			expectedErr: true,
		},
		{
			name:        "valid multi sig is valid",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				sigs := []string{
					"v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
					"v2,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
					tp.header.Get("webhook-signature"), // valid signature
					"v1,Ceo5qEr07ixe2NLpvHk3FH9bwy/WavXrAFQ/9tdO6mc=",
				}
				tp.header.Set("webhook-signature", strings.Join(sigs, " "))
			},
			expectedErr: false,
		},
		{
			name:        "entry without a comma followed by a valid signature is valid",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Set("webhook-signature", "v1 "+tp.header.Get("webhook-signature"))
			},
			expectedErr: false,
		},
		{
			name:        "malformed entry followed by a valid signature is valid",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Set("webhook-signature", ",,, v1,not,base64  "+tp.header.Get("webhook-signature"))
			},
			expectedErr: false,
		},
		{
			name:        "valid signature with another version is invalid",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Set("webhook-signature", strings.Replace(tp.header.Get("webhook-signature"), "v1,", "v2,", 1))
			},
			expectedErr: true,
		},
		{
			name:        "valid signature with trailing data is invalid",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Set("webhook-signature", tp.header.Get("webhook-signature")+",extra")
			},
			expectedErr: true,
		},
		{
			name:               "old timestamp passes when ignoring tolerance",
			testPayload:        newTestPayload(time.Now().Add(tolerance * -1)),
			noEnforceTimestamp: true,
			expectedErr:        false,
		},
		{
			name:               "new timestamp passes when ignoring tolerance",
			testPayload:        newTestPayload(time.Now().Add(tolerance * 1)),
			noEnforceTimestamp: true,
			expectedErr:        false,
		},
		{
			name:               "valid timestamp passes when ignoring tolerance",
			testPayload:        newTestPayload(time.Now()),
			noEnforceTimestamp: true,
			expectedErr:        false,
		},
		{
			name:        "invalid timestamp fails when ignoring tolerance",
			testPayload: newTestPayload(time.Now()),
			modifyPayload: func(tp *testPayload) {
				tp.header.Set("webhook-timestamp", fmt.Sprint(time.Date(2000, 1, 1, 0, 0, 0, 0, time.UTC).Unix()))
			},
			noEnforceTimestamp: true,
			expectedErr:        true,
		},
	}

	for _, tc := range testCases {
		if tc.modifyPayload != nil {
			tc.modifyPayload(tc.testPayload)
		}

		wh, err := standardwebhooks.NewWebhook(tc.testPayload.secret)
		if err != nil {
			t.Error(err)
			continue
		}
		if tc.noEnforceTimestamp {
			err = wh.VerifyIgnoringTimestamp(tc.testPayload.payload, tc.testPayload.header)
		} else {
			err = wh.Verify(tc.testPayload.payload, tc.testPayload.header)
		}
		if err != nil && !tc.expectedErr {
			t.Errorf("%s: failed with err %s but shouldn't have", tc.name, err.Error())
		} else if err == nil && tc.expectedErr {
			t.Errorf("%s: didn't error but should have", tc.name)
		}
	}
}

func TestWebhookPrefix(t *testing.T) {
	tp := newTestPayload(time.Now())

	wh, err := standardwebhooks.NewWebhook(tp.secret)
	if err != nil {
		t.Fatal(err)
	}

	err = wh.Verify(tp.payload, tp.header)
	if err != nil {
		t.Fatal(err)
	}

	wh, err = standardwebhooks.NewWebhook(fmt.Sprintf("whsec_%s", tp.secret))
	if err != nil {
		t.Fatal(err)
	}

	err = wh.Verify(tp.payload, tp.header)
	if err != nil {
		t.Fatal(err)
	}
}

func TestWebhookSign(t *testing.T) {
	key := "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
	msgID := "msg_p5jXN8AQM9LWM0D4loKWxJek"
	timestamp := time.Unix(1614265330, 0)
	payload := []byte(`{"test": 2432232314}`)
	expected := "v1,g0hM9SsE+OTPJTGt/tmIKtSyZlE3uFJELVlNIOLJ1OE="

	wh, err := standardwebhooks.NewWebhook(key)
	if err != nil {
		t.Fatal(err)
	}

	signature, err := wh.Sign(msgID, timestamp, payload)
	if err != nil {
		t.Fatal(err)
	}

	if signature != expected {
		t.Fatalf("signature %s != expected signature %s", signature, expected)
	}

}

func TestWebhookNoEmptySecrets(t *testing.T) {
	_, err := standardwebhooks.NewWebhook("")
	if err == nil {
		t.Fatal("Expected error")
	}
}

func TestWebhookNoEmptyPrefixedSecrets(t *testing.T) {
	_, err := standardwebhooks.NewWebhook("whsec_")
	if err == nil {
		t.Fatal("Expected error")
	}
}

func TestWebhookNewRawNoEmptyPrefixedSecrets(t *testing.T) {
	_, err := standardwebhooks.NewWebhookRaw([]byte(""))
	if err == nil {
		t.Fatal("Expected error")
	}
}

func TestChallengeResponse(t *testing.T) {
	challenge := "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
	expected := "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="

	wh, err := standardwebhooks.NewWebhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
	if err != nil {
		t.Fatal(err)
	}

	response := wh.EndpointVerificationResponse(challenge)
	if response != expected {
		t.Fatalf("challenge response %s != expected challenge response %s", response, expected)
	}
}

func TestVerifyChallengeResponse(t *testing.T) {
	challenge := "0njH7N3kV5eXPB+/eh+eoA2Ahvz2JgNp0iC/BLOl6yM="
	header := "v1,2WoO4Z1kjgoudcvk4PX/8LDD6FNJDEUHMtYqZpNx/FA="

	wh, err := standardwebhooks.NewWebhook("whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw")
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
}
