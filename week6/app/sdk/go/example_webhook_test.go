package standardwebhooks_test

import (
	"fmt"
	"log"
	"net/http"
	"time"

	standardwebhooks "github.com/standard-webhooks/standard-webhooks/libraries/go"
)

const (
	secretKey = "MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"
)

// Example_signatureFlow describes the full flow of signature and verification
// of a webhook payload by verifying the timestamp also.
func Example_signatureFlow() {
	var (
		ts = time.Now()
		id = "1234567890"
	)

	wh, err := standardwebhooks.NewWebhook(secretKey)
	if err != nil {
		log.Fatal(err)
	}

	payload := `{"type": "example.created", "timestamp":"2023-09-28T19:20:22+00:00", "data":{"str":"string","bool":true,"int":42}}`

	// signing the payload with the webhook handler
	signature, err := wh.Sign(id, ts, []byte(payload))
	if err != nil {
		log.Fatal(err)
	}

	// generating the http header carrier
	header := http.Header{}
	header.Set(standardwebhooks.HeaderWebhookID, id)
	header.Set(standardwebhooks.HeaderWebhookSignature, signature)
	header.Set(standardwebhooks.HeaderWebhookTimestamp, fmt.Sprint(ts.Unix()))

	// http request is sent to consumer

	// consumer verifies the signature
	err = wh.Verify([]byte(payload), header)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("done")
	// Output: done
}
