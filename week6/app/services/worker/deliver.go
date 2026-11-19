package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"time"

	"switchyard/worker/internal/standardwebhooks"
)

// send makes one attempt. It returns the HTTP status, or an error text when
// no response arrived.
func (w *Worker) send(d Delivery, attemptID string) (int, string) {
	var body bytes.Buffer
	if err := json.Compact(&body, d.Event); err != nil {
		return 0, "event is not JSON"
	}
	var event struct {
		ID string `json:"id"`
	}
	_ = json.Unmarshal(d.Event, &event)

	signer, err := standardwebhooks.NewWebhook(d.Endpoint.Secret)
	if err != nil {
		return 0, "endpoint secret is not usable"
	}
	stamp := time.Unix(w.now(), 0)
	signature, err := signer.Sign(d.DeliveryID, stamp, body.Bytes())
	if err != nil {
		return 0, "could not sign"
	}

	req, err := http.NewRequest(http.MethodPost, d.Endpoint.URL, bytes.NewReader(body.Bytes()))
	if err != nil {
		return 0, "endpoint url is not usable"
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("webhook-id", d.DeliveryID)
	req.Header.Set("webhook-timestamp", strconv.FormatInt(stamp.Unix(), 10))
	req.Header.Set("webhook-signature", signature)
	req.Header.Set("x-switchyard-attempt", attemptID)
	req.Header.Set("x-switchyard-event", event.ID)

	resp, err := w.merchants.Do(req)
	if err != nil {
		return 0, fmt.Sprintf("no response: %T", err)
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 1<<16))
	return resp.StatusCode, ""
}

// outcomeOf maps an attempt to the outcome reported to the router.
func outcomeOf(a *attempt) string {
	switch {
	case !a.Finished:
		return "unknown"
	case a.Error != "":
		return "failed"
	case a.Status >= 200 && a.Status < 300:
		return "delivered"
	default:
		return "failed"
	}
}

func resultOf(a *attempt) Result {
	return Result{DeliveryID: a.Delivery, AttemptID: a.ID, Outcome: outcomeOf(a), Status: a.Status, Error: a.Error}
}
