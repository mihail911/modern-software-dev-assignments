package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"switchyard/worker/internal/standardwebhooks"
)

const testSecret = "whsec_MfKQ9r8GKYqrTwjUPD8ILPZIo2LaLaSw"

func newTestWorker(t *testing.T) *Worker {
	t.Helper()
	dir := t.TempDir()
	journal, err := OpenJournal(dir)
	if err != nil {
		t.Fatal(err)
	}
	logFile, err := os.Create(filepath.Join(dir, "worker.jsonl"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { logFile.Close() })
	return &Worker{
		id:        "wrk_test",
		journal:   journal,
		merchants: &http.Client{Timeout: 2 * time.Second, Transport: &http.Transport{DisableKeepAlives: true}},
		logFile:   logFile,
	}
}

func testDelivery(url string) Delivery {
	event := json.RawMessage(`{"id": "evt_0123456789abcd", "type": "payment.captured", "created": 1793577600,
		"data": {"object": "payment", "payment_id": "pay_1", "amount_minor": 1250, "currency": "USD"}}`)
	return Delivery{
		DeliveryID: "dlv_0123456789abcd",
		AttemptNo:  1,
		Event:      event,
		Endpoint:   Endpoint{ID: "ep_test", URL: url, Secret: testSecret},
	}
}

func TestOutcomeOfAFinishedAttempt(t *testing.T) {
	for status := 200; status < 600; status++ {
		a := &attempt{Delivery: "dlv_x", ID: "att_x", Finished: true, Status: status}
		want := "failed"
		if status >= 200 && status < 300 {
			want = "delivered"
		}
		got := outcomeOf(a)
		t.Logf("status %d -> %s", status, got)
		if got != want {
			t.Errorf("status %d: outcome %q, want %q", status, got, want)
		}
	}
}

func TestOutcomeOfAnAttemptWithNoResponse(t *testing.T) {
	for _, problem := range []string{"no response: *url.Error", "endpoint secret is not usable", "could not sign"} {
		a := &attempt{Delivery: "dlv_x", ID: "att_x", Finished: true, Error: problem}
		got := resultOf(a)
		t.Logf("%q -> %+v", problem, got)
		if got.Outcome != "failed" {
			t.Errorf("%q: outcome %q, want failed", problem, got.Outcome)
		}
	}
}

func TestSendSignsTheRequest(t *testing.T) {
	verifier, err := standardwebhooks.NewWebhook(testSecret)
	if err != nil {
		t.Fatal(err)
	}
	var seen http.Header
	var body []byte
	merchant := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
		seen = r.Header.Clone()
		body, _ = io.ReadAll(r.Body)
		rw.WriteHeader(http.StatusOK)
	}))
	defer merchant.Close()

	w := newTestWorker(t)
	status, problem := w.send(testDelivery(merchant.URL+"/hooks/ep_test"), "att_0123456789abcd")
	if status != 200 || problem != "" {
		t.Fatalf("send returned %d %q", status, problem)
	}
	for name, values := range seen {
		t.Logf("header %s: %s", name, strings.Join(values, ", "))
	}
	t.Logf("body %s", body)
	if err := verifier.Verify(body, seen); err != nil {
		t.Errorf("signature does not verify: %v", err)
	}
	for header, want := range map[string]string{
		"webhook-id":           "dlv_0123456789abcd",
		"x-switchyard-attempt": "att_0123456789abcd",
		"x-switchyard-event":   "evt_0123456789abcd",
		"Content-Type":         "application/json",
	} {
		if got := seen.Get(header); got != want {
			t.Errorf("%s is %q, want %q", header, got, want)
		}
	}
	if strings.Contains(string(body), "\n") || strings.Contains(string(body), ": ") {
		t.Errorf("body is not compact: %s", body)
	}
}

func TestSendReturnsTheMerchantsStatus(t *testing.T) {
	for _, want := range []int{200, 201, 204, 301, 400, 404, 410, 429, 500, 502, 503} {
		merchant := httptest.NewServer(http.HandlerFunc(func(rw http.ResponseWriter, r *http.Request) {
			rw.WriteHeader(want)
		}))
		w := newTestWorker(t)
		status, problem := w.send(testDelivery(merchant.URL), "att_1")
		merchant.Close()
		t.Logf("merchant answered %d: send returned %d %q", want, status, problem)
		if status != want || problem != "" {
			t.Errorf("merchant answered %d: send returned %d %q", want, status, problem)
		}
	}
}

func TestSendToAnEndpointThatIsNotThere(t *testing.T) {
	merchant := httptest.NewServer(http.NotFoundHandler())
	url := merchant.URL
	merchant.Close()
	w := newTestWorker(t)
	status, problem := w.send(testDelivery(url), "att_1")
	t.Logf("send returned %d %q", status, problem)
	if status != 0 || !strings.HasPrefix(problem, "no response") {
		t.Errorf("send returned %d %q", status, problem)
	}
}

func TestSendWithASecretThatIsNotUsable(t *testing.T) {
	w := newTestWorker(t)
	d := testDelivery("http://127.0.0.1:9/")
	d.Endpoint.Secret = "whsec_not base64!"
	status, problem := w.send(d, "att_1")
	t.Logf("send returned %d %q", status, problem)
	if status != 0 || problem != "endpoint secret is not usable" {
		t.Errorf("send returned %d %q", status, problem)
	}
}

func TestAttemptIdsDoNotRepeat(t *testing.T) {
	w := newTestWorker(t)
	seen := map[string]bool{}
	for i := 0; i < 400; i++ {
		id := w.attemptID()
		t.Logf("record %d: %s", w.journal.n+1, id)
		if seen[id] {
			t.Fatalf("attempt id %s was issued twice", id)
		}
		seen[id] = true
		if err := w.journal.Append(Record{Rec: "attempt.started", At: 1, Lease: "lse_x", Delivery: "dlv_x", Attempt: id}); err != nil {
			t.Fatal(err)
		}
	}
}

// Support and merchants quote attempt ids back to us, so the id a worker
// gives its nth record is the same on every build (ADR-0011).
func TestAttemptIdsAreTheSameOnEveryBuild(t *testing.T) {
	w := newTestWorker(t)
	for i, want := range []string{"att_3d0a77e1c2f7a0", "att_0e755246f593a2"} {
		got := w.attemptID()
		t.Logf("record %d: %s", w.journal.n+1, got)
		if got != want {
			t.Errorf("attempt id %d is %s, want %s", i+1, got, want)
		}
		if err := w.journal.Append(Record{Rec: "attempt.started", At: 1, Lease: "lse_x", Delivery: "dlv_x", Attempt: got}); err != nil {
			t.Fatal(err)
		}
	}

}
