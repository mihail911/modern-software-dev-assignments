// worker: leases deliveries from the router, signs and sends them, and
// reports the results.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"
)

var release = "2.3.1"

type Worker struct {
	mu        sync.Mutex
	id        string
	clock     int64
	crashAt   int
	crashHits int
	journal   *Journal
	queue     *Queue
	merchants *http.Client
	logFile   *os.File
}

func (w *Worker) now() int64 {
	if w.clock != 0 {
		return w.clock
	}
	return time.Now().Unix()
}

// crash implements the crash points listed in contracts/harness.md.
func (w *Worker) crash(point int) {
	if w.crashAt != point {
		return
	}
	w.crashHits--
	if w.crashHits <= 0 {
		os.Exit(86)
	}
}

func (w *Worker) log(op string, fields map[string]any) {
	line := map[string]any{"t": w.now(), "worker": w.id, "op": op}
	for k, v := range fields {
		line[k] = v
	}
	data, _ := json.Marshal(line)
	w.logFile.Write(append(data, '\n'))
}

func (w *Worker) attemptID() string {
	sum := sha256.Sum256([]byte(fmt.Sprintf("switchyard:att:%s:%d", w.id, w.journal.n+1)))
	return "att_" + hex.EncodeToString(sum[:])[:14]
}

// tick settles anything still held, then leases one delivery and attempts it.
func (w *Worker) tick() (int, error) {
	did, err := w.settle()
	if err != nil {
		return did, err
	}
	lease, err := w.queue.Lease(w.id)
	if err != nil {
		return did, err
	}
	if lease.LeaseID == "" || lease.Delivery == nil {
		return did, nil
	}
	w.crash(1)
	d := lease.Delivery
	w.log("lease.opened", map[string]any{"lease": lease.LeaseID, "delivery": d.DeliveryID})
	w.crash(2)

	w.crash(3)
	a := &attempt{Delivery: d.DeliveryID, ID: w.attemptID()}
	started := Record{Rec: "attempt.started", At: w.now(), Lease: lease.LeaseID, Delivery: a.Delivery, Attempt: a.ID}
	if err := w.journal.Append(started); err != nil {
		return did, err
	}
	w.log("attempt.started", map[string]any{
		"lease": lease.LeaseID, "delivery": a.Delivery, "attempt": a.ID,
		"attempt_no": d.AttemptNo, "endpoint": d.Endpoint.ID, "url": d.Endpoint.URL,
	})
	w.crash(4)

	a.Status, a.Error = w.send(*d, a.ID)
	a.Finished = true
	w.crash(5)

	finished := Record{Rec: "attempt.finished", At: w.now(), Lease: lease.LeaseID, Delivery: a.Delivery, Attempt: a.ID, Status: a.Status, Error: a.Error}
	if err := w.journal.Append(finished); err != nil {
		return did, err
	}
	fields := map[string]any{"lease": lease.LeaseID, "delivery": a.Delivery, "attempt": a.ID, "status": a.Status}
	if a.Error != "" {
		fields["error"] = a.Error
	}
	w.log("attempt.finished", fields)
	w.crash(6)

	results := []Result{resultOf(a)}
	did++

	if err := w.queue.Report(lease.LeaseID, results); err != nil {
		return did, err // the journal still holds the lease; the next tick settles it
	}
	w.crash(7)
	if err := w.journal.Append(Record{Rec: "lease.reported", At: w.now(), Lease: lease.LeaseID}); err != nil {
		return did, err
	}
	w.log("lease.reported", map[string]any{"lease": lease.LeaseID, "outcome": results[0].Outcome})
	if err := w.queue.Close(lease.LeaseID); err != nil {
		return did, err
	}
	w.crash(8)
	if err := w.journal.Append(Record{Rec: "lease.closed", At: w.now(), Lease: lease.LeaseID}); err != nil {
		return did, err
	}
	w.log("lease.closed", map[string]any{"lease": lease.LeaseID})
	w.crash(9)
	return did, nil
}

func (w *Worker) state() map[string]any {
	held := []map[string]any{}
	for _, lease := range w.journal.Held() {
		held = append(held, map[string]any{
			"lease": lease.ID, "delivery": lease.Deliveries[0], "reported": lease.Reported,
		})
	}
	return map[string]any{"service": "worker", "now": w.now(), "journal_attempts": w.journal.n, "held": held}
}

func reply(rw http.ResponseWriter, status int, body any) {
	data, _ := json.Marshal(body)
	rw.Header().Set("Content-Type", "application/json")
	rw.Header().Set("Content-Length", strconv.Itoa(len(data)))
	rw.WriteHeader(status)
	rw.Write(data)
}

func (w *Worker) serve(rw http.ResponseWriter, req *http.Request) {
	w.mu.Lock()
	defer w.mu.Unlock()
	var body map[string]any
	if req.Method == http.MethodPost {
		raw, _ := io.ReadAll(req.Body)
		if len(raw) > 0 && json.Unmarshal(raw, &body) != nil {
			reply(rw, 400, map[string]any{"error": "body is not JSON"})
			return
		}
	}
	number := func(key string, fallback float64) float64 {
		if v, ok := body[key].(float64); ok {
			return v
		}
		return fallback
	}
	switch req.Method + " " + req.URL.Path {
	case "GET /__admin/health":
		reply(rw, 200, map[string]any{"service": "worker", "release": release})
	case "GET /__admin/state":
		reply(rw, 200, w.state())
	case "POST /__admin/clock":
		w.clock = int64(number("now", 0))
		reply(rw, 200, map[string]any{"now": w.clock})
	case "POST /__admin/crash_at":
		w.crashAt = int(number("point", 0))
		w.crashHits = int(number("hits", 1))
		reply(rw, 200, map[string]any{"armed": w.crashAt})
	case "POST /__admin/tick":
		did, err := w.tick()
		if err != nil {
			w.log("tick.failed", map[string]any{"error": err.Error()})
			reply(rw, 200, map[string]any{"did": did, "error": err.Error()})
			return
		}
		reply(rw, 200, map[string]any{"did": did})
	default:
		reply(rw, 404, map[string]any{"error": "not found"})
	}
}

func env(name, fallback string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return fallback
}

func main() {
	stateDir := env("SWITCHYARD_STATE", "./state")
	base, _ := strconv.Atoi(env("SWITCHYARD_PORT_BASE", "7100"))

	journal, err := OpenJournal(stateDir)
	if err != nil {
		fmt.Fprintln(os.Stderr, "worker: journal:", err)
		os.Exit(1)
	}
	if err := os.MkdirAll(filepath.Join(stateDir, "logs"), 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "worker: logs:", err)
		os.Exit(1)
	}
	logPath := filepath.Join(stateDir, "logs", "worker.jsonl")
	logFile, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		fmt.Fprintln(os.Stderr, "worker: log:", err)
		os.Exit(1)
	}

	// No connection reuse: a request is never silently repeated on a stale connection.
	oneShot := func() *http.Client {
		return &http.Client{Timeout: 5 * time.Second, Transport: &http.Transport{DisableKeepAlives: true}}
	}
	clock, _ := strconv.ParseInt(env("SWITCHYARD_CLOCK", "0"), 10, 64)
	w := &Worker{
		id:        env("SWITCHYARD_WORKER_ID", "wrk_1"),
		clock:     clock,
		journal:   journal,
		queue:     &Queue{base: fmt.Sprintf("http://127.0.0.1:%d", base+2), http: oneShot()},
		merchants: oneShot(),
		logFile:   logFile,
	}
	w.log("worker.started", map[string]any{"held": len(journal.Held())})

	server := &http.Server{Addr: fmt.Sprintf("127.0.0.1:%d", base+3), Handler: http.HandlerFunc(w.serve)}
	if err := server.ListenAndServe(); err != nil {
		fmt.Fprintln(os.Stderr, "worker:", err)
		os.Exit(1)
	}
}
