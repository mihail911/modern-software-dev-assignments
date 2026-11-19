package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

// Client for the router's queue.

type Endpoint struct {
	ID     string `json:"id"`
	URL    string `json:"url"`
	Secret string `json:"secret"`
}

type Delivery struct {
	DeliveryID string          `json:"delivery_id"`
	AttemptNo  int             `json:"attempt_no"`
	Event      json.RawMessage `json:"event"`
	Endpoint   Endpoint        `json:"endpoint"`
}

type Lease struct {
	LeaseID   string    `json:"lease_id"`
	ExpiresAt int64     `json:"expires_at"`
	Delivery  *Delivery `json:"delivery"`
}

type Result struct {
	DeliveryID string `json:"delivery_id"`
	AttemptID  string `json:"attempt_id"`
	Outcome    string `json:"outcome"`
	Status     int    `json:"status,omitempty"`
	Error      string `json:"error,omitempty"`
}

type Queue struct {
	base string
	http *http.Client
}

func (q *Queue) post(path string, in any, out any) error {
	body, err := json.Marshal(in)
	if err != nil {
		return err
	}
	resp, err := q.http.Post(q.base+path, "application/json", bytes.NewReader(body))
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("router %s: status %d", path, resp.StatusCode)
	}
	if out == nil {
		return nil
	}
	return json.Unmarshal(data, out)
}

func (q *Queue) Lease(workerID string) (*Lease, error) {
	var lease Lease
	err := q.post("/queue/lease", map[string]any{"worker_id": workerID}, &lease)
	return &lease, err
}

func (q *Queue) Report(leaseID string, results []Result) error {
	return q.post("/queue/report", map[string]any{"lease_id": leaseID, "results": results}, nil)
}

func (q *Queue) Close(leaseID string) error {
	return q.post("/queue/close", map[string]any{"lease_id": leaseID}, nil)
}
