package main

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
)

// The journal is the worker's own durable record of what it holds and what it
// has done with it: one line per attempt, numbered by attempt, written again
// each time the attempt moves on. The last line with a given n is the attempt
// as it stands. A lease is one delivery, and a lease with no attempt started is
// not in the journal.
type Record struct {
	N          int      `json:"n"`
	Rec        string   `json:"rec"`
	At         int64    `json:"at"`
	Lease      string   `json:"lease,omitempty"`
	Deliveries []string `json:"deliveries,omitempty"`
	Delivery   string   `json:"delivery,omitempty"`
	Attempt    string   `json:"attempt,omitempty"`
	Status     int      `json:"status,omitempty"`
	Error      string   `json:"error,omitempty"`
}

type attempt struct {
	Delivery string
	ID       string
	Finished bool
	Status   int
	Error    string
}

type heldLease struct {
	ID         string
	Deliveries []string
	Attempts   []*attempt
	Reported   bool
}

type line struct {
	N        int    `json:"n"`
	At       int64  `json:"at"`
	Lease    string `json:"lease"`
	Delivery string `json:"delivery"`
	Attempt  string `json:"attempt"`
	Status   *int   `json:"status,omitempty"`
	Error    string `json:"error,omitempty"`
	Reported bool   `json:"reported,omitempty"`
	Closed   bool   `json:"closed,omitempty"`
}

type Journal struct {
	file *os.File
	n    int // attempts numbered so far
	open map[int]*line
}

func OpenJournal(stateDir string) (*Journal, error) {
	dir := filepath.Join(stateDir, "worker")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, err
	}
	path := filepath.Join(dir, "journal.jsonl")
	j := &Journal{open: map[int]*line{}}
	if old, err := os.Open(path); err == nil {
		scanner := bufio.NewScanner(old)
		scanner.Buffer(make([]byte, 0, 1<<20), 1<<24)
		for scanner.Scan() {
			var l line
			if json.Unmarshal(scanner.Bytes(), &l) != nil || l.N == 0 {
				continue // a torn final line
			}
			if l.N > j.n {
				j.n = l.N
			}
			if l.Closed {
				delete(j.open, l.N)
			} else {
				kept := l
				j.open[l.N] = &kept
			}
		}
		old.Close()
	}
	file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	j.file = file
	return j, nil
}

func (j *Journal) write(l *line) error {
	data, err := json.Marshal(l)
	if err != nil {
		return err
	}
	_, err = j.file.Write(append(data, '\n'))
	return err
}

func (j *Journal) ofLease(lease string) []*line {
	var found []*line
	for _, l := range j.open {
		if l.Lease == lease {
			found = append(found, l)
		}
	}
	sort.Slice(found, func(a, b int) bool { return found[a].N < found[b].N })
	return found
}

func (j *Journal) Append(rec Record) error {
	switch rec.Rec {
	case "attempt.started":
		j.n++
		l := &line{N: j.n, At: rec.At, Lease: rec.Lease, Delivery: rec.Delivery, Attempt: rec.Attempt}
		j.open[l.N] = l
		return j.write(l)
	case "attempt.finished":
		for _, l := range j.ofLease(rec.Lease) {
			if l.Attempt == rec.Attempt {
				status := rec.Status
				l.At, l.Status, l.Error = rec.At, &status, rec.Error
				return j.write(l)
			}
		}
	case "lease.reported":
		for _, l := range j.ofLease(rec.Lease) {
			l.At, l.Reported = rec.At, true
			if err := j.write(l); err != nil {
				return err
			}
		}
	case "lease.closed":
		for _, l := range j.ofLease(rec.Lease) {
			l.At, l.Closed = rec.At, true
			if err := j.write(l); err != nil {
				return err
			}
			delete(j.open, l.N)
		}
	}
	return nil
}

// Held returns the leases that are not closed, oldest first.
func (j *Journal) Held() []*heldLease {
	numbers := make([]int, 0, len(j.open))
	for n := range j.open {
		numbers = append(numbers, n)
	}
	sort.Ints(numbers)
	var out []*heldLease
	byLease := map[string]*heldLease{}
	for _, n := range numbers {
		l := j.open[n]
		lease := byLease[l.Lease]
		if lease == nil {
			lease = &heldLease{ID: l.Lease}
			byLease[l.Lease] = lease
			out = append(out, lease)
		}
		a := &attempt{Delivery: l.Delivery, ID: l.Attempt, Finished: l.Status != nil, Error: l.Error}
		if l.Status != nil {
			a.Status = *l.Status
		}
		lease.Deliveries = append(lease.Deliveries, l.Delivery)
		lease.Attempts = append(lease.Attempts, a)
		lease.Reported = lease.Reported || l.Reported
	}
	return out
}
