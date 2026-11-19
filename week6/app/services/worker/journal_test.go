package main

import (
	"os"
	"path/filepath"
	"testing"
)

func mustAppend(t *testing.T, j *Journal, rec Record) {
	t.Helper()
	if err := j.Append(rec); err != nil {
		t.Fatal(err)
	}
	t.Logf("journal %d: %s lease=%s delivery=%s attempt=%s status=%d", j.n, rec.Rec, rec.Lease, rec.Delivery, rec.Attempt, rec.Status)
}

func attemptOn(t *testing.T, j *Journal, lease, delivery, id string, status int) {
	t.Helper()
	mustAppend(t, j, Record{Rec: "attempt.started", At: 10, Lease: lease, Delivery: delivery, Attempt: id})
	if status != 0 {
		mustAppend(t, j, Record{Rec: "attempt.finished", At: 10, Lease: lease, Delivery: delivery, Attempt: id, Status: status})
	}
}

func TestAnAttemptIsHeldAfterReopening(t *testing.T) {
	dir := t.TempDir()
	j, err := OpenJournal(dir)
	if err != nil {
		t.Fatal(err)
	}
	attemptOn(t, j, "lse_1", "dlv_a", "att_1", 200)

	again, err := OpenJournal(dir)
	if err != nil {
		t.Fatal(err)
	}
	held := again.Held()
	if len(held) != 1 || held[0].ID != "lse_1" {
		t.Fatalf("held %+v, want lse_1", held)
	}
	if a := held[0].Attempts[0]; !a.Finished || a.Status != 200 || a.Delivery != "dlv_a" {
		t.Errorf("attempt replayed as %+v", a)
	}
}

func TestAnAttemptWithNoResultIsHeldUnfinished(t *testing.T) {
	dir := t.TempDir()
	j, _ := OpenJournal(dir)
	attemptOn(t, j, "lse_1", "dlv_a", "att_1", 0)
	again, _ := OpenJournal(dir)
	held := again.Held()
	if len(held) != 1 || held[0].Attempts[0].Finished {
		t.Fatalf("held %+v, want one unfinished attempt", held)
	}
}

func TestAClosedLeaseIsNotHeld(t *testing.T) {
	dir := t.TempDir()
	j, _ := OpenJournal(dir)
	attemptOn(t, j, "lse_1", "dlv_a", "att_1", 200)
	mustAppend(t, j, Record{Rec: "lease.reported", At: 10, Lease: "lse_1"})
	mustAppend(t, j, Record{Rec: "lease.closed", At: 10, Lease: "lse_1"})
	again, _ := OpenJournal(dir)
	if held := again.Held(); len(held) != 0 {
		t.Errorf("held %d leases, want none", len(held))
	}
}

func TestAReportedLeaseIsHeldUntilItIsClosed(t *testing.T) {
	dir := t.TempDir()
	j, _ := OpenJournal(dir)
	attemptOn(t, j, "lse_1", "dlv_a", "att_1", 503)
	mustAppend(t, j, Record{Rec: "lease.reported", At: 10, Lease: "lse_1"})
	again, _ := OpenJournal(dir)
	held := again.Held()
	if len(held) != 1 || !held[0].Reported {
		t.Fatalf("held %+v, want one reported lease", held)
	}
}

func TestAttemptNumbersContinueAfterReopening(t *testing.T) {
	dir := t.TempDir()
	j, _ := OpenJournal(dir)
	for _, id := range []string{"att_1", "att_2", "att_3"} {
		attemptOn(t, j, "lse_"+id, "dlv_"+id, id, 200)
		mustAppend(t, j, Record{Rec: "lease.reported", At: 10, Lease: "lse_" + id})
		mustAppend(t, j, Record{Rec: "lease.closed", At: 10, Lease: "lse_" + id})
	}
	again, _ := OpenJournal(dir)
	if again.n != 3 {
		t.Errorf("attempts numbered so far %d, want 3", again.n)
	}
}

func TestATornFinalLineIsSkipped(t *testing.T) {
	dir := t.TempDir()
	j, _ := OpenJournal(dir)
	attemptOn(t, j, "lse_1", "dlv_a", "att_1", 0)
	file, err := os.OpenFile(filepath.Join(dir, "worker", "journal.jsonl"), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	file.WriteString(`{"n":2,"at":11,"lea`)
	file.Close()
	again, err := OpenJournal(dir)
	if err != nil {
		t.Fatal(err)
	}
	if held := again.Held(); len(held) != 1 || again.n != 1 {
		t.Errorf("held %+v and n=%d, want lse_1 and 1", held, again.n)
	}
}
