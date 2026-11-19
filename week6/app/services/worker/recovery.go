package main

// settle finishes every lease the journal still holds: it reports what the
// journal knows about the attempt and closes the lease. It runs before new
// work is leased, so after a restart the worker first accounts for what it
// held when it stopped.
func (w *Worker) settle() (int, error) {
	did := 0
	for _, lease := range w.journal.Held() {
		if len(lease.Attempts) > 0 && !lease.Reported {
			results := make([]Result, 0, len(lease.Attempts))
			for _, a := range lease.Attempts {
				results = append(results, resultOf(a))
			}
			if err := w.queue.Report(lease.ID, results); err != nil {
				return did, err
			}
			w.crash(7)
			if err := w.journal.Append(Record{Rec: "lease.reported", At: w.now(), Lease: lease.ID}); err != nil {
				return did, err
			}
			w.log("lease.reported", map[string]any{"lease": lease.ID, "outcome": results[0].Outcome})
		}
		if err := w.queue.Close(lease.ID); err != nil {
			return did, err
		}
		w.crash(8)
		if err := w.journal.Append(Record{Rec: "lease.closed", At: w.now(), Lease: lease.ID}); err != nil {
			return did, err
		}
		w.log("lease.closed", map[string]any{"lease": lease.ID})
		w.crash(9)
		did++
	}
	return did, nil
}
