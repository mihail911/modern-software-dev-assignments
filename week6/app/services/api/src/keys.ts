// Idempotency keys (ADR-0013): for each merchant, the keys it has sent in the
// last KEY_WINDOW_SECONDS and the event each one was accepted as.

export const KEY_WINDOW_SECONDS = 24 * 3600;

interface Held {
  eventId: string;
  at: number;
}

export class RecentKeys {
  private held = new Map<string, Held>();

  private static slot(merchantId: string, key: string): string {
    return `${merchantId}\u0000${key}`;
  }

  // The event accepted under this merchant's key, if it is still in the window.
  find(merchantId: string, key: string, now: number): string | undefined {
    const slot = RecentKeys.slot(merchantId, key);
    const held = this.held.get(slot);
    if (!held) return undefined;
    if (now - held.at >= KEY_WINDOW_SECONDS) {
      this.held.delete(slot);
      return undefined;
    }
    return held.eventId;
  }

  remember(merchantId: string, key: string, eventId: string, now: number): void {
    this.held.set(RecentKeys.slot(merchantId, key), { eventId, at: now });
  }

  get size(): number {
    return this.held.size;
  }
}
