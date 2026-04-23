// Browser-side persistence for the cockpit's live message tail.
//
// The bus does not yet expose a "recent messages with content" endpoint, so we
// snapshot the SSE-fed liveTail into localStorage and reload it on mount. This
// gives the user a populated thread on first paint instead of an empty view.
// Survives F5; resets when the user clears browser data.

import { api, type BusMessage } from './api';

const STORAGE_KEY = 'qo.cockpit.liveTail.v1';
const CAP = 100;

export function loadTail(): BusMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.slice(-CAP) as BusMessage[];
  } catch {
    return [];
  }
}

export function saveTail(tail: BusMessage[]): void {
  try {
    const trimmed = tail.slice(-CAP);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch {
    // quota or serialization issue — silently ignore
  }
}

export function clearTail(): void {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore
  }
}

// Server-first hydration helper: pulls the recent bus tail from the QO
// backend ring buffer (cap 200, populated by a background subscriber).
// Falls back to the localStorage snapshot if the network call fails so
// the cockpit still shows *something* on first paint when offline.
//
// NOTE: App.tsx still uses the synchronous loadTail() for its initial
// state — this helper is intended for an async post-mount refresh that
// a follow-up commit can wire up.
export async function loadTailFromServer(): Promise<BusMessage[]> {
  try {
    return await api.recentMessages(100);
  } catch {
    return loadTail();
  }
}
