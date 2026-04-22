// Auto-reconnecting SSE subscription. Mirrors the IDE inbox pattern.
// EventSource is patched in main.tsx to inject auth token.

export interface SseHandle {
  close(): void;
}

export function subscribeSSE<T = unknown>(
  url: string,
  onMessage: (data: T) => void,
  options: { onError?: (err: Event) => void; reconnectMs?: number } = {},
): SseHandle {
  const reconnect = options.reconnectMs ?? 5000;
  let stopped = false;
  let es: EventSource | null = null;

  const open = () => {
    if (stopped) return;
    es = new EventSource(url);
    es.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(ev.data) as T;
        onMessage(parsed);
      } catch {
        // ignore unparseable lines
      }
    };
    es.onerror = (err) => {
      options.onError?.(err);
      es?.close();
      es = null;
      if (!stopped) setTimeout(open, reconnect);
    };
  };

  open();

  return {
    close() {
      stopped = true;
      es?.close();
      es = null;
    },
  };
}

// Throttle helper for triggering visual pulses without spamming setState.
export function throttle<F extends (...args: unknown[]) => void>(fn: F, ms: number): F {
  let last = 0;
  return ((...args: Parameters<F>) => {
    const now = Date.now();
    if (now - last >= ms) {
      last = now;
      fn(...args);
    }
  }) as F;
}
