// Shared by the streaming App and offline stream-to-render fixtures.
export function applyRoundtableEvent(roundtable, event) {
  if (event.type === 'roundtable_accounting') {
    if (!event.data || typeof event.data !== 'object' || Array.isArray(event.data)) return roundtable;
    return { ...roundtable, call_accounting: event.data };
  }
  if (event.type === 'roundtable_complete') return { ...roundtable, status: 'completed' };
  if (event.type === 'roundtable_error') return { ...roundtable, status: 'error', error: event.message };
  return roundtable;
}
