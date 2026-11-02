// Shared by the streaming App and offline stream-to-render fixtures.
export function applyRoundtableEvent(roundtable, event) {
  if (event.type === 'roundtable_accounting') {
    if (!event.data || typeof event.data !== 'object' || Array.isArray(event.data)) {
      return { ...roundtable, call_accounting: {} };
    }
    return { ...roundtable, call_accounting: event.data };
  }
  // The backend sends the run's real outcome; older streams without it keep the legacy meaning.
  const outcome = event.data && typeof event.data === 'object' && !Array.isArray(event.data) ? event.data : {};
  if (event.type === 'roundtable_complete') {
    return {
      ...roundtable,
      status: outcome.status || 'completed',
      failure_reason: outcome.failure_reason || null,
      failed_seats: Array.isArray(outcome.failed_seats) ? outcome.failed_seats : [],
    };
  }
  if (event.type === 'roundtable_error') {
    return {
      ...roundtable,
      status: outcome.status || 'error',
      error: event.message,
      failure_reason: outcome.failure_reason || null,
      failed_seats: Array.isArray(outcome.failed_seats) ? outcome.failed_seats : [],
    };
  }
  return roundtable;
}
