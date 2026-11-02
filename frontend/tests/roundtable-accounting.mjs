import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { applyRoundtableEvent } from '../src/utils/roundtableState.js';
import { createServer } from 'vite';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
try {
  const { default: Timeline } = await server.ssrLoadModule('/src/components/RoundtableTimeline.jsx');
  const html = renderToStaticMarkup(React.createElement(Timeline, { roundtable: {
    status: 'completed', rounds: [], call_accounting: {
      predicted_calls: 6, attempted_calls: 4, failed_calls: 1, max_calls_per_run: 14,
      quota_units: 'unknown'
    }
  }}));
  assert.match(html, /Predicted: 6/);
  assert.match(html, /Attempted: 4/);
  assert.match(html, /Failed: 1/);
  assert.match(html, /unknown/i);
  assert.match(html, /6/);
  assert.match(html, /4/);
  assert.doesNotMatch(html, /quota usage: 0/i);
  const render = (roundtable) => renderToStaticMarkup(React.createElement(Timeline, { roundtable }));
  for (const data of [null, [], 'invalid']) {
    for (const type of ['roundtable_complete', 'roundtable_error']) {
      let state = { status: 'running', rounds: [], call_accounting: { predicted_calls: 6, attempted_calls: 0, failed_calls: 0 } };
      state = applyRoundtableEvent(state, { type: 'roundtable_accounting', data });
      state = applyRoundtableEvent(state, { type, message: 'synthetic failure' });
      assert.match(render(state), /Attempted: unknown/);
      assert.match(render(state), /Failed: unknown/);
    }
  }
  const blocked = render({ status: 'error', error: 'Predicted call budget exceeded', rounds: [], call_accounting: {
    predicted_calls: 6, attempted_calls: 0, failed_calls: 0, quota_units: 0
  }});
  assert.match(blocked, /Attempted: 0/);
  assert.match(blocked, /Failed: 0/);
  assert.match(blocked, /role="alert">Predicted call budget exceeded/);
  assert.doesNotMatch(blocked, /quota usage: 0/i);
  const unknown = render({ status: 'aborted', rounds: [], call_accounting: { predicted_calls: -1, failed_calls: '1' } });
  for (const label of ['Predicted', 'Attempted', 'Failed']) assert.match(unknown, new RegExp(`${label}: unknown`));
  assert.doesNotMatch(render({ rounds: [], status: 'completed' }), /Run call counts/);
  assert.equal(render(null), '');
  let state = { status: 'running', rounds: [] };
  assert.deepEqual(applyRoundtableEvent(state, { type: 'roundtable_accounting', data: null }).call_accounting, {});
  assert.deepEqual(applyRoundtableEvent(state, { type: 'roundtable_accounting', data: [] }).call_accounting, {});
  const missing = applyRoundtableEvent(state, { type: 'roundtable_complete' });
  assert.doesNotMatch(render(missing), /Run call counts/);
  if (process.env.WOR402_STREAM_EVIDENCE_DIR) {
    for (const terminal of ['chair_complete', 'roundtable_aborted', 'roundtable_budget_exceeded']) {
      const events = JSON.parse(readFileSync(`${process.env.WOR402_STREAM_EVIDENCE_DIR}/${terminal}.json`, 'utf8'));
      state = events.reduce(applyRoundtableEvent, { status: 'running', rounds: [] });
      const streamed = render(state);
      assert.match(streamed, terminal === 'roundtable_budget_exceeded' ? /Attempted: 0/ : /Attempted: 4/);
      assert.match(streamed, terminal === 'roundtable_budget_exceeded' ? /Failed: 0/ : /Failed: 1/);
      assert.match(streamed, /quota usage: unknown/);
      // WOR-1016: terminal events now carry the run's real outcome.
      const expected = { chair_complete: 'completed', roundtable_aborted: 'aborted', roundtable_budget_exceeded: 'budget_exceeded' };
      assert.equal(state.status, expected[terminal]);
      if (terminal !== 'chair_complete') assert.match(streamed, /role="alert"/);
      const missingEvents = JSON.parse(readFileSync(`${process.env.WOR402_STREAM_EVIDENCE_DIR}/${terminal}-missing.json`, 'utf8'));
      const missingState = missingEvents.reduce(applyRoundtableEvent, { status: 'running', rounds: [] });
      assert.match(render(missingState), /Attempted: unknown/);
      assert.match(render(missingState), /Failed: unknown/);
    }
  }
  // WOR-1016 outcome presentation.
  const base = { status: 'running', rounds: [] };
  const legacy = applyRoundtableEvent(base, { type: 'roundtable_complete' });
  assert.equal(legacy.status, 'completed');
  assert.match(render(legacy), /status-badge complete/);
  const degraded = applyRoundtableEvent(base, {
    type: 'roundtable_complete',
    data: { status: 'degraded', failure_reason: 'seat failures: Skeptic', failed_seats: ['Skeptic'] },
  });
  assert.equal(degraded.status, 'degraded');
  assert.deepEqual(degraded.failed_seats, ['Skeptic']);
  assert.match(render(degraded), /status-badge degraded/);
  assert.match(render(degraded), /seat failures: Skeptic/);
  assert.doesNotMatch(render(degraded), /status-badge complete/);
  const failed = applyRoundtableEvent(base, {
    type: 'roundtable_error',
    message: 'Roundtable failed: no council seat succeeded in round 1',
    data: { status: 'failed', failure_reason: 'no council seat succeeded in round 1', failed_seats: ['Builder'] },
  });
  assert.equal(failed.status, 'failed');
  assert.match(render(failed), /status-badge failed/);
  assert.match(render(failed), /role="alert">Roundtable failed: no council seat succeeded in round 1/);
  assert.doesNotMatch(render(failed), /status-badge complete/);
  const budget = applyRoundtableEvent(base, { type: 'roundtable_error', message: 'Predicted call budget exceeded', data: { status: 'budget_exceeded' } });
  assert.match(render(budget), /Budget rejected/);
  const chairFailed = render({ ...failed, chair_final: { model: 'm', content: '', error: true, error_message: 'x' } });
  assert.match(chairFailed, /Chair failed: x/);
  const modFailed = render({ ...degraded, moderator: { model: 'm', content: '', error: true, error_message: 'mod timeout' } });
  assert.match(modFailed, /Moderator failed: mod timeout/);
  console.log('Synthetic timeline accounting and App event-state fixtures passed');
} finally { await server.close(); }
