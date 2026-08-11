/* ==========================================================================
   RASID — Federated Intrusion Detection Console

   Talks only to the documented GUI adapter (uavids-gui-api-v1). It never loads
   checkpoints, preprocessing objects, full dataset rows, or cryptographic material, and
   it never computes a verdict, a probability, or a metric in the browser.

   Honesty rules enforced here:
     - Every dashboard verdict comes from a locked-test demo endpoint and the frozen model.
     - Dataset attack families are ground truth only; the model remains binary.
     - Backend counters (adapter process lifetime) and session counters (this tab)
       are displayed separately and never summed.
     - presentation_mode / inference_mode / replayed stay visible and distinct.
   ========================================================================== */

'use strict';

/* ── Config ─────────────────────────────────────────────────────────── */

const params = new URLSearchParams(location.search);
const API_BASE =
  params.get('api') ||
  window.GUI_API_BASE ||
  'http://127.0.0.1:8090/api/gui/v1';

const SNAPSHOT_POLL_MS = 1000;
const EVENT_POLL_MS = 1200;
const RECORDED_RELEASE_MS = 650;   // pacing for the all-at-once recorded batch
const TIMELINE_MAX = 240;
const TIMELINE_WINDOW_MS = 5 * 60 * 1000;
const REQUEST_TIMEOUT_MS = 4000;

// Rate histogram: 40 buckets of 3 s = a rolling two-minute window. At the
// traffic replay's default 1.5 s interval a bucket holds ~2 verdicts, so bars have
// visible height within seconds of a demo starting, and 40 of them across a
// full-width panel gives the dense spiky profile of the Grafana reference.
const VOLUME_BUCKET_MS = 3000;
const VOLUME_BUCKETS = 40;
const VOLUME_MAX_POINTS = 4000;

/* ── API client ─────────────────────────────────────────────────────── */

class ApiError extends Error {
  constructor(status, code, message) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  let response;
  try {
    response = await fetch(API_BASE + path, {
      ...options,
      signal: controller.signal,
      headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
    });
  } catch (cause) {
    throw new ApiError(0, 'network_unreachable', 'Cannot reach the GUI adapter');
  } finally {
    clearTimeout(timer);
  }

  const text = await response.text();
  let body = {};
  if (text) {
    try { body = JSON.parse(text); }
    catch { throw new ApiError(response.status, 'invalid_json', 'Adapter returned malformed JSON'); }
  }
  if (!response.ok) {
    const err = body && body.error ? body.error : {};
    throw new ApiError(response.status, err.code || 'unknown_error', err.message || ('HTTP ' + response.status));
  }
  return body;
}

const api = {
  health:          ()   => request('/health'),
  snapshot:        ()   => request('/snapshot'),
  events:          (n)  => request('/events?after_seq=' + n),
  federatedEvents: (n)  => request('/federated/events?after_seq=' + n),
  demoCatalog:     ()   => request('/demo/catalog'),
  demoNextNormal:  ()   => request('/demo/traffic/next', { method: 'POST', body: '{}' }),
  demoReset:       ()   => request('/demo/reset', { method: 'POST', body: '{}' }),
};

/* ── State ──────────────────────────────────────────────────────────── */

const state = {
  snapshot: null,
  demoCatalog: null,
  error: null,
  connected: false,

  // session (this tab only)
  history: [],            // recent predictions for the timeline
  volume: [],             // { t: epoch ms, attack: bool } for the rate histogram
  seenPredictions: new Set(),
  // locked-test traffic replay
  intervalMs: 1500,
  busy: false,
  trafficRunning: false,
  trafficTimer: null,

  // event feeds
  inferenceCursor: 0,
  federatedCursor: 0,
  inferenceLog: [],
  federatedLog: [],
  pendingFederated: [],   // buffered recorded batch awaiting paced release
  federatedPaced: false,

  alertsExpanded: false,
  severityFilter: 'all',
  view: 'dashboard',
};

/* ── Small helpers ──────────────────────────────────────────────────── */

const $ = (id) => document.getElementById(id);
const num = (v) => (typeof v === 'number' && isFinite(v) ? v.toLocaleString('en-US') : '—');
const pct = (v, d = 1) => (typeof v === 'number' && isFinite(v) ? (v * 100).toFixed(d) + '%' : '—');
const fixed = (v, d = 4) => (typeof v === 'number' && isFinite(v) ? v.toFixed(d) : '—');

function clockOf(iso) {
  if (!iso) return '—';
  const dt = new Date(iso);
  return isNaN(dt) ? '—' : dt.toLocaleTimeString('en-GB', { hour12: false });
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function svgEl(tag, attrs = {}) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/* ── Tooltip layer ──────────────────────────────────────────────────── */

let tipNode = null;

function showTip(html, event) {
  if (!tipNode) {
    tipNode = el('div', 'tip');
    document.body.appendChild(tipNode);
  }
  tipNode.innerHTML = html;
  tipNode.style.display = 'block';
  const pad = 14;
  const rect = tipNode.getBoundingClientRect();
  let x = event.clientX + pad;
  let y = event.clientY + pad;
  if (x + rect.width > window.innerWidth - 8) x = event.clientX - rect.width - pad;
  if (y + rect.height > window.innerHeight - 8) y = event.clientY - rect.height - pad;
  tipNode.style.left = Math.max(8, x) + 'px';
  tipNode.style.top = Math.max(8, y) + 'px';
}

function hideTip() {
  if (tipNode) tipNode.style.display = 'none';
}

function attachTip(node, htmlFn) {
  node.addEventListener('mousemove', (e) => showTip(htmlFn(), e));
  node.addEventListener('mouseleave', hideTip);
}

/* ── Charts ─────────────────────────────────────────────────────────────
   Data marks use a diverging blue <-> red pair split at the decision threshold.
   Position against the threshold rule is the primary channel; color reinforces
   it, so the encoding survives color-vision deficiency.
   ------------------------------------------------------------------------ */

function renderThresholdScale(container, probability, threshold) {
  container.textContent = '';
  const W = 480, H = 74, padX = 10;
  const trackY = 34, trackH = 12;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  const x = (v) => padX + v * (W - padX * 2);

  svg.appendChild(svgEl('rect', {
    x: padX, y: trackY, width: W - padX * 2, height: trackH,
    fill: cssVar('--line-soft'),
  }));

  if (typeof probability === 'number' && isFinite(probability)) {
    const attack = probability >= threshold;
    svg.appendChild(svgEl('rect', {
      x: padX, y: trackY, width: Math.max(4, x(probability) - padX), height: trackH,
      fill: attack ? cssVar('--mark-attack') : cssVar('--mark-normal'),
    }));
  }

  // Threshold rule (a 2px surface gap keeps it separate from the fill)
  const tx = x(threshold);
  svg.appendChild(svgEl('line', {
    x1: tx, y1: trackY - 9, x2: tx, y2: trackY + trackH + 9,
    stroke: cssVar('--surface'), 'stroke-width': 5, 'stroke-linecap': 'round',
  }));
  svg.appendChild(svgEl('line', {
    x1: tx, y1: trackY - 8, x2: tx, y2: trackY + trackH + 8,
    stroke: cssVar('--mark-rule'), 'stroke-width': 2, 'stroke-dasharray': '4 3', 'stroke-linecap': 'round',
  }));

  const label = svgEl('text', {
    x: tx, y: trackY - 14, 'text-anchor': 'middle',
    fill: cssVar('--ink-muted'), 'font-size': 11, 'font-weight': 600,
    'font-family': 'system-ui, sans-serif',
  });
  label.textContent = 'threshold ' + threshold;
  svg.appendChild(label);

  for (const t of [0, 0.5, 1]) {
    const tick = svgEl('text', {
      x: x(t), y: trackY + trackH + 20,
      'text-anchor': t === 0 ? 'start' : t === 1 ? 'end' : 'middle',
      fill: cssVar('--ink-muted'), 'font-size': 10.5, 'font-family': 'system-ui, sans-serif',
    });
    tick.textContent = t.toFixed(1);
    svg.appendChild(tick);
  }

  container.appendChild(svg);
}

function renderTimeline(container, history, threshold) {
  container.textContent = '';
  if (!history.length) {
    container.appendChild(el('div', 'chart-empty',
      'No test flows scored yet. Press Start Traffic to begin the locked-test replay.'));
    return;
  }

  // Wide viewBox for the full-width presentation panel.
  const W = 1100, H = 300;
  const padL = 34, padR = 12, padT = 12, padB = 26;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  svg.setAttribute('aria-label',
    `Attack probability for the last ${history.length} verdicts against the ${threshold} decision threshold`);

  const y = (v) => padT + (1 - v) * plotH;
  const slot = plotW / Math.max(1, history.length);
  const barW = Math.max(3, Math.min(14, slot - 2));   // 2px surface gap between bars

  // Recessive gridlines
  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    svg.appendChild(svgEl('line', {
      x1: padL, y1: y(g), x2: W - padR, y2: y(g),
      stroke: cssVar('--grid'), 'stroke-width': 1,
    }));
    const t = svgEl('text', {
      x: padL - 8, y: y(g) + 3.5, 'text-anchor': 'end',
      fill: cssVar('--ink-muted'), 'font-size': 10.5, 'font-family': 'system-ui, sans-serif',
    });
    t.textContent = g.toFixed(2);
    svg.appendChild(t);
  }

  history.forEach((p, i) => {
    const attack = p.label === 'Attack';
    const cx = padL + i * slot + (slot - barW) / 2;
    const top = y(p.attack_probability);
    const h = Math.max(2, y(0) - top);
    const bar = svgEl('rect', {
      x: cx, y: top, width: barW, height: h,
      fill: attack ? cssVar('--mark-attack') : cssVar('--mark-normal'),
    });
    bar.style.cursor = 'crosshair';
    attachTip(bar, () =>
      `<strong>${p.label}</strong> · p = <strong>${p.attack_probability.toFixed(4)}</strong>` +
      `<br><span class="tip-k">record</span> ${p.record_id}` +
      `<br><span class="tip-k">source</span> ${p.source || '—'}` +
      `<br><span class="tip-k">time</span> ${clockOf(p.timestamp_utc)}` +
      (p.dataset ? `<br><span class="tip-k">ground truth</span> ${p.dataset.ground_truth_label}` : '') +
      (p.demo && p.demo.target_client_id
        ? `<br><span class="tip-k">controlled target</span> ${p.demo.target_client_id}` : '') +
      (p.demo ? `<br><span class="tip-k">outcome</span> ${p.demo.outcome.replace('_', ' ')}` : ''));
    svg.appendChild(bar);

    if (p.demo && p.demo.kind === 'controlled_attack') {
      const markerX = cx + barW / 2;
      svg.appendChild(svgEl('line', {
        x1: markerX, y1: padT, x2: markerX, y2: y(0),
        stroke: cssVar('--mark-attack'), 'stroke-width': 2, 'stroke-dasharray': '3 3',
      }));
      const marker = svgEl('path', {
        d: `M ${markerX - 6} ${padT + 1} L ${markerX + 6} ${padT + 1} L ${markerX} ${padT + 11} Z`,
        fill: cssVar('--mark-attack'),
      });
      attachTip(marker, () =>
        `<strong>Controlled ${p.dataset.ground_truth_label}</strong>` +
        `<br><span class="tip-k">target</span> ${p.demo.target_client_id}` +
        `<br><span class="tip-k">model verdict</span> ${p.label}` +
        `<br><span class="tip-k">outcome</span> ${p.demo.outcome.replace('_', ' ')}`);
      svg.appendChild(marker);
    }
  });

  // Threshold rule on top, with a surface halo so it stays readable over bars
  svg.appendChild(svgEl('line', {
    x1: padL, y1: y(threshold), x2: W - padR, y2: y(threshold),
    stroke: cssVar('--surface'), 'stroke-width': 4,
  }));
  svg.appendChild(svgEl('line', {
    x1: padL, y1: y(threshold), x2: W - padR, y2: y(threshold),
    stroke: cssVar('--mark-rule'), 'stroke-width': 2, 'stroke-dasharray': '5 4',
  }));

  // Direct-label the most recent verdict only (never every point)
  const last = history[history.length - 1];
  const lastX = padL + (history.length - 1) * slot + slot / 2;
  const lastY = y(last.attack_probability);
  const tag = svgEl('text', {
    x: Math.min(lastX, W - padR - 4), y: Math.max(padT + 10, lastY - 8),
    'text-anchor': 'end', 'font-size': 11.5, 'font-weight': 700,
    'font-family': 'system-ui, sans-serif',
    fill: last.label === 'Attack' ? cssVar('--mark-attack') : cssVar('--mark-normal'),
  });
  tag.textContent = last.attack_probability.toFixed(2);
  svg.appendChild(tag);

  const axis = svgEl('line', {
    x1: padL, y1: y(0), x2: W - padR, y2: y(0),
    stroke: cssVar('--line'), 'stroke-width': 1,
  });
  svg.appendChild(axis);

  const older = svgEl('text', {
    x: padL, y: H - 6, fill: cssVar('--ink-muted'), 'font-size': 10.5,
    'font-family': 'system-ui, sans-serif',
  });
  older.textContent = clockOf(history[0].timestamp_utc);
  svg.appendChild(older);

  const newer = svgEl('text', {
    x: W - padR, y: H - 6, 'text-anchor': 'end',
    fill: cssVar('--ink-muted'), 'font-size': 10.5, 'font-family': 'system-ui, sans-serif',
  });
  newer.textContent = clockOf(history[history.length - 1].timestamp_utc);
  svg.appendChild(newer);

  container.appendChild(svg);
}

/* Rolling rate histogram. The detection timeline plots one bar per verdict, so
   it says nothing about arrival rate — a burst and a trickle look identical
   there. This buckets by arrival time instead, stacking Normal under Attack so
   total column height reads as total traffic. */
function bucketVolume(now = Date.now()) {
  const end = Math.floor(now / VOLUME_BUCKET_MS) * VOLUME_BUCKET_MS + VOLUME_BUCKET_MS;
  const start = end - VOLUME_BUCKETS * VOLUME_BUCKET_MS;
  const buckets = Array.from({ length: VOLUME_BUCKETS }, (_, i) => ({
    start: start + i * VOLUME_BUCKET_MS, normal: 0, attack: 0,
  }));
  for (const point of state.volume) {
    if (point.t < start || point.t >= end) continue;
    const slot = Math.floor((point.t - start) / VOLUME_BUCKET_MS);
    if (slot < 0 || slot >= VOLUME_BUCKETS) continue;
    buckets[slot][point.attack ? 'attack' : 'normal'] += 1;
  }
  return { buckets, start, end };
}

function renderVolumeChart(container, buckets) {
  container.textContent = '';
  const peak = Math.max(1, ...buckets.map((b) => b.normal + b.attack));

  // Wide viewBox: this panel spans the full dashboard, and the SVG scales to
  // width with height:auto — a squarer box would render it ~600px tall.
  const W = 1100, H = 210;
  const padL = 34, padR = 10, padT = 10, padB = 26;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  svg.setAttribute('aria-label',
    `Verdicts per ${VOLUME_BUCKET_MS / 1000} seconds over the last ` +
    `${(VOLUME_BUCKETS * VOLUME_BUCKET_MS) / 60000} minutes, Normal and Attack stacked`);

  const y = (v) => padT + (1 - v / peak) * plotH;
  const slot = plotW / VOLUME_BUCKETS;
  const barW = slot;   // contiguous: adjacent buckets are adjacent time, not separate events

  // Gridlines at whole counts only — fractional verdicts do not exist.
  const step = Math.max(1, Math.ceil(peak / 4));
  for (let g = 0; g <= peak; g += step) {
    svg.appendChild(svgEl('line', {
      x1: padL, y1: y(g), x2: W - padR, y2: y(g),
      stroke: cssVar('--grid'), 'stroke-width': 1,
    }));
    const label = svgEl('text', {
      x: padL - 7, y: y(g) + 3.5, 'text-anchor': 'end',
      fill: cssVar('--ink-muted'), 'font-size': 10, 'font-family': 'system-ui, sans-serif',
    });
    label.textContent = String(g);
    svg.appendChild(label);
  }

  buckets.forEach((bucket, i) => {
    const x = padL + i * slot + (slot - barW) / 2;
    const clock = clockOf(new Date(bucket.start).toISOString());
    let cursor = y(0);
    // Normal first (bottom), Attack stacked above it.
    for (const [key, colour] of [['normal', '--mark-normal'], ['attack', '--mark-attack']]) {
      const count = bucket[key];
      if (!count) continue;
      const h = (count / peak) * plotH;
      const rect = svgEl('rect', {
        x, y: cursor - h, width: barW, height: h, fill: cssVar(colour),
      });
      rect.style.cursor = 'crosshair';
      attachTip(rect, () =>
        `<strong>${clock}</strong> · ${VOLUME_BUCKET_MS / 1000}s bucket` +
        `<br><span class="tip-k">normal</span> ${bucket.normal}` +
        `<br><span class="tip-k">attack</span> ${bucket.attack}` +
        `<br><span class="tip-k">total</span> ${bucket.normal + bucket.attack}`);
      svg.appendChild(rect);
      cursor -= h;
    }
  });

  svg.appendChild(svgEl('line', {
    x1: padL, y1: y(0), x2: W - padR, y2: y(0),
    stroke: cssVar('--line'), 'stroke-width': 1,
  }));

  // Time axis: only the ends, so the labels never collide.
  for (const [bucket, anchor, x] of [
    [buckets[0], 'start', padL],
    [buckets[buckets.length - 1], 'end', W - padR],
  ]) {
    const t = svgEl('text', {
      x, y: H - 6, 'text-anchor': anchor === 'start' ? 'start' : 'end',
      fill: cssVar('--ink-muted'), 'font-size': 10, 'font-family': 'system-ui, sans-serif',
    });
    t.textContent = clockOf(new Date(bucket.start).toISOString());
    svg.appendChild(t);
  }

  container.appendChild(svg);
}

/* Grafana-style "current | total" block beside the series. `current` is the last
   complete bucket, not the one still filling — a partial bucket always reads low
   and would look like traffic dropping off. */
function renderVolumeStats(container, buckets) {
  container.textContent = '';
  const settled = buckets[buckets.length - 2] || { normal: 0, attack: 0 };
  const totals = buckets.reduce(
    (acc, b) => ({ normal: acc.normal + b.normal, attack: acc.attack + b.attack }),
    { normal: 0, attack: 0 });

  const table = el('table', 'series-table');
  const group = el('colgroup');
  for (const cls of ['col-name', 'col-num', 'col-num']) group.appendChild(el('col', cls));
  table.appendChild(group);

  const head = el('tr');
  head.appendChild(el('th', null, ''));
  head.appendChild(el('th', 'series-num', 'current'));
  head.appendChild(el('th', 'series-num', 'total'));
  table.appendChild(head);

  for (const [key, label, swatch] of [
    ['normal', 'Normal', 'swatch-normal'],
    ['attack', 'Attack', 'swatch-attack'],
  ]) {
    const row = el('tr');
    const name = el('td', 'series-name');
    name.appendChild(el('i', 'swatch ' + swatch));
    name.appendChild(document.createTextNode(label));
    row.appendChild(name);
    row.appendChild(el('td', 'series-num mono', String(settled[key])));
    row.appendChild(el('td', 'series-num mono', String(totals[key])));
    table.appendChild(row);
  }

  const sum = el('tr', 'series-total');
  sum.appendChild(el('td', 'series-name', 'All'));
  sum.appendChild(el('td', 'series-num mono', String(settled.normal + settled.attack)));
  sum.appendChild(el('td', 'series-num mono', String(totals.normal + totals.attack)));
  table.appendChild(sum);

  container.appendChild(table);
  container.appendChild(el('p', 'series-note',
    `current = verdicts in the last complete ${VOLUME_BUCKET_MS / 1000}s bucket`));
}

function renderRejectionChart(container, rejections) {
  container.textContent = '';
  const entries = Object.entries(rejections).sort((a, b) => b[1] - a[1]);

  if (!entries.length) {
    container.appendChild(el('div', 'chart-empty',
      'No rejected federated messages reported yet.'));
    return;
  }

  // Magnitude comparison: bar length carries the value, so a single hue is
  // correct here — a categorical ramp would imply identity that is not there.
  const rowH = 30, gap = 6, labelW = 168, valW = 34;
  const W = 560;
  const H = entries.length * (rowH + gap);
  const max = Math.max(...entries.map((e) => e[1]));
  const barMax = W - labelW - valW - 12;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  svg.setAttribute('aria-label', 'Rejected federated messages by category');

  entries.forEach(([category, count], i) => {
    const yTop = i * (rowH + gap);
    const w = Math.max(3, (count / max) * barMax);

    const label = svgEl('text', {
      x: labelW - 12, y: yTop + rowH / 2 + 4, 'text-anchor': 'end',
      fill: cssVar('--ink-2'), 'font-size': 12.5, 'font-family': 'ui-monospace, monospace',
    });
    label.textContent = category.replace(/_/g, ' ');
    svg.appendChild(label);

    const bar = svgEl('rect', {
      x: labelW, y: yTop + 5, width: w, height: rowH - 10,
      fill: cssVar('--mark-normal'),
    });
    bar.style.cursor = 'crosshair';
    attachTip(bar, () =>
      `<strong>${category}</strong><br><span class="tip-k">rejected messages</span> <strong>${count}</strong>`);
    svg.appendChild(bar);

    const value = svgEl('text', {
      x: labelW + w + 9, y: yTop + rowH / 2 + 4,
      fill: cssVar('--ink'), 'font-size': 13, 'font-weight': 700,
      'font-family': 'ui-monospace, monospace',
    });
    value.textContent = String(count);
    svg.appendChild(value);
  });

  container.appendChild(svg);
}

/* ── Prediction recording ───────────────────────────────────────────── */

function recordPrediction(prediction, options = {}) {
  if (!prediction || !prediction.prediction_id) return;
  // This dashboard now visualises only traceable locked-test demonstration rows.
  if (!prediction.dataset || !prediction.demo) return;
  const activeSession = state.snapshot && state.snapshot.dataset_demo
    ? state.snapshot.dataset_demo.session_id : null;
  if (activeSession && prediction.demo.session_id !== activeSession) return;
  if (state.seenPredictions.has(prediction.prediction_id)) return;
  state.seenPredictions.add(prediction.prediction_id);

  state.history.push(prediction);
  state.history.sort((left, right) => {
    const a = Date.parse(left.timestamp_utc);
    const b = Date.parse(right.timestamp_utc);
    return (isNaN(a) ? 0 : a) - (isNaN(b) ? 0 : b);
  });
  const cutoff = Date.now() - TIMELINE_WINDOW_MS;
  state.history = state.history.filter((item) => {
    const stamped = Date.parse(item.timestamp_utc);
    return isNaN(stamped) || stamped >= cutoff;
  });
  if (state.history.length > TIMELINE_MAX) {
    state.history.splice(0, state.history.length - TIMELINE_MAX);
  }

  // Arrival time drives the rate histogram. Fall back to now when the adapter
  // sends no timestamp, so a verdict is never silently dropped from the counts.
  const stamped = Date.parse(prediction.timestamp_utc);
  state.volume.push({
    t: isNaN(stamped) ? Date.now() : stamped,
    attack: prediction.label === 'Attack',
  });
  if (state.volume.length > VOLUME_MAX_POINTS) {
    state.volume.splice(0, state.volume.length - VOLUME_MAX_POINTS);
  }

  if (options.fromThisTab) state.sessionRecords += 1;
}

/* ── Rendering: sidebar status ──────────────────────────────────────────
   The inference_mode and presentation_mode indicators live here rather than in
   a top bar. GUI_MINIMUM_REQUIREMENTS.md requires them to stay visible and
   distinct from each other, which is why they are two separate rows.
   ------------------------------------------------------------------------ */

function renderStatus() {
  const snap = state.snapshot;
  const linkState = $('link-state');
  const linkLabel = linkState.querySelector('.link-label');

  if (state.connected && snap) {
    linkState.className = 'link-state is-up';
    linkLabel.textContent = 'Adapter connected';
  } else {
    linkState.className = 'link-state is-down';
    linkLabel.textContent = 'Adapter unreachable';
  }

  const inference = $('s-inference');
  const mode = $('s-mode');

  if (!snap) {
    inference.innerHTML = '<span class="pill pill-down">unavailable</span>';
    mode.innerHTML = '<span class="pill pill-down">unavailable</span>';
    return;
  }

  inference.innerHTML = snap.backend.inference_mode === 'live_model'
    ? '<span class="pill pill-live">live model</span>'
    : `<span class="pill pill-replay">${snap.backend.inference_mode}</span>`;

  mode.innerHTML = snap.presentation_mode === 'live'
    ? '<span class="pill pill-live">live federated</span>'
    : '<span class="pill pill-replay">recorded evidence</span>';
}

/* ── Rendering: dashboard ───────────────────────────────────────────── */

function renderVerdict() {
  const snap = state.snapshot;
  const threshold = snap ? snap.model.decision_threshold : 0.42;
  const prediction = state.history.length
    ? state.history[state.history.length - 1]
    : (snap && snap.inference.latest_prediction) || null;

  const panel = $('verdict-panel');
  const label = $('verdict-label');

  if (!prediction) {
    panel.className = 'panel panel-verdict';
    label.className = 'verdict-label';
    label.textContent = 'IDLE';
    $('verdict-prob').textContent = '—';
    $('verdict-origin').textContent = 'awaiting input';
    $('verdict-record').textContent = '—';
    $('verdict-source').textContent = '—';
    $('verdict-conf').textContent = '—';
    $('verdict-imputed').textContent = '—';
    renderThresholdScale($('verdict-scale'), null, threshold);
    return;
  }

  const attack = prediction.label === 'Attack';
  panel.className = 'panel panel-verdict ' + (attack ? 'is-attack' : 'is-normal');
  label.className = 'verdict-label ' + (attack ? 'is-attack' : 'is-normal');
  label.textContent = prediction.label.toUpperCase();

  $('verdict-prob').textContent = prediction.attack_probability.toFixed(3);
  $('verdict-origin').textContent = prediction.replayed
    ? 'recorded fixture input · scored live'
    : 'injected input · scored live';
  $('verdict-record').textContent = prediction.record_id || '—';
  $('verdict-source').textContent = prediction.source || 'not supplied';
  $('verdict-conf').textContent = pct(prediction.confidence, 2);
  $('verdict-imputed').textContent =
    prediction.missing_features_imputed + ' of ' + (snap ? snap.model.feature_count : 15);

  renderThresholdScale($('verdict-scale'), prediction.attack_probability, prediction.decision_threshold);
}

function renderKpis() {
  const snap = state.snapshot;
  if (!snap) return;
  const inf = snap.dataset_demo || snap.inference;
  const total = inf.records_processed || 0;

  $('k-records').textContent = num(total);
  $('k-attacks').textContent = num(inf.attack_count);
  $('k-normal').textContent = num(inf.normal_count);
  // One send is one flow record, so this counts sends. sessionPackets is the
  // number of packets those records *describe* — a flood record summarises
  // thousands of them, so showing it here made one click look like thousands.
  const remaining = inf.remaining ? inf.remaining['Normal Traffic'] : null;
  $('k-packets').textContent = num(remaining);
  $('k-packets-sub').textContent = 'locked-test demonstration pool';
  if ($('traffic-remaining')) $('traffic-remaining').textContent = num(remaining);

  $('k-attack-rate').textContent = total
    ? pct(inf.attack_count / total) + ' of records' : 'no records yet';
  $('k-normal-rate').textContent = total
    ? pct(inf.normal_count / total) + ' of records' : 'no records yet';
}

/* Bands for the Attack confidence column. This is the model's certainty, not a
   threat rating — a 0.99 says the model is sure it is an attack, not that the
   attack is severe. The band names avoid severity words for that reason. */
const CONFIDENCE_BANDS = [
  { min: 0.90, label: 'Very high', cls: 'conf-very-high' },
  { min: 0.70, label: 'High',      cls: 'conf-high' },
  { min: 0.00, label: 'Marginal',  cls: 'conf-marginal' },
];

function confidenceBand(probability) {
  return CONFIDENCE_BANDS.find((band) => probability >= band.min) ||
    CONFIDENCE_BANDS[CONFIDENCE_BANDS.length - 1];
}

const ALERT_COLUMNS = [
  'Time / Flow', 'Controlled target', 'Dataset label', 'Recorded route',
  'Packets Tx / Rx / lost', 'Throughput', 'Drop rate', 'P(Attack)',
];
const ALERT_DEFAULT_LIMIT = 10;

function availableAlerts() {
  return (state.snapshot && state.snapshot.dataset_demo && state.snapshot.dataset_demo.recent_alerts) || [];
}

function csvCell(value) {
  return '"' + String(value == null ? '' : value).replace(/"/g, '""') + '"';
}

function exportAlerts() {
  const alerts = availableAlerts();
  if (!alerts.length) return;
  const anchor = document.createElement('a');
  anchor.href = API_BASE + '/demo/alerts.csv';
  anchor.download = 'uavids-detected-alerts.csv';
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
}

function renderAlerts() {
  const table = $('alert-table');
  table.textContent = '';
  const alerts = availableAlerts();
  const shown = state.alertsExpanded ? alerts : alerts.slice(0, ALERT_DEFAULT_LIMIT);
  const summary = $('alert-summary');
  const toggle = $('btn-alert-history');
  const exportButton = $('btn-alert-export');

  summary.textContent = alerts.length > ALERT_DEFAULT_LIMIT && !state.alertsExpanded
    ? `latest ${ALERT_DEFAULT_LIMIT} of ${alerts.length}`
    : `${alerts.length} alert${alerts.length === 1 ? '' : 's'}`;
  toggle.hidden = alerts.length <= ALERT_DEFAULT_LIMIT;
  toggle.textContent = state.alertsExpanded ? `Show latest ${ALERT_DEFAULT_LIMIT}` : 'Show full history';
  toggle.setAttribute('aria-expanded', String(state.alertsExpanded));
  exportButton.disabled = alerts.length === 0;

  const head = el('tr');
  for (const column of ALERT_COLUMNS) head.appendChild(el('th', null, column));
  table.appendChild(head);

  if (!alerts.length) {
    const row = el('tr');
    const cell = el('td', 'empty-cell', 'No attack alerts recorded by the adapter yet.');
    cell.setAttribute('colspan', String(ALERT_COLUMNS.length));
    row.appendChild(cell);
    table.appendChild(row);
    return;
  }

  const threshold = (state.snapshot && state.snapshot.model.decision_threshold) || 0.42;

  for (const alert of shown) {
    const probability = alert.attack_probability;
    const dataset = alert.dataset || {};
    const demo = alert.demo || {};
    const row = el('tr');

    const time = el('td');
    time.appendChild(el('span', 'mono', clockOf(alert.timestamp_utc)));
    time.appendChild(el('span', 'cell-sub mono', 'Flow ' + (dataset.flow_id ?? '—')));
    row.appendChild(time);
    row.appendChild(el('td', 'mono', demo.target_client_id || 'baseline'));
    const truth = el('td');
    truth.appendChild(el('span', 'row-tag ' +
      (dataset.ground_truth_label === 'Normal Traffic' ? '' : 'row-tag-warn'),
    dataset.ground_truth_label || '—'));
    if (demo.outcome === 'false_alarm') truth.appendChild(el('span', 'cell-sub', 'false alarm'));
    row.appendChild(truth);
    row.appendChild(el('td', 'mono route-cell',
      `${dataset.recorded_source || '—'} → ${dataset.recorded_destination || '—'}`));
    row.appendChild(el('td', 'mono',
      `${num(dataset.tx_packets)} / ${num(dataset.rx_packets)} / ${num(dataset.lost_packets)}`));
    row.appendChild(el('td', 'mono',
      typeof dataset.throughput_kbps === 'number' ? dataset.throughput_kbps.toFixed(3) + ' Kbps' : '—'));
    row.appendChild(el('td', 'mono', pct(dataset.packet_drop_rate, 1)));
    const probabilityCell = el('td');
    probabilityCell.appendChild(el('span', 'conf-chip conf-very-high', pct(probability, 1)));
    probabilityCell.appendChild(el('span', 'cell-sub mono', `threshold ${threshold}`));
    row.appendChild(probabilityCell);

    table.appendChild(row);
  }
}

function renderVolume() {
  const { buckets } = bucketVolume();
  renderVolumeChart($('volume-chart'), buckets);
  renderVolumeStats($('volume-stats'), buckets);
  $('volume-window').textContent =
    `last ${(VOLUME_BUCKETS * VOLUME_BUCKET_MS) / 60000} min · ` +
    `${VOLUME_BUCKET_MS / 1000}s buckets`;
}

function renderDashboard() {
  renderKpis();
  renderTimeline($('timeline-chart'), state.history,
    state.snapshot ? state.snapshot.model.decision_threshold : 0.42);
  renderAlerts();
  renderConsole();
}

/* ── Rendering: AI data ─────────────────────────────────────────────── */

function renderModelIdentity() {
  const container = $('model-identity');
  container.textContent = '';
  const snap = state.snapshot;
  if (!snap) return;

  const technical = snap.evidence && snap.evidence.technical;
  const architecture = technical && Array.isArray(technical.architecture)
    ? technical.architecture.join('–') : '—';

  const rows = [
    ['Model ID', snap.model.model_id],
    ['Version', snap.model.model_version],
    ['Architecture', architecture],
    ['Binary labels', snap.model.labels.join(' / ')],
    ['Frozen threshold', String(snap.model.decision_threshold)],
    ['Checkpoint SHA-256', technical ? technical.checkpoint_sha256 : '—'],
  ];

  for (const [key, value] of rows) {
    container.appendChild(el('dt', null, key));
    container.appendChild(el('dd', null, value == null ? '—' : String(value)));
  }
}

function renderLockedTestMetrics(locked) {
  const container = $('locked-test-metrics');
  container.textContent = '';
  const metrics = locked.federated_fedavg;
  const rows = [
    ['Macro F1', metrics.macro_f1],
    ['Attack precision', metrics.attack_precision],
    ['Attack recall', metrics.attack_recall],
    ['False positive rate', metrics.fpr],
  ];

  for (const [key, value] of rows) {
    const tile = el('div', 'metric');
    tile.appendChild(el('div', 'metric-key', key));
    tile.appendChild(el('div', 'metric-val', pct(value, 2)));
    container.appendChild(tile);
  }
  $('locked-test-rows').textContent = num(locked.rows) + ' locked-test flows';
}

function renderConfusionMatrix(locked) {
  const container = $('confusion-matrix');
  container.textContent = '';
  const source = locked.federated_fedavg;

  container.appendChild(el('div', 'cm-h', ''));
  container.appendChild(el('div', 'cm-h', 'predicted attack'));
  container.appendChild(el('div', 'cm-h', 'predicted normal'));

  const cells = [
    ['actual attack', [['tp', source.tp, 'is-hit'], ['fn', source.fn, 'is-miss']]],
    ['actual normal', [['fp', source.fp, 'is-miss'], ['tn', source.tn, 'is-hit']]],
  ];

  for (const [rowLabel, entries] of cells) {
    container.appendChild(el('div', 'cm-rh', rowLabel));
    for (const [key, value, cls] of entries) {
      const cell = el('div', 'cm-cell ' + cls);
      cell.appendChild(el('div', 'cm-val', num(value)));
      cell.appendChild(el('div', 'cm-lab', key));
      container.appendChild(cell);
    }
  }

  $('cm-rows').textContent = num(locked.rows) + ' flows';
}

function renderComparison(locked) {
  const container = $('comparison-bars');
  container.textContent = '';
  const entries = [
    ['Local-only mean', locked.local_only.mean_macro_f1, 'comparison-local'],
    ['Federated FedAvg', locked.federated_fedavg.macro_f1, 'comparison-federated'],
    ['Centralized', locked.centralized.macro_f1, 'comparison-centralized'],
  ];
  for (const [label, value, className] of entries) {
    const row = el('div', 'comparison-row ' + className);
    row.appendChild(el('div', 'comparison-label', label));
    const track = el('div', 'comparison-track');
    const bar = el('div', 'comparison-fill');
    bar.style.width = Math.max(0, Math.min(100, value * 100)) + '%';
    track.appendChild(bar);
    row.appendChild(track);
    row.appendChild(el('div', 'comparison-value', pct(value, 2)));
    container.appendChild(row);
  }

  const gain = locked.macro_f1_deltas.fedavg_vs_local_mean * 100;
  const gap = Math.abs(locked.macro_f1_deltas.fedavg_vs_centralized * 100);
  const conclusion = $('comparison-conclusion');
  conclusion.textContent = '';
  const lead = el('strong', null, 'Mixed result. ');
  conclusion.appendChild(lead);
  conclusion.appendChild(document.createTextNode(
    `FedAvg improved on the isolated-client mean by ${gain.toFixed(2)} percentage points, ` +
    `but remained ${gap.toFixed(2)} points below centralized pooling.`));
}

function renderFeatureList() {
  const list = $('feature-list');
  list.textContent = '';
  const snap = state.snapshot;
  if (!snap) return;
  for (const feature of snap.model.features) list.appendChild(el('li', null, feature));
  $('feature-count').textContent = snap.model.feature_count + ' inputs';
}

function appendKv(container, rows) {
  container.textContent = '';
  for (const [key, value] of rows) {
    container.appendChild(el('dt', null, key));
    container.appendChild(el('dd', null, value == null ? '—' : String(value)));
  }
}

function renderTechnicalDetails(evidence) {
  renderModelIdentity();
  renderFeatureList();
  const technical = evidence.technical;
  appendKv($('preprocessing-details'), [
    ['Fit rows', num(technical.preprocessor_fit_rows)],
    ['Fit scope', technical.preprocessor_fit_scope],
    ['Privacy limitation', technical.privacy_limitation],
  ]);

  const list = $('client-score-list');
  list.textContent = '';
  for (const client of evidence.locked_test.local_only.clients) {
    const row = el('div', 'client-score-row');
    row.appendChild(el('span', 'mono', client.client_id));
    row.appendChild(el('strong', null, pct(client.macro_f1, 2)));
    list.appendChild(row);
  }
}

function renderClientSummary(locked) {
  const container = $('client-summary');
  container.textContent = '';
  const summary = locked.local_only;
  const entries = [
    ['Best', summary.best_model.replace('local/', ''), summary.best_macro_f1],
    ['Five-client mean', 'local-only baseline', summary.mean_macro_f1],
    ['Worst', summary.worst_model.replace('local/', ''), summary.worst_macro_f1],
  ];
  for (const [label, client, value] of entries) {
    const card = el('div', 'client-summary-card');
    card.appendChild(el('div', 'metric-key', label));
    card.appendChild(el('div', 'client-summary-value', pct(value, 2)));
    card.appendChild(el('div', 'client-summary-name', client));
    container.appendChild(card);
  }
}

function renderSecurityEvidence(security) {
  const badges = $('algorithm-badges');
  badges.textContent = '';
  const algorithms = security.algorithms;
  const items = [
    ['Key establishment', algorithms.kem],
    ['Authentication', algorithms.signature],
    ['Key derivation', algorithms.kdf],
    ['Payload protection', algorithms.aead],
  ];
  for (const [role, name] of items) {
    const badge = el('div', 'algorithm-badge');
    badge.appendChild(el('span', null, role));
    badge.appendChild(el('strong', null, name));
    badges.appendChild(badge);
  }

  const summary = $('security-summary');
  summary.textContent = '';
  const statements = [
    `${security.authenticated_clients} authenticated clients completed training`,
    `${security.rejected_messages} controlled malicious messages were safely rejected`,
    `Plain and secure aggregation matched exactly (max difference ${security.maximum_plain_secure_difference}; tolerance ${security.tolerance})`,
  ];
  for (const statement of statements) summary.appendChild(el('div', 'security-proof', statement));
  $('security-result').textContent = security.rejected_messages + ' / ' +
    security.rejected_messages + ' rejected safely';
  renderRejectionChart($('reject-chart'), security.rejection_categories);
}

function renderAiData() {
  const evidence = state.snapshot && state.snapshot.evidence;
  const available = !!(evidence && evidence.available);
  $('evidence-unavailable').hidden = available;
  $('evidence-content').hidden = !available;
  $('evidence-state').textContent = available ? 'verified saved evidence' : 'evidence unavailable';
  if (!available) return;

  renderLockedTestMetrics(evidence.locked_test);
  renderComparison(evidence.locked_test);
  renderConfusionMatrix(evidence.locked_test);
  renderClientSummary(evidence.locked_test);
  renderSecurityEvidence(evidence.security_test);
  renderTechnicalDetails(evidence);
}

/* ── Rendering: console ─────────────────────────────────────────────── */

const SEVERITIES = ['all', 'info', 'warning', 'error'];

function buildSeverityFilters() {
  const container = $('severity-filters');
  container.textContent = '';
  for (const severity of SEVERITIES) {
    const button = el('button', 'filter-btn' + (state.severityFilter === severity ? ' is-active' : ''),
      severity === 'all' ? 'All severities' : severity);
    button.addEventListener('click', () => {
      state.severityFilter = severity;
      buildSeverityFilters();
      renderConsole();
    });
    container.appendChild(button);
  }
}

function logRow(entry) {
  const severity = entry.severity || 'info';
  const row = el('div', 'row row-' + severity);
  row.appendChild(el('div', 'row-seq', '#' + entry.seq));

  const main = el('div', 'row-main');
  const type = el('div', 'row-type', entry.event_type);
  if (entry.recorded) type.appendChild(el('span', 'row-tag', 'recorded'));
  main.appendChild(type);

  const bits = [];
  if (entry.timestamp_utc) bits.push(clockOf(entry.timestamp_utc));
  if (entry.client_id) bits.push(entry.client_id);
  else if (entry.source) bits.push(entry.source);
  if (entry.round != null && entry.round !== 0) bits.push('round ' + entry.round);

  const payload = entry.payload || {};
  const summary = summarizePayload(entry.event_type, payload);
  if (summary) bits.push(summary);

  main.appendChild(el('div', 'row-meta', bits.join('  ·  ')));
  row.appendChild(main);
  return row;
}

function summarizePayload(type, payload) {
  if (type === 'prediction_completed') {
    return `${payload.label} · p ${payload.attack_probability.toFixed(3)} · ${payload.record_id}`;
  }
  if (type === 'security_message_rejected') {
    return `${payload.category} at ${payload.endpoint}`;
  }
  if (type === 'client_training_completed') {
    return `macro F1 ${fixed(payload.macro_f1)} · ${payload.training_ms != null ? payload.training_ms.toFixed(0) + ' ms' : ''}`;
  }
  if (type === 'round_metrics') {
    return `macro F1 ${fixed(payload.macro_f1)} · FPR ${fixed(payload.fpr)}`;
  }
  if (type === 'aggregation_completed') {
    return `${num(payload.total_samples)} samples · ${payload.aggregation_ms} ms`;
  }
  if (type === 'client_registered') {
    return `${payload.profile || ''} · ${num(payload.samples)} rows`;
  }
  if (type === 'update_rejected') {
    return payload.reason || '';
  }
  if (type === 'server_waiting_for_clients') {
    return 'missing: ' + (payload.missing_clients || []).join(', ');
  }
  if (type === 'demo_completed') {
    return `${payload.rounds} rounds · ${payload.total_runtime_seconds != null ? payload.total_runtime_seconds.toFixed(1) + ' s' : ''}`;
  }
  return '';
}

function paintLog(container, entries) {
  const follow = $('autoscroll').checked;
  container.textContent = '';

  const filtered = state.severityFilter === 'all'
    ? entries
    : entries.filter((e) => (e.severity || 'info') === state.severityFilter);

  if (!filtered.length) {
    container.appendChild(el('div', 'empty', 'No events match the current filter.'));
    return;
  }

  for (const entry of filtered) container.appendChild(logRow(entry));
  if (follow) container.scrollTop = container.scrollHeight;
}

function renderConsole() {
  paintLog($('inference-log'), state.inferenceLog);
  paintLog($('federated-log'), state.federatedLog);
  $('inf-log-count').textContent = state.inferenceLog.length + ' events';
  $('fed-log-count').textContent = state.federatedLog.length + ' events';

  const note = $('paced-note');
  if (state.federatedPaced && state.pendingFederated.length) {
    note.hidden = false;
    note.textContent =
      `Recorded federated evidence. The adapter returns the whole batch at once, so it is ` +
      `released here at presentation pace — ${state.pendingFederated.length} events still queued. ` +
      `Ordering is by sequence number; recorded events carry no timestamps.`;
  } else if (state.federatedPaced) {
    note.hidden = false;
    note.textContent =
      'Recorded federated evidence, released at presentation pace. Ordering is by sequence ' +
      'number; recorded events carry no timestamps.';
  } else {
    note.hidden = true;
  }
}

/* ── Render everything ──────────────────────────────────────────────── */

function render() {
  renderStatus();

  const banner = $('error-banner');
  if (state.error) {
    banner.hidden = false;
    banner.textContent = `Dashboard notice — ${state.error}.`;
  } else {
    banner.hidden = true;
  }

  if (state.view === 'dashboard') renderDashboard();
  else renderAiData();
}

/* ── Polling ────────────────────────────────────────────────────────── */

async function pollSnapshot() {
  try {
    const snapshot = await api.snapshot();
    state.snapshot = snapshot;
    state.connected = true;
    state.error = null;
    // Pick up predictions this tab did not originate (a replay pressed elsewhere,
    // or a second console) so the timeline reflects everything the adapter scored.
    if (snapshot.inference.latest_prediction) {
      recordPrediction(snapshot.inference.latest_prediction);
    }
  } catch (error) {
    state.connected = false;
    state.error = error.message;
  }
  render();
}

async function pollInferenceEvents() {
  if (!state.connected) return;
  try {
    const page = await api.events(state.inferenceCursor);
    if (page.events && page.events.length) {
      for (const event of page.events) {
        if (event.event_type === 'prediction_completed' && event.payload) {
          recordPrediction(event.payload);
        }
      }
      state.inferenceLog.push(...page.events);
      if (state.inferenceLog.length > 500) {
        state.inferenceLog.splice(0, state.inferenceLog.length - 500);
      }
    }
    state.inferenceCursor = page.last_seq != null ? page.last_seq : state.inferenceCursor;
  } catch { /* transient; the snapshot poll owns the connection banner */ }
}

async function pollFederatedEvents() {
  if (!state.connected) return;
  try {
    const page = await api.federatedEvents(state.federatedCursor);
    const events = page.events || [];
    state.federatedPaced = page.data_mode === 'replay';

    if (events.length) {
      if (state.federatedPaced) state.pendingFederated.push(...events);
      else state.federatedLog.push(...events);
    }
    state.federatedCursor = page.last_seq != null ? page.last_seq : state.federatedCursor;
  } catch { /* transient */ }
}

function releasePendingFederated() {
  if (!state.pendingFederated.length) return;
  state.federatedLog.push(state.pendingFederated.shift());
  if (state.view === 'dashboard') renderConsole();
}

/* ── Injector ───────────────────────────────────────────────────────── */

/* ── Locked-test traffic replay ─────────────────────────────────────────── */

function setTrafficBusy(busy) {
  state.busy = busy;
  const start = $('btn-traffic');
  const reset = $('btn-traffic-reset');
  if (start) start.disabled = busy || !state.connected;
  if (reset) reset.disabled = busy || !state.connected;
}

async function sendNextNormal() {
  if (state.busy || !state.connected || !state.trafficRunning) return;
  setTrafficBusy(true);
  try {
    const prediction = await api.demoNextNormal();
    recordPrediction(prediction, { fromThisTab: true });
    $('traffic-state').textContent = 'running · locked-test Normal';
    render();
  } catch (error) {
    if (error.code === 'demo_pool_exhausted') {
      pauseTraffic();
      $('traffic-state').textContent = 'complete · pool exhausted';
    } else {
      state.error = error.message;
      pauseTraffic();
    }
    render();
  } finally {
    setTrafficBusy(false);
  }
}

function scheduleTraffic() {
  if (!state.trafficRunning) return;
  state.trafficTimer = setTimeout(async () => {
    await sendNextNormal();
    scheduleTraffic();
  }, state.intervalMs);
}

function startTraffic() {
  if (state.trafficRunning || !state.connected) return;
  state.trafficRunning = true;
  $('btn-traffic').textContent = 'Pause Traffic';
  $('btn-traffic').classList.add('is-running');
  $('traffic-state').textContent = 'starting · locked-test Normal';
  sendNextNormal().then(scheduleTraffic);
}

function pauseTraffic() {
  state.trafficRunning = false;
  clearTimeout(state.trafficTimer);
  if ($('btn-traffic')) {
    $('btn-traffic').textContent = 'Start Traffic';
    $('btn-traffic').classList.remove('is-running');
  }
  if ($('traffic-state') && !$('traffic-state').textContent.includes('complete')) {
    $('traffic-state').textContent = 'paused';
  }
}

async function resetTraffic() {
  if (state.busy || !state.connected) return;
  pauseTraffic();
  setTrafficBusy(true);
  try {
    const session = await api.demoReset();
    state.history = [];
    state.volume = [];
    state.seenPredictions = new Set();
    state.inferenceLog = [];
    state.inferenceCursor = session.last_event_seq || state.inferenceCursor;
    if (state.snapshot) state.snapshot.dataset_demo = session;
    $('traffic-state').textContent = 'ready · stream restarted';
    render();
  } catch (error) {
    state.error = error.message;
    render();
  } finally {
    setTrafficBusy(false);
  }
}

function openAttackerView() {
  const url = 'attacker.html?api=' + encodeURIComponent(API_BASE);
  const popup = window.open(url, 'rasid-attacker', 'popup,width=1380,height=900,resizable=yes,scrollbars=yes');
  if (!popup) {
    state.error = 'The browser blocked Attacker View. Allow pop-ups for this local page and try again';
    render();
  }
}

/* ── Wiring ─────────────────────────────────────────────────────────── */

function switchView(view) {
  state.view = view;
  for (const button of document.querySelectorAll('.nav-item[data-view]')) {
    const isActive = button.dataset.view === view;
    button.classList.toggle('is-active', isActive);
    button.setAttribute('aria-selected', String(isActive));
  }
  for (const section of document.querySelectorAll('.view')) {
    section.classList.toggle('is-active', section.id === 'view-' + view);
  }
  render();
}

/* ── Theme ──────────────────────────────────────────────────────────────
   The inline script in <head> sets data-theme before first paint; this only
   handles switching. Charts read their colours from CSS custom properties at
   draw time, so a switch has to re-render them — nothing repaints itself.
   ------------------------------------------------------------------------ */

const THEME_KEY = 'rasid-theme';

function currentTheme() {
  return document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'light';
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const button = $('theme-toggle');
  if (button) {
    button.setAttribute('aria-pressed', String(theme === 'dark'));
    $('theme-label').textContent = theme === 'dark' ? 'Light theme' : 'Dark theme';
  }
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* private mode */ }
  render();
}

function wire() {
  for (const button of document.querySelectorAll('.nav-item[data-view]')) {
    button.addEventListener('click', () => switchView(button.dataset.view));
  }

  const themeButton = $('theme-toggle');
  if (themeButton) {
    themeButton.addEventListener('click',
      () => applyTheme(currentTheme() === 'dark' ? 'light' : 'dark'));
  }

  for (const header of document.querySelectorAll('.row-header')) {
    header.addEventListener('click', () => {
      const body = $('section-' + header.dataset.section);
      if (!body) return;
      const collapsed = body.classList.toggle('is-collapsed');
      header.setAttribute('aria-expanded', String(!collapsed));
    });
  }

  $('btn-traffic').addEventListener('click', () =>
    (state.trafficRunning ? pauseTraffic() : startTraffic()));
  $('btn-traffic-reset').addEventListener('click', resetTraffic);
  $('btn-attacker-view').addEventListener('click', openAttackerView);
  $('btn-alert-history').addEventListener('click', () => {
    state.alertsExpanded = !state.alertsExpanded;
    renderAlerts();
  });
  $('btn-alert-export').addEventListener('click', exportAlerts);

  const rate = $('traffic-rate');
  rate.addEventListener('input', () => {
    state.intervalMs = Number(rate.value);
    $('traffic-rate-out').textContent = (state.intervalMs / 1000).toFixed(2).replace(/0$/, '') + ' s';
  });

  $('btn-clear-console').addEventListener('click', () => {
    state.inferenceLog = [];
    state.federatedLog = [];
    renderConsole();
  });

  buildSeverityFilters();
  // Sync the button's label with whatever the pre-paint script chose.
  applyTheme(currentTheme());

  window.addEventListener('resize', () => {
    if (state.view === 'dashboard') renderDashboard();
  });
}

/* ── Boot ───────────────────────────────────────────────────────────── */

async function boot() {
  wire();
  render();

  try {
    await api.health();
    state.demoCatalog = await api.demoCatalog();
    state.connected = true;
  } catch (error) {
    state.connected = false;
    state.error = error.message;
  }

  await pollSnapshot();
  await pollInferenceEvents();
  await pollFederatedEvents();
  render();

  setInterval(pollSnapshot, SNAPSHOT_POLL_MS);
  setInterval(pollInferenceEvents, EVENT_POLL_MS);
  setInterval(pollFederatedEvents, EVENT_POLL_MS);
  setInterval(releasePendingFederated, RECORDED_RELEASE_MS);
}

document.addEventListener('DOMContentLoaded', boot);
