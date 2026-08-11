/* ==========================================================================
   RASID — Federated Intrusion Detection Console

   Talks only to the documented GUI adapter (uavids-gui-api-v1). It never loads
   checkpoints, preprocessing objects, datasets, or cryptographic material, and
   it never computes a verdict, a probability, or a metric in the browser.

   Honesty rules enforced here:
     - Every verdict comes from POST /predictions or POST /replay/next.
     - The injector emits model INPUT only. A profile names the traffic shape the
       operator selected; the model is binary and never identifies an attack family.
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
const TIMELINE_MAX = 60;
const REQUEST_TIMEOUT_MS = 4000;

// Rate histogram: 40 buckets of 3 s = a rolling two-minute window. At the
// injector's default 1.5 s interval a bucket holds ~2 verdicts, so bars have
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
  predict:         (b)  => request('/predictions', { method: 'POST', body: JSON.stringify(b) }),
  replayNext:      ()   => request('/replay/next', { method: 'POST', body: '{}' }),
  events:          (n)  => request('/events?after_seq=' + n),
  federatedEvents: (n)  => request('/federated/events?after_seq=' + n),
};

/* ── Traffic profiles ───────────────────────────────────────────────────
   Sampling envelopes are measured from the committed validation fixture
   gui_integration/examples/replay_records.json — rows 01/03/05 (which the frozen
   model scores Normal) and rows 02/04/06 (which it scores Attack). Feature
   magnitudes are not invented.

   Derived features are computed, not sampled, using relationships that hold
   exactly in that fixture, so an emitted vector is internally self-consistent
   rather than 15 unrelated random numbers.
   ------------------------------------------------------------------------ */

const PROFILES = [
  {
    id: 'normal',
    name: 'NORMAL TRAFFIC',
    desc: 'A normal traffic, handful of packets, seperated and not consecutive',
  },
  {
    id: 'borderline',
    name: 'COINFLIP',
    desc: 'Sometimes the packet is malicious, sometimes not',
  },
  {
    id: 'flood',
    name: 'FLOOD ATTACK',
    desc: 'Lots of packets are sent to the drone. Also known as a DOS attack',
  },
];

/** Plain-language name for a profile id, for status text. */
function profileLabel(profileId) {
  const profile = PROFILES.find((p) => p.id === profileId);
  return profile ? profile.name.toLowerCase() : profileId;
}

const NORMAL_ENV = {
  duration: [2.0, 180.0], packets: [1, 8], packetSize: [30, 46],
  delay: [1.3, 2.1], jitter: [0.0, 0.66], hop: [0.0, 0.0], lossRate: [0.0, 0.15],
};
const FLOOD_ENV = {
  duration: [3400, 3500], packets: [4200, 5850], packetSize: [76, 76],
  delay: [0.0095, 0.014], jitter: [0.006, 0.0105], hop: [0.031, 0.039], lossRate: [0.0003, 0.0017],
};

const uniform = (lo, hi) => lo + Math.random() * (hi - lo);
const lerp = (lo, hi, t) => lo * (1 - t) + hi * t;
const logLerp = (lo, hi, t) =>
  Math.exp(Math.log(Math.max(lo, 1e-6)) * (1 - t) + Math.log(Math.max(hi, 1e-6)) * t);
const sig = (v, d = 6) => (v === 0 ? 0 : Number(v.toPrecision(d)));

function drawEnvelope(profileId) {
  if (profileId === 'normal' || profileId === 'flood') {
    const e = profileId === 'normal' ? NORMAL_ENV : FLOOD_ENV;
    return {
      duration: uniform(...e.duration),
      packets: Math.round(uniform(...e.packets)),
      packetSize: uniform(...e.packetSize),
      delay: uniform(...e.delay),
      jitter: uniform(...e.jitter),
      hop: uniform(...e.hop),
      lossRate: uniform(...e.lossRate),
    };
  }
  // Borderline. Duration, packet count, delay and jitter each span two or more
  // orders of magnitude between the clusters, so they blend geometrically — the
  // linear midpoint would collapse back onto the normal cluster. Packet size and
  // hop count are near-linear and blend linearly.
  const t = uniform(0.35, 0.65);
  return {
    duration: logLerp(uniform(...NORMAL_ENV.duration), uniform(...FLOOD_ENV.duration), t),
    packets: Math.max(1, Math.round(logLerp(uniform(...NORMAL_ENV.packets), uniform(...FLOOD_ENV.packets), t))),
    packetSize: lerp(uniform(...NORMAL_ENV.packetSize), uniform(...FLOOD_ENV.packetSize), t),
    delay: logLerp(uniform(...NORMAL_ENV.delay), uniform(...FLOOD_ENV.delay), t),
    jitter: logLerp(uniform(...NORMAL_ENV.jitter), uniform(...FLOOD_ENV.jitter), t),
    hop: lerp(uniform(...NORMAL_ENV.hop), uniform(...FLOOD_ENV.hop), t),
    lossRate: lerp(uniform(...NORMAL_ENV.lossRate), uniform(...FLOOD_ENV.lossRate), t),
  };
}

function generateVector(profileId) {
  const d = drawEnvelope(profileId);
  const flowDuration = sig(d.duration);
  const txPackets = Math.max(1, d.packets);
  const lostPackets = Math.min(txPackets, Math.round(txPackets * d.lossRate));
  const rxPackets = Math.max(0, txPackets - lostPackets);
  const txBytes = Math.round(txPackets * d.packetSize);
  const rxBytes = Math.round(rxPackets * d.packetSize);

  return {
    'FlowDuration/s': flowDuration,
    TxPackets: txPackets,
    RxPackets: rxPackets,
    LostPackets: lostPackets,
    TxBytes: txBytes,
    RxBytes: rxBytes,
    'TxPacketRate/s': sig(txPackets / flowDuration),
    'RxPacketRate/s': sig(rxPackets / flowDuration),
    'TxByteRate/s': sig(txBytes / flowDuration),
    'RxByteRate/s': sig(rxBytes / flowDuration),
    'MeanDelay/s': sig(d.delay),
    'MeanJitter/s': sig(d.jitter),
    'Throughput/Kbps': sig((rxBytes * 8) / 1000 / flowDuration),
    PacketDropRate: sig(txPackets === 0 ? 0 : lostPackets / txPackets),
    AverageHopCount: sig(d.hop),
  };
}

const READOUT_FEATURES = [
  'FlowDuration/s', 'TxPackets', 'RxPackets', 'LostPackets',
  'MeanDelay/s', 'MeanJitter/s', 'Throughput/Kbps', 'PacketDropRate',
];

/* ── State ──────────────────────────────────────────────────────────── */

const state = {
  snapshot: null,
  error: null,
  connected: false,

  // session (this tab only)
  history: [],            // recent predictions for the timeline
  volume: [],             // { t: epoch ms, attack: bool } for the rate histogram
  seenPredictions: new Set(),
  sessionRecords: 0,
  sessionPackets: 0,
  lastVector: null,
  lastProfile: null,

  // injector
  profile: 'normal',
  intervalMs: 1500,
  streaming: false,
  streamTimer: null,
  busy: false,

  // event feeds
  inferenceCursor: 0,
  federatedCursor: 0,
  inferenceLog: [],
  federatedLog: [],
  pendingFederated: [],   // buffered recorded batch awaiting paced release
  federatedPaced: false,

  // derived from federated events
  selectedClient: null,   // client_id selected in the topology
  clientInfo: {},         // client_id -> { profile, samples }
  clientTraining: {},     // client_id -> latest client_training_completed payload
  roundMetrics: [],       // { round, ...metrics }
  rejections: {},         // category -> count

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
      'No verdicts yet. Inject traffic or replay the fixture to populate the timeline.'));
    return;
  }

  const W = 620, H = 220;
  const padL = 34, padR = 12, padT = 12, padB = 26;
  const plotW = W - padL - padR, plotH = H - padT - padB;
  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img' });
  svg.setAttribute('aria-label',
    `Attack probability for the last ${history.length} verdicts against the ${threshold} decision threshold`);

  const y = (v) => padT + (1 - v) * plotH;
  const slot = plotW / TIMELINE_MAX;
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
      (p.replayed ? '<br><span class="tip-k">replayed fixture input</span>' : ''));
    svg.appendChild(bar);
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
  older.textContent = 'older';
  svg.appendChild(older);

  const newer = svgEl('text', {
    x: W - padR, y: H - 6, 'text-anchor': 'end',
    fill: cssVar('--ink-muted'), 'font-size': 10.5, 'font-family': 'system-ui, sans-serif',
  });
  newer.textContent = 'latest';
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
    label.textContent = category;
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

/* ── Derived state from federated events ────────────────────────────── */

function mineFederatedEvent(event) {
  const type = event.event_type;
  const payload = event.payload || {};
  const cid = event.client_id;

  if (type === 'client_registered' && cid) {
    state.clientInfo[cid] = {
      profile: payload.profile || null,
      samples: typeof payload.samples === 'number' ? payload.samples : null,
    };
  }

  if (type === 'aggregation_started' && payload.sample_counts) {
    for (const [id, samples] of Object.entries(payload.sample_counts)) {
      state.clientInfo[id] = { ...(state.clientInfo[id] || {}), samples };
    }
  }

  if (type === 'client_training_completed' && cid) {
    state.clientTraining[cid] = { ...payload, round: event.round };
  }

  if (type === 'round_metrics') {
    const existing = state.roundMetrics.findIndex((r) => r.round === event.round);
    const record = { round: event.round, ...payload };
    if (existing >= 0) state.roundMetrics[existing] = record;
    else state.roundMetrics.push(record);
    state.roundMetrics.sort((a, b) => a.round - b.round);
  }

  if (type === 'demo_completed' && payload.final_metrics) {
    const record = { round: event.round, final: true, ...payload.final_metrics };
    const existing = state.roundMetrics.findIndex((r) => r.final);
    if (existing >= 0) state.roundMetrics[existing] = record;
    else state.roundMetrics.push(record);
  }

  if (type === 'security_message_rejected') {
    const category = payload.category || 'unspecified';
    state.rejections[category] = (state.rejections[category] || 0) + 1;
  }
}

/* ── Prediction recording ───────────────────────────────────────────── */

function recordPrediction(prediction, options = {}) {
  if (!prediction || !prediction.prediction_id) return;
  if (state.seenPredictions.has(prediction.prediction_id)) return;
  state.seenPredictions.add(prediction.prediction_id);

  state.history.push(prediction);
  if (state.history.length > TIMELINE_MAX) state.history.shift();

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

  if (options.fromThisTab) {
    state.sessionRecords += 1;
    if (options.vector) {
      const tx = options.vector.TxPackets || 0;
      const rx = options.vector.RxPackets || 0;
      state.sessionPackets += tx + rx;
    }
  }
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
  const inf = snap.inference;
  const total = inf.records_processed || 0;

  $('k-records').textContent = num(total);
  $('k-attacks').textContent = num(inf.attack_count);
  $('k-normal').textContent = num(inf.normal_count);
  // One send is one flow record, so this counts sends. sessionPackets is the
  // number of packets those records *describe* — a flood record summarises
  // thousands of them, so showing it here made one click look like thousands.
  $('k-packets').textContent = num(state.sessionRecords);
  $('k-packets-sub').textContent = state.sessionPackets
    ? `this browser session · describing ${num(state.sessionPackets)} packets`
    : 'this browser session only';

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

const ALERT_COLUMNS = ['Time', 'Record', 'Sent as', 'Attack confidence', 'Probability', 'Over threshold', 'Input'];

function renderAlerts() {
  const table = $('alert-table');
  table.textContent = '';
  const alerts = (state.snapshot && state.snapshot.inference.recent_alerts) || [];

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

  for (const alert of alerts) {
    const probability = alert.attack_probability;
    const band = confidenceBand(probability);
    const row = el('tr');

    row.appendChild(el('td', 'mono', clockOf(alert.timestamp_utc)));
    row.appendChild(el('td', 'mono cell-id', alert.record_id));
    row.appendChild(el('td', 'mono', alert.source || '—'));

    const confidence = el('td');
    const chip = el('span', 'conf-chip ' + band.cls, band.label);
    confidence.appendChild(chip);
    confidence.appendChild(el('span', 'conf-pct mono', pct(probability, 1)));
    row.appendChild(confidence);

    row.appendChild(el('td', 'mono', probability.toFixed(4)));
    row.appendChild(el('td', 'mono', '+' + (probability - threshold).toFixed(4)));

    const input = el('td');
    input.appendChild(el('span', 'row-tag', alert.replayed ? 'replayed' : 'injected'));
    if (alert.missing_features_imputed) {
      input.appendChild(el('span', 'row-tag row-tag-warn',
        alert.missing_features_imputed + ' imputed'));
    }
    row.appendChild(input);

    table.appendChild(row);
  }
}

/* ── Federated network topology ─────────────────────────────────────────
   Hub and spokes: five clients around one control center, with no client-to-
   client edges — that absence is the architecture, not a simplification.

   Node state uses the status palette. Every node carries its state as text
   beside the color, so the encoding never rests on hue alone.
   ------------------------------------------------------------------------ */

/* Configured container profiles from phase4/DEVICE_PROFILES.md. These are fixed
   deployment facts, not run telemetry, and the detail panel labels them as such.
   Device-inspired resource caps — not hardware emulation. */
const DEVICE_PROFILES = {
  'uav-client-1': { device: 'Raspberry Pi 4 Model B',   cpu: '0.65 CPU', memory: '512 MiB' },
  'uav-client-2': { device: 'NVIDIA Jetson Nano',       cpu: '0.55 CPU', memory: '512 MiB' },
  'uav-client-3': { device: 'Raspberry Pi Zero 2 W',    cpu: '0.35 CPU', memory: '448 MiB' },
  'uav-client-4': { device: 'NVIDIA Jetson Orin Nano',  cpu: '1.00 CPU', memory: '768 MiB' },
  'uav-client-5': { device: 'NXP NavQPlus',             cpu: '0.45 CPU', memory: '512 MiB' },
};

const STATE_BUCKET = {
  complete: 'complete',
  round_complete: 'complete',
  error: 'error',
  waiting: 'wait',
};

function stateBucket(clientState) {
  return STATE_BUCKET[clientState] || 'active';
}

const BUCKET_COLORS = {
  complete: { stroke: '--ok',        fill: '--ok-soft' },
  active:   { stroke: '--accent',    fill: '--accent-soft' },
  wait:     { stroke: '--ink-muted', fill: '--idle-soft' },
  error:    { stroke: '--crit',      fill: '--crit-soft' },
};

function renderTopology() {
  const container = $('fed-topology');
  container.textContent = '';
  const fed = state.snapshot && state.snapshot.federated;

  if (!fed || !fed.clients.length) {
    container.appendChild(el('div', 'chart-empty', 'Federated telemetry unavailable.'));
    return;
  }

  const W = 560, H = 400;
  const cx = W / 2, cy = H / 2;
  const orbit = 148;
  const nodeW = 116, nodeH = 52;
  const hubW = 132, hubH = 60;

  const svg = svgEl('svg', { viewBox: `0 0 ${W} ${H}`, role: 'group' });
  svg.setAttribute('aria-label',
    `Federated topology: ${fed.clients.length} clients connected to one control center`);

  // One vertex at the top, the rest stepped evenly around the circle.
  const clients = fed.clients;
  const step = (Math.PI * 2) / clients.length;
  const positions = clients.map((client, index) => {
    const angle = -Math.PI / 2 + index * step;
    return { client, x: cx + Math.cos(angle) * orbit, y: cy + Math.sin(angle) * orbit };
  });

  // Edges first, so nodes paint over their endpoints.
  for (const { client, x, y } of positions) {
    const bucket = stateBucket(client.state);
    const active = bucket === 'complete' || bucket === 'active';
    svg.appendChild(svgEl('line', {
      x1: cx, y1: cy, x2: x, y2: y,
      stroke: cssVar(active ? BUCKET_COLORS[bucket].stroke : '--line'),
      'stroke-width': active ? 2 : 1.5,
      'stroke-dasharray': bucket === 'wait' ? '4 4' : 'none',
      opacity: active ? 0.55 : 1,
    }));
  }

  // Hub
  svg.appendChild(svgEl('rect', {
    x: cx - hubW / 2, y: cy - hubH / 2, width: hubW, height: hubH,
    fill: cssVar('--surface'), stroke: cssVar('--ink'), 'stroke-width': 2,
  }));
  const hubLabel = svgEl('text', {
    x: cx, y: cy - 4, 'text-anchor': 'middle', fill: cssVar('--ink'),
    'font-size': 13, 'font-weight': 700, 'font-family': 'system-ui, sans-serif',
  });
  hubLabel.textContent = 'CONTROL CENTER';
  svg.appendChild(hubLabel);

  const hubSub = svgEl('text', {
    x: cx, y: cy + 13, 'text-anchor': 'middle', fill: cssVar('--ink-muted'),
    'font-size': 11, 'font-family': 'system-ui, sans-serif',
  });
  hubSub.textContent = 'FedAvg aggregator';
  svg.appendChild(hubSub);

  // Client nodes
  for (const { client, x, y } of positions) {
    const bucket = stateBucket(client.state);
    const colors = BUCKET_COLORS[bucket];
    const selected = state.selectedClient === client.client_id;

    const group = svgEl('g', {
      class: 'node-box', tabindex: '0', role: 'button',
      'aria-label': `${client.client_id}, state ${client.state}`,
    });

    group.appendChild(svgEl('rect', {
      class: 'node-face',
      x: x - nodeW / 2, y: y - nodeH / 2, width: nodeW, height: nodeH,
      fill: cssVar(colors.fill),
      stroke: cssVar(selected ? '--ink' : colors.stroke),
      'stroke-width': selected ? 3 : 2,
    }));

    const idLabel = svgEl('text', {
      x, y: y - 5, 'text-anchor': 'middle', fill: cssVar('--ink'),
      'font-size': 12.5, 'font-weight': 700, 'font-family': 'ui-monospace, monospace',
      'pointer-events': 'none',
    });
    idLabel.textContent = client.client_id.replace('uav-', '');
    group.appendChild(idLabel);

    const stateLabel = svgEl('text', {
      x, y: y + 12, 'text-anchor': 'middle', fill: cssVar('--ink-2'),
      'font-size': 10.5, 'font-family': 'system-ui, sans-serif', 'pointer-events': 'none',
    });
    stateLabel.textContent = client.state.replace(/_/g, ' ');
    group.appendChild(stateLabel);

    const select = () => {
      state.selectedClient = client.client_id;
      renderTopology();
      renderClientDetail();
    };
    group.addEventListener('click', select);
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); }
    });

    const info = state.clientInfo[client.client_id] || {};
    attachTip(group, () =>
      `<strong>${client.client_id}</strong><br>` +
      `<span class="tip-k">state</span> ${client.state.replace(/_/g, ' ')}` +
      (info.samples != null ? `<br><span class="tip-k">local rows</span> ${num(info.samples)}` : '') +
      '<br><span class="tip-k">click for detail</span>');

    svg.appendChild(group);
  }

  container.appendChild(svg);
}

function detailRow(list, key, value, absentText) {
  list.appendChild(el('dt', null, key));
  const present = value !== null && value !== undefined && value !== '';
  list.appendChild(el('dd', present ? null : 'absent',
    present ? String(value) : (absentText || 'not reported in this run')));
}

function renderClientDetail() {
  const container = $('client-detail');
  container.textContent = '';
  const fed = state.snapshot && state.snapshot.federated;
  const client = fed && fed.clients.find((c) => c.client_id === state.selectedClient);

  if (!client) {
    container.appendChild(el('div', 'detail-empty',
      'Select a node in the network to inspect that client.'));
    return;
  }

  const info = state.clientInfo[client.client_id] || {};
  const training = state.clientTraining[client.client_id] || {};
  const configured = DEVICE_PROFILES[client.client_id];
  const bucket = stateBucket(client.state);

  const head = el('div', 'detail-head');
  head.appendChild(el('div', 'detail-id', client.client_id));
  const chipClass = { complete: 'chip-complete', error: 'chip-error', wait: 'chip-wait' }[bucket]
    || 'chip-active';
  head.appendChild(el('span', 'chip ' + chipClass, client.state.replace(/_/g, ' ')));
  container.appendChild(head);

  // Configured deployment facts — constant across runs.
  if (configured) {
    const group = el('div', 'detail-group');
    group.appendChild(el('h3', null, 'Configured profile'));
    const list = el('dl', 'detail-rows');
    detailRow(list, 'Device inspiration', configured.device);
    detailRow(list, 'CPU limit', configured.cpu);
    detailRow(list, 'Memory limit', configured.memory);
    group.appendChild(list);
    container.appendChild(group);
  }

  // Reported by the current run — may be absent in a recorded excerpt.
  const reported = el('div', 'detail-group');
  reported.appendChild(el('h3', null, 'Reported by this run'));
  const list = el('dl', 'detail-rows');
  detailRow(list, 'Local rows', info.samples != null ? num(info.samples) : null);

  // A recorded excerpt carries training telemetry for only some clients. Six
  // identical "not reported" rows read as a broken panel, so collapse the whole
  // absent block into one statement instead.
  const hasTraining = ['round', 'macro_f1', 'loss', 'training_ms', 'update_bytes']
    .some((key) => training[key] != null);

  if (hasTraining) {
    detailRow(list, 'Round', training.round != null ? String(training.round) : null);
    detailRow(list, 'Local macro F1', training.macro_f1 != null ? fixed(training.macro_f1) : null);
    detailRow(list, 'Local loss', training.loss != null ? fixed(training.loss) : null);
    detailRow(list, 'Training time',
      training.training_ms != null ? training.training_ms.toFixed(0) + ' ms' : null);
    detailRow(list, 'Update size',
      training.update_bytes != null ? num(training.update_bytes) + ' B' : null);
  }
  reported.appendChild(list);

  if (!hasTraining) {
    reported.appendChild(el('p', 'detail-absent',
      state.snapshot && state.snapshot.presentation_mode === 'live'
        ? 'No training round reported for this client yet.'
        : 'This client’s per-round training telemetry is not part of the recorded excerpt. A live federated run reports it for all five.'));
  }
  container.appendChild(reported);

  container.appendChild(el('p', 'detail-note',
    'Configured profile is a fixed container resource cap from the deployment ' +
    'documentation. Run values come from this run’s federated events; a recorded ' +
    'excerpt does not carry every client’s telemetry.'));
}

function renderFederation() {
  const fed = state.snapshot && state.snapshot.federated;
  renderTopology();
  renderClientDetail();
  if (!fed) return;

  $('fed-round').textContent = 'round ' + fed.current_round + ' of ' + fed.total_rounds;
  $('updates-text').textContent = fed.updates_received + ' / ' + fed.updates_expected;
  const ratio = fed.updates_expected ? fed.updates_received / fed.updates_expected : 0;
  $('updates-meter').style.width = Math.round(ratio * 100) + '%';
  $('local-data-statement').textContent = fed.local_data_statement || '';
}

function renderRejections() {
  renderRejectionChart($('reject-chart'), state.rejections);
  const fed = state.snapshot && state.snapshot.federated;
  const reported = fed && fed.security ? fed.security.rejected_messages : null;
  const mined = Object.values(state.rejections).reduce((a, b) => a + b, 0);
  $('reject-total').textContent = (reported != null ? reported : mined) + ' total';
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
  renderVerdict();
  renderKpis();
  renderTimeline($('timeline-chart'), state.history,
    state.snapshot ? state.snapshot.model.decision_threshold : 0.42);
  renderVolume();
  renderAlerts();
  renderFederation();
  renderRejections();
  renderConsole();
}

/* ── Rendering: AI data ─────────────────────────────────────────────── */

function renderModelIdentity() {
  const container = $('model-identity');
  container.textContent = '';
  const snap = state.snapshot;
  if (!snap) return;

  const rows = [
    ['Model ID', snap.model.model_id],
    ['Version', snap.model.model_version],
    ['Task', snap.model.task],
    ['Labels', snap.model.labels.join(' · ')],
    ['Positive class', snap.model.positive_class],
    ['Decision threshold', String(snap.model.decision_threshold)],
    ['Inference mode', snap.backend.inference_mode],
    ['Adapter started', clockOf(snap.backend.started_utc)],
    ['API version', snap.api_version],
  ];

  for (const [key, value] of rows) {
    container.appendChild(el('dt', null, key));
    container.appendChild(el('dd', null, value == null ? '—' : String(value)));
  }
}

function renderFrozenMetrics() {
  const container = $('frozen-metrics');
  container.textContent = '';
  const snap = state.snapshot;
  if (!snap) return;

  const metrics = snap.model.frozen_validation_metrics || {};
  const rows = [
    ['Macro F1', metrics.macro_f1],
    ['Attack precision', metrics.attack_precision],
    ['Attack recall', metrics.attack_recall],
    ['False positive rate', metrics.fpr],
  ];

  for (const [key, value] of rows) {
    const tile = el('div', 'metric');
    tile.appendChild(el('div', 'metric-key', key));
    tile.appendChild(el('div', 'metric-val', typeof value === 'number' ? value.toFixed(4) : '—'));
    container.appendChild(tile);
  }

  $('metrics-source').textContent = snap.model.metrics_note ? 'saved evidence' : '';
}

function renderConfusionMatrix() {
  const container = $('confusion-matrix');
  container.textContent = '';

  const source = state.roundMetrics.find((r) => r.final) ||
    state.roundMetrics[state.roundMetrics.length - 1];

  if (!source || source.tp == null) {
    container.appendChild(el('div', 'chart-empty',
      'No round_metrics event received yet — the confusion matrix arrives with the federated event feed.'));
    $('cm-rows').textContent = '';
    return;
  }

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

  $('cm-rows').textContent = num(source.rows) + ' validation rows';
}

const ROUND_METRIC_COLUMNS = [
  ['macro_f1', 'Macro F1'], ['accuracy', 'Accuracy'], ['balanced_accuracy', 'Balanced acc.'],
  ['attack_precision', 'Atk precision'], ['attack_recall', 'Atk recall'], ['attack_f1', 'Atk F1'],
  ['normal_precision', 'Nrm precision'], ['normal_recall', 'Nrm recall'], ['normal_f1', 'Nrm F1'],
  ['roc_auc', 'ROC AUC'], ['pr_auc', 'PR AUC'], ['log_loss', 'Log loss'],
  ['fpr', 'FPR'], ['fnr', 'FNR'],
];

function renderRoundMetricsTable() {
  const table = $('round-metrics-table');
  table.textContent = '';

  if (!state.roundMetrics.length) {
    const body = document.createElement('tbody');
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.textContent = 'Waiting for round_metrics events from the federated feed.';
    cell.style.color = cssVar('--ink-muted');
    row.appendChild(cell);
    body.appendChild(row);
    table.appendChild(body);
    $('round-metrics-tag').textContent = '';
    return;
  }

  const head = document.createElement('thead');
  const headRow = document.createElement('tr');
  headRow.appendChild(el('th', null, 'Metric'));
  for (const record of state.roundMetrics) {
    headRow.appendChild(el('th', null, record.final ? 'final' : 'round ' + record.round));
  }
  head.appendChild(headRow);
  table.appendChild(head);

  const body = document.createElement('tbody');
  for (const [key, label] of ROUND_METRIC_COLUMNS) {
    const row = document.createElement('tr');
    row.appendChild(el('td', 'name', label));
    for (const record of state.roundMetrics) {
      row.appendChild(el('td', 'num', fixed(record[key])));
    }
    body.appendChild(row);
  }
  table.appendChild(body);

  $('round-metrics-tag').textContent = state.roundMetrics.length + ' evaluations';
}

function renderClientTable() {
  const table = $('client-table');
  table.textContent = '';

  const ids = Object.keys({ ...state.clientInfo, ...state.clientTraining }).sort();
  if (!ids.length) {
    const body = document.createElement('tbody');
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.textContent = 'Waiting for client events from the federated feed.';
    cell.style.color = cssVar('--ink-muted');
    row.appendChild(cell);
    body.appendChild(row);
    table.appendChild(body);
    return;
  }

  const columns = ['Client', 'Device profile', 'Local rows', 'Round', 'Local macro F1',
    'Local loss', 'Train ms', 'Update bytes'];
  const head = document.createElement('thead');
  const headRow = document.createElement('tr');
  for (const c of columns) headRow.appendChild(el('th', null, c));
  head.appendChild(headRow);
  table.appendChild(head);

  const body = document.createElement('tbody');
  for (const id of ids) {
    const info = state.clientInfo[id] || {};
    const training = state.clientTraining[id] || {};
    const row = document.createElement('tr');
    row.appendChild(el('td', 'name', id));
    row.appendChild(el('td', null, info.profile || '—'));
    row.appendChild(el('td', 'num', info.samples != null ? num(info.samples) : '—'));
    row.appendChild(el('td', 'num', training.round != null ? String(training.round) : '—'));
    row.appendChild(el('td', 'num', fixed(training.macro_f1)));
    row.appendChild(el('td', 'num', fixed(training.loss)));
    row.appendChild(el('td', 'num', training.training_ms != null ? training.training_ms.toFixed(1) : '—'));
    row.appendChild(el('td', 'num', training.update_bytes != null ? num(training.update_bytes) : '—'));
    body.appendChild(row);
  }
  table.appendChild(body);
}

function renderFeatureList() {
  const list = $('feature-list');
  list.textContent = '';
  const snap = state.snapshot;
  if (!snap) return;
  for (const feature of snap.model.features) list.appendChild(el('li', null, feature));
  $('feature-count').textContent = snap.model.feature_count + ' inputs';
}

function renderSecurity() {
  const container = $('security-kv');
  container.textContent = '';
  const fed = state.snapshot && state.snapshot.federated;
  if (!fed || !fed.security) return;

  const security = fed.security;
  const algorithms = security.algorithms || {};
  const rows = [
    ['Mode', security.mode],
    ['Status', security.status],
    ['Key establishment', algorithms.kem || 'not reported'],
    ['Client authentication', algorithms.signature || 'not reported'],
    ['Payload protection', algorithms.aead || 'not reported'],
    ['Key derivation', algorithms.kdf || 'not reported'],
    ['Authenticated clients', String(security.authenticated_clients)],
    ['Rejected messages', String(security.rejected_messages)],
  ];

  for (const [key, value] of rows) {
    container.appendChild(el('dt', null, key));
    container.appendChild(el('dd', null, value == null ? '—' : String(value)));
  }
}

function renderAiData() {
  renderModelIdentity();
  renderFrozenMetrics();
  renderConfusionMatrix();
  renderRoundMetricsTable();
  renderClientTable();
  renderFeatureList();
  renderSecurity();
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
    banner.textContent =
      `Adapter unreachable — ${state.error}. Start it with: python -m gui_integration.backend ` +
      `--allowed-origins ${location.origin}. Values below are the last known state and are not live.`;
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
      // Mine derived state immediately — the tables and charts should be correct
      // straight away even though the log itself is released at presentation pace.
      for (const event of events) mineFederatedEvent(event);

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

function setBusy(busy) {
  state.busy = busy;
  $('btn-single').disabled = busy || !state.connected;
  $('btn-replay').disabled = busy || !state.connected;
}

function renderReadout(vector, profileId) {
  const readout = $('inject-readout');
  const grid = $('readout-grid');
  readout.hidden = false;
  grid.textContent = '';

  for (const feature of READOUT_FEATURES) {
    const cell = el('div');
    cell.appendChild(el('dt', null, feature));
    const value = vector[feature];
    cell.appendChild(el('dd', null,
      typeof value === 'number' ? (Number.isInteger(value) ? num(value) : value.toPrecision(4)) : '—'));
    grid.appendChild(cell);
  }
  $('inject-state').textContent =
    (state.streaming ? 'sending · ' : 'sent one · ') + profileLabel(profileId);
}

async function injectOnce() {
  if (state.busy || !state.connected) return;
  setBusy(true);
  const profileId = state.profile;
  const vector = generateVector(profileId);
  try {
    const prediction = await api.predict({
      record_id: 'inj-' + Date.now().toString(36),
      source: 'sim:' + profileId,
      features: vector,
    });
    recordPrediction(prediction, { fromThisTab: true, vector });
    state.lastVector = vector;
    state.lastProfile = profileId;
    renderReadout(vector, profileId);
    render();
  } catch (error) {
    state.error = error.message;
    render();
  } finally {
    setBusy(false);
  }
}

async function replayOnce() {
  if (state.busy || !state.connected) return;
  setBusy(true);
  try {
    const result = await api.replayNext();
    recordPrediction(result.prediction, { fromThisTab: true });
    $('inject-state').textContent =
      `saved packet ${result.position} of ${result.total_records}`;
    render();
  } catch (error) {
    state.error = error.message;
    render();
  } finally {
    setBusy(false);
  }
}

function scheduleStream() {
  if (!state.streaming) return;
  state.streamTimer = setTimeout(async () => {
    await injectOnce();
    scheduleStream();
  }, state.intervalMs);
}

function startStream() {
  if (state.streaming) return;
  state.streaming = true;
  $('btn-stream').textContent = 'Stop sending';
  $('btn-stream').classList.add('is-running');
  $('inject-state').textContent = 'sending · ' + profileLabel(state.profile);
  injectOnce().then(scheduleStream);
}

function stopStream() {
  state.streaming = false;
  clearTimeout(state.streamTimer);
  $('btn-stream').textContent = 'Start sending';
  $('btn-stream').classList.remove('is-running');
  $('inject-state').textContent = 'idle';
}

/* ── Wiring ─────────────────────────────────────────────────────────── */

function buildProfileButtons() {
  const container = $('profile-buttons');
  container.textContent = '';
  for (const profile of PROFILES) {
    const button = el('button', 'profile-btn' + (state.profile === profile.id ? ' is-active' : ''),
      profile.name);
    button.addEventListener('click', () => {
      state.profile = profile.id;
      buildProfileButtons();
      $('profile-desc').textContent = profile.desc;
      if (state.streaming) $('inject-state').textContent = 'sending · ' + profileLabel(profile.id);
    });
    container.appendChild(button);
  }
  const active = PROFILES.find((p) => p.id === state.profile);
  $('profile-desc').textContent = active ? active.desc : '';
}

function switchView(view) {
  state.view = view;
  for (const button of document.querySelectorAll('.nav-item')) {
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
  for (const button of document.querySelectorAll('.nav-item')) {
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

  $('btn-stream').addEventListener('click', () => (state.streaming ? stopStream() : startStream()));
  $('btn-single').addEventListener('click', injectOnce);
  $('btn-replay').addEventListener('click', replayOnce);

  const rate = $('inject-rate');
  rate.addEventListener('input', () => {
    state.intervalMs = Number(rate.value);
    $('inject-rate-out').textContent = (state.intervalMs / 1000).toFixed(1) + ' s';
  });

  $('btn-clear-console').addEventListener('click', () => {
    state.inferenceLog = [];
    state.federatedLog = [];
    renderConsole();
  });

  buildProfileButtons();
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
