'use strict';

const params = new URLSearchParams(location.search);
const API_BASE = params.get('api') || 'http://127.0.0.1:8090/api/gui/v1';
const $ = (id) => document.getElementById(id);
const CLIENT_IDS = Array.from({ length: 5 }, (_, i) => `uav-client-${i + 1}`);

const PROFILES = {
  'uav-client-1': { device: 'Raspberry Pi 4 Model B', spec: 'Quad Cortex-A72; 1/2/4/8 GB variants', cpu: '0.65 CPU', memory: '512 MiB', samples: 1230, source: '192.168.0.26', counts: [597,255,17,23,338] },
  'uav-client-2': { device: 'NVIDIA Jetson Nano', spec: 'Quad Cortex-A57 up to 1.43 GHz; 4 GB LPDDR4', cpu: '0.55 CPU', memory: '512 MiB', samples: 1046, source: '192.168.0.25', counts: [264,367,42,25,348] },
  'uav-client-3': { device: 'Raspberry Pi Zero 2 W', spec: 'Quad Cortex-A53 at 1 GHz; 512 MB LPDDR2', cpu: '0.35 CPU', memory: '448 MiB', samples: 591, source: '192.168.0.100', counts: [145,161,3,162,120] },
  'uav-client-4': { device: 'NVIDIA Jetson Orin Nano 8GB', spec: 'Six-core Cortex-A78AE; 8 GB module', cpu: '1.00 CPU', memory: '768 MiB', samples: 2027, source: '192.168.0.5', counts: [201,329,1429,0,68] },
  'uav-client-5': { device: 'NXP NavQPlus', spec: 'i.MX 8M Plus; Dronecode connectors; 8 GB LPDDR4', cpu: '0.45 CPU', memory: '512 MiB', samples: 1254, source: '192.168.0.32', counts: [619,252,15,0,368] },
};
const DIST_LABELS = ['Normal','Blackhole','Flooding','Sybil','Wormhole'];
const state = { connected: false, snapshot: null, catalog: null, selected: null, busy: false };

async function request(path, options = {}) {
  const response = await fetch(API_BASE + path, {
    ...options,
    headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ? body.error.message : `HTTP ${response.status}`);
  return body;
}

function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
function svgEl(name, attrs) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function clientState(id) {
  const clients = state.snapshot && state.snapshot.federated ? state.snapshot.federated.clients : [];
  const item = clients.find((client) => client.client_id === id);
  return item ? item.state : 'not reported';
}

function drawTopology() {
  const host = $('attacker-topology');
  host.textContent = '';
  const svg = svgEl('svg', { viewBox: '0 0 760 410', role: 'img' });
  const center = { x: 380, y: 205 };
  const points = [{x:380,y:55},{x:635,y:145},{x:540,y:350},{x:220,y:350},{x:125,y:145}];
  points.forEach((point) => svg.appendChild(svgEl('line', {
    x1: center.x, y1: center.y, x2: point.x, y2: point.y,
    stroke: css('--line'), 'stroke-width': 2,
  })));
  const hub = svgEl('g', {});
  hub.appendChild(svgEl('circle', { cx:center.x, cy:center.y, r:54, fill:css('--surface-sunk'), stroke:css('--accent'), 'stroke-width':3 }));
  const hubTitle = svgEl('text', { x:center.x, y:center.y - 3, 'text-anchor':'middle', fill:css('--ink'), 'font-size':14, 'font-weight':800 });
  hubTitle.textContent = 'CONTROL CENTER'; hub.appendChild(hubTitle);
  const hubSub = svgEl('text', { x:center.x, y:center.y + 17, 'text-anchor':'middle', fill:css('--ink-muted'), 'font-size':11 });
  hubSub.textContent = 'FedAvg'; hub.appendChild(hubSub); svg.appendChild(hub);

  CLIENT_IDS.forEach((id, index) => {
    const point = points[index];
    const group = svgEl('g', { class: 'topology-client' + (state.selected === id ? ' is-selected' : ''), tabindex: 0, role: 'button', 'aria-label': `Select ${id}` });
    group.appendChild(svgEl('circle', { cx:point.x, cy:point.y, r:43, fill:css('--surface'), stroke: state.selected === id ? css('--accent') : css('--mark-normal'), 'stroke-width':3 }));
    const title = svgEl('text', { x:point.x, y:point.y - 3, 'text-anchor':'middle', fill:css('--ink'), 'font-size':13, 'font-weight':800 });
    title.textContent = `CLIENT ${index + 1}`; group.appendChild(title);
    const status = svgEl('text', { x:point.x, y:point.y + 17, 'text-anchor':'middle', fill:css('--ink-muted'), 'font-size':10 });
    status.textContent = clientState(id).replaceAll('_',' '); group.appendChild(status);
    group.addEventListener('click', () => selectClient(id));
    group.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') selectClient(id); });
    svg.appendChild(group);
  });
  host.appendChild(svg);
}

function selectClient(id) {
  state.selected = id;
  $('selected-target').textContent = id;
  $('attack-type').disabled = false;
  $('send-attack').disabled = !state.connected || state.busy;
  renderClient();
  drawTopology();
}

function renderClient() {
  if (!state.selected) return;
  const p = PROFILES[state.selected];
  const host = $('attacker-client-detail');
  host.textContent = '';
  const head = document.createElement('div'); head.className = 'client-title';
  const heading = document.createElement('div');
  const h2 = document.createElement('h2'); h2.textContent = state.selected;
  const sub = document.createElement('p'); sub.textContent = p.device + ' inspired profile';
  heading.append(h2, sub);
  const badge = document.createElement('span'); badge.className = 'client-state'; badge.textContent = clientState(state.selected).replaceAll('_',' ');
  head.append(heading, badge); host.appendChild(head);
  const facts = document.createElement('div'); facts.className = 'client-facts';
  [['Manufacturer specification',p.spec],['Docker CPU cap',p.cpu],['Docker memory cap',p.memory],['Local training rows',p.samples.toLocaleString()],['Training source key',p.source],['Hardware claim','Device-inspired, not emulated']].forEach(([key,value]) => {
    const cell = document.createElement('div'); cell.className = 'client-fact';
    const k = document.createElement('span'); k.textContent = key; const v = document.createElement('strong'); v.textContent = value;
    cell.append(k,v); facts.appendChild(cell);
  });
  host.appendChild(facts);
  const dist = document.createElement('div'); dist.className = 'distribution';
  const title = document.createElement('h3'); title.textContent = 'Local training distribution'; dist.appendChild(title);
  const max = Math.max(...p.counts);
  p.counts.forEach((count,index) => {
    const row = document.createElement('div'); row.className = 'dist-row';
    const label = document.createElement('span'); label.textContent = DIST_LABELS[index];
    const bar = document.createElement('span'); bar.className = 'dist-bar'; const fill = document.createElement('i'); fill.style.width = `${100 * count / max}%`; bar.appendChild(fill);
    const value = document.createElement('b'); value.textContent = count.toLocaleString(); row.append(label,bar,value); dist.appendChild(row);
  });
  host.appendChild(dist);
}

function renderResult(prediction) {
  const host = $('attack-result');
  const d = prediction.dataset; const demo = prediction.demo;
  host.hidden = false; host.textContent = '';
  host.className = 'panel attack-result-panel ' + (demo.outcome === 'detected' ? 'is-detected' : 'is-missed');
  const head = document.createElement('div'); head.className = 'result-head';
  const title = document.createElement('h2'); title.textContent = `${d.ground_truth_label} sent toward ${demo.target_client_id}`;
  const verdict = document.createElement('span'); verdict.className = 'result-verdict'; verdict.textContent = `Model: ${prediction.label} · ${demo.outcome.toUpperCase().replace('_',' ')}`;
  head.append(title, verdict); host.appendChild(head);
  const grid = document.createElement('div'); grid.className = 'result-grid';
  [['Flow ID',d.flow_id],['Recorded source',d.recorded_source],['Recorded destination',d.recorded_destination],['Attack probability',(prediction.attack_probability*100).toFixed(2)+'%'],['Frozen threshold',prediction.decision_threshold],['Model ID',prediction.model_id]].forEach(([key,value]) => {
    const cell = document.createElement('div'); const k = document.createElement('span'); k.textContent = key; const v = document.createElement('strong'); v.textContent = value; cell.append(k,v); grid.appendChild(cell);
  });
  host.appendChild(grid);
}

async function sendAttack() {
  if (!state.selected || state.busy || !state.connected) return;
  state.busy = true; $('send-attack').disabled = true; $('send-attack').textContent = 'Scoring held-out flow…';
  try {
    const prediction = await request('/demo/attacks', { method:'POST', body:JSON.stringify({ client_id:state.selected, attack_type:$('attack-type').value }) });
    renderResult(prediction); $('attacker-error').hidden = true;
    await refresh();
  } catch (error) { $('attacker-error').hidden = false; $('attacker-error').textContent = error.message; }
  finally { state.busy = false; $('send-attack').disabled = !state.selected || !state.connected; $('send-attack').textContent = 'Send Next Held-out Attack Flow'; }
}

async function refresh() {
  try {
    state.snapshot = await request('/snapshot'); state.connected = true;
    $('attacker-status').className = 'attacker-status is-up'; $('attacker-status').lastChild.textContent = 'Adapter connected · frozen model ready';
    const fed = state.snapshot.federated; $('attacker-round').textContent = `round ${fed.current_round} / ${fed.total_rounds} · ${fed.data_mode}`;
    const remaining = state.snapshot.dataset_demo.remaining[$('attack-type').value];
    $('attack-pool-status').textContent = remaining == null ? 'locked test split' : `${remaining} unused ${$('attack-type').value.replace(' Attack','')} rows`;
    $('send-attack').disabled = !state.selected || state.busy;
    drawTopology(); if (state.selected) renderClient();
  } catch (error) {
    state.connected = false; $('attacker-status').className = 'attacker-status is-down'; $('attacker-status').lastChild.textContent = 'Adapter unavailable'; $('send-attack').disabled = true;
    $('attacker-error').hidden = false; $('attacker-error').textContent = error.message;
  }
}

async function boot() {
  try {
    state.catalog = await request('/demo/catalog');
    const select = $('attack-type');
    state.catalog.attack_types.forEach((type) => { const option = document.createElement('option'); option.value = type; option.textContent = type; select.appendChild(option); });
    select.addEventListener('change', refresh);
  } catch (error) { $('attacker-error').hidden = false; $('attacker-error').textContent = error.message; }
  $('send-attack').addEventListener('click', sendAttack);
  $('attacker-theme').addEventListener('click', () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next); try { localStorage.setItem('rasid-theme', next); } catch (e) {} drawTopology();
  });
  await refresh(); setInterval(refresh, 1200);
}

document.addEventListener('DOMContentLoaded', boot);
