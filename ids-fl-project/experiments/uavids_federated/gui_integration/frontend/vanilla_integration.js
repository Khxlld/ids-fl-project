// Minimal framework-neutral integration example. Adapt state rendering to your UI.
const API_BASE = window.UAVIDS_API_BASE || "http://127.0.0.1:8090/api/gui/v1";

let inferenceCursor = 0;
let federatedCursor = 0;

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (options.body !== undefined) headers["Content-Type"] = "application/json";
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
  });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error?.message || `HTTP ${response.status}`);
  return value;
}

export async function getSnapshot() {
  return api("/snapshot");
}

export async function classifyRecord(recordId, source, features) {
  return api("/predictions", {
    method: "POST",
    body: JSON.stringify({ record_id: recordId, source, features }),
  });
}

export async function replayNext() {
  return api("/replay/next", { method: "POST", body: "{}" });
}

export async function pollEvents(onInferenceEvents, onFederatedEvents) {
  const [inference, federated] = await Promise.all([
    api(`/events?after_seq=${inferenceCursor}`),
    api(`/federated/events?after_seq=${federatedCursor}`),
  ]);
  inferenceCursor = inference.last_seq;
  federatedCursor = federated.last_seq;
  onInferenceEvents(inference.events, inference);
  onFederatedEvents(federated.events, federated);
}

// Example polling loop with explicit disconnect handling.
export function startPolling(renderSnapshot, renderError) {
  let stopped = false;
  async function tick() {
    try {
      renderSnapshot(await getSnapshot());
    } catch (error) {
      renderError({ backendAvailable: false, message: String(error) });
    } finally {
      if (!stopped) window.setTimeout(tick, 1000);
    }
  }
  tick();
  return () => { stopped = true; };
}
