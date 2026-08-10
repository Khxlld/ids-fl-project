(() => {
  "use strict";

  const DEFAULT_CONFIG = {
    apiBase: "http://127.0.0.1:8090/api/gui/v1",
    pollMs: 1200,
    requestTimeoutMs: 1800,
  };

  const state = {
    config: { ...DEFAULT_CONFIG },
    replayData: null,
    mode: "recorded",
    activeView: "monitor",
    liveStatus: "checking",
    liveFailures: 0,
    snapshot: null,
    replayStageIndex: 0,
    replayPredictionCursor: 0,
    replayPlaying: false,
    replaySpeed: 1,
    replayTimer: null,
    liveTimer: null,
    predictionHistory: [],
    inferenceEvents: [],
    federatedEvents: [],
    seenInferenceEvents: new Set(),
    seenFederatedEvents: new Set(),
    inferenceCursor: 0,
    federatedCursor: 0,
    sessionStarted: Date.now(),
    busy: false,
  };

  const $ = (id) => document.getElementById(id);
  const all = (selector) => Array.from(document.querySelectorAll(selector));

  function setText(id, value) {
    const node = $(id);
    if (node) node.textContent = value ?? "Unavailable";
  }

  function finiteNumber(value) {
    return typeof value === "number" && Number.isFinite(value);
  }

  function pct(value, digits = 2) {
    return finiteNumber(value) ? `${(value * 100).toFixed(digits)}%` : "Unavailable";
  }

  function number(value, digits = 0) {
    return finiteNumber(value)
      ? value.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits })
      : "Unavailable";
  }

  function compactId(value) {
    if (!value) return "Unavailable";
    const text = String(value);
    return text.length > 22 ? `${text.slice(0, 10)}…${text.slice(-6)}` : text;
  }

  function formatTime(value) {
    if (!value) return "Recorded";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "Recorded";
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function humanize(value) {
    if (!value) return "Unavailable";
    return String(value)
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) => letter.toUpperCase());
  }

  function normalizedState(value) {
    const safe = String(value || "unavailable").toLowerCase().replace(/[^a-z0-9_-]/g, "_");
    const aliases = {
      registered: "ready",
      connected: "ready",
      client_ready: "ready",
      training_local: "training",
      update_sent: "update_received",
      update_received: "update_received",
      completed: "complete",
    };
    return aliases[safe] || safe;
  }

  function stateLabel(value) {
    const labels = {
      waiting: "Waiting",
      ready: "Ready",
      receiving: "Receiving model",
      training: "Local training",
      sending: "Sending update",
      update_received: "Update received",
      complete: "Complete",
      failed: "Failed",
      unavailable: "Unavailable",
    };
    const normalized = normalizedState(value);
    return labels[normalized] || humanize(normalized);
  }

  function eventKey(scope, event) {
    if (Number.isInteger(event?.seq)) return `${scope}:${event.seq}`;
    return `${scope}:${event?.event_type || "unknown"}:${event?.timestamp_utc || "none"}:${event?.client_id || "none"}`;
  }

  function appendPrediction(prediction) {
    if (!prediction || !["Normal", "Attack"].includes(prediction.label)) return;
    const id = prediction.prediction_id || `${prediction.record_id}:${prediction.timestamp_utc}`;
    if (state.predictionHistory.some((item) => (item.prediction_id || `${item.record_id}:${item.timestamp_utc}`) === id)) return;
    state.predictionHistory.push(prediction);
    if (state.predictionHistory.length > 30) state.predictionHistory.shift();
  }

  function appendInferenceEvents(events) {
    for (const event of Array.isArray(events) ? events : []) {
      const key = eventKey("inference", event);
      if (state.seenInferenceEvents.has(key)) continue;
      state.seenInferenceEvents.add(key);
      state.inferenceEvents.push(event);
      if (event.event_type === "prediction_completed") appendPrediction(event.payload);
    }
    if (state.inferenceEvents.length > 120) state.inferenceEvents = state.inferenceEvents.slice(-120);
  }

  function appendFederatedEvents(events) {
    for (const event of Array.isArray(events) ? events : []) {
      const key = eventKey("federated", event);
      if (state.seenFederatedEvents.has(key)) continue;
      state.seenFederatedEvents.add(key);
      state.federatedEvents.push(event);
    }
    state.federatedEvents.sort((a, b) => (a.seq || 0) - (b.seq || 0));
    if (state.federatedEvents.length > 160) state.federatedEvents = state.federatedEvents.slice(-160);
  }

  async function fetchJson(url, options = {}) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), state.config.requestTimeoutMs);
    try {
      const response = await fetch(url, {
        cache: "no-store",
        ...options,
        headers: {
          Accept: "application/json",
          ...(options.body ? { "Content-Type": "application/json" } : {}),
          ...(options.headers || {}),
        },
        signal: controller.signal,
      });
      const payload = await response.json().catch(() => null);
      if (!response.ok) {
        const message = payload?.error?.message || `Request failed with HTTP ${response.status}`;
        throw new Error(message);
      }
      if (!payload || typeof payload !== "object") throw new Error("The server returned an invalid JSON response.");
      return payload;
    } finally {
      window.clearTimeout(timeout);
    }
  }

  function api(path) {
    return `${state.config.apiBase.replace(/\/$/, "")}${path}`;
  }

  function showBanner(message, kind = "warning", actionLabel = "", action = null) {
    const banner = $("statusBanner");
    banner.classList.remove("hidden", "error");
    if (kind === "error") banner.classList.add("error");
    setText("statusBannerText", message);
    const button = $("statusBannerAction");
    if (actionLabel && typeof action === "function") {
      button.classList.remove("hidden");
      button.textContent = actionLabel;
      button.onclick = action;
    } else {
      button.classList.add("hidden");
      button.onclick = null;
    }
  }

  function hideBanner() {
    $("statusBanner").classList.add("hidden");
  }

  function visibleReplayEvents(index) {
    const sequences = new Set();
    for (let i = 0; i <= index; i += 1) {
      for (const seq of state.replayData.replay_stages[i].event_sequences || []) sequences.add(seq);
    }
    return state.replayData.federated_events.filter((event) => sequences.has(event.seq));
  }

  function buildReplaySnapshot() {
    const data = state.replayData;
    const stage = data.replay_stages[state.replayStageIndex];
    const clients = data.clients.map((client) => ({ ...client, state: stage.client_state }));
    const predictions = state.predictionHistory;
    const latest = predictions.at(-1) || null;
    const alerts = predictions.filter((item) => item.label === "Attack").slice(-20).reverse();
    const security = data.evidence.security;
    const rejectionCount = finiteNumber(stage.rejected_messages) ? stage.rejected_messages : 0;
    const authenticated = state.replayStageIndex >= 1 ? security.authenticated_clients : 0;
    const normalCount = predictions.filter((item) => item.label === "Normal").length;
    const attackCount = predictions.filter((item) => item.label === "Attack").length;

    return {
      schema_version: "uavids-gui-snapshot-v1",
      generated_utc: new Date().toISOString(),
      api_version: "uavids-gui-api-v1",
      presentation_mode: "replay",
      backend: {
        available: false,
        inference_mode: "recorded_results",
        federated_upstream_available: false,
      },
      model: { ...data.model, available: false },
      inference: {
        records_processed: predictions.length,
        normal_count: normalCount,
        attack_count: attackCount,
        latest_prediction: latest,
        recent_alerts: alerts,
      },
      federated: {
        data_mode: "replay",
        upstream_available: false,
        run_id: "23795a6c-4d27-4d27-96e1-1da6954d7c9c",
        state: stage.id === "complete" || state.replayStageIndex > 6 ? "completed" : stage.id,
        current_round: stage.round,
        total_rounds: 3,
        updates_received: stage.updates,
        updates_expected: 5,
        clients,
        local_data_statement: "Each logical client trains from its own partition; raw training rows are not sent to the server during federated rounds.",
        global_model_metrics: {
          ...data.evidence.demo_validation,
          source: data.evidence.demo_validation.source,
          note: data.evidence.demo_validation.note,
        },
        security: {
          mode: "secure",
          status: "recorded_verified",
          algorithms: security.algorithms,
          authenticated_clients: authenticated,
          rejected_messages: rejectionCount,
          note: "ML-KEM establishes key material; ML-DSA authenticates identities; AES-GCM protects model exchanges.",
        },
      },
    };
  }

  function applyReplayStage(index, { announce = false } = {}) {
    const max = state.replayData.replay_stages.length - 1;
    state.replayStageIndex = Math.min(Math.max(index, 0), max);
    const stage = state.replayData.replay_stages[state.replayStageIndex];
    if (Number.isInteger(stage.prediction_index)) appendPrediction(state.replayData.predictions[stage.prediction_index]);
    state.federatedEvents = visibleReplayEvents(state.replayStageIndex);
    state.snapshot = buildReplaySnapshot();
    if (announce) showBanner(`Recorded stage: ${stage.title}`, "warning");
    render();
  }

  function switchToRecorded({ preserveStage = true } = {}) {
    stopLivePolling();
    state.mode = "recorded";
    state.liveStatus = state.liveStatus === "checking" ? "offline" : state.liveStatus;
    state.inferenceCursor = 0;
    state.federatedCursor = 0;
    state.seenInferenceEvents.clear();
    state.seenFederatedEvents.clear();
    state.inferenceEvents = [];
    state.predictionHistory = [];
    state.replayPredictionCursor = 0;
    if (!preserveStage) state.replayStageIndex = 0;
    applyReplayStage(state.replayStageIndex);
  }

  async function tryLive({ userInitiated = false } = {}) {
    stopReplay();
    state.liveStatus = "checking";
    renderMode();
    if (userInitiated) showBanner("Checking the documented GUI adapter…", "warning");
    try {
      const [health, snapshot] = await Promise.all([
        fetchJson(api("/health")),
        fetchJson(api("/snapshot")),
      ]);
      if (health.ok !== true || health.model_available !== true || snapshot.backend?.available !== true) {
        throw new Error("The adapter is reachable, but its frozen model is unavailable.");
      }
      state.mode = "live";
      state.liveStatus = "connected";
      state.liveFailures = 0;
      state.snapshot = snapshot;
      state.predictionHistory = [];
      state.inferenceEvents = [];
      state.federatedEvents = [];
      state.seenInferenceEvents.clear();
      state.seenFederatedEvents.clear();
      state.inferenceCursor = 0;
      state.federatedCursor = 0;
      if (snapshot.inference?.latest_prediction) appendPrediction(snapshot.inference.latest_prediction);
      hideBanner();
      render();
      scheduleLivePoll(150);
    } catch (error) {
      state.liveStatus = "offline";
      if (state.mode !== "recorded") switchToRecorded();
      showBanner(
        `Live adapter unavailable: ${safeErrorMessage(error)} Recorded mode remains ready.`,
        "error",
        "Retry live",
        () => tryLive({ userInitiated: true }),
      );
      renderMode();
    }
  }

  function safeErrorMessage(error) {
    const message = error instanceof Error ? error.message : "Connection failed.";
    if (/abort/i.test(message)) return "request timed out.";
    return message.endsWith(".") ? message : `${message}.`;
  }

  function scheduleLivePoll(delay = state.config.pollMs) {
    stopLivePolling();
    state.liveTimer = window.setTimeout(pollLive, delay);
  }

  function stopLivePolling() {
    if (state.liveTimer) window.clearTimeout(state.liveTimer);
    state.liveTimer = null;
  }

  async function pollLive() {
    if (state.mode !== "live") return;
    try {
      const [snapshot, inferencePage, federatedPage] = await Promise.all([
        fetchJson(api("/snapshot")),
        fetchJson(api(`/events?after_seq=${state.inferenceCursor}`)),
        fetchJson(api(`/federated/events?after_seq=${state.federatedCursor}`)),
      ]);
      state.snapshot = snapshot;
      state.liveStatus = "connected";
      state.liveFailures = 0;
      appendInferenceEvents(inferencePage.events);
      appendFederatedEvents(federatedPage.events);
      if (Number.isInteger(inferencePage.last_seq)) state.inferenceCursor = Math.max(state.inferenceCursor, inferencePage.last_seq);
      if (Number.isInteger(federatedPage.last_seq)) state.federatedCursor = Math.max(state.federatedCursor, federatedPage.last_seq);
      if (snapshot.inference?.latest_prediction) appendPrediction(snapshot.inference.latest_prediction);
      hideBanner();
      render();
    } catch (error) {
      state.liveFailures += 1;
      state.liveStatus = "reconnecting";
      showBanner(
        `The live adapter stopped responding (${state.liveFailures}). The last valid state is frozen while reconnection continues.`,
        "error",
        "Use recorded demo",
        () => switchToRecorded(),
      );
      renderMode();
    } finally {
      if (state.mode === "live") scheduleLivePoll(state.liveFailures ? 2800 : state.config.pollMs);
    }
  }

  function stopReplay() {
    if (state.replayTimer) window.clearTimeout(state.replayTimer);
    state.replayTimer = null;
    state.replayPlaying = false;
    renderReplayControls();
  }

  function scheduleReplay() {
    if (!state.replayPlaying || state.mode !== "recorded") return;
    const delay = 3000 / state.replaySpeed;
    state.replayTimer = window.setTimeout(() => {
      const max = state.replayData.replay_stages.length - 1;
      if (state.replayStageIndex >= max) {
        stopReplay();
        return;
      }
      applyReplayStage(state.replayStageIndex + 1);
      scheduleReplay();
    }, delay);
  }

  function toggleReplay() {
    if (state.mode !== "recorded") return;
    state.replayPlaying = !state.replayPlaying;
    if (state.replayPlaying) scheduleReplay();
    else if (state.replayTimer) window.clearTimeout(state.replayTimer);
    renderReplayControls();
  }

  function restartReplay() {
    stopReplay();
    state.predictionHistory = [];
    state.replayPredictionCursor = 0;
    state.inferenceEvents = [];
    state.federatedEvents = [];
    applyReplayStage(0, { announce: true });
  }

  async function analyzeNext() {
    if (state.busy) return;
    state.busy = true;
    $("analyzeNextButton").disabled = true;
    try {
      if (state.mode === "live") {
        const response = await fetchJson(api("/replay/next"), {
          method: "POST",
          body: "{}",
        });
        appendPrediction(response.prediction);
        state.snapshot = await fetchJson(api("/snapshot"));
        showBanner("Verified replay input was evaluated now by the live frozen model.", "warning");
      } else {
        const prediction = state.replayData.predictions[state.replayPredictionCursor % state.replayData.predictions.length];
        state.replayPredictionCursor += 1;
        appendPrediction(prediction);
        const event = {
          seq: state.inferenceEvents.length + 1,
          event_type: "prediction_completed",
          severity: prediction.label === "Attack" ? "warning" : "info",
          timestamp_utc: prediction.timestamp_utc,
          source: prediction.source,
          payload: prediction,
          recorded: true,
        };
        state.inferenceEvents.push(event);
        state.snapshot = buildReplaySnapshot();
      }
      render();
    } catch (error) {
      showBanner(`Prediction request failed: ${safeErrorMessage(error)}`, "error");
    } finally {
      state.busy = false;
      $("analyzeNextButton").disabled = false;
    }
  }

  function renderMode() {
    const dot = $("connectionDot");
    dot.className = "connection-dot";
    const isLive = state.mode === "live";
    $("liveModeButton").classList.toggle("active", isLive);
    $("replayModeButton").classList.toggle("active", !isLive);
    $("replayControls").classList.toggle("inactive", isLive);

    if (isLive && state.liveStatus === "connected") {
      dot.classList.add("live");
      setText("modeTitle", "Live Adapter");
      const telemetry = state.snapshot?.presentation_mode === "live" ? "live federated telemetry" : "recorded federated telemetry";
      setText("modeDetail", `Real model available • ${telemetry}`);
    } else if (isLive) {
      dot.classList.add("offline");
      setText("modeTitle", "Live Adapter");
      setText("modeDetail", state.liveStatus === "checking" ? "Checking connection…" : "Reconnecting… last state frozen");
    } else {
      if (state.liveStatus === "offline") dot.classList.add("offline");
      setText("modeTitle", "Recorded Demo");
      const stage = state.replayData?.replay_stages?.[state.replayStageIndex];
      setText("modeDetail", stage ? `Clearly labelled replay • ${stage.title}` : "Loading verified evidence…");
    }

    const model = state.snapshot?.model || state.replayData?.model;
    setText("headerModel", model?.model_id || "Unavailable");
    setText("footerSource", isLive ? `Source: ${state.config.apiBase}` : "Source: verified recorded project evidence");
  }

  function renderMonitor() {
    const snapshot = state.snapshot;
    const inference = snapshot?.inference || {};
    const model = snapshot?.model || state.replayData.model;
    const latest = inference.latest_prediction || state.predictionHistory.at(-1) || null;
    const panel = $("verdictPanel");
    panel.classList.remove("state-empty", "state-normal", "state-attack");

    if (!latest) {
      panel.classList.add("state-empty");
      setText("predictionSourceChip", state.mode === "live" ? "Live model ready" : "Recorded evidence ready");
      setText("confidenceValue", "—");
      setText("verdictKicker", "Awaiting inference");
      setText("verdictLabel", "No verdict");
      setText("verdictExplanation", "Use the verified sample control to exercise the deployed binary model.");
      setText("latestRecord", "Unavailable");
      setText("latestSource", "Unavailable");
      setText("attackProbability", "Unavailable");
      $("probabilityOrbit").style.setProperty("--probability", "0");
    } else {
      const isAttack = latest.label === "Attack";
      panel.classList.add(isAttack ? "state-attack" : "state-normal");
      const modeLabel = state.mode === "live"
        ? (latest.replayed ? "Replayed input • live model" : "Live input")
        : "Recorded real-model result";
      setText("predictionSourceChip", modeLabel);
      setText("confidenceValue", pct(latest.confidence, 1));
      setText("verdictKicker", isAttack ? "Threshold crossed" : "Below attack threshold");
      setText("verdictLabel", latest.label);
      setText(
        "verdictExplanation",
        isAttack
          ? `Attack probability reached ${pct(latest.attack_probability, 1)}, above the frozen ${number(latest.decision_threshold, 2)} threshold.`
          : `Attack probability stayed at ${pct(latest.attack_probability, 1)}, below the frozen ${number(latest.decision_threshold, 2)} threshold.`,
      );
      setText("latestRecord", latest.record_id || "Unavailable");
      setText("latestSource", latest.source || "Unavailable");
      setText("attackProbability", pct(latest.attack_probability, 2));
      $("probabilityOrbit").style.setProperty("--probability", String(Math.max(0, Math.min(1, latest.confidence || 0))));
    }

    const inferenceMode = snapshot?.backend?.inference_mode;
    setText(
      "inferenceModeChip",
      inferenceMode === "live_model" ? "LIVE FROZEN MODEL" : (state.mode === "recorded" ? "RECORDED RESULT" : "Unavailable"),
    );
    setText("recordsProcessed", number(inference.records_processed ?? state.predictionHistory.length));
    setText("normalCount", number(inference.normal_count ?? state.predictionHistory.filter((item) => item.label === "Normal").length));
    setText("attackCount", number(inference.attack_count ?? state.predictionHistory.filter((item) => item.label === "Attack").length));
    setText("thresholdValue", finiteNumber(model?.decision_threshold) ? model.decision_threshold.toFixed(2) : "Unavailable");
    renderProbabilityChart(model?.decision_threshold ?? 0.42);
    renderAlerts(inference.recent_alerts || []);
  }

  function renderProbabilityChart(threshold) {
    const svgGroup = $("probabilitySeries");
    while (svgGroup.firstChild) svgGroup.removeChild(svgGroup.firstChild);
    const history = state.predictionHistory.slice(-12);
    $("emptyChart").classList.toggle("hidden", history.length > 0);
    if (!history.length) return;

    const ns = "http://www.w3.org/2000/svg";
    const minX = 58;
    const maxX = 610;
    const minY = 24;
    const maxY = 206;
    const x = (index) => history.length === 1 ? (minX + maxX) / 2 : minX + (index * (maxX - minX)) / (history.length - 1);
    const y = (probability) => maxY - Math.max(0, Math.min(1, probability || 0)) * (maxY - minY);
    const points = history.map((item, index) => `${x(index)},${y(item.attack_probability)}`).join(" ");

    const polyline = document.createElementNS(ns, "polyline");
    polyline.setAttribute("points", points);
    polyline.setAttribute("class", "probability-path");
    svgGroup.append(polyline);

    history.forEach((item, index) => {
      const circle = document.createElementNS(ns, "circle");
      circle.setAttribute("cx", String(x(index)));
      circle.setAttribute("cy", String(y(item.attack_probability)));
      circle.setAttribute("r", "6");
      circle.setAttribute("class", "probability-dot");
      circle.setAttribute("fill", item.label === "Attack" ? "#ff6e78" : "#24c59b");
      const title = document.createElementNS(ns, "title");
      title.textContent = `${item.record_id || "record"}: ${pct(item.attack_probability, 2)} attack probability`;
      circle.append(title);
      svgGroup.append(circle);
    });

    const thresholdY = y(threshold);
    const thresholdLine = document.querySelector(".threshold-line");
    const thresholdLabel = document.querySelector(".threshold-label");
    thresholdLine.setAttribute("y1", String(thresholdY));
    thresholdLine.setAttribute("y2", String(thresholdY));
    thresholdLabel.setAttribute("y", String(thresholdY - 8));
  }

  function renderAlerts(alerts) {
    const list = $("alertList");
    list.replaceChildren();
    const safeAlerts = (Array.isArray(alerts) ? alerts : []).filter((item) => item?.label === "Attack").slice(0, 20);
    setText("alertCount", String(safeAlerts.length));
    if (!safeAlerts.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      const symbol = document.createElement("span");
      symbol.className = "empty-symbol";
      symbol.textContent = "✓";
      const strong = document.createElement("strong");
      strong.textContent = "No attack alerts yet";
      const copy = document.createElement("p");
      copy.textContent = "Alerts will appear when the model crosses its frozen threshold.";
      empty.append(symbol, strong, copy);
      list.append(empty);
      return;
    }
    for (const alert of safeAlerts) {
      const item = document.createElement("article");
      item.className = "alert-item";
      const marker = document.createElement("span");
      marker.className = "alert-marker";
      const main = document.createElement("div");
      main.className = "alert-main";
      const title = document.createElement("strong");
      title.textContent = alert.record_id || "Attack prediction";
      const source = document.createElement("span");
      source.textContent = alert.source || "Source unavailable";
      main.append(title, source);
      const side = document.createElement("div");
      const confidence = document.createElement("div");
      confidence.className = "alert-confidence";
      confidence.textContent = pct(alert.confidence, 1);
      const time = document.createElement("time");
      time.className = "alert-time";
      time.textContent = formatTime(alert.timestamp_utc);
      side.append(confidence, time);
      item.append(marker, main, side);
      list.append(item);
    }
  }

  function liveStageDescription(federated) {
    const current = federated?.state || "unknown";
    const titles = {
      waiting_for_clients: "Waiting for five clients",
      running: `Federated round ${federated?.current_round || 0}`,
      aggregating: "FedAvg aggregation in progress",
      completed: "Federated demonstration complete",
      failed: "Federated demonstration stopped",
    };
    return {
      title: titles[current] || humanize(current),
      caption: federated?.upstream_available
        ? "This state is being read from the live federated backend."
        : "The adapter is supplying verified recorded federated telemetry.",
    };
  }

  function renderFederation() {
    const federated = state.snapshot?.federated || {};
    const security = federated.security || {};
    const replayClients = state.replayData.clients;
    const supplied = Array.isArray(federated.clients) ? federated.clients : [];
    const clients = replayClients.map((known) => {
      const current = supplied.find((item) => item.client_id === known.client_id);
      return { ...known, ...(current || {}), state: current?.state || "unavailable" };
    });

    const stack = $("clientStack");
    stack.replaceChildren();
    clients.forEach((client, index) => {
      const node = document.createElement("article");
      const clientState = normalizedState(client.state);
      node.className = `client-node ${clientState}`;
      const icon = document.createElement("span");
      icon.className = "client-icon";
      icon.textContent = String(index + 1).padStart(2, "0");
      const copy = document.createElement("div");
      copy.className = "client-copy";
      const name = document.createElement("strong");
      name.textContent = client.client_id;
      const detail = document.createElement("span");
      detail.textContent = `${number(client.samples)} local samples`;
      copy.append(name, detail);
      const badge = document.createElement("span");
      badge.className = "state-badge";
      badge.textContent = stateLabel(clientState);
      node.append(icon, copy, badge);
      stack.append(node);
    });

    setText("localDataStatement", federated.local_data_statement || "Local data statement unavailable.");
    setText("federationRunId", `Run ${compactId(federated.run_id)}`);
    setText("currentRound", number(federated.current_round));
    setText("totalRounds", number(federated.total_rounds));
    setText("updatesReceived", number(federated.updates_received));
    setText("updatesExpected", number(federated.updates_expected));
    const expected = Number(federated.updates_expected) || 0;
    const received = Number(federated.updates_received) || 0;
    $("updateFill").style.width = `${expected ? Math.min(100, (received / expected) * 100) : 0}%`;
    setText("aggregationState", humanize(federated.state || "waiting"));
    setText("exchangeMode", security.mode === "plain" ? "Plain comparison" : "Secure exchange");

    const stage = state.mode === "recorded"
      ? state.replayData.replay_stages[state.replayStageIndex]
      : liveStageDescription(federated);
    setText("replayStageIndex", state.mode === "recorded" ? `STAGE ${String(state.replayStageIndex + 1).padStart(2, "0")}` : "LIVE STATE");
    setText("replayStageTitle", stage.title);
    setText("replayStageCaption", stage.caption);

    const metrics = federated.global_model_metrics || {};
    setText("demoMacroF1", pct(metrics.macro_f1, 2));
    setText("demoAttackRecall", pct(metrics.attack_recall, 2));
    setText("demoFpr", pct(metrics.fpr, 2));
    setText("metricsNote", metrics.note || "Validation telemetry unavailable.");
    setText("metricsModeChip", federated.data_mode === "live" ? "Live demo validation" : "Recorded validation");
    renderFederatedEvents();
  }

  function renderFederatedEvents() {
    const list = $("federatedEventList");
    list.replaceChildren();
    const events = state.federatedEvents.filter((event) => event.event_type !== "security_message_rejected").slice(-7).reverse();
    setText("federatedEventCount", `${state.federatedEvents.length} events`);
    if (!events.length) {
      const empty = document.createElement("p");
      empty.className = "metric-note";
      empty.textContent = "No federated events are available at this stage.";
      list.append(empty);
      return;
    }
    for (const event of events) {
      const row = document.createElement("div");
      row.className = "compact-event";
      const seq = document.createElement("b");
      seq.textContent = String(event.seq || "—").padStart(2, "0");
      const copy = document.createElement("div");
      const title = document.createElement("strong");
      title.textContent = humanize(event.event_type);
      const detail = document.createElement("span");
      detail.textContent = event.summary || [event.client_id, event.round ? `round ${event.round}` : null].filter(Boolean).join(" • ") || "Event details unavailable";
      copy.append(title, detail);
      row.append(seq, copy);
      list.append(row);
    }
  }

  function rejectionCounts() {
    const counts = {};
    for (const event of state.federatedEvents) {
      if (event.event_type !== "security_message_rejected") continue;
      const category = event.category || event.payload?.category;
      if (!category) continue;
      counts[category] = (counts[category] || 0) + 1;
    }
    return counts;
  }

  function renderSecurity() {
    const security = state.snapshot?.federated?.security || {};
    const isPlain = security.mode === "plain";
    setText("securityModeBadge", isPlain ? "Plain comparison mode" : (state.mode === "live" ? "Secure mode • live" : "Secure mode • recorded"));
    $("securityModeBadge").classList.toggle("plain", isPlain);
    setText("authenticatedClients", isPlain ? "Not applicable" : `${number(security.authenticated_clients)} / 5`);
    setText("rejectedMessages", number(security.rejected_messages));
    setText("securityEventMode", state.mode === "live" ? "Live events" : "Recorded events");

    const counts = rejectionCounts();
    const bars = $("rejectionBars");
    bars.replaceChildren();
    const categories = Object.keys(state.replayData.evidence.security.rejection_categories);
    const max = Math.max(1, ...categories.map((category) => counts[category] || 0));
    for (const category of categories) {
      const count = counts[category] || 0;
      const row = document.createElement("div");
      row.className = "rejection-row";
      const label = document.createElement("span");
      label.textContent = humanize(category);
      const track = document.createElement("div");
      const fill = document.createElement("i");
      fill.style.width = `${(count / max) * 100}%`;
      track.append(fill);
      const value = document.createElement("b");
      value.textContent = String(count);
      row.append(label, track, value);
      bars.append(row);
    }
    renderSecurityEvents();
  }

  function renderSecurityEvents() {
    const list = $("securityEventList");
    list.replaceChildren();
    const events = state.federatedEvents.filter((event) => event.event_type === "security_message_rejected").slice(-7).reverse();
    if (!events.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.style.minHeight = "210px";
      const symbol = document.createElement("span");
      symbol.className = "empty-symbol";
      symbol.textContent = "—";
      const title = document.createElement("strong");
      title.textContent = "No rejection event at this stage";
      const body = document.createElement("p");
      body.textContent = state.mode === "recorded" ? "Advance to the security-probe stage." : "The backend has not reported a rejection.";
      empty.append(symbol, title, body);
      list.append(empty);
      return;
    }
    for (const event of events) {
      const row = document.createElement("div");
      row.className = "security-event";
      const marker = document.createElement("i");
      const copy = document.createElement("div");
      const title = document.createElement("strong");
      const category = event.category || event.payload?.category;
      title.textContent = humanize(category || "message rejected");
      const detail = document.createElement("span");
      detail.textContent = event.summary || "Rejected safely before aggregation.";
      copy.append(title, detail);
      const seq = document.createElement("time");
      seq.textContent = `#${event.seq || "—"}`;
      row.append(marker, copy, seq);
      list.append(row);
    }
  }

  function renderEvidence() {
    const evidence = state.replayData.evidence;
    const locked = evidence.locked_test;
    const local = locked.local_mean_macro_f1;
    const federated = locked.federated.macro_f1;
    const centralized = locked.centralized.macro_f1;
    $("localComparisonBar").style.width = `${local * 100}%`;
    $("federatedComparisonBar").style.width = `${federated * 100}%`;
    $("centralizedComparisonBar").style.width = `${centralized * 100}%`;
    setText("localComparisonValue", pct(local, 2));
    setText("federatedComparisonValue", pct(federated, 2));
    setText("centralizedComparisonValue", pct(centralized, 2));
    setText("federatedGain", `${((federated - local) * 100).toFixed(2)} points`);
    setText("centralizedGap", `${((centralized - federated) * 100).toFixed(2)} points`);
    setText("lockedAccuracy", pct(locked.federated.accuracy, 2));
    setText("lockedAttackPrecision", pct(locked.federated.attack_precision, 2));
    setText("lockedAttackRecall", pct(locked.federated.attack_recall, 2));
    setText("lockedFpr", pct(locked.federated.fpr, 2));
    setText("lockedTestNote", locked.note);

    const security = evidence.security;
    setText("aggregationDifference", number(security.maximum_plain_secure_difference, 1));
    setText("messageExpansion", `${security.mean_message_expansion_percent.toFixed(1)}%`);
    renderTimingList(security.operation_timings_ms);
  }

  function renderTimingList(timings) {
    const list = $("timingList");
    list.replaceChildren();
    const order = ["encapsulation", "decapsulation", "signing", "verification", "encryption", "decryption"];
    const max = Math.max(...order.map((key) => timings[key] || 0), 1);
    for (const key of order) {
      const value = timings[key];
      const row = document.createElement("div");
      row.className = "timing-row";
      const label = document.createElement("span");
      label.textContent = humanize(key);
      const track = document.createElement("div");
      const fill = document.createElement("i");
      fill.style.width = `${(value / max) * 100}%`;
      track.append(fill);
      const display = document.createElement("b");
      display.textContent = `${value.toFixed(3)} ms`;
      row.append(label, track, display);
      list.append(row);
    }
  }

  function renderReplayControls() {
    const stages = state.replayData?.replay_stages || [];
    const count = stages.length;
    setText("replayPlayIcon", state.replayPlaying ? "Ⅱ" : "▶");
    setText("replayPlayLabel", state.replayPlaying ? "Pause" : "Play");
    setText("stageCounter", `${state.replayStageIndex + 1} / ${count || 1}`);
    $("stageFill").style.width = `${count ? ((state.replayStageIndex + 1) / count) * 100 : 0}%`;
    $("replayPrevious").disabled = state.mode !== "recorded" || state.replayStageIndex === 0;
    $("replayNext").disabled = state.mode !== "recorded" || state.replayStageIndex >= count - 1;
  }

  function render() {
    renderMode();
    renderReplayControls();
    renderMonitor();
    renderFederation();
    renderSecurity();
    renderEvidence();
    document.documentElement.dataset.appReady = "true";
    document.documentElement.dataset.mode = state.mode;
  }

  function selectView(view) {
    if (!["monitor", "federation", "security", "evidence"].includes(view)) return;
    state.activeView = view;
    all(".view-button").forEach((button) => button.classList.toggle("active", button.dataset.view === view));
    all(".view").forEach((section) => section.classList.toggle("active", section.id === `view-${view}`));
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function bindControls() {
    all(".view-button").forEach((button) => button.addEventListener("click", () => selectView(button.dataset.view)));
    $("liveModeButton").addEventListener("click", () => tryLive({ userInitiated: true }));
    $("replayModeButton").addEventListener("click", () => {
      hideBanner();
      switchToRecorded();
    });
    $("replayPlay").addEventListener("click", toggleReplay);
    $("replayPrevious").addEventListener("click", () => {
      stopReplay();
      applyReplayStage(state.replayStageIndex - 1);
    });
    $("replayNext").addEventListener("click", () => {
      stopReplay();
      applyReplayStage(state.replayStageIndex + 1);
    });
    $("replayRestart").addEventListener("click", restartReplay);
    $("replaySpeed").addEventListener("change", (event) => {
      state.replaySpeed = Number(event.target.value) || 1;
      if (state.replayPlaying) {
        if (state.replayTimer) window.clearTimeout(state.replayTimer);
        scheduleReplay();
      }
    });
    $("analyzeNextButton").addEventListener("click", analyzeNext);
  }

  function updateClock() {
    const elapsed = Math.max(0, Date.now() - state.sessionStarted);
    const totalSeconds = Math.floor(elapsed / 1000);
    const hours = String(Math.floor(totalSeconds / 3600)).padStart(2, "0");
    const minutes = String(Math.floor((totalSeconds % 3600) / 60)).padStart(2, "0");
    const seconds = String(totalSeconds % 60).padStart(2, "0");
    setText("sessionClock", `${hours}:${minutes}:${seconds}`);
  }

  async function loadConfig() {
    try {
      const runtime = await fetchJson("/runtime-config.json");
      state.config = { ...DEFAULT_CONFIG, ...runtime };
    } catch {
      state.config = { ...DEFAULT_CONFIG };
    }
  }

  async function init() {
    bindControls();
    window.setInterval(updateClock, 1000);
    updateClock();
    await loadConfig();
    state.replayData = await fetchJson("data/replay.json");
    if (state.replayData.schema_version !== "anas-attempt-recorded-demo-v1") {
      throw new Error("Recorded demonstration schema is incompatible.");
    }
    const query = new URLSearchParams(window.location.search);
    const requestedStage = Number.parseInt(query.get("stage") || "0", 10);
    const initialStage = Number.isInteger(requestedStage)
      ? Math.min(Math.max(requestedStage, 0), state.replayData.replay_stages.length - 1)
      : 0;
    applyReplayStage(initialStage);
    const requestedView = query.get("view");
    if (["monitor", "federation", "security", "evidence"].includes(requestedView)) selectView(requestedView);
    if (query.get("mode") !== "recorded") await tryLive();
  }

  window.addEventListener("error", () => {
    document.documentElement.dataset.runtimeError = "true";
  });
  window.addEventListener("unhandledrejection", () => {
    document.documentElement.dataset.runtimeError = "true";
  });

  init().catch((error) => {
    document.documentElement.dataset.appReady = "false";
    document.documentElement.dataset.runtimeError = "true";
    showBanner(`The presentation console could not start: ${safeErrorMessage(error)}`, "error");
  });
})();
