# RASID — Federated Intrusion Detection Console

Operator-facing GUI for the binary UAV intrusion-detection model, the five-client
federated demonstration, and the post-quantum secure transport.

It is a **pure client**. It talks only to the documented GUI adapter
(`uavids-gui-api-v1`, see [`../API_CONTRACT.md`](../API_CONTRACT.md)) and never
loads a checkpoint, preprocessing object, dataset, partition, key, signature, or
ciphertext. No verdict, probability, or metric is ever computed in the browser.

Zero dependencies — plain HTML, CSS and JavaScript served by a stdlib Python
static server. Nothing to install, nothing to build.

## Run it

Two processes: the adapter (port 8090) and this console (port 3000).

```bash
# terminal 1 — the adapter, from experiments/uavids_federated
python -m gui_integration.backend --host 127.0.0.1 --port 8090 \
    --allowed-origins http://localhost:3000

# terminal 2 — this console
python3 serve.py
```

Then open **http://localhost:3000**.

> Use `localhost`, not `127.0.0.1`, unless you passed the matching origin to the
> adapter. They are different origins to the browser and CORS treats them as such.
> The real adapter allows both by default; `dev_mock_backend.py` hardcodes
> `http://localhost:3000` only.

On Windows the adapter launcher is
`.\gui_integration\scripts\start_backend.ps1 -FrontendOrigin http://localhost:3000`.

### Without the frozen artifacts

If the adapter cannot start, the committed fixture server renders the same API:

```bash
python3 ../dev_mock_backend.py
```

**Its verdicts come from a stand-in scorer and mean nothing.** It exists so the
layout can be developed. Never present on it — the real adapter prints a
`gui_backend_ready` line with the true `model_id`; the mock prints a MOCK banner.

### Options

```bash
python3 serve.py --port 3001 --no-browser
```

Point at a non-default adapter with a query string:
`http://localhost:3000/?api=http://127.0.0.1:9000/api/gui/v1`

## The two sections

**Dashboard** — the live operating picture, top to bottom: latest verdict as a
hero figure with the attack probability placed against the 0.42 decision
threshold; the adapter's counters plus this session's injected packet count; a
60-verdict detection timeline; the packet injector; verdicts-over-time; rejected
federated messages by category; the attack alert log; the five-client network
with click-to-inspect; and both raw event feeds.

**AI Data** — the model itself. Identity and version, frozen validation metrics,
the confusion matrix, the full round-metrics table (macro F1, per-class
precision/recall/F1, ROC-AUC, PR-AUC, log-loss, FPR/FNR), per-client training
telemetry, the 15 input features, and the communication-security summary.

### Verdicts over time

A rolling two-minute window in 5-second buckets, Normal stacked under Attack, so
column height reads as total traffic. The detection timeline plots one bar per
verdict and therefore says nothing about rate — a burst and a trickle look
identical there. This panel is the one that shows arrival rate.

It counts what **this tab** has seen since it was opened: its own injections plus
anything else the adapter scored while it was watching. That is not the adapter's
lifetime total, and the two are never added together. `current` is the last
*complete* bucket — the one still filling always reads low and would look like
traffic falling off a cliff.

### The alert log

Time, record, what was sent, attack confidence, probability, margin over the
threshold, and whether the input was injected or replayed.

**Attack confidence is the model's certainty, not a threat rating.** A 0.99 means
the model is sure this is an attack, not that the attack is severe — the bands
(`Marginal` / `High` / `Very high`) are named to avoid implying otherwise, and
they use one deepening hue rather than a green-amber-red severity ramp.

### The event feeds

Both feeds live at the bottom of the Dashboard, with independent `after_seq`
cursors, severity filtering, and follow-tail.

## The packet injector

You pick a kind of traffic, the browser makes up one flow record matching it, and
that record is sent to `POST /predictions` where the **real frozen model** scores
it. Nothing travels over a real network and no container is involved — the 15
numbers go straight to the model, which answers Normal or Attack.

The three buttons are **Normal traffic**, **Borderline**, and **Flood attack**.
"Send a real saved packet" is different: it calls `POST /replay/next`, which
scores an actual recorded validation row rather than a made-up one.

The made-up numbers are not arbitrary. Their ranges are measured from the two
clusters in the committed fixture `../examples/replay_records.json` — rows
01/03/05 (scored Normal) and 02/04/06 (scored Attack).

Derived features are computed rather than sampled, using relationships that hold
exactly in that fixture:

```
TxPacketRate = TxPackets / FlowDuration     Throughput/Kbps = RxBytes * 8 / 1000 / FlowDuration
TxByteRate   = TxBytes   / FlowDuration     PacketDropRate  = LostPackets / TxPackets
```

so one sent record holds together instead of being 15 unrelated random numbers.
**Borderline** sits between the two clusters, close to the 0.42 line, so the
answer genuinely varies — that is the honest moment in a demo, not a rigged flip.

**The button name says what the operator chose to send, not what the model
found.** The model only answers Normal or Attack and never names an attack type.

Injecting never involves the Docker containers, federated learning, or the
cryptography. It is one flow record scored by the frozen global model on the
adapter host. The five clients are training participants only — there is no
per-client inference path, and `source` is free text echoed back unchanged.

## Things the API cannot give you

Deliberate omissions, so nothing on screen is fabricated:

- **No attack families.** The model outputs `Normal` or `Attack` only. The
  categorical breakdown on the Dashboard is *rejected federated protocol
  messages* by category — a real mechanism with real counts — not traffic classes.
- **No packet totals from the backend.** The API never returns feature values, so
  the only packet figure available is the one this tab put into its own records.
  The "Records you sent" tile counts **sends** — one press of *Send one* is one
  flow record, so it goes up by one. The packet figure is the sub-line, phrased
  as *describing N packets*, because a single flood record summarises thousands
  of packets and counting them as "packets injected" made one click look like
  ten thousand. Neither figure is ever added to the adapter's counters.
- **No live accuracy.** Injected and replayed input has no ground truth, so no
  accuracy is computed from this screen. Every metric shown is saved validation
  evidence and is labelled as such.

## Behaviour in replay mode

When Phase 4/5 is not running, the adapter serves recorded federated evidence and
reports `presentation_mode: "replay"`. Two consequences the console handles:

1. **All 27 recorded events arrive on the first poll.** The event feed releases them
   at presentation pace so the sequence unfolds, with a visible note saying so.
   Derived tables and charts populate immediately regardless.
2. **Recorded events carry no timestamps.** `backend.py` constructs them with a
   sequence number only, so the log orders by `seq`, never by time.

The recorded excerpt contains registration and training detail for
`uav-client-4` only. The other four show "profile not reported" rather than an
invented value; a live Phase 4/5 run fills all five.

## Themes

The control at the foot of the sidebar switches light and dark. The choice is
stored in `localStorage` under `rasid-theme`; with nothing stored the page
follows the operating system. A small inline script in `<head>` sets
`data-theme` on `<html>` before first paint, so the page never flashes the wrong
theme on load.

Charts read their colours from CSS custom properties at draw time, so switching
theme re-renders them — `applyTheme()` calls `render()` for that reason.

**Dark is Grafana.** The surface, ink, gridline and series values are sampled
from the reference dashboard rather than eyeballed: page `#161719`, panel
`#212124`, sunk `#262628`, ink `#d8d9da`, chart gridline `#464648`, green
`#7eb26d`, yellow `#eab839`, red `#e24d42`.

The one deliberate departure is the data-mark blue. The reference's own bars are
`#1f78c1`, which measures 3.45:1 against `#212124` — too thin to project. Dark
uses Grafana's `#5794f2` at 5.29:1 instead. The pair scores normal-vision ΔE
45.5, protan ΔE 51.0, deutan ΔE 60.5 against the panel — all checks pass.

## Layout

Arranged in the rhythm of the Grafana reference: a stat strip across the top,
then rows of panels with tight uniform gutters, panel titles centred with the
context tag pinned right, and collapsible section rows (**Alerts**,
**Federation**, **Log**) that fold away when you want the screen for something
else.

## Colour

Data marks use a diverging blue ↔ red pair split at the decision threshold,
validated against the white chart surface: CVD separation ΔE 27.5 (protan),
normal-vision ΔE 35.3, both poles above 3:1 contrast.

Position against the threshold rule is the primary channel and colour reinforces
it, so the timeline stays readable under colour-vision deficiency. Green/red
status colours — which fail deutan separation as a pair — are used only on the
verdict chip and client chips, where the state is always spelled out in text
beside the colour.

## Claims the UI must keep

- Binary Normal/Attack only — no attack family, technique, or actor.
- `presentation_mode`, `inference_mode` and per-prediction `replayed` stay visible
  and distinct.
- Backend counters (adapter process lifetime) and session counters (this tab) are
  shown separately and never summed.
- Saved validation metrics are labelled as saved evidence, not live accuracy.
- Five logical clients are dataset partitions on device-inspired container
  profiles, not verified physical UAV identities.
- Security covers federated model exchange in transit only — not poisoning by an
  authenticated client, not HTTP metadata, not production readiness.

## Files

```
index.html    structure and static copy
styles.css    light technical-report theme, sized for projection
app.js        API client, injector, event mining, SVG charts, rendering
serve.py      stdlib static server (no proxying, no credentials)
```
