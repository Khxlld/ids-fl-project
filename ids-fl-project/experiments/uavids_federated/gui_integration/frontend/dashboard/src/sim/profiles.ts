/**
 * Synthetic traffic generation for the live simulation panel.
 *
 * IMPORTANT HONESTY BOUNDARY
 * --------------------------
 * These profiles generate *model inputs only*. Every verdict shown in the UI is
 * produced by the real frozen model behind POST /predictions — nothing here
 * fabricates a prediction, a probability, or a metric.
 *
 * The sampling envelopes below are taken from the two clusters visible in the
 * committed validation fixture (gui_integration/examples/replay_records.json):
 * records 01/03/05 (which the frozen model scores Normal) and records 02/04/06
 * (which it scores Attack). We do not invent feature magnitudes.
 *
 * Derived features are computed rather than sampled, using the relationships
 * that hold exactly in the recorded fixture:
 *   TxPacketRate  = TxPackets / FlowDuration
 *   RxPacketRate  = RxPackets / FlowDuration
 *   TxByteRate    = TxBytes   / FlowDuration
 *   RxByteRate    = RxBytes   / FlowDuration
 *   Throughput    = RxBytes * 8 / 1000 / FlowDuration   (Kbps)
 *   PacketDropRate= LostPackets / TxPackets
 * so an emitted vector is internally self-consistent, not 15 independent
 * random numbers.
 *
 * The model is binary. A profile name describes the *traffic shape being
 * injected by the operator*; it is never a claim that the model identified an
 * attack family.
 */

import type { FeatureVector } from '../api/types'

export type ProfileId = 'nominal' | 'transitional' | 'flood'

export interface Profile {
  id: ProfileId
  name: string
  description: string
  /** What the recorded evidence suggests, stated as a tendency, never a promise. */
  expectation: string
}

export const PROFILES: Profile[] = [
  {
    id: 'nominal',
    name: 'NOMINAL LINK',
    description:
      'Sparse, low-volume telemetry with high per-hop delay and jitter. Envelope taken from recorded validation rows 01/03/05.',
    expectation: 'Recorded rows in this envelope score Normal',
  },
  {
    id: 'transitional',
    name: 'TRANSITIONAL',
    description:
      'Interpolated between the two recorded clusters. Deliberately near the decision boundary — the verdict genuinely varies.',
    expectation: 'Outcome not predetermined; expect both verdicts',
  },
  {
    id: 'flood',
    name: 'SUSTAINED FLOOD',
    description:
      'Long-lived, high-volume flow: thousands of packets, sub-20 ms delay, near-zero loss. Envelope taken from recorded rows 02/04/06.',
    expectation: 'Recorded rows in this envelope score Attack',
  },
]

/** Sampling envelopes measured from the committed fixture. */
const NOMINAL = {
  duration: [2.0, 180.0],
  packets: [1, 8],
  packetSize: [30, 46],
  delay: [1.3, 2.1],
  jitter: [0.0, 0.66],
  hop: [0.0, 0.0],
  lossRate: [0.0, 0.15],
} as const

const FLOOD = {
  duration: [3400, 3500],
  packets: [4200, 5850],
  packetSize: [76, 76],
  delay: [0.0095, 0.014],
  jitter: [0.006, 0.0105],
  hop: [0.031, 0.039],
  lossRate: [0.0003, 0.0017],
} as const

const uniform = (lo: number, hi: number) => lo + Math.random() * (hi - lo)

/** Log-space interpolation, for the heavy-tailed count/duration axes. */
const logLerp = (lo: number, hi: number, t: number) =>
  Math.exp(Math.log(Math.max(lo, 1e-6)) * (1 - t) + Math.log(Math.max(hi, 1e-6)) * t)

const lerp = (lo: number, hi: number, t: number) => lo * (1 - t) + hi * t

interface Draw {
  duration: number
  packets: number
  packetSize: number
  delay: number
  jitter: number
  hop: number
  lossRate: number
}

function draw(profile: ProfileId): Draw {
  if (profile === 'nominal') {
    return {
      duration: uniform(...NOMINAL.duration),
      packets: Math.round(uniform(...NOMINAL.packets)),
      packetSize: uniform(...NOMINAL.packetSize),
      delay: uniform(...NOMINAL.delay),
      jitter: uniform(...NOMINAL.jitter),
      hop: uniform(...NOMINAL.hop),
      lossRate: uniform(...NOMINAL.lossRate),
    }
  }
  if (profile === 'flood') {
    return {
      duration: uniform(...FLOOD.duration),
      packets: Math.round(uniform(...FLOOD.packets)),
      packetSize: uniform(...FLOOD.packetSize),
      delay: uniform(...FLOOD.delay),
      jitter: uniform(...FLOOD.jitter),
      hop: uniform(...FLOOD.hop),
      lossRate: uniform(...FLOOD.lossRate),
    }
  }
  // Transitional: blend the two envelopes with a randomized mix so successive
  // injections sweep across the region where the threshold actually decides.
  //
  // Duration, packet count, delay and jitter each span two or more orders of
  // magnitude between the clusters, so they are blended geometrically — the
  // linear midpoint of 1.7 s and 0.012 s is 0.86 s, which sits nowhere near the
  // middle of that range and would collapse the profile back onto the nominal
  // cluster. Packet size and hop count are near-linear and blend linearly.
  const t = uniform(0.35, 0.65)
  return {
    duration: logLerp(uniform(...NOMINAL.duration), uniform(...FLOOD.duration), t),
    packets: Math.max(
      1,
      Math.round(logLerp(uniform(...NOMINAL.packets), uniform(...FLOOD.packets), t)),
    ),
    packetSize: lerp(uniform(...NOMINAL.packetSize), uniform(...FLOOD.packetSize), t),
    delay: logLerp(uniform(...NOMINAL.delay), uniform(...FLOOD.delay), t),
    jitter: logLerp(uniform(...NOMINAL.jitter), uniform(...FLOOD.jitter), t),
    hop: lerp(uniform(...NOMINAL.hop), uniform(...FLOOD.hop), t),
    lossRate: lerp(uniform(...NOMINAL.lossRate), uniform(...FLOOD.lossRate), t),
  }
}

/** Round to a fixed number of significant digits, matching fixture precision. */
const sig = (value: number, digits = 6): number =>
  value === 0 ? 0 : Number(value.toPrecision(digits))

export function generateVector(profile: ProfileId): FeatureVector {
  const d = draw(profile)

  const flowDuration = sig(d.duration)
  const txPackets = Math.max(1, d.packets)
  const lostPackets = Math.min(txPackets, Math.round(txPackets * d.lossRate))
  const rxPackets = Math.max(0, txPackets - lostPackets)
  const txBytes = Math.round(txPackets * d.packetSize)
  const rxBytes = Math.round(rxPackets * d.packetSize)

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
  }
}

/** Provenance string returned unchanged by the API, so the origin stays visible. */
export const sourceFor = (profile: ProfileId) => `sim:${profile}`

/** Feature subset surfaced in the injected-vector readout (the rest stay in the payload). */
export const HEADLINE_FEATURES: (keyof FeatureVector)[] = [
  'FlowDuration/s',
  'TxPackets',
  'RxPackets',
  'LostPackets',
  'MeanDelay/s',
  'MeanJitter/s',
  'Throughput/Kbps',
  'PacketDropRate',
]
