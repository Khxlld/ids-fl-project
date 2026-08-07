import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, submitPrediction } from '../api/client'
import type { FeatureVector, Prediction } from '../api/types'
import { generateVector, sourceFor, type ProfileId } from '../sim/profiles'

export interface SimulatorState {
  running: boolean
  profile: ProfileId
  intervalMs: number
  injected: number
  lastVector: FeatureVector | null
  error: string | null
}

interface Options {
  onPrediction: (prediction: Prediction) => void
}

/**
 * Drives the live simulation: generates a synthetic feature vector on each tick
 * and submits it to the real model via POST /predictions.
 *
 * Ticks are self-scheduling rather than setInterval, so a slow round trip
 * cannot stack overlapping in-flight requests.
 */
export function useSimulator({ onPrediction }: Options) {
  const [running, setRunning] = useState(false)
  const [profile, setProfile] = useState<ProfileId>('nominal')
  const [intervalMs, setIntervalMs] = useState(1200)
  const [injected, setInjected] = useState(0)
  const [lastVector, setLastVector] = useState<FeatureVector | null>(null)
  const [error, setError] = useState<string | null>(null)

  const seq = useRef(0)
  const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  // Latest values read inside the loop without re-arming it on every change.
  const live = useRef({ running, profile, intervalMs, onPrediction })
  live.current = { running, profile, intervalMs, onPrediction }

  const injectOnce = useCallback(async (): Promise<void> => {
    const active = live.current.profile
    const vector = generateVector(active)
    setLastVector(vector)
    seq.current += 1
    const recordId = `sim-${String(seq.current).padStart(4, '0')}`
    try {
      const prediction = await submitPrediction(vector, recordId, sourceFor(active))
      live.current.onPrediction(prediction)
      setInjected((count) => count + 1)
      setError(null)
    } catch (cause) {
      setError(cause instanceof ApiError ? cause.message : String(cause))
      setRunning(false)
    }
  }, [])

  useEffect(() => {
    if (!running) {
      clearTimeout(timer.current)
      return
    }
    let stopped = false

    async function tick() {
      await injectOnce()
      if (!stopped && live.current.running) {
        timer.current = setTimeout(tick, live.current.intervalMs)
      }
    }

    tick()
    return () => {
      stopped = true
      clearTimeout(timer.current)
    }
  }, [running, injectOnce])

  const reset = useCallback(() => {
    seq.current = 0
    setInjected(0)
    setLastVector(null)
    setError(null)
  }, [])

  return {
    state: { running, profile, intervalMs, injected, lastVector, error } satisfies SimulatorState,
    setProfile,
    setIntervalMs,
    start: () => setRunning(true),
    stop: () => setRunning(false),
    injectOnce,
    reset,
  }
}
