import { useCallback, useRef, useState } from 'react'
import type { Prediction } from '../api/types'

const MAX_HISTORY = 400

/**
 * Verdict history for the current browser session.
 *
 * This is distinct from the backend's own process-lifetime counters in
 * /snapshot: it holds only what this tab has observed since it loaded, and it
 * resets on reload. The UI labels the two separately rather than blending them.
 */
export function useSessionHistory() {
  const [history, setHistory] = useState<Prediction[]>([])
  const seen = useRef<Set<string>>(new Set())

  const record = useCallback((prediction: Prediction) => {
    if (seen.current.has(prediction.prediction_id)) return
    seen.current.add(prediction.prediction_id)
    setHistory((previous) => {
      const next = [...previous, prediction]
      return next.length > MAX_HISTORY ? next.slice(next.length - MAX_HISTORY) : next
    })
  }, [])

  const reset = useCallback(() => {
    seen.current = new Set()
    setHistory([])
  }, [])

  return { history, record, reset }
}
