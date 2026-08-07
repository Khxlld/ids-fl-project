import { useEffect, useRef, useState } from 'react'
import { ApiError, getSnapshot } from '../api/client'
import type { Snapshot } from '../api/types'

export interface SnapshotState {
  snapshot: Snapshot | null
  loading: boolean
  error: ApiError | null
}

// Frontend flow step 2 (API_CONTRACT.md): poll /snapshot about once per second.
export function useSnapshot(intervalMs = 1000): SnapshotState {
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [error, setError] = useState<ApiError | null>(null)
  const [loading, setLoading] = useState(true)
  const stopped = useRef(false)

  useEffect(() => {
    stopped.current = false
    let timer: ReturnType<typeof setTimeout>

    async function tick() {
      try {
        const next = await getSnapshot()
        if (stopped.current) return
        setSnapshot(next)
        setError(null)
      } catch (cause) {
        if (stopped.current) return
        setError(cause instanceof ApiError ? cause : new ApiError(0, 'unknown_error', String(cause)))
      } finally {
        if (!stopped.current) {
          setLoading(false)
          timer = setTimeout(tick, intervalMs)
        }
      }
    }

    tick()
    return () => {
      stopped.current = true
      clearTimeout(timer)
    }
  }, [intervalMs])

  return { snapshot, loading, error }
}
