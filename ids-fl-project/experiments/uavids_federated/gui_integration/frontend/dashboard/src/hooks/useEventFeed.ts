import { useEffect, useRef, useState } from 'react'

// Generic poller for the two independently-cursored event feeds
// (/events and /federated/events, API_CONTRACT.md: "Keep separate inference and federated cursors").
export function useEventFeed<TEvent>(
  fetchPage: (afterSeq: number) => Promise<{ events: TEvent[]; last_seq: number }>,
  intervalMs = 1500,
  maxBuffered = 50,
) {
  const [events, setEvents] = useState<TEvent[]>([])
  const cursor = useRef(0)
  const stopped = useRef(false)

  useEffect(() => {
    stopped.current = false
    let timer: ReturnType<typeof setTimeout>

    async function tick() {
      try {
        const page = await fetchPage(cursor.current)
        if (stopped.current) return
        if (page.events.length > 0) {
          // The API returns each page in ascending seq order; this list is
          // newest-first, so reverse the page before prepending. Without this a
          // multi-event page keeps its ascending order at the head of the list
          // and the log reads out of sequence.
          const newestFirst = [...page.events].reverse()
          setEvents((previous) => [...newestFirst, ...previous].slice(0, maxBuffered))
        }
        cursor.current = page.last_seq
      } catch {
        // Transient poll failures are surfaced via the snapshot health check instead.
      } finally {
        if (!stopped.current) timer = setTimeout(tick, intervalMs)
      }
    }

    tick()
    return () => {
      stopped.current = true
      clearTimeout(timer)
    }
  }, [fetchPage, intervalMs, maxBuffered])

  return events
}
