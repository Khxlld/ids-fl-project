import type { FederatedEvent, InferenceEvent } from '../api/types'
import { Empty, Panel, formatClock } from './primitives'

/**
 * Display prominence is read from the reported severity and, failing that, from
 * the event type itself. Saved fixture events carry no severity field, so the
 * adapter defaults them to "info" — which would render a rejected security
 * message identically to a routine round start. Keying off the event type is a
 * presentation decision about an existing fact; no severity value is invented,
 * and the event type is always shown verbatim beside it.
 */
function classify(event: { severity: string; event_type: string }): 'crit' | 'warn' | 'info' {
  if (event.severity === 'critical' || event.severity === 'error') return 'crit'
  if (event.severity === 'warning') return 'warn'
  if (/(rejected|error|failed|invalid|denied)/.test(event.event_type)) return 'warn'
  return 'info'
}

const TAG = { crit: 'ERR', warn: 'WRN', info: 'INF' } as const

export function FederatedEventConsole({
  events,
  dataMode,
}: {
  events: FederatedEvent[]
  dataMode: string | undefined
}) {
  return (
    <Panel
      title="Federated & security event log"
      note={dataMode ? `${dataMode} · ${events.length} shown` : `${events.length} shown`}
      flush
    >
      {events.length === 0 ? (
        <Empty>no federated events received</Empty>
      ) : (
        <div className="log tall">
          {events.map((event) => {
            const level = classify(event)
            return (
              <div key={`${event.run_id}-${event.seq}`} className="logrow">
                <span className={`logrow-sev sev-${level}`}>{TAG[level]}</span>
                <span className="logrow-main">{event.event_type.replace(/_/g, ' ')}</span>
                <span className="logrow-sub">
                  {event.client_id ?? event.source} · round {event.round}
                </span>
                <span className="logrow-t">#{event.seq}</span>
              </div>
            )
          })}
        </div>
      )}
    </Panel>
  )
}

export function InferenceEventConsole({ events }: { events: InferenceEvent[] }) {
  return (
    <Panel title="Inference event log" note={`${events.length} shown`} flush>
      {events.length === 0 ? (
        <Empty>no inference events yet</Empty>
      ) : (
        <div className="log tall">
          {events.map((event) => {
            const payload = event.payload as { label?: string; attack_probability?: number }
            const isAttack = payload.label === 'Attack'
            return (
              <div key={event.seq} className="logrow">
                <span className={isAttack ? 'logrow-sev sev-crit' : 'logrow-sev sev-ok'}>
                  {isAttack ? 'ATK' : 'NRM'}
                </span>
                <span className="logrow-main">{event.event_type.replace(/_/g, ' ')}</span>
                <span className="logrow-sub">
                  {event.source ?? 'unattributed'}
                  {payload.attack_probability !== undefined
                    ? ` · p=${payload.attack_probability.toFixed(3)}`
                    : ''}
                </span>
                <span className="logrow-t">{formatClock(event.timestamp_utc)}</span>
              </div>
            )
          })}
        </div>
      )}
    </Panel>
  )
}
