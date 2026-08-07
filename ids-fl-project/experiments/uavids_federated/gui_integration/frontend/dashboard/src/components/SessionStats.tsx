import type { InferenceState, Prediction } from '../api/types'
import { Cell, Panel, pct } from './primitives'

/**
 * Two independent counters, deliberately not blended:
 *  - session: what this browser tab has observed since it loaded
 *  - backend: the adapter's own process-lifetime totals from /snapshot
 */
export function SessionStats({
  history,
  inference,
}: {
  history: Prediction[]
  inference: InferenceState | undefined
}) {
  const attacks = history.filter((prediction) => prediction.label === 'Attack').length
  const normals = history.length - attacks
  const meanProbability =
    history.length === 0
      ? undefined
      : history.reduce((sum, prediction) => sum + prediction.attack_probability, 0) / history.length

  return (
    <Panel title="Counters" note="session vs backend lifetime" flush>
      <div className="cells">
        <Cell value={history.length} label="Session records" />
        <Cell value={normals} label="Session normal" tone={normals > 0 ? 'ok' : 'default'} />
        <Cell value={attacks} label="Session attack" tone={attacks > 0 ? 'crit' : 'default'} />
        <Cell
          value={history.length === 0 ? '—' : pct(attacks / history.length, 0)}
          label="Session attack share"
        />
        <Cell
          value={meanProbability === undefined ? '—' : meanProbability.toFixed(3)}
          label="Mean probability"
        />
      </div>
      <div className="cells">
        <Cell value={inference?.records_processed ?? '—'} label="Backend processed" />
        <Cell value={inference?.normal_count ?? '—'} label="Backend normal" />
        <Cell value={inference?.attack_count ?? '—'} label="Backend attack" />
        <Cell value={inference?.recent_alerts.length ?? '—'} label="Alerts buffered" />
        <Cell value={history.length > 0 ? 'YES' : 'NO'} label="Session active" />
      </div>
      <p className="disclose" style={{ padding: '10px 13px 12px', margin: 0 }}>
        <strong>Session</strong> counts reset when this page reloads. <strong>Backend</strong>{' '}
        counts cover the adapter process lifetime and include records submitted by any client. Neither
        is a live accuracy measurement — no ground-truth labels exist for injected or replayed input.
      </p>
    </Panel>
  )
}
