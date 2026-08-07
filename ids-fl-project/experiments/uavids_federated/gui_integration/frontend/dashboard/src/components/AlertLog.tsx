import type { Prediction } from '../api/types'
import { Empty, Panel, formatClock } from './primitives'

export function AlertLog({ alerts }: { alerts: Prediction[] }) {
  return (
    <Panel title="Attack alerts" note={`${alerts.length} buffered`} flush>
      {alerts.length === 0 ? (
        <Empty>no attack classifications yet</Empty>
      ) : (
        <div className="log short">
          {alerts.map((alert) => (
            <div key={alert.prediction_id} className="logrow">
              <span className="logrow-sev sev-crit">ATK</span>
              <span className="logrow-main">{alert.record_id}</span>
              <span className="logrow-sub">
                p={alert.attack_probability.toFixed(3)} · {alert.source ?? 'unattributed'}
              </span>
              <span className="logrow-t">{formatClock(alert.timestamp_utc)}</span>
            </div>
          ))}
        </div>
      )}
      <p className="disclose" style={{ padding: '10px 13px 12px', margin: 0 }}>
        Buffer holds the most recent alerts from the adapter. No feature values or raw tensors are
        returned in alert payloads.
      </p>
    </Panel>
  )
}
