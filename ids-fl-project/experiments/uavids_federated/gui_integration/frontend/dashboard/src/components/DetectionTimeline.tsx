import type { Prediction } from '../api/types'
import { Panel, formatClock } from './primitives'

const WINDOW = 72

/**
 * Per-record attack probability across the session, with the frozen decision
 * threshold drawn as a rule. Bars, not a line: each bar is one discrete
 * classification, and the threshold crossing is the thing to read.
 */
export function DetectionTimeline({
  history,
  threshold,
}: {
  history: Prediction[]
  threshold: number
}) {
  const window = history.slice(-WINDOW)
  const first = window[0]
  const last = window[window.length - 1]

  return (
    <Panel
      title="Detection timeline"
      note={`last ${window.length} of ${history.length} this session`}
    >
      <div className="strip">
        <div className="strip-plot">
          {window.length === 0 && <span className="strip-empty">no records classified yet</span>}
          {window.map((prediction, index) => {
            const isAttack = prediction.label === 'Attack'
            return (
              <div
                key={prediction.prediction_id}
                className={[
                  'strip-bar',
                  isAttack ? 'is-attack' : 'is-normal',
                  index === window.length - 1 ? 'is-latest' : '',
                ]
                  .filter(Boolean)
                  .join(' ')}
                style={{ height: `${Math.max(1.5, prediction.attack_probability * 100)}%` }}
                title={`${prediction.record_id} · ${prediction.label} · p=${prediction.attack_probability.toFixed(4)}`}
              />
            )
          })}
          <div className="strip-threshold" style={{ bottom: `${threshold * 100}%` }}>
            <span>threshold {threshold.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div className="strip-axis">
        <span>{first ? formatClock(first.timestamp_utc) : '--:--:--'}</span>
        <span>bar height = attack probability · outlined bar = latest</span>
        <span>{last ? formatClock(last.timestamp_utc) : '--:--:--'}</span>
      </div>
    </Panel>
  )
}
