import type { Prediction } from '../api/types'
import { Panel, formatClock, pct } from './primitives'

/** Shape + word carry the verdict, so it never rests on colour alone. */
const GLYPH = { Attack: '▲', Normal: '●', idle: '○' }

function Gauge({ prediction, threshold }: { prediction: Prediction | null; threshold: number }) {
  const probability = prediction?.attack_probability ?? 0
  const isAttack = prediction?.label === 'Attack'

  return (
    <div className="gauge">
      <div className="gauge-scale">
        <div
          className={`gauge-fill ${isAttack ? 'is-attack' : 'is-normal'}`}
          style={{ width: `${Math.min(100, Math.max(0, probability * 100))}%` }}
        />
        <div className="gauge-threshold" style={{ left: `${threshold * 100}%` }} />
      </div>
      <div className="gauge-marks">
        <span>0.00</span>
        <span>attack probability</span>
        <span>1.00</span>
      </div>
      <div className="gauge-legend">
        <span>
          Measured <i>{prediction ? prediction.attack_probability.toFixed(4) : '—'}</i>
        </span>
        <span>
          Frozen threshold <i>{threshold.toFixed(2)}</i>
        </span>
        <span>
          Margin{' '}
          <i>
            {prediction ? (prediction.attack_probability - threshold >= 0 ? '+' : '') : ''}
            {prediction ? (prediction.attack_probability - threshold).toFixed(4) : '—'}
          </i>
        </span>
      </div>
    </div>
  )
}

export function VerdictPanel({
  prediction,
  threshold,
}: {
  prediction: Prediction | null
  threshold: number
}) {
  const label = prediction?.label
  const word = label ?? 'STANDBY'
  const tone = label === 'Attack' ? 'is-attack' : label === 'Normal' ? 'is-normal' : 'is-idle'
  const glyph = label ? GLYPH[label] : GLYPH.idle

  const origin = !prediction
    ? '—'
    : prediction.replayed
      ? 'Recorded fixture row'
      : prediction.source?.startsWith('sim:')
        ? 'Synthetic — operator injected'
        : 'Submitted record'

  return (
    <Panel
      title="Current verdict"
      note={prediction ? `${prediction.record_id} · ${formatClock(prediction.timestamp_utc)}` : 'awaiting first record'}
    >
      <div className="verdict">
        <div className="verdict-mark">
          <span className={`verdict-word ${tone}`}>{word.toUpperCase()}</span>
          <span className={`verdict-glyph ${tone}`}>
            <span aria-hidden="true">{glyph}</span>
            {label ? `classified ${label.toLowerCase()}` : 'no record yet'}
          </span>
          <span className="verdict-conf">
            confidence <b>{prediction ? pct(prediction.confidence, 2) : '—'}</b>
          </span>
        </div>

        <div className="verdict-meta">
          <div>
            <div className="meta-k">Record</div>
            <div className="meta-v">{prediction?.record_id ?? '—'}</div>
          </div>
          <div>
            <div className="meta-k">Source</div>
            <div className="meta-v">{prediction?.source ?? '—'}</div>
          </div>
          <div>
            <div className="meta-k">Input origin</div>
            <div className="meta-v">{origin}</div>
          </div>
          <div>
            <div className="meta-k">Scored by</div>
            <div className="meta-v">{prediction?.inference_mode ?? '—'}</div>
          </div>
          <div>
            <div className="meta-k">Model</div>
            <div className="meta-v">{prediction?.model_id ?? '—'}</div>
          </div>
          <div>
            <div className="meta-k">Imputed features</div>
            <div className="meta-v">
              {prediction ? `${prediction.missing_features_imputed} / 15` : '—'}
            </div>
          </div>
        </div>
      </div>

      <Gauge prediction={prediction} threshold={threshold} />

      <p className="disclose">
        <strong>Binary detector.</strong> The model emits Normal or Attack only — it does not
        identify an attack family, and confidence is the probability assigned to the displayed
        class, not a guarantee of correctness.
      </p>
    </Panel>
  )
}
