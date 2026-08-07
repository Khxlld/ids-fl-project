import { useState } from 'react'
import { ApiError, replayNext, submitPrediction } from '../api/client'
import { FEATURE_NAMES, type FeatureVector, type Prediction } from '../api/types'
import { HEADLINE_FEATURES, PROFILES, type ProfileId } from '../sim/profiles'
import type { SimulatorState } from '../hooks/useSimulator'
import { Panel } from './primitives'

function parseVector(text: string): FeatureVector {
  let parsed: unknown
  try {
    parsed = JSON.parse(text)
  } catch {
    throw new Error('not valid JSON')
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('expected a JSON object of the 15 feature names')
  }
  const record = parsed as Record<string, unknown>
  for (const name of FEATURE_NAMES) {
    if (!(name in record)) throw new Error(`missing feature: ${name}`)
    const value = record[name]
    if (value !== null && typeof value !== 'number') {
      throw new Error(`${name} must be a number or null`)
    }
  }
  return record as unknown as FeatureVector
}

const compact = (value: number | null): string => {
  if (value === null) return 'null'
  if (value === 0) return '0'
  const magnitude = Math.abs(value)
  if (magnitude >= 1000) return value.toFixed(0)
  if (magnitude >= 1) return value.toFixed(2)
  if (magnitude >= 0.001) return value.toFixed(4)
  return value.toExponential(1)
}

interface Props {
  state: SimulatorState
  setProfile: (profile: ProfileId) => void
  setIntervalMs: (ms: number) => void
  start: () => void
  stop: () => void
  injectOnce: () => void
  onPrediction: (prediction: Prediction) => void
  disabled: boolean
}

export function SimulationPanel({
  state,
  setProfile,
  setIntervalMs,
  start,
  stop,
  injectOnce,
  onPrediction,
  disabled,
}: Props) {
  const [showCustom, setShowCustom] = useState(false)
  const [customText, setCustomText] = useState('')
  const [customError, setCustomError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  async function handleReplay() {
    setBusy(true)
    try {
      const response = await replayNext()
      onPrediction(response.prediction)
    } catch (cause) {
      setCustomError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setBusy(false)
    }
  }

  async function handleCustom() {
    setCustomError(null)
    let vector: FeatureVector
    try {
      vector = parseVector(customText)
    } catch (cause) {
      setCustomError(cause instanceof Error ? cause.message : String(cause))
      return
    }
    setBusy(true)
    try {
      onPrediction(await submitPrediction(vector, undefined, 'manual-entry'))
    } catch (cause) {
      setCustomError(cause instanceof ApiError ? cause.message : String(cause))
    } finally {
      setBusy(false)
    }
  }

  const perSecond = (1000 / state.intervalMs).toFixed(1)

  return (
    <Panel
      title="Live traffic injector"
      note={state.running ? `RUNNING · ${state.injected} sent` : `${state.injected} sent`}
    >
      <div className="callout">
        <span>
          Generates <strong>synthetic model input only</strong>. Feature envelopes are sampled from
          the recorded validation clusters and every vector is internally consistent (rates and
          throughput derived from counts, bytes and duration). Verdicts come from the real frozen
          model — nothing here fabricates a prediction.
        </span>
      </div>

      <div className="field">
        <div className="field-label">Traffic profile</div>
        <div className="profiles">
          {PROFILES.map((profile) => (
            <button
              key={profile.id}
              type="button"
              className={`profile${state.profile === profile.id ? ' is-on' : ''}`}
              onClick={() => setProfile(profile.id)}
              aria-pressed={state.profile === profile.id}
            >
              <span className="profile-dot" />
              <span>
                <span className="profile-name">{profile.name}</span>
                <span className="profile-desc">{profile.description}</span>
                <span className="profile-expect">↳ {profile.expectation}</span>
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="field">
        <div className="field-label">
          Injection rate
          <em>
            {(state.intervalMs / 1000).toFixed(2)} s · ≈{perSecond} rec/s
          </em>
        </div>
        <input
          className="slider"
          type="range"
          min={250}
          max={3000}
          step={50}
          value={state.intervalMs}
          onChange={(event) => setIntervalMs(Number(event.target.value))}
        />
      </div>

      <div className="btnrow">
        {state.running ? (
          <button type="button" className="btn is-stop" onClick={stop}>
            ■ Stop stream
          </button>
        ) : (
          <button type="button" className="btn is-run" onClick={start} disabled={disabled}>
            ▶ Start stream
          </button>
        )}
        <button type="button" className="btn" onClick={injectOnce} disabled={disabled || state.running}>
          Inject one
        </button>
        <button type="button" className="btn" onClick={handleReplay} disabled={disabled || busy}>
          Replay fixture
        </button>
      </div>

      {state.error && <p className="errbar" style={{ marginTop: 12, marginBottom: 0 }}>{state.error}</p>}

      <div className="field" style={{ marginTop: 14, marginBottom: 0 }}>
        <div className="field-label">
          Last vector emitted
          <em>{state.lastVector ? '15 features sent' : 'none yet'}</em>
        </div>
        {state.lastVector ? (
          <div className="vecgrid">
            {HEADLINE_FEATURES.map((name) => (
              <div key={name}>
                <div className="vec-k">{name}</div>
                <div className="vec-v">{compact(state.lastVector![name])}</div>
              </div>
            ))}
          </div>
        ) : (
          <p className="disclose" style={{ margin: 0 }}>
            Start the stream or inject a single record to see the emitted feature vector.
          </p>
        )}
      </div>

      <button type="button" className="linkish" onClick={() => setShowCustom((on) => !on)}>
        {showCustom ? '− Hide manual vector entry' : '+ Manual vector entry'}
      </button>

      {showCustom && (
        <div style={{ marginTop: 10 }}>
          <textarea
            className="vector"
            spellCheck={false}
            placeholder={'{\n  "FlowDuration/s": 176.62,\n  "TxPackets": 7,\n  ...all 15 features\n}'}
            value={customText}
            onChange={(event) => setCustomText(event.target.value)}
            aria-label="Custom feature vector JSON"
          />
          {customError && (
            <p className="errbar" style={{ marginTop: 8, marginBottom: 0 }}>
              {customError}
            </p>
          )}
          <div className="btnrow" style={{ marginTop: 8 }}>
            <button type="button" className="btn" onClick={handleCustom} disabled={disabled || busy}>
              Classify vector
            </button>
            <button
              type="button"
              className="btn"
              onClick={() => setCustomText(JSON.stringify(state.lastVector ?? {}, null, 2))}
              disabled={!state.lastVector}
            >
              Load last
            </button>
          </div>
          <p className="disclose">
            All 15 feature names are required exactly; use <code>null</code> for a value the frozen
            training-only imputer should fill in.
          </p>
        </div>
      )}
    </Panel>
  )
}
