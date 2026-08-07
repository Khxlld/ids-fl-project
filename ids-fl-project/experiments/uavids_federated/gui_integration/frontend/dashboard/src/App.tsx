import { useCallback, useEffect } from 'react'
import { getFederatedEvents, getInferenceEvents } from './api/client'
import { useEventFeed } from './hooks/useEventFeed'
import { useSessionHistory } from './hooks/useSessionHistory'
import { useSimulator } from './hooks/useSimulator'
import { useSnapshot } from './hooks/useSnapshot'
import { CommandBar } from './components/CommandBar'
import { VerdictPanel } from './components/VerdictPanel'
import { DetectionTimeline } from './components/DetectionTimeline'
import { SessionStats } from './components/SessionStats'
import { SimulationPanel } from './components/SimulationPanel'
import { AlertLog } from './components/AlertLog'
import { ModelCard } from './components/ModelCard'
import { FederatedPanel } from './components/FederatedPanel'
import { SecurityPanel } from './components/SecurityPanel'
import { FederatedEventConsole, InferenceEventConsole } from './components/EventConsole'
import './App.css'

const FALLBACK_THRESHOLD = 0.42

function App() {
  const { snapshot, loading, error } = useSnapshot()
  const { history, record } = useSessionHistory()
  const simulator = useSimulator({ onPrediction: record })

  const fetchFederated = useCallback((afterSeq: number) => getFederatedEvents(afterSeq), [])
  const fetchInference = useCallback((afterSeq: number) => getInferenceEvents(afterSeq), [])
  const federatedEvents = useEventFeed(fetchFederated)
  const inferenceEvents = useEventFeed(fetchInference)

  // Pick up predictions this tab did not originate (replay presses elsewhere,
  // another client) so the timeline reflects everything the backend has scored.
  const latestFromBackend = snapshot?.inference.latest_prediction
  useEffect(() => {
    if (latestFromBackend) record(latestFromBackend)
  }, [latestFromBackend, record])

  const connected = snapshot !== null && error === null
  const threshold = snapshot?.model.decision_threshold ?? FALLBACK_THRESHOLD
  const displayed = history.length > 0 ? history[history.length - 1] : (latestFromBackend ?? null)

  return (
    <div className="console">
      <CommandBar snapshot={snapshot} connected={connected} simulating={simulator.state.running} />

      <main className="console-body">
        {error && (
          <div className="console-wide" style={{ gridTemplateColumns: '1fr' }}>
            <div className="errbar" style={{ margin: 0 }}>
              BACKEND UNREACHABLE — {error.message}. The adapter must be running on{' '}
              <code>127.0.0.1:8090</code>. Displayed values are the last known state and are not live.
            </div>
          </div>
        )}

        {loading && !error && (
          <div className="console-wide" style={{ gridTemplateColumns: '1fr' }}>
            <div className="callout">
              <span>Connecting to the GUI adapter…</span>
            </div>
          </div>
        )}

        <div className="console-col">
          <VerdictPanel prediction={displayed} threshold={threshold} />
          <DetectionTimeline history={history} threshold={threshold} />
          <SessionStats history={history} inference={snapshot?.inference} />
        </div>

        <div className="console-col">
          <SimulationPanel
            state={simulator.state}
            setProfile={simulator.setProfile}
            setIntervalMs={simulator.setIntervalMs}
            start={simulator.start}
            stop={simulator.stop}
            injectOnce={simulator.injectOnce}
            onPrediction={record}
            disabled={!connected}
          />
          <AlertLog alerts={snapshot?.inference.recent_alerts ?? []} />
        </div>

        <div className="console-wide">
          <ModelCard model={snapshot?.model} />
          <FederatedPanel federated={snapshot?.federated} />
          <SecurityPanel security={snapshot?.federated.security} />
        </div>

        <div className="console-wide two">
          <InferenceEventConsole events={inferenceEvents} />
          <FederatedEventConsole
            events={federatedEvents}
            dataMode={snapshot?.federated.data_mode}
          />
        </div>
      </main>

      <footer className="footer">
        <span>
          <b>Detection scope.</b> Binary Normal/Attack only. No attack family, technique, or actor is
          identified. Confidence is the probability assigned to the displayed class.
        </span>
        <span>
          <b>Injected traffic.</b> Simulation profiles emit synthetic model input sampled from
          recorded validation envelopes. Verdicts are always produced by the real frozen model.
        </span>
        <span>
          <b>Federated clients.</b> Five logical partitions on device-inspired container profiles —
          not verified physical UAV identities. Raw training rows are never uploaded.
        </span>
        <span>
          <b>Status.</b> Research demonstrator on a lab dataset. Not production infrastructure, not a
          validated operational detector, and not a privacy guarantee.
        </span>
      </footer>
    </div>
  )
}

export default App
