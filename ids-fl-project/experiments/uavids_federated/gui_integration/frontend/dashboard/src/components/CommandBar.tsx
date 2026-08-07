import { useEffect, useState } from 'react'
import type { Snapshot } from '../api/types'
import { Readout } from './primitives'

function useClock(): string {
  const [now, setNow] = useState(() => new Date())
  useEffect(() => {
    const timer = setInterval(() => setNow(new Date()), 1000)
    return () => clearInterval(timer)
  }, [])
  return now.toLocaleTimeString('en-GB', { hour12: false })
}

export function CommandBar({
  snapshot,
  connected,
  simulating,
}: {
  snapshot: Snapshot | null
  connected: boolean
  simulating: boolean
}) {
  const clock = useClock()
  const federated = snapshot?.federated

  return (
    <header className="cmdbar">
      <div className="cmdbar-id">
        <span className="cmdbar-mark">
          UAV<span>IDS</span>
        </span>
        <span className="cmdbar-sub">Federated intrusion detection · operations console</span>
      </div>

      <div className="cmdbar-slots">
        <Readout
          label="Link"
          value={connected ? 'CONNECTED' : 'NO SIGNAL'}
          tone={connected ? 'ok' : 'crit'}
        />
        <Readout
          label="Detection"
          value={snapshot?.model.available ? 'LIVE MODEL' : 'UNAVAILABLE'}
          tone={snapshot?.model.available ? 'ok' : 'crit'}
        />
        <Readout
          label="FL telemetry"
          value={snapshot ? snapshot.presentation_mode.toUpperCase() : '—'}
          tone={snapshot?.presentation_mode === 'live' ? 'ok' : 'dim'}
        />
        <Readout
          label="Crypto"
          value={federated ? federated.security.mode.toUpperCase() : '—'}
          tone={federated?.security.mode === 'secure' ? 'ok' : 'dim'}
        />
        <Readout
          label="Injector"
          value={simulating ? 'RUNNING' : 'IDLE'}
          tone={simulating ? 'warn' : 'dim'}
        />
        <Readout label="Time" value={clock} />
      </div>
    </header>
  )
}
