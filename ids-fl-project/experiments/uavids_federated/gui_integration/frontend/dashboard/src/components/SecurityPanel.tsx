import type { SecurityState } from '../api/types'
import { Cell, Empty, KeyValue, Panel } from './primitives'

const ALGO_ROLE: Record<string, string> = {
  kem: 'Key encapsulation',
  signature: 'Client authentication',
  aead: 'Message protection',
  kdf: 'Key derivation',
}

const ALGO_ORDER = ['kem', 'signature', 'aead', 'kdf']

export function SecurityPanel({ security }: { security: SecurityState | undefined }) {
  if (!security) {
    return (
      <Panel title="Communication security">
        <Empty>awaiting security telemetry</Empty>
      </Panel>
    )
  }

  const secure = security.mode === 'secure'
  const algorithms = security.algorithms

  return (
    <Panel title="Communication security" note={security.status.replace(/_/g, ' ')}>
      <div className="cells" style={{ marginBottom: 12 }}>
        <Cell
          value={security.mode.toUpperCase()}
          label="Channel mode"
          tone={secure ? 'ok' : 'default'}
        />
        <Cell value={security.authenticated_clients} label="Authenticated" />
        <Cell
          value={security.rejected_messages}
          label="Rejected msgs"
          tone={security.rejected_messages > 0 ? 'warn' : 'default'}
        />
      </div>

      {secure && algorithms ? (
        <KeyValue
          items={ALGO_ORDER.filter((role) => role in algorithms).map((role) => [
            ALGO_ROLE[role] ?? role,
            algorithms[role as keyof typeof algorithms],
          ])}
        />
      ) : (
        <p className="disclose" style={{ marginTop: 0 }}>
          Plain comparison mode: federated messages carry no cryptographic protection. This is the
          Phase 4 baseline used for comparison against the secure path.
        </p>
      )}

      <p className="disclose">{security.note}</p>
      <p className="disclose">
        <strong>Scope.</strong> This protects federated model exchange in transit. It does not detect
        poisoning by an already-authenticated client, does not conceal HTTP metadata, and does not
        make this production infrastructure. No key material, shared secret, signature, or ciphertext
        is exposed to this interface.
      </p>
    </Panel>
  )
}
