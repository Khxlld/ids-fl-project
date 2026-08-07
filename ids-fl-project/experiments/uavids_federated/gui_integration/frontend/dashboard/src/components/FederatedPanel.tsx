import type { FederatedState } from '../api/types'
import { Cell, Empty, Panel, pct } from './primitives'

function ledClass(state: string): string {
  if (state === 'error') return 'node-led is-err'
  if (state === 'complete' || state === 'round_complete') return 'node-led is-ok'
  if (state === 'waiting') return 'node-led'
  return 'node-led is-run'
}

export function FederatedPanel({ federated }: { federated: FederatedState | undefined }) {
  if (!federated) {
    return (
      <Panel title="Federated learning">
        <Empty>awaiting federated telemetry</Empty>
      </Panel>
    )
  }

  const roundPct =
    federated.total_rounds > 0
      ? Math.min(100, (federated.current_round / federated.total_rounds) * 100)
      : 0

  return (
    <Panel
      title="Federated learning"
      note={federated.data_mode === 'live' ? 'LIVE UPSTREAM' : 'RECORDED EVIDENCE'}
    >
      <div className="nodes">
        {federated.clients.map((client) => (
          <div key={client.client_id} className="node">
            <div className="node-id">{client.client_id}</div>
            <div className="node-state">
              <span className={ledClass(client.state)} />
              {client.state.replace(/_/g, ' ')}
            </div>
          </div>
        ))}
      </div>

      <div className="field-label">
        Round progress
        <em>
          {federated.current_round} / {federated.total_rounds}
        </em>
      </div>
      <div className="bar">
        <div className="bar-fill" style={{ width: `${roundPct}%` }} />
      </div>

      <div className="cells" style={{ marginTop: 12 }}>
        <Cell value={federated.clients.length} label="Clients" />
        <Cell
          value={`${federated.updates_received}/${federated.updates_expected}`}
          label="Updates in"
        />
        <Cell value={pct(federated.global_model_metrics.macro_f1, 1)} label="Macro F1" />
        <Cell value={pct(federated.global_model_metrics.attack_recall, 1)} label="Attack recall" />
        <Cell value={pct(federated.global_model_metrics.fpr, 1)} label="FPR" tone="warn" />
      </div>

      <p className="disclose">
        <strong>{federated.local_data_statement}</strong>
      </p>
      <p className="disclose">
        Metrics source: <code>{federated.global_model_metrics.source}</code>.{' '}
        {federated.global_model_metrics.note} Logical clients are dataset partitions and
        device-inspired container profiles, not verified physical UAV identities. The shared
        preprocessor was fitted centrally on pooled training features — this is not
        privacy-preserving federated preprocessing.
      </p>
    </Panel>
  )
}
