import type { ModelInfo } from '../api/types'
import { Empty, KeyValue, Panel, pct } from './primitives'

export function ModelCard({ model }: { model: ModelInfo | undefined }) {
  if (!model) {
    return (
      <Panel title="Detection engine">
        <Empty>model metadata unavailable</Empty>
      </Panel>
    )
  }

  return (
    <Panel title="Detection engine" note={model.available ? 'loaded' : 'unavailable'}>
      <KeyValue
        items={[
          ['Model ID', model.model_id],
          ['Version', model.model_version],
          ['Task', 'binary classification'],
          ['Classes', model.labels.join(' / ')],
          ['Positive class', model.positive_class],
          ['Decision threshold', model.decision_threshold.toFixed(2)],
          ['Input features', String(model.feature_count)],
          ['Aggregation', 'FedAvg (frozen Phase 3)'],
        ]}
      />

      <div className="field-label" style={{ margin: '14px 0 6px' }}>
        Frozen validation evidence
      </div>
      <KeyValue
        items={[
          ['Macro F1', pct(model.frozen_validation_metrics.macro_f1, 2)],
          ['Attack precision', pct(model.frozen_validation_metrics.attack_precision, 2)],
          ['Attack recall', pct(model.frozen_validation_metrics.attack_recall, 2)],
          ['False-positive rate', pct(model.frozen_validation_metrics.fpr, 2)],
        ]}
      />

      <p className="disclose">
        <strong>{model.metrics_note}</strong> These figures come from a saved validation run against
        held-out data. They do not describe the records being classified on this screen, and they are
        not a production guarantee.
      </p>
    </Panel>
  )
}
