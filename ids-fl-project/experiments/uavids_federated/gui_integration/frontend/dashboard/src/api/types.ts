// Mirrors experiments/uavids_federated/gui_integration/api/openapi.json.
// Only fields the dashboard actually reads are typed; unknown fields must be ignored (API_CONTRACT.md).

export type Label = 'Normal' | 'Attack'
export type PresentationMode = 'live' | 'replay'
export type InferenceMode = 'live_model'
export type SecurityMode = 'secure' | 'plain'
export type ClientState = string

export interface FeatureVector {
  'FlowDuration/s': number | null
  TxPackets: number | null
  RxPackets: number | null
  LostPackets: number | null
  TxBytes: number | null
  RxBytes: number | null
  'TxPacketRate/s': number | null
  'RxPacketRate/s': number | null
  'TxByteRate/s': number | null
  'RxByteRate/s': number | null
  'MeanDelay/s': number | null
  'MeanJitter/s': number | null
  'Throughput/Kbps': number | null
  PacketDropRate: number | null
  AverageHopCount: number | null
}

export const FEATURE_NAMES: (keyof FeatureVector)[] = [
  'FlowDuration/s',
  'TxPackets',
  'RxPackets',
  'LostPackets',
  'TxBytes',
  'RxBytes',
  'TxPacketRate/s',
  'RxPacketRate/s',
  'TxByteRate/s',
  'RxByteRate/s',
  'MeanDelay/s',
  'MeanJitter/s',
  'Throughput/Kbps',
  'PacketDropRate',
  'AverageHopCount',
]

export interface Prediction {
  schema_version: 'uavids-gui-prediction-v1'
  prediction_id: string
  timestamp_utc: string
  record_id: string
  source: string | null
  label: Label
  confidence: number
  attack_probability: number
  decision_threshold: number
  missing_features_imputed: number
  model_id: string
  model_version: string
  inference_mode: InferenceMode
  replayed: boolean
}

export interface ReplayResponse {
  schema_version: 'uavids-gui-replay-v1'
  position: number
  total_records: number
  wrapped: boolean
  prediction: Prediction
}

export interface Health {
  schema_version: 'uavids-gui-health-v1'
  ok: boolean
  api_version: string
  model_available: boolean
}

export interface ModelInfo {
  available: boolean
  model_id: string
  model_version: string
  task: string
  labels: Label[]
  positive_class: Label
  feature_count: number
  features: string[]
  decision_threshold: number
  frozen_validation_metrics: {
    macro_f1: number
    attack_precision: number
    attack_recall: number
    fpr: number
  }
  metrics_note: string
}

export interface InferenceState {
  records_processed: number
  normal_count: number
  attack_count: number
  latest_prediction: Prediction | null
  recent_alerts: Prediction[]
}

export interface FederatedClient {
  client_id: string
  state: ClientState
}

export interface SecurityState {
  mode: SecurityMode
  status: string
  algorithms: {
    aead: string
    kdf: string
    kem: string
    signature: string
  } | null
  authenticated_clients: number
  rejected_messages: number
  note: string
}

export interface GlobalModelMetrics {
  macro_f1?: number
  attack_precision?: number
  attack_recall?: number
  fpr?: number
  source: string
  note: string
}

export interface FederatedState {
  data_mode: PresentationMode
  upstream_available: boolean
  run_id: string | null
  state: string
  current_round: number
  total_rounds: number
  updates_received: number
  updates_expected: number
  clients: FederatedClient[]
  local_data_statement: string
  global_model_metrics: GlobalModelMetrics
  security: SecurityState
}

export interface Snapshot {
  schema_version: 'uavids-gui-snapshot-v1'
  generated_utc: string
  api_version: string
  presentation_mode: PresentationMode
  backend: {
    available: boolean
    started_utc: string
    inference_mode: InferenceMode
    federated_upstream_available: boolean
  }
  model: ModelInfo
  inference: InferenceState
  federated: FederatedState
}

export interface InferenceEvent {
  schema_version: string
  seq: number
  timestamp_utc: string
  event_type: string
  severity: 'info' | 'warning' | string
  source: string | null
  payload: Record<string, unknown>
}

export interface EventPage {
  schema_version: 'uavids-gui-events-v1'
  events: InferenceEvent[]
  last_seq: number
}

export interface FederatedEvent {
  schema_version: string
  seq: number
  recorded: boolean
  run_id: string | null
  source: string
  event_type: string
  severity: 'info' | 'warning' | string
  round: number
  client_id: string | null
  payload: Record<string, unknown>
}

export interface FederatedEventPage {
  schema_version: 'uavids-gui-federated-events-v1'
  data_mode: PresentationMode
  run_id: string | null
  events: FederatedEvent[]
  last_seq: number
}

export interface ApiErrorBody {
  schema_version: 'uavids-gui-error-v1'
  error: { code: string; message: string }
}
