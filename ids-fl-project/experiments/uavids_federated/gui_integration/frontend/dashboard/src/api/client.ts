import type {
  ApiErrorBody,
  EventPage,
  FeatureVector,
  FederatedEventPage,
  Health,
  Prediction,
  ReplayResponse,
  Snapshot,
} from './types'

// Override with VITE_API_BASE if the adapter runs on a non-default host/port.
export const API_BASE: string =
  (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://127.0.0.1:8090/api/gui/v1'

export class ApiError extends Error {
  code: string
  status: number
  constructor(status: number, code: string, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }
}

async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...options,
      headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
    })
  } catch (cause) {
    throw new ApiError(0, 'network_unreachable', 'Cannot reach the GUI backend adapter')
  }
  const text = await response.text()
  const body = text ? (JSON.parse(text) as unknown) : {}
  if (!response.ok) {
    const errorBody = body as ApiErrorBody
    throw new ApiError(
      response.status,
      errorBody.error?.code ?? 'unknown_error',
      errorBody.error?.message ?? `HTTP ${response.status}`,
    )
  }
  return body as T
}

export const getHealth = () => request<Health>('/health')

export const getSnapshot = () => request<Snapshot>('/snapshot')

export const submitPrediction = (features: FeatureVector, recordId?: string, source?: string) =>
  request<Prediction>('/predictions', {
    method: 'POST',
    body: JSON.stringify({
      ...(recordId ? { record_id: recordId } : {}),
      ...(source ? { source } : {}),
      features,
    }),
  })

export const replayNext = () =>
  request<ReplayResponse>('/replay/next', { method: 'POST', body: '{}' })

export const getInferenceEvents = (afterSeq: number) =>
  request<EventPage>(`/events?after_seq=${afterSeq}`)

export const getFederatedEvents = (afterSeq: number) =>
  request<FederatedEventPage>(`/federated/events?after_seq=${afterSeq}`)
