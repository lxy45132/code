// 与后端 DTO 对齐（hybrid_inpaint/interfaces/api/schemas）。

export interface InpaintResponse {
  job_id: string
  output_url: string
  latency_ms: number
  model_version: string
  mask_ratio: number
  input_size: [number, number]
  warnings: string[]
}

export interface TaskAccepted {
  task_id: string
  status: string
  task_url: string
}

export interface TaskStatus {
  task_id: string
  status: 'PENDING' | 'STARTED' | 'RETRY' | 'SUCCESS' | 'FAILURE'
  result: (InpaintResponse & { status: string; error_code?: string; message?: string }) | null
}

export interface HistoryItem {
  job_id: string
  created_at: string
  mode: string
  batch_id: string | null
  output_url: string
  model_version: string
  width: number
  height: number
  mask_ratio: number
  latency_ms: number
  note: string | null
}

export interface HistoryPage {
  items: HistoryItem[]
  limit: number
  offset: number
}

export interface ModelInfo {
  model_id: string | null
  model_version: string | null
  ready: boolean
  device: string
  last_error: string | null
}

export interface HealthResponse {
  status: string
  model: ModelInfo
  storage_backend: string
  inference_mode: string
}

export interface BatchItem {
  item_index: number
  image_name: string
  status: 'pending' | 'ok' | 'failed' | 'cancelled' | 'skipped'
  job_id: string | null
  error_code: string | null
  latency_ms: number | null
  mask_ratio: number | null
  output_url: string | null
}

export interface BatchDetail {
  batch_id: string
  status: 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled'
  total: number
  done: number
  failed: number
  created_at: string
  finished_at: string | null
  note: string | null
  report_url: string | null
  items: BatchItem[]
}

export interface BatchCreated {
  batch_id: string
  status: string
  total: number
  events_url: string
}

export interface BatchEvent {
  id: string
  event: string
  data: Record<string, unknown>
}
