import { defineStore } from 'pinia'
import { cancelBatch, createBatch, eventsUrl, getBatch } from '../api/batches'
import { ApiError } from '../api/client'
import type { BatchDetail, BatchEvent } from '../api/types'

export interface BatchPair {
  image: File
  mask: File
}

const EVENT_TYPES = [
  'item_started',
  'item_done',
  'item_failed',
  'progress',
  'cancelled',
  'finished',
] as const

const TERMINAL = ['succeeded', 'failed', 'cancelled']

interface BatchState {
  images: File[]
  masks: File[]
  note: string
  batchId: string
  detail: BatchDetail | null
  submitting: boolean
  running: boolean
  error: string
  itemRunning: Record<number, boolean>
  sseFailures: number
  polling: boolean
  lastEventId: string
  events: BatchEvent[]
}

let source: EventSource | null = null
let pollTimer: ReturnType<typeof setInterval> | null = null

export const useBatchStore = defineStore('batch', {
  state: (): BatchState => ({
    images: [],
    masks: [],
    note: '',
    batchId: '',
    detail: null,
    submitting: false,
    running: false,
    error: '',
    itemRunning: {},
    sseFailures: 0,
    polling: false,
    lastEventId: '',
    events: [],
  }),
  getters: {
    pairs(state): BatchPair[] {
      const count = Math.min(state.images.length, state.masks.length)
      return Array.from({ length: count }, (_, i) => ({
        image: state.images[i],
        mask: state.masks[i],
      }))
    },
    progress(state): number {
      if (!state.detail || state.detail.total === 0) return 0
      return Math.round(((state.detail.done + state.detail.failed) / state.detail.total) * 100)
    },
    finished(state): boolean {
      return !!state.detail && TERMINAL.includes(state.detail.status)
    },
  },
  actions: {
    setImages(files: File[]) {
      this.images = files
    },
    setMasks(files: File[]) {
      this.masks = files
    },
    async submit() {
      this.error = ''
      if (this.images.length === 0 || this.images.length !== this.masks.length) {
        this.error = '图片与掩码数量必须一致且非空'
        return
      }
      this.submitting = true
      try {
        const created = await createBatch(this.images, this.masks, this.note)
        this.batchId = created.batch_id
        this.detail = await getBatch(created.batch_id)
        this.running = !this.finished
        this.itemRunning = {}
        this.events = []
        this.sseFailures = 0
        if (this.running) this.openEvents()
      } catch (error) {
        this.error = error instanceof ApiError ? error.message : '提交失败，请重试'
      } finally {
        this.submitting = false
      }
    },
    openEvents() {
      this.closeEvents()
      // EventSource 断线自动重连并携带 Last-Event-ID（服务端从游标回放）。
      source = new EventSource(eventsUrl(this.batchId))
      for (const type of EVENT_TYPES) {
        source.addEventListener(type, (event: MessageEvent) => this.onEvent(type, event))
      }
      source.onerror = () => {
        this.sseFailures += 1
        if (this.sseFailures >= 3) {
          // 3 次失败退回 2s 轮询兜底（docs/03 §8）
          this.closeEvents()
          this.startPolling()
        }
      }
    },
    onEvent(type: string, event: MessageEvent) {
      this.sseFailures = 0
      this.lastEventId = event.lastEventId || this.lastEventId
      let data: Record<string, unknown> = {}
      try {
        data = JSON.parse(event.data)
      } catch {
        return
      }
      this.events.push({ id: this.lastEventId, event: type, data })
      const index = Number(data.index ?? -1)
      const detail = this.detail
      if (type === 'item_started' && index >= 0) {
        this.itemRunning[index] = true
      } else if (type === 'item_done' && detail && index >= 0) {
        this.itemRunning[index] = false
        const item = detail.items.find((it) => it.item_index === index)
        if (item) {
          item.status = 'ok'
          item.job_id = String(data.job_id ?? '')
          item.output_url = item.job_id ? `/api/v1/outputs/${item.job_id}` : null
          item.latency_ms = Number(data.latency_ms ?? 0)
        }
        detail.done += 1
      } else if (type === 'item_failed' && detail && index >= 0) {
        this.itemRunning[index] = false
        const item = detail.items.find((it) => it.item_index === index)
        if (item) {
          item.status = 'failed'
          item.error_code = String(data.error_code ?? 'internal_error')
        }
        detail.failed += 1
      } else if (type === 'cancelled' && detail && index >= 0) {
        this.itemRunning[index] = false
        const item = detail.items.find((it) => it.item_index === index)
        if (item) item.status = 'cancelled'
      } else if (type === 'finished') {
        this.running = false
        this.closeEvents()
        void this.refresh()
      }
    },
    startPolling() {
      if (pollTimer) return
      this.polling = true
      pollTimer = setInterval(() => void this.refresh(), 2000)
    },
    async refresh() {
      if (!this.batchId) return
      try {
        this.detail = await getBatch(this.batchId)
        if (this.finished) {
          this.running = false
          this.stopPolling()
          this.closeEvents()
        }
      } catch {
        // 轮询期间的瞬时错误忽略，下一轮重试
      }
    },
    async cancel() {
      if (!this.batchId) return
      try {
        await cancelBatch(this.batchId)
      } catch (error) {
        this.error = error instanceof ApiError ? error.message : '取消失败'
      }
    },
    closeEvents() {
      source?.close()
      source = null
    },
    stopPolling() {
      if (pollTimer) clearInterval(pollTimer)
      pollTimer = null
      this.polling = false
    },
    reset() {
      this.closeEvents()
      this.stopPolling()
      this.images = []
      this.masks = []
      this.note = ''
      this.batchId = ''
      this.detail = null
      this.running = false
      this.error = ''
      this.itemRunning = {}
      this.events = []
      this.lastEventId = ''
      this.sseFailures = 0
    },
  },
})
