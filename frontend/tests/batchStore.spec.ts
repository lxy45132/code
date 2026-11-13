import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useBatchStore } from '../src/stores/batch'
import { cancelBatch, createBatch, getBatch } from '../src/api/batches'
import type { BatchDetail } from '../src/api/types'

vi.mock('../src/api/batches', () => ({
  createBatch: vi.fn(),
  getBatch: vi.fn(),
  cancelBatch: vi.fn(),
  eventsUrl: (id: string) => `/api/v1/batches/${id}/events`,
  reportUrl: (id: string) => `/api/v1/batches/${id}/report`,
}))

class FakeEventSource {
  static instances: FakeEventSource[] = []
  listeners: Record<string, Array<(event: MessageEvent) => void>> = {}
  onerror: (() => void) | null = null
  closed = false
  url: string

  constructor(url: string) {
    this.url = url
    FakeEventSource.instances.push(this)
  }
  addEventListener(type: string, cb: (event: MessageEvent) => void) {
    ;(this.listeners[type] ??= []).push(cb)
  }
  close() {
    this.closed = true
  }
  emit(type: string, data: unknown, id = '1-0') {
    for (const cb of this.listeners[type] ?? []) {
      cb({ data: JSON.stringify(data), lastEventId: id } as MessageEvent)
    }
  }
}

function detail(status: BatchDetail['status'], done = 0, failed = 0): BatchDetail {
  return {
    batch_id: 'b'.repeat(32),
    status,
    total: 2,
    done,
    failed,
    created_at: '2026-11-13T12:00:00+00:00',
    finished_at: null,
    note: null,
    report_url: null,
    items: [0, 1].map((i) => ({
      item_index: i,
      image_name: `${i}.png`,
      status: 'pending' as const,
      job_id: null,
      error_code: null,
      latency_ms: null,
      mask_ratio: null,
      output_url: null,
    })),
  }
}

beforeEach(() => {
  setActivePinia(createPinia())
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
})

afterEach(() => {
  vi.useRealTimers()
  vi.clearAllMocks()
  vi.unstubAllGlobals()
})

describe('batch store', () => {
  it('rejects mismatched pair counts', async () => {
    const store = useBatchStore()
    store.setImages([new File(['x'], 'a.png')])
    await store.submit()
    expect(store.error).toContain('数量必须一致')
    expect(createBatch).not.toHaveBeenCalled()
  })

  it('submit opens SSE and tracks item events', async () => {
    vi.mocked(createBatch).mockResolvedValue({
      batch_id: 'b'.repeat(32),
      status: 'queued',
      total: 2,
      events_url: '/events',
    })
    vi.mocked(getBatch)
      .mockResolvedValueOnce(detail('queued'))
      .mockResolvedValue(detail('succeeded', 1, 1))

    const store = useBatchStore()
    store.setImages([new File(['x'], 'a.png'), new File(['y'], 'b.png')])
    store.setMasks([new File(['m'], 'm0.png'), new File(['m'], 'm1.png')])
    await store.submit()

    expect(store.running).toBe(true)
    const source = FakeEventSource.instances[0]
    expect(source.url).toContain('/events')

    source.emit('item_started', { index: 0 }, '1-0')
    expect(store.itemRunning[0]).toBe(true)

    source.emit('item_done', { index: 0, job_id: 'j0', latency_ms: 5 }, '2-0')
    expect(store.itemRunning[0]).toBe(false)
    expect(store.detail?.items[0].status).toBe('ok')
    expect(store.detail?.done).toBe(1)
    expect(store.lastEventId).toBe('2-0')

    source.emit('item_failed', { index: 1, error_code: 'validation_error' }, '3-0')
    expect(store.detail?.items[1].status).toBe('failed')
    expect(store.detail?.failed).toBe(1)

    source.emit('finished', { status: 'succeeded' }, '4-0')
    await vi.waitFor(() => expect(store.running).toBe(false))
    expect(source.closed).toBe(true)
    expect(store.detail?.status).toBe('succeeded')
  })

  it('falls back to polling after 3 SSE errors', async () => {
    vi.useFakeTimers()
    vi.mocked(createBatch).mockResolvedValue({
      batch_id: 'b'.repeat(32),
      status: 'queued',
      total: 2,
      events_url: '/events',
    })
    vi.mocked(getBatch)
      .mockResolvedValueOnce(detail('running'))
      .mockResolvedValue(detail('succeeded', 2, 0))

    const store = useBatchStore()
    store.setImages([new File(['x'], 'a.png')])
    store.setMasks([new File(['m'], 'm.png')])
    await store.submit()

    const source = FakeEventSource.instances[0]
    source.onerror?.()
    source.onerror?.()
    expect(store.polling).toBe(false)
    source.onerror?.()
    expect(store.polling).toBe(true)
    expect(source.closed).toBe(true)

    await vi.advanceTimersByTimeAsync(2100)
    expect(getBatch).toHaveBeenCalledTimes(2)
    expect(store.finished).toBe(true)
    expect(store.polling).toBe(false)
  })

  it('cancel calls API', async () => {
    const store = useBatchStore()
    store.batchId = 'b'.repeat(32)
    await store.cancel()
    expect(cancelBatch).toHaveBeenCalledWith('b'.repeat(32))
  })
})
