import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useInpaintStore } from '../src/stores/inpaint'
import { ApiError } from '../src/api/client'
import { getTask, postInpaint } from '../src/api/inpaint'
import type { InpaintResponse } from '../src/api/types'

vi.mock('../src/api/inpaint', () => ({
  postInpaint: vi.fn(),
  getTask: vi.fn(),
  isTaskAccepted: (value: object) => 'task_id' in value,
}))

const result: InpaintResponse = {
  job_id: 'a'.repeat(32),
  output_url: '/api/v1/outputs/' + 'a'.repeat(32),
  latency_ms: 12.5,
  model_version: 'v1',
  mask_ratio: 0.25,
  input_size: [8, 8],
  warnings: [],
}

beforeEach(() => {
  setActivePinia(createPinia())
  URL.createObjectURL = vi.fn(() => 'blob:fake')
  URL.revokeObjectURL = vi.fn()
})

afterEach(() => {
  vi.clearAllMocks()
})

describe('inpaint store', () => {
  it('submit without image sets error', async () => {
    const store = useInpaintStore()
    await store.submit(new Blob())
    expect(store.error).toBe('请先选择图片')
    expect(postInpaint).not.toHaveBeenCalled()
  })

  it('inline submit stores result', async () => {
    vi.mocked(postInpaint).mockResolvedValue(result)
    const store = useInpaintStore()
    store.setImage(new File(['x'], 'a.png', { type: 'image/png' }))
    expect(store.imageUrl).toBe('blob:fake')

    await store.submit(new Blob())
    expect(store.result).toEqual(result)
    expect(store.loading).toBe(false)
    expect(store.error).toBe('')
  })

  it('api error surfaces message', async () => {
    vi.mocked(postInpaint).mockRejectedValue(new ApiError('resource_limit', '文件过大', 413))
    const store = useInpaintStore()
    store.setImage(new File(['x'], 'a.png', { type: 'image/png' }))
    await store.submit(new Blob())
    expect(store.error).toBe('文件过大')
    expect(store.result).toBeNull()
  })

  it('worker mode polls task until success', async () => {
    vi.mocked(postInpaint).mockResolvedValue({
      task_id: 't1',
      status: 'queued',
      task_url: '/api/v1/tasks/t1',
    })
    vi.mocked(getTask).mockResolvedValue({
      task_id: 't1',
      status: 'SUCCESS',
      result: { ...result, status: 'ok' },
    })
    const store = useInpaintStore()
    store.setImage(new File(['x'], 'a.png', { type: 'image/png' }))
    await store.submit(new Blob())
    expect(store.taskId).toBe('t1')
    expect(store.result?.job_id).toBe(result.job_id)
  })

  it('reset clears state and revokes object url', async () => {
    const store = useInpaintStore()
    store.setImage(new File(['x'], 'a.png', { type: 'image/png' }))
    store.reset()
    expect(store.imageFile).toBeNull()
    expect(store.imageUrl).toBe('')
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:fake')
  })
})
