import { defineStore } from 'pinia'
import { getTask, isTaskAccepted, postInpaint } from '../api/inpaint'
import { ApiError } from '../api/client'
import type { InpaintResponse } from '../api/types'

interface InpaintState {
  imageFile: File | null
  imageUrl: string
  note: string
  loading: boolean
  error: string
  result: InpaintResponse | null
  taskId: string
}

export const useInpaintStore = defineStore('inpaint', {
  state: (): InpaintState => ({
    imageFile: null,
    imageUrl: '',
    note: '',
    loading: false,
    error: '',
    result: null,
    taskId: '',
  }),
  actions: {
    setImage(file: File | null) {
      if (this.imageUrl) URL.revokeObjectURL(this.imageUrl)
      this.imageFile = file
      this.imageUrl = file ? URL.createObjectURL(file) : ''
      this.result = null
      this.error = ''
    },
    async submit(mask: Blob) {
      if (!this.imageFile) {
        this.error = '请先选择图片'
        return
      }
      this.loading = true
      this.error = ''
      this.result = null
      try {
        const response = await postInpaint(this.imageFile, mask, this.note)
        if (isTaskAccepted(response)) {
          // worker 模式：轮询任务结果（2s 间隔，SSE 单图通道 C7f 接入）
          this.taskId = response.task_id
          this.result = await this.pollTask(response.task_id)
        } else {
          this.result = response
        }
      } catch (error) {
        this.error = error instanceof ApiError ? error.message : '请求失败，请重试'
      } finally {
        this.loading = false
      }
    },
    async pollTask(taskId: string): Promise<InpaintResponse | null> {
      for (let attempt = 0; attempt < 150; attempt += 1) {
        const status = await getTask(taskId)
        if (status.status === 'SUCCESS' && status.result) {
          if (status.result.status === 'ok') return status.result
          throw new ApiError(
            status.result.error_code ?? 'internal_error',
            status.result.message ?? '任务失败',
            500,
          )
        }
        if (status.status === 'FAILURE') {
          throw new ApiError('internal_error', '任务执行失败', 500)
        }
        await new Promise((resolve) => setTimeout(resolve, 2000))
      }
      throw new ApiError('timeout', '任务超时未完成', 504)
    },
    reset() {
      this.setImage(null)
      this.note = ''
      this.result = null
      this.error = ''
      this.taskId = ''
    },
  },
})
