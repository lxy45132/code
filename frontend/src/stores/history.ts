import { defineStore } from 'pinia'
import { deleteHistory, listHistory } from '../api/history'
import { ApiError } from '../api/client'
import type { HistoryItem } from '../api/types'

interface HistoryState {
  items: HistoryItem[]
  limit: number
  offset: number
  loading: boolean
  error: string
  hasMore: boolean
}

export const useHistoryStore = defineStore('history', {
  state: (): HistoryState => ({
    items: [],
    limit: 20,
    offset: 0,
    loading: false,
    error: '',
    hasMore: false,
  }),
  actions: {
    async load() {
      this.loading = true
      this.error = ''
      try {
        const page = await listHistory(this.limit, this.offset)
        this.items = page.items
        this.hasMore = page.items.length === this.limit
      } catch (error) {
        this.error = error instanceof ApiError ? error.message : '加载失败'
      } finally {
        this.loading = false
      }
    },
    async setPage(offset: number) {
      this.offset = offset
      await this.load()
    },
    async remove(jobId: string, deleteFiles: boolean) {
      try {
        await deleteHistory(jobId, deleteFiles)
        await this.load()
      } catch (error) {
        this.error = error instanceof ApiError ? error.message : '删除失败'
      }
    },
  },
})
