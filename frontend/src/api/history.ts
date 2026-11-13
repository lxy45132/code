import { client } from './client'
import type { HistoryItem, HistoryPage } from './types'

export async function listHistory(limit = 20, offset = 0): Promise<HistoryPage> {
  const response = await client.get('/history', { params: { limit, offset } })
  return response.data
}

export async function getHistory(jobId: string): Promise<HistoryItem> {
  const response = await client.get(`/history/${jobId}`)
  return response.data
}

export async function deleteHistory(jobId: string, deleteFiles = false): Promise<void> {
  await client.delete(`/history/${jobId}`, { params: { delete_files: deleteFiles } })
}

export function exportUrl(): string {
  return `${client.defaults.baseURL}/history/export`
}
