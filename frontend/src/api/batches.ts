import { client } from './client'
import type { BatchCreated, BatchDetail } from './types'

export async function createBatch(
  images: File[],
  masks: Blob[],
  note: string,
): Promise<BatchCreated> {
  const form = new FormData()
  images.forEach((image) => form.append('images', image, image.name))
  masks.forEach((mask, index) => form.append('masks', mask, `mask_${index}.png`))
  form.append('note', note)
  const response = await client.post('/batches', form)
  return response.data
}

export async function getBatch(batchId: string): Promise<BatchDetail> {
  const response = await client.get(`/batches/${batchId}`)
  return response.data
}

export async function cancelBatch(batchId: string): Promise<void> {
  await client.post(`/batches/${batchId}/cancel`)
}

export function reportUrl(batchId: string): string {
  return `${client.defaults.baseURL}/batches/${batchId}/report`
}

export function eventsUrl(batchId: string): string {
  return `${client.defaults.baseURL}/batches/${batchId}/events`
}
