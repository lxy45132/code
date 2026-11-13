import { client } from './client'
import type { InpaintResponse, TaskAccepted, TaskStatus } from './types'

// inline 模式同步返回 InpaintResponse；worker 模式 202 返回 TaskAccepted。
export async function postInpaint(
  image: File | Blob,
  mask: Blob,
  note: string,
): Promise<InpaintResponse | TaskAccepted> {
  const form = new FormData()
  form.append('image', image, 'image.png')
  form.append('mask', mask, 'mask.png')
  form.append('note', note)
  const response = await client.post('/inpaint', form)
  return response.data
}

export function isTaskAccepted(value: InpaintResponse | TaskAccepted): value is TaskAccepted {
  return 'task_id' in value
}

export async function getTask(taskId: string): Promise<TaskStatus> {
  const response = await client.get(`/tasks/${taskId}`)
  return response.data
}

export async function getHealth() {
  const response = await client.get('/health')
  return response.data
}
