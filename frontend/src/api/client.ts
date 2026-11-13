import axios from 'axios'

// 统一错误体 {code, message, detail?}（docs/03 §3）。
export class ApiError extends Error {
  code: string
  status: number
  detail?: unknown

  constructor(code: string, message: string, status: number, detail?: unknown) {
    super(message)
    this.code = code
    this.status = status
    this.detail = detail
  }
}

export const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || '/api/v1',
  timeout: 120_000,
})

client.interceptors.request.use((config) => {
  const apiKey = import.meta.env.VITE_API_KEY
  if (apiKey) {
    config.headers['X-API-Key'] = apiKey
  }
  return config
})

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (axios.isAxiosError(error) && error.response) {
      const body = error.response.data as { code?: string; message?: string; detail?: unknown }
      throw new ApiError(
        body?.code ?? 'http_error',
        body?.message ?? error.message,
        error.response.status,
        body?.detail,
      )
    }
    throw new ApiError('network_error', '无法连接后端服务', 0)
  },
)
