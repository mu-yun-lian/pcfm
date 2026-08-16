import type { Job } from '../types'

export interface ApiEnvelope {
  ok: boolean
  message?: string
  [key: string]: unknown
}

export async function api<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  })
  const raw = await response.text()
  let data: ApiEnvelope
  try {
    data = raw ? (JSON.parse(raw) as ApiEnvelope) : ({} as ApiEnvelope)
  } catch {
    throw new Error('本地服务返回了无法读取的响应（HTTP ' + response.status + '）。请重启服务后重试。')
  }
  if (!response.ok || data.ok === false) {
    const message = (data.message as string) || '操作失败'
    const requestId = (data as { request_id?: string }).request_id
    throw new Error(requestId ? message + '（错误编号：' + requestId + '）' : message)
  }
  return data as T
}

export async function pollJob(jobId: string, timeoutMs = 10 * 60 * 1000): Promise<Job> {
  const deadline = Date.now() + timeoutMs
  while (true) {
    const data = await api<{ job: Job }>('/api/jobs/' + encodeURIComponent(jobId))
    const job = data.job
    if (job.status === 'succeeded') return job
    if (job.status === 'failed') throw new Error(job.error_message || '任务失败')
    if (job.status === 'cancelled') throw new Error('任务已取消')
    if (Date.now() > deadline) throw new Error('任务超时，请重试')
    await new Promise((r) => setTimeout(r, 1500))
  }
}
