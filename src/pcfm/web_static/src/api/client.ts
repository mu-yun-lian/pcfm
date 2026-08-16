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
    throw new Error((data.message as string) || '操作失败')
  }
  return data as T
}

export async function pollJob(jobId: string): Promise<Job> {
  while (true) {
    const data = await api<{ job: Job }>('/api/jobs/' + encodeURIComponent(jobId))
    const job = data.job
    if (job.status === 'succeeded') return job
    if (job.status === 'failed') throw new Error(job.error_message || '任务失败')
    if (job.status === 'cancelled') throw new Error('任务已取消')
    await new Promise((r) => setTimeout(r, 1500))
  }
}
