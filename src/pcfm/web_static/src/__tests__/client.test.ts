import { afterEach, describe, expect, it, vi } from 'vitest'
import { api, pollJob } from '../api/client'

function stubFetch(body: unknown, ok = true, status = 200) {
  vi.stubGlobal(
    'fetch',
    vi.fn().mockResolvedValue({
      ok,
      status,
      text: async () => JSON.stringify(body),
    }),
  )
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.useRealTimers()
})

describe('api request_id 透传', () => {
  it('在错误信息后追加错误编号', async () => {
    stubFetch({ ok: false, message: '服务器出错', request_id: 'req-123' }, false, 500)
    await expect(api('/x')).rejects.toThrow('服务器出错（错误编号：req-123）')
  })

  it('无 request_id 时仅返回 message', async () => {
    stubFetch({ ok: false, message: '操作失败' }, false, 500)
    await expect(api('/x')).rejects.toThrow('操作失败')
  })

  it('解析失败时提示重启服务', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({ ok: false, status: 500, text: async () => 'not-json' }),
    )
    await expect(api('/x')).rejects.toThrow('本地服务返回了无法读取的响应')
  })
})

describe('pollJob', () => {
  it('任务成功时返回 job', async () => {
    stubFetch({ ok: true, job: { job_id: 'j1', status: 'succeeded', result: { n: 1 } } })
    const job = await pollJob('j1')
    expect(job.status).toBe('succeeded')
  })

  it('任务失败时抛出 error_message', async () => {
    stubFetch({ ok: true, job: { job_id: 'j1', status: 'failed', error_message: '失败原因' } })
    await expect(pollJob('j1')).rejects.toThrow('失败原因')
  })

  it('任务取消时抛出取消提示', async () => {
    stubFetch({ ok: true, job: { job_id: 'j1', status: 'cancelled' } })
    await expect(pollJob('j1')).rejects.toThrow('任务已取消')
  })

  it('超时抛出超时提示', async () => {
    vi.useFakeTimers()
    stubFetch({ ok: true, job: { job_id: 'j1', status: 'running' } })
    const promise = pollJob('j1', 3000)
    const rejection = expect(promise).rejects.toThrow('任务超时，请重试')
    await vi.advanceTimersByTimeAsync(10000)
    await rejection
  })
})
