import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

vi.mock('../api/client', () => ({
  api: vi.fn(),
  pollJob: vi.fn(),
}))

import { api, pollJob } from '../api/client'
import { useAppStore } from '../stores/app'

const apiMock = vi.mocked(api)
const pollJobMock = vi.mocked(pollJob)

function emptyConversation() {
  return {
    messages: [],
    source_counts: { confirmed: 0 },
    active_version: null,
    status: 'insufficient_evidence',
    status_text: '尚未建立人物模型',
  }
}

function mockPersonFlow(personId = 'p1') {
  apiMock.mockImplementation(async (path: string) => {
    if (path === '/api/conversation/people') return { person: { person_id: personId, name: 'X' } }
    if (path === '/api/people') return { people: [{ person_id: personId, name: 'X' }] }
    if (path === `/api/people/${personId}`) return { person: { person_id: personId, name: 'X' } }
    if (path === `/api/people/${personId}/conversation` || path === `/api/people/${personId}/conversation?light=1&full_messages=1`) return { conversation: emptyConversation() }
    if (path === `/api/people/${personId}/conversation/sessions`) return { sessions: [] }
    if (path === `/api/people/${personId}/conversation/search`) return { job_id: 'j1' }
    return {}
  })
  pollJobMock.mockResolvedValue({ job_id: 'j1', status: 'succeeded' } as never)
}

describe('createPerson 系统搜索分支', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('system_search 创建后触发搜索任务并提示完成', async () => {
    mockPersonFlow()
    const store = useAppStore()
    await store.createPerson({ name: 'X', source_mode: 'system_search' })
    expect(pollJobMock).toHaveBeenCalledWith('j1')
    expect(store.toast?.message).toBe('人物已创建，公开资料搜索任务已完成。')
    expect(store.toast?.error).toBeFalsy()
  })

  it('user_provided 创建后不触发搜索任务', async () => {
    mockPersonFlow()
    const store = useAppStore()
    await store.createPerson({ name: 'X', source_mode: 'user_provided' })
    expect(pollJobMock).not.toHaveBeenCalled()
    expect(store.toast?.message).toBe('人物已创建。')
  })

  it('搜索任务失败时提示错误但不影响创建', async () => {
    mockPersonFlow()
    pollJobMock.mockRejectedValue(new Error('搜索失败'))
    const store = useAppStore()
    await store.createPerson({ name: 'X', source_mode: 'system_search' })
    expect(store.toast?.message).toBe('搜索失败')
    expect(store.toast?.error).toBe(true)
  })
})

describe('AI 助手不绑定人物对话模型', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('setDialogueModel 在助手下不调用 API', async () => {
    const store = useAppStore()
    store.person = { person_id: 'assistant', name: 'AI 助手', avatar: '' }
    await store.setDialogueModel('srv:model')
    expect(apiMock).not.toHaveBeenCalled()
    expect(store.toast?.message).toContain('AI 助手')
    expect(store.toast?.error).toBe(true)
  })

  it('clearDialogueModel 在助手下不调用 API', async () => {
    const store = useAppStore()
    store.person = { person_id: 'assistant', name: 'AI 助手', avatar: '' }
    await store.clearDialogueModel()
    expect(apiMock).not.toHaveBeenCalled()
    expect(store.toast?.message).toContain('AI 助手')
  })
})

describe('processAllMaterials 提取后重读来源（防陈旧引用）', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('提取候选并刷新后，从刷新后的会话确认候选', async () => {
    const sourceId = 'SRC'
    const candidateId = 'CAND'
    const makeSource = (review: string, candidates: unknown[]) => ({
      source_id: sourceId,
      title: 'T',
      review_status: review,
      content_authenticity: 'verbatim_transcript',
      source_locator: '第 1 段',
      source_url: 'https://example.com/x',
      llm_response_event_candidates: candidates,
    })
    const makeConv = (review: string, candidates: unknown[]) => ({
      messages: [],
      source_counts: { confirmed: review === 'confirmed' ? 1 : 0 },
      active_version: null,
      status: 'insufficient_evidence',
      status_text: '尚未建立人物模型',
      sources: [makeSource(review, candidates)],
    })

    let refreshCount = 0
    apiMock.mockImplementation(async (path: string) => {
      if (path === '/api/people/p1/processing-progress') return { progress: {} }
      if (path === `/api/people/p1/conversation/sources/${sourceId}/review`) return {}
      if (path === '/api/people/p1/conversation?full=1') {
        refreshCount++
        if (refreshCount === 1) return { conversation: makeConv('confirmed', []) }
        return { conversation: makeConv('confirmed', [{ candidate_id: candidateId, review_status: 'pending' }]) }
      }
      if (path === `/api/people/p1/conversation/sources/${sourceId}/extract-candidates`) return { job_id: 'j-extract' }
      if (path === `/api/people/p1/conversation/sources/${sourceId}/candidates/${candidateId}/review`) return {}
      if (path === '/api/people/p1/conversation/sessions') return { sessions: [] }
      return {}
    })
    pollJobMock.mockResolvedValue({ job_id: 'j', status: 'succeeded' } as never)

    const store = useAppStore()
    store.person = { person_id: 'p1', name: 'X', avatar: '' }
    store.modelServices.roles.material_processing = 'srv:model'
    store.conversation = makeConv('pending', []) as never

    await store.processAllMaterials()

    expect(apiMock).toHaveBeenCalledWith(
      `/api/people/p1/conversation/sources/${sourceId}/candidates/${candidateId}/review`,
      expect.objectContaining({ method: 'POST', body: JSON.stringify({ decision: 'confirmed' }) }),
    )
    expect(store.toast?.message).toContain('确认 1 条候选并形成版本')
  })
})
