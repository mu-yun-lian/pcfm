import { defineStore } from 'pinia'
import { api, pollJob } from '../api/client'
import type {
  Person,
  Conversation,
  Comparison,
  Session,
  Capabilities,
  ModelServicesState,
  ModelService,
  ModelOption,
  OptimizationCandidate,
} from '../types'

const APP_VERSION = '0.10.0-simulation-v5'

let toastTimer: ReturnType<typeof setTimeout> | undefined
let progressTimer: ReturnType<typeof setInterval> | undefined

export const useAppStore = defineStore('app', {
  state: () => ({
    people: [] as Person[],
    person: null as Person | null,
    conversation: null as Conversation | null,
    comparison: null as Comparison | null,
    sessions: [] as Session[],
    archivedPeople: [] as Person[],
    capabilities: {} as Capabilities,
    modelServices: { services: [], roles: {} } as ModelServicesState,
    archiveTarget: null as Person | null,
    permanentDeleteTarget: null as Person | null,
    activeDialog: '' as string,
    versionMismatch: false as boolean,
    drafts: {} as Record<string, string>,
    processing: { visible: false, text: '', percent: 0 },
    composerFocusRequest: 0,
    activeJobId: null as string | null,
    creatingPerson: false as boolean,
    toast: null as null | { message: string; error: boolean; action?: () => Promise<void> },
  }),

  getters: {
    isAssistant(state): boolean {
      return state.person?.person_id === 'assistant'
    },
    enabledModelOptions(state): ModelOption[] {
      return state.modelServices.services.flatMap((service) =>
        (service.enabled_models || []).map((modelId) => ({
          ref: service.service_id + ':' + modelId,
          label: service.display_name + ' · ' + modelId,
          service,
          modelId,
          ready: service.call_readiness === 'ready' && service.last_probe_model === modelId,
        })),
      )
    },
    currentModelLabel(): string {
      const ref = this.conversation?.dialogue_model_ref || ''
      const option = this.enabledModelOptions.find((item) => item.ref === ref)
      return option ? option.modelId + (option.ready ? '' : '（需验证）') + ' ▾' : '模型 ▾'
    },
    currentName(): string {
      return this.person?.name || this.people.find((p) => p.person_id === this.person?.person_id)?.name || '当前人物'
    },
  },

  actions: {
    openDialog(name: string) {
      this.activeDialog = name
      this.creatingPerson = false
    },
    closeDialog() {
      this.activeDialog = ''
      this.creatingPerson = false
    },
    openCreatePerson() {
      this.creatingPerson = true
      this.activeDialog = 'person'
    },
    requestComposerFocus() {
      this.composerFocusRequest++
    },

    showToast(message: string, error = false, action?: () => Promise<void>) {
      if (toastTimer) clearTimeout(toastTimer)
      this.toast = { message, error, action }
      toastTimer = setTimeout(() => {
        this.toast = null
      }, action ? 8000 : 3600)
    },
    hideToast() {
      if (toastTimer) clearTimeout(toastTimer)
      this.toast = null
    },

    async cancelActiveJob() {
      if (!this.activeJobId) return
      const jobId = this.activeJobId
      this.activeJobId = null
      try {
        await api('/api/jobs/' + encodeURIComponent(jobId) + '/cancel', { method: 'POST', body: '{}' })
      } catch (error) {
        this.showToast((error as Error).message, true)
      }
    },

    async bootstrap() {
      try {
        await Promise.all([
          this.loadModelServices(),
          this.loadPeople(),
          this.refreshArchiveCount(),
          this.checkAppVersion(),
        ])
      } catch (error) {
        this.showToast((error as Error).message, true)
      }
      setInterval(() => this.checkAppVersion(), 60000)
    },

    async checkAppVersion() {
      try {
        const response = await fetch('/api/health', { cache: 'no-store' })
        const data = await response.json()
        this.versionMismatch = data.app_version !== APP_VERSION
        this.capabilities = data.capabilities || {}
      } catch {
        /* regular API errors remain visible through normal actions */
      }
    },

    async loadModelServices() {
      const data = await api<{ model_services: ModelServicesState }>('/api/model-services')
      this.modelServices = data.model_services
    },

    async loadPeople(selectId?: string) {
      const data = await api<{ people: Person[] }>('/api/people')
      this.people = data.people
      if (selectId && this.people.find((p) => p.person_id === selectId)) {
        await this.selectPerson(selectId)
      } else if (!this.person && this.people.length) {
        await this.selectPerson(this.people[0].person_id)
      } else if (!this.people.length) {
        this.person = null
        this.conversation = null
        this.comparison = null
      }
    },

    async selectPerson(personId: string) {
      const [personData, conversationData] = await Promise.all([
        api<{ person: Person }>('/api/people/' + encodeURIComponent(personId)),
        api<{ conversation: Conversation }>('/api/people/' + encodeURIComponent(personId) + '/conversation'),
      ])
      this.person = personData.person
      this.conversation = conversationData.conversation
      this.comparison = null
      await this.loadSessions()
    },

    async selectAssistant() {
      const data = await api<{ conversation: Conversation }>('/api/assistant/conversation')
      this.person = { person_id: 'assistant', name: 'AI 助手', avatar: '/default-person-avatar.png' }
      this.conversation = data.conversation
      this.comparison = null
      this.sessions = []
      if (!this.conversation.messages.length) {
        this.conversation.messages = [
          {
            message_id: 'assistant-greeting',
            role: 'assistant',
            text: '我是 AI 助手，帮你操作这个系统。 · 建人物 · 加材料 · 搜索 · 归档 / 恢复 / 永久删除 · 列出人物 / 归档列表',
            status: 'answered',
            answer_status: 'assistant',
          },
        ]
      }
    },

    async refreshAssistant() {
      const data = await api<{ conversation: Conversation }>('/api/assistant/conversation')
      this.conversation = data.conversation
    },

    async refreshConversation() {
      if (!this.person) return
      const data = await api<{ conversation: Conversation }>(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation',
      )
      this.conversation = data.conversation
      const summary = this.people.find((p) => p.person_id === this.person!.person_id)
      if (summary && this.conversation) {
        summary.conversation_version = this.conversation.active_version ?? null
        summary.source_count = this.conversation.source_counts.confirmed
        summary.message_count = this.conversation.messages.length
        summary.last_message = this.conversation.messages[this.conversation.messages.length - 1]?.text || ''
        summary.conversation_status = this.conversation.status
        summary.conversation_status_text = this.conversation.status_text
      }
      await this.loadSessions()
    },

    async sendMessage(text: string, lookup: boolean) {
      const person = this.person
      const conversation = this.conversation
      if (!person || !conversation) return
      if (this.isAssistant) {
        await api('/api/assistant/message', { method: 'POST', body: JSON.stringify({ text }) })
        await this.refreshAssistant()
        await this.loadPeople()
        return
      }
      const res = await api<{ job_id: string }>(
        '/api/people/' + encodeURIComponent(person.person_id) + '/conversation/messages',
        {
          method: 'POST',
          body: JSON.stringify({
            text,
            reality_lookup_requested: lookup,
            dialogue_model_ref: conversation.dialogue_model_ref || '',
          }),
        },
      )
      conversation.messages.push({
        message_id: 'optimistic-' + Date.now(),
        role: 'user',
        text,
        status: 'answered',
        answer_status: 'answered',
      })
      conversation.messages.push({
        message_id: 'generating-' + Date.now(),
        role: 'assistant',
        text: '生成中…',
        status: 'generating',
        answer_status: 'generating',
      })
      this.activeJobId = res.job_id
      try {
        await pollJob(res.job_id)
        await this.refreshConversation()
        if (lookup) {
          const lastAssistant = this.conversation?.messages.filter((m) => m.role === 'assistant').pop()
          if (lastAssistant) setTimeout(() => this.runRealityLookup(lastAssistant.message_id), 60)
        }
      } catch (err) {
        await this.refreshConversation()
        throw err
      } finally {
        this.activeJobId = null
      }
    },

    async startNewConversation() {
      if (!this.person) return
      await api('/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/new', {
        method: 'POST',
        body: '{}',
      })
      this.activeDialog = ''
      await this.refreshConversation()
      this.showToast('新对话已开始。')
    },

    async runRealityLookup(messageId: string) {
      if (!this.person) return
      const data = await api<{ comparison: Comparison }>(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/messages/' + encodeURIComponent(messageId) + '/reality',
        { method: 'POST', body: '{}' },
      )
      await this.refreshConversation()
      if (data.comparison.status === 'candidate_found') this.openComparison(data.comparison)
      else this.showToast('未找到可核验的现实回答。系统没有生成伪造对照。')
    },

    openComparison(comparison: Comparison) {
      if (!comparison || comparison.status !== 'candidate_found') return
      this.comparison = comparison
    },
    closeComparison() {
      this.comparison = null
    },

    async handleComparisonAction(action: string) {
      const person = this.person
      const comparison = this.comparison
      if (!person || !comparison) return
      const created = await api<{ candidate: OptimizationCandidate }>(
        '/api/people/' + encodeURIComponent(person.person_id) + '/conversation/messages/' + encodeURIComponent(comparison.message_id) + '/optimization',
        {
          method: 'POST',
          body: JSON.stringify({ comparison_candidate_id: comparison.selected_candidate_id || '' }),
        },
      )
      if (action === 'candidate') {
        this.showToast('已加入待优化资料；当前人物版本没有改变。')
      } else {
        const decision = action === 'reference' ? 'reference_only' : 'not_same_question'
        await api(
          '/api/people/' + encodeURIComponent(person.person_id) + '/conversation/optimization/' + encodeURIComponent(created.candidate.candidate_id) + '/review',
          { method: 'POST', body: JSON.stringify({ decision }) },
        )
        this.showToast(action === 'reference' ? '已仅保存为参考，未修改模型。' : '已标记为不是同一个问题。')
      }
      this.closeComparison()
      await this.refreshConversation()
      this.activeDialog = 'sources'
    },

    async sendFeedback(messageId: string) {
      if (!this.person) return
      await api(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/messages/' + encodeURIComponent(messageId) + '/feedback',
        { method: 'POST', body: JSON.stringify({ value: 'helpful' }) },
      )
      this.showToast('已保存反馈。后续可以扩展为更细的错误类型。')
      await this.refreshConversation()
    },

    async reviewSource(sourceId: string, decision: string) {
      if (!this.person) return
      await api(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sources/' + encodeURIComponent(sourceId) + '/review',
        { method: 'POST', body: JSON.stringify({ decision }) },
      )
      this.showToast(decision === 'confirmed' ? '资料已确认；只有参数训练资料会形成新的探索性版本。' : '资料已拒绝。')
      await this.refreshConversation()
    },

    async extractSource(sourceId: string) {
      if (!this.person) return
      await api(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sources/' + encodeURIComponent(sourceId) + '/extract-candidates',
        { method: 'POST', body: '{}' },
      )
      await this.refreshConversation()
      this.showToast('候选已生成，仍需逐条核对原文位置后才能进入模型。')
    },

    async reviewEventCandidate(sourceId: string, candidateId: string, decision: string) {
      if (!this.person) return
      await api(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sources/' + encodeURIComponent(sourceId) + '/candidates/' + encodeURIComponent(candidateId) + '/review',
        { method: 'POST', body: JSON.stringify({ decision }) },
      )
      await this.refreshConversation()
      this.showToast(decision === 'confirmed' ? '候选已与原文逐字核对并形成新人物版本。' : '候选已拒绝。', false)
    },

    startProgressPolling() {
      this.stopProgressPolling()
      if (!this.person) return
      this.processing.visible = true
      const tick = async () => {
        try {
          const data = await api<{ progress: any }>(
            '/api/people/' + encodeURIComponent(this.person!.person_id) + '/processing-progress',
          )
          const p = data.progress || {}
          if (p.active) {
            const total = p.total_chunks || 1
            this.processing.percent = Math.round(((p.chunk || 0) / total) * 100)
            this.processing.text = '正在处理「' + (p.title || '') + '」第 ' + (p.chunk || 0) + '/' + total + ' 块…'
          } else if (p.status === 'done') {
            this.processing.percent = 100
            this.processing.text = '处理完成'
          }
        } catch {
          /* polling failure ignored */
        }
      }
      tick()
      progressTimer = setInterval(tick, 1200)
    },
    stopProgressPolling() {
      if (progressTimer) {
        clearInterval(progressTimer)
        progressTimer = undefined
      }
      this.processing.visible = false
      this.processing.percent = 0
      this.processing.text = ''
    },

    async processAllMaterials() {
      if (!this.person || !this.conversation) return
      this.startProgressPolling()
      const base = '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sources'
      try {
        for (const source of this.conversation?.sources || []) {
          if (source.review_status === 'pending') {
            await api(base + '/' + encodeURIComponent(source.source_id) + '/review', {
              method: 'POST',
              body: JSON.stringify({ decision: 'confirmed' }),
            })
          }
        }
        await this.refreshConversation()
        for (const source of this.conversation?.sources || []) {
          if (source.review_status === 'confirmed' && !(source.llm_response_event_candidates || []).length) {
            const extractRes = await api<{ job_id: string }>(
              base + '/' + encodeURIComponent(source.source_id) + '/extract-candidates',
              { method: 'POST', body: '{}' },
            )
            await pollJob(extractRes.job_id)
          }
        }
        await this.refreshConversation()
        let confirmed = 0
        for (const source of this.conversation?.sources || []) {
          for (const candidate of source.llm_response_event_candidates || []) {
            if ((candidate.review_status || 'pending') === 'pending') {
              await api(
                base + '/' + encodeURIComponent(source.source_id) + '/candidates/' + encodeURIComponent(candidate.candidate_id) + '/review',
                { method: 'POST', body: JSON.stringify({ decision: 'confirmed' }) },
              )
              confirmed++
            }
          }
        }
        await this.refreshConversation()
        this.showToast(confirmed ? '处理完成：确认 ' + confirmed + ' 条候选，已形成新人物版本。' : '处理完成：没有待确认的候选。')
      } catch (error) {
        this.showToast((error as Error).message, true)
      } finally {
        this.stopProgressPolling()
      }
    },

    async reviewOptimization(candidateId: string, decision: string) {
      if (!this.person) return
      const data = await api<{ candidate: OptimizationCandidate }>(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/optimization/' + encodeURIComponent(candidateId) + '/review',
        { method: 'POST', body: JSON.stringify({ decision }) },
      )
      if (data.candidate.status === 'accepted_exploratory') {
        this.showToast('已生成探索性版本 v' + data.candidate.new_version + '；人物响应准确性仍未验证。')
      } else if (data.candidate.status === 'failed_validation') {
        this.showToast('优化未通过：' + (data.candidate.validation_reasons || []).join('、'), true)
      } else {
        this.showToast('候选已处理，当前版本未改变。')
      }
      await this.refreshConversation()
    },

    async reviewOptimizationStyle(candidateId: string, decision: string) {
      if (!this.person) return
      const data = await api<{ candidate: OptimizationCandidate }>(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/optimization/' + encodeURIComponent(candidateId) + '/style-review',
        { method: 'POST', body: JSON.stringify({ decision }) },
      )
      const status = data.candidate.surface_extraction?.status
      this.showToast(
        status === 'accepted_exploratory'
          ? '表达样本已单独验证并形成新风格版本；内容模型未改变。'
          : '表达样本已单独拒绝，内容模型不受影响。',
        status === 'failed_validation',
      )
      await this.refreshConversation()
    },

    async rollbackVersion(version: number) {
      if (!this.person) return
      await api('/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/versions/' + version + '/rollback', {
        method: 'POST',
        body: '{}',
      })
      this.showToast('已回退到版本 v' + version + '。')
      await this.refreshConversation()
    },

    async submitTextSource(body: Record<string, unknown>) {
      if (!this.person) return
      await api('/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sources/text', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      this.showToast('资料已保存为待审核，尚未进入人物版本。')
      await this.refreshConversation()
    },
    async submitFileSource(body: Record<string, unknown>) {
      if (!this.person) return
      await api('/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sources/file', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      this.showToast('文件已按事件候选整理并进入待审核队列。')
      await this.refreshConversation()
    },
    async submitUrlSource(body: Record<string, unknown>) {
      if (!this.person) return
      await api('/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sources/url', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      this.showToast('网页快照已按事件候选整理并进入待审核队列。')
      await this.refreshConversation()
    },

    async submitPerson(body: Record<string, unknown>) {
      const person = this.person
      if (!person) return
      await api('/api/people/' + encodeURIComponent(person.person_id), {
        method: 'PUT',
        body: JSON.stringify(body),
      })
      this.activeDialog = ''
      await this.loadPeople(person.person_id)
      this.showToast('人物信息已更新。')
    },

    async createPerson(body: Record<string, unknown>) {
      const data = await api<{ person: Person }>('/api/conversation/people', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      this.creatingPerson = false
      this.activeDialog = ''
      await this.loadPeople(data.person.person_id)
      this.showToast('人物已创建。')
    },

    async uploadAvatar(dataUrl: string) {
      const person = this.person
      if (!person) return
      await api('/api/people/' + encodeURIComponent(person.person_id) + '/avatar', {
        method: 'POST',
        body: JSON.stringify({ avatar: dataUrl }),
      })
      person.avatar = '/api/people/' + person.person_id + '/avatar'
      await this.loadPeople(person.person_id)
      this.showToast('头像已更新。')
    },
    async removeAvatar() {
      const person = this.person
      if (!person) return
      await api('/api/people/' + encodeURIComponent(person.person_id) + '/avatar', {
        method: 'POST',
        body: JSON.stringify({ avatar: '' }),
      })
      person.avatar = ''
      await this.loadPeople(person.person_id)
      this.showToast('已移除头像。')
    },

    async importBackup(payload: unknown) {
      const data = await api<{ person: Person }>('/api/import-product', {
        method: 'POST',
        body: JSON.stringify({ payload }),
      })
      await this.loadPeople(data.person.person_id)
      this.showToast('人物备份已加载。')
    },

    requestArchive(personId: string) {
      this.archiveTarget = this.people.find((p) => p.person_id === personId) || null
    },
    async archiveSelectedPerson() {
      const target = this.archiveTarget
      if (!target) return
      await api('/api/people/' + encodeURIComponent(target.person_id), { method: 'DELETE' })
      this.activeDialog = ''
      this.archiveTarget = null
      this.person = null
      this.conversation = null
      this.comparison = null
      await this.loadPeople()
      await this.refreshArchiveCount()
      this.showToast('“' + target.name + '”已移入归档。', false, async () => {
        await api('/api/archived-people/' + encodeURIComponent(target.person_id) + '/restore', {
          method: 'POST',
          body: '{}',
        })
        await this.loadPeople(target.person_id)
        await this.refreshArchiveCount()
      })
    },
    async restoreArchived(personId: string) {
      await api('/api/archived-people/' + encodeURIComponent(personId) + '/restore', {
        method: 'POST',
        body: '{}',
      })
      this.activeDialog = ''
      await this.loadPeople(personId)
      await this.refreshArchiveCount()
      this.showToast('人物及其资料、对话和版本已恢复。')
    },
    async permanentlyDelete(personId: string, expectedName: string) {
      await api('/api/archived-people/' + encodeURIComponent(personId), {
        method: 'DELETE',
        body: JSON.stringify({ expected_name: expectedName }),
      })
      this.activeDialog = ''
      this.permanentDeleteTarget = null
      await this.loadArchive()
      this.showToast('归档人物已永久删除，无法恢复。')
    },

    async refreshArchiveCount(): Promise<Person[]> {
      const data = await api<{ people: Person[] }>('/api/archived-people')
      this.archivedPeople = data.people
      return data.people
    },
    async loadArchive(): Promise<Person[]> {
      return this.refreshArchiveCount()
    },

    async loadSessions() {
      if (!this.person) return
      const data = await api<{ sessions: Session[] }>(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sessions',
      )
      this.sessions = data.sessions
    },
    async switchSession(sessionId: string) {
      if (!this.person) return
      await api(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sessions/' + encodeURIComponent(sessionId) + '/switch',
        { method: 'POST', body: '{}' },
      )
      await this.refreshConversation()
    },
    async renameSession(sessionId: string, title: string) {
      if (!this.person) return
      await api(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sessions/' + encodeURIComponent(sessionId) + '/rename',
        { method: 'POST', body: JSON.stringify({ title }) },
      )
      await this.loadSessions()
      await this.refreshConversation()
    },
    async deleteSession(sessionId: string) {
      if (!this.person) return
      await api(
        '/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/sessions/' + encodeURIComponent(sessionId),
        { method: 'DELETE', body: '{}' },
      )
      await this.loadSessions()
      await this.refreshConversation()
    },

    async setDialogueModel(modelRef: string) {
      if (!this.person) return
      await api('/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/model', {
        method: 'POST',
        body: JSON.stringify({ model_ref: modelRef }),
      })
      await this.refreshConversation()
      await this.loadModelServices()
      this.showToast('已切换此人物的对话模型；历史回答和人物版本没有改变。')
    },
    async clearDialogueModel() {
      if (!this.person) return
      await api('/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/model', {
        method: 'POST',
        body: JSON.stringify({ model_ref: '' }),
      })
      await this.refreshConversation()
      await this.loadModelServices()
      this.showToast('此人物已改为不使用对话模型；历史回答和人物版本没有改变。')
    },

    async modelServiceAction(serviceId: string, action: string, payload: Record<string, unknown> = {}) {
      const data = await api<{ job_id: string }>('/api/model-services/' + encodeURIComponent(serviceId) + '/' + action, {
        method: 'POST',
        body: JSON.stringify(payload),
      })
      const job = await pollJob(data.job_id)
      const result = (job.result || {}) as any
      const verifiedForPerson =
        action === 'test' && result.status === 'connected' && !!this.person && !!payload.model_id
      if (verifiedForPerson && this.person) {
        const modelRef = serviceId + ':' + payload.model_id
        await api('/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/model', {
          method: 'POST',
          body: JSON.stringify({ model_ref: modelRef }),
        })
        await this.refreshConversation()
      }
      await this.loadModelServices()
      const failed = result.status === 'unavailable'
      this.showToast(verifiedForPerson ? '真实调用验证成功，已用于当前人物。' : result.message || '模型列表已刷新。', failed)
    },

    async submitModelService(body: Record<string, unknown>) {
      const data = await api<{ service: ModelService }>('/api/model-services', {
        method: 'POST',
        body: JSON.stringify(body),
      })
      await this.loadModelServices()
      this.showToast('已保存，正在自动读取模型并验证…')
      await this.autoVerifyService(String(data.service?.service_id))
    },

    async autoVerifyService(serviceId: string) {
      try {
        const refreshRes = await api<{ job_id: string }>('/api/model-services/' + encodeURIComponent(serviceId) + '/refresh-models', {
          method: 'POST',
          body: '{}',
        })
        await pollJob(refreshRes.job_id)
        await this.loadModelServices()
        const service = this.modelServices.services.find((s) => s.service_id === serviceId)
        const models = service?.enabled_models || service?.models || []
        const modelId = (service?.default_model as string) || models[0]
        if (!modelId) {
          this.showToast('已保存，但没读到模型；可手动填模型 ID 后刷新。')
          return
        }
        const testRes = await api<{ job_id: string }>('/api/model-services/' + encodeURIComponent(serviceId) + '/test', {
          method: 'POST',
          body: JSON.stringify({ model_id: modelId }),
        })
        const testJob = await pollJob(testRes.job_id)
        const result = (testJob.result || {}) as any
        if (result.status === 'connected' && this.person) {
          const modelRef = serviceId + ':' + modelId
          await api('/api/people/' + encodeURIComponent(this.person.person_id) + '/conversation/model', {
            method: 'POST',
            body: JSON.stringify({ model_ref: modelRef }),
          })
          await this.refreshConversation()
        }
        await this.loadModelServices()
        const ok = result.status === 'connected'
        this.showToast(
          ok ? (this.person ? '已保存并自动验证，已用于当前人物。' : '已保存并自动验证通过。') : result.message || '已保存，但验证未通过。',
          !ok,
        )
      } catch (error) {
        this.showToast('已保存，但自动验证失败：' + (error as Error).message, true)
      }
    },

    async setModelRole(role: string, modelRef: string) {
      await api('/api/model-roles', { method: 'POST', body: JSON.stringify({ role, model_ref: modelRef }) })
      await this.loadModelServices()
    },
  },
})
