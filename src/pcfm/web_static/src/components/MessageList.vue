<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import { useAppStore } from '../stores/app'
import { statusLabel, humanStatus } from '../lib/labels'
import type { Message } from '../types'

const store = useAppStore()
const messagesEl = ref<HTMLElement>()
const expanded = ref<Set<string>>(new Set())
const runningReality = ref('')
const feedingBack = ref('')

const messages = computed(() => store.conversation?.messages || [])
const suggestions = computed(() => store.person?.recommended_questions || [])

const emptyHint = computed(() =>
  store.conversation?.active_version
    ? '系统会把当前消息作为完整会话状态的增量，再结合历史事件、人物公开取向和外部知识组织回答。'
    : '尚未建立人物模型；选择对话模型后仍可正常回答，但会标记为通用知识而非人物预测。',
)

function toggleEvidence(id: string) {
  const next = new Set(expanded.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  expanded.value = next
}

function evidenceMeta(m: Message): string {
  const first = statusLabel(m.answer_status || m.status)
  const second =
    m.person_prediction_status && m.person_prediction_status !== 'not_available'
      ? '证据支持 ' + Number(m.confidence).toFixed(2) + '（非准确率）'
      : '非人物预测'
  return first + ' · ' + second
}

function modelUsageText(m: Message): string {
  const usage = m.model_usage || {}
  return usage.total_calls && usage.total_calls > 0
    ? '本次大模型调用 ' + usage.total_calls + ' 次'
    : '本次未调用大模型'
}

function dialogueModelText(m: Message): string {
  return m.dialogue_model_provider && m.dialogue_model_id
    ? m.dialogue_model_provider + ' · ' + m.dialogue_model_id
    : '未选择'
}

function fillSuggestion(text: string) {
  if (store.person) store.drafts[store.person.person_id] = text
  store.requestComposerFocus()
}

async function runReality(m: Message) {
  runningReality.value = m.message_id
  try {
    await store.runRealityLookup(m.message_id)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    runningReality.value = ''
  }
}

async function feedback(m: Message) {
  if (m.feedback) return
  feedingBack.value = m.message_id
  try {
    await store.sendFeedback(m.message_id)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    feedingBack.value = ''
  }
}

function onAvatarError(event: Event) {
  ;(event.target as HTMLImageElement).src = '/default-person-avatar.png'
}

watch(
  () => store.conversation?.messages.map((m) => m.message_id + ':' + m.text).join('|'),
  () => {
    nextTick(() => {
      if (messagesEl.value) messagesEl.value.scrollTop = messagesEl.value.scrollHeight
    })
  },
  { immediate: true },
)
</script>

<template>
  <div ref="messagesEl" class="messages" aria-live="polite">
    <!-- AI assistant conversation -->
    <template v-if="store.isAssistant">
      <div v-if="!messages.length" class="messages-empty">
        <div>
          <strong>AI 助手</strong>
          <p>说个意图，我列步骤帮你操作：建人物 / 加材料 / 搜索 / 归档 / 恢复 / 永久删除。</p>
        </div>
      </div>
      <template v-else>
        <article v-for="m in messages" :key="m.message_id" class="message-row" :class="m.role">
          <div v-if="m.role === 'user'" class="user-bubble">{{ m.text }}</div>
          <template v-else>
            <img class="assistant-avatar" :src="'/default-person-avatar.png'" alt="" />
            <div class="assistant-body"><div class="answer">{{ m.text }}</div></div>
          </template>
        </article>
      </template>
    </template>

    <!-- person conversation -->
    <template v-else>
      <div v-if="!messages.length" class="messages-empty">
        <div>
          <strong>现在可以直接开始对话</strong>
          <p>{{ emptyHint }}</p>
          <div v-if="suggestions.length" class="suggestion-list">
            <p>推荐测试问题</p>
            <button v-for="s in suggestions" :key="s.text" class="suggestion-chip" @click="fillSuggestion(s.text)">
              <span>{{ s.label }}</span>{{ s.text }}
            </button>
          </div>
        </div>
      </div>

      <article
        v-for="m in messages"
        :key="m.message_id"
        class="message-row"
        :class="[m.role, { refused: m.status === 'refused' }]"
      >
        <div v-if="m.role === 'user'" class="user-bubble">{{ m.text }}</div>
        <template v-else>
          <img
            class="assistant-avatar"
            :src="store.person?.avatar || '/default-person-avatar.png'"
            alt=""
            @error="onAvatarError"
          />
          <div class="assistant-body">
            <div class="answer">{{ m.text }}</div>
            <button v-if="m.status === 'generating' && store.activeJobId" class="mini-button" type="button" @click="store.cancelActiveJob()">停止生成</button>

            <div v-if="m.uncertainties && m.uncertainties.length" class="plain-notice">
              不确定项：{{ m.uncertainties.join('；') }}
            </div>

            <div v-if="m.status !== 'generating'" class="message-actions">
              <button @click="toggleEvidence(m.message_id)">依据</button>
              <button :disabled="runningReality === m.message_id" @click="runReality(m)">
                {{ runningReality === m.message_id ? '查找中…' : '现实回答' }}
              </button>
              <button :disabled="!!m.feedback || feedingBack === m.message_id" @click="feedback(m)">
                {{ m.feedback ? '已反馈' : '反馈' }}
              </button>
              <button
                v-if="m.comparison && m.comparison.status === 'candidate_found'"
                @click="store.openComparison(m.comparison)"
              >
                发现可核验回答
              </button>
            </div>

            <div v-if="expanded.has(m.message_id)" class="evidence-details">
              <p class="evidence-meta">{{ evidenceMeta(m) }}</p>
              <template v-if="m.evidence && m.evidence.length">
                <p v-for="(item, i) in m.evidence" :key="i">
                  <strong>{{ item.title }}</strong><br />
                  响应事件：{{ item.event_id || '未记录' }} · 说话人：{{ item.speaker || '未记录' }} · 日期：{{ item.date || '未记录' }} · 位置：{{ item.locator }}<br />
                  候选采用分数：{{ Number(item.support_score).toFixed(2) }}
                  <a v-if="item.url" :href="item.url" target="_blank" rel="noreferrer">打开来源</a>
                </p>
              </template>
              <p v-else>本次没有可用于支持预测内容的直接证据。</p>
              <p>
                回应动作：{{ m.structured_prediction?.speech_act?.label || '未输出' }}；立场：{{ m.structured_prediction?.stance?.label || '未输出' }}；回答路径：{{ statusLabel(m.answer_status || m.status) }}；知识来源：{{ m.knowledge_source || 'none' }}
              </p>
              <p>
                内容模型：{{ m.model_kind }}；对话模型：{{ dialogueModelText(m) }}；{{ modelUsageText(m) }}；表达状态：{{ humanStatus(m.style_status) }}；准确性：{{ humanStatus(m.response_accuracy_status) }}
              </p>
            </div>
          </div>
        </template>
      </article>
    </template>
  </div>
</template>
