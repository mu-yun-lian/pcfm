<script setup lang="ts">
import { computed } from 'vue'
import { useAppStore } from '../stores/app'
import type { CollectionState } from '../types'

const store = useAppStore()

const avatarSrc = computed(() => store.person?.avatar || '/default-person-avatar.png')
const personName = computed(() => (store.isAssistant ? 'AI 助手' : store.person?.name || ''))
const versionText = computed(() => {
  if (store.isAssistant) return '操作员助手'
  return store.conversation?.active_version ? '模型版本 v' + store.conversation.active_version : '未建立模型版本'
})
const sourceCountText = computed(() => {
  if (store.isAssistant) return '建人物 · 加材料 · 搜索 · 归档'
  return (store.conversation?.source_counts?.confirmed ?? 0) + ' 份已确认资料'
})
const sessionTitle = computed(() => store.conversation?.session_title || '新对话')
const sessionCount = computed(() => (store.conversation?.messages?.length || 0) + ' 条消息')
const modelPickerLabel = computed(() => (store.isAssistant ? '助手模型' : store.currentModelLabel))

const collectionStatusText = computed(() => {
  if (store.isAssistant) return '说个意图，我列步骤帮你操作。'
  const conv = store.conversation
  const collection = (store.person?.collection || conv?.profile?.collection || {}) as CollectionState
  const count = collection.candidate_count || 0
  const messages: Record<string, string> = {
    candidates_found: '系统已找到 ' + count + ' 条公开资料候选；核验原文前不会用于训练。',
    no_candidates: '系统已完成公开搜索，但没有找到可用候选。',
    temporarily_unavailable: '公开资料搜索暂时不可用，可以稍后重试或自行提供资料。',
    search_ready: '搜索服务已配置；结果只进入待审核候选资料。',
    awaiting_user_materials: '等待用户提供原始资料；系统会自动提取待审核响应事件。',
    verified_demo_materials_loaded: '已载入可追溯的一手演示资料；预测仍属探索性，准确性尚未验证。',
  }
  const modelView = conv?.public_response_model || {}
  const stepHint = conv?.active_version ? '' : ' 还差几步就能对话：加材料 → 「一键处理全部材料」→ 形成版本。'
  return (
    (messages[collection.status || ''] || collection.message || '资料状态尚未记录。') +
    stepHint +
    ' 当前模拟层：' +
    (modelView.event_frame_count || 0) +
    ' 个事件原子、' +
    (modelView.value_atom_count || 0) +
    ' 个单事件公开取向原子、' +
    (modelView.value_orientation_count || 0) +
    ' 个聚合公开取向、' +
    (modelView.preference_structure_count || 0) +
    ' 个明确取舍结构、' +
    (modelView.knowledge_claim_count || 0) +
    ' 条人物公开使用的知识主张。'
  )
})

const modelUnavailable = computed(() => {
  const modelRef = store.conversation?.dialogue_model_ref || ''
  if (!modelRef) return false
  const service = store.modelServices.services.find((s) => modelRef.startsWith(s.service_id + ':'))
  const selectedModelId = modelRef.split(':').slice(1).join(':')
  return !!service && !(service.call_readiness === 'ready' && service.last_probe_model === selectedModelId)
})
const modelUnavailableName = computed(() => {
  const modelRef = store.conversation?.dialogue_model_ref || ''
  const service = store.modelServices.services.find((s) => modelRef.startsWith(s.service_id + ':'))
  return service?.display_name || ''
})

function onAvatarError(event: Event) {
  ;(event.target as HTMLImageElement).src = '/default-person-avatar.png'
}

function openModelPicker() {
  store.loadModelServices().then(() => store.openDialog('model'))
}
</script>

<template>
  <header class="chat-header">
    <div class="person-heading">
      <img :src="avatarSrc" alt="人物头像" @error="onAvatarError" />
      <div class="person-heading-text">
        <h2>{{ personName }}</h2>
        <p>
          <span>{{ versionText }}</span><span class="dot">·</span><span>{{ sourceCountText }}</span>
        </p>
      </div>
    </div>
    <div class="header-right">
      <span class="session-title">{{ sessionTitle }}</span>
      <span class="session-count">{{ sessionCount }}</span>
      <span class="model-state" v-if="store.isAssistant">AI 助手（工具调用）</span>
      <span class="model-state insufficient" v-else-if="modelUnavailable">
        对话模型需验证：<strong>{{ modelUnavailableName }}</strong>
        <button type="button" class="text-button" @click="openModelPicker">处理</button>
      </span>
      <span
        class="model-state"
        :class="{ insufficient: store.conversation?.status === 'insufficient_evidence' }"
        v-else
      >
        {{ store.conversation?.status_text }}
      </span>
      <details class="menu">
        <summary aria-label="更多操作">⋯</summary>
        <div class="menu-body">
          <button type="button" @click="store.openDialog('newConversation')">新对话</button>
          <button type="button" @click="store.openDialog('sources')">人物资料</button>
          <button type="button" @click="store.openDialog('versions')">版本</button>
          <button type="button" @click="store.openDialog('advanced')">高级</button>
          <button type="button" class="menu-model" @click="openModelPicker">{{ modelPickerLabel }}</button>
        </div>
      </details>
    </div>
  </header>

  <details class="model-status">
    <summary>模型状态</summary>
    <div class="model-status-body">
      <p class="trust-note">分层预测：非生成式人物模型先预测回应动作、立场、主张和理由，再由独立表达层改写；准确性未验证时只标记为探索性。</p>
      <p class="collection-status">{{ collectionStatusText }}</p>
    </div>
  </details>
</template>
