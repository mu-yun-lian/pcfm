<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppStore } from '../stores/app'
import { api } from '../api/client'
import { fileToBase64 } from '../lib/file'
import { shortTime } from '../lib/labels'
import type { Session } from '../types'

const store = useAppStore()

const personSearch = ref('')
const sessionSearch = ref('')
const openMenus = ref<string | null>(null)
const draggingId = ref<string | null>(null)
const zoneActive = ref(false)
const fileInput = ref<HTMLInputElement>()

const visiblePeople = computed(() => {
  const q = personSearch.value.trim().toLowerCase()
  return store.people.filter((p) => (p.name + ' ' + (p.last_message || '')).toLowerCase().includes(q))
})

const visibleSessions = computed(() => {
  const q = sessionSearch.value.trim().toLowerCase()
  return store.sessions.filter((s) => (s.title || '新对话').toLowerCase().includes(q))
})

const archiveCount = computed(() => store.archivedPeople.length)

function onAvatarError(event: Event) {
  ;(event.target as HTMLImageElement).src = '/default-person-avatar.png'
}

async function selectPerson(id: string) {
  openMenus.value = null
  if (id === 'assistant') await store.selectAssistant()
  else await store.selectPerson(id)
}

function editPerson(id: string) {
  openMenus.value = null
  store.selectPerson(id).then(() => store.openDialog('person'))
}

function toggleMenu(id: string) {
  openMenus.value = openMenus.value === id ? null : id
}

function requestArchive(id: string) {
  openMenus.value = null
  store.requestArchive(id)
  store.openDialog('archiveConfirm')
}

async function openArchive() {
  await store.loadArchive()
  store.openDialog('archive')
}

async function renameSession(s: Session) {
  const current = s.title || ''
  const title = window.prompt('新标题：', current)
  if (title === null) return
  await store.renameSession(s.session_id, title)
}

async function deleteSession(s: Session) {
  if (!window.confirm('删除这个会话？消息将无法恢复。')) return
  await store.deleteSession(s.session_id)
}

function onImportFile(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) importBackup(file)
  input.value = ''
}

async function importBackup(file: File) {
  try {
    const payload = JSON.parse(await file.text())
    await store.importBackup(payload)
  } catch (error) {
    store.showToast((error as Error).message, true)
  }
}

// --- drag-to-archive ---
function onDragStart(event: DragEvent, personId: string) {
  draggingId.value = personId
  if (event.dataTransfer) {
    event.dataTransfer.effectAllowed = 'move'
    event.dataTransfer.setData('text/plain', personId)
  }
}
function onDragEnd() {
  draggingId.value = null
  zoneActive.value = false
}
function onZoneDragOver(event: DragEvent) {
  if (draggingId.value) {
    event.preventDefault()
    zoneActive.value = true
  }
}
function onZoneDragLeave() {
  zoneActive.value = false
}
function onZoneDrop(event: DragEvent) {
  event.preventDefault()
  zoneActive.value = false
  const personId = event.dataTransfer?.getData('text/plain') || draggingId.value
  draggingId.value = null
  if (personId) requestArchive(personId)
}

// --- drop files onto a person card ---
function onCardDragOver(event: DragEvent) {
  if (event.dataTransfer && Array.from(event.dataTransfer.types).includes('Files')) {
    event.preventDefault()
    ;(event.currentTarget as HTMLElement).classList.add('file-over')
  }
}
function onCardDragLeave(event: DragEvent) {
  const card = event.currentTarget as HTMLElement
  if (!card.contains(event.relatedTarget as Node | null)) card.classList.remove('file-over')
}
async function onCardDrop(event: DragEvent, personId: string) {
  const card = event.currentTarget as HTMLElement
  card.classList.remove('file-over')
  const files = event.dataTransfer ? Array.from(event.dataTransfer.files) : []
  if (!files.length) return
  if (personId === 'assistant') {
    store.showToast('AI 助手不接收资料文件。', true)
    return
  }
  await store.selectPerson(personId)
  for (const file of files) {
    try {
      const content_base64 = await fileToBase64(file)
      await api('/api/people/' + encodeURIComponent(personId) + '/conversation/sources/file', {
        method: 'POST',
        body: JSON.stringify({
          filename: file.name,
          content_base64,
          speaker: store.person?.name || '',
          source_date: '',
          dataset_role: 'model_source',
          content_authenticity: 'unverified_material',
          speaker_scope: 'single_speaker_entire_document',
        }),
      })
    } catch (error) {
      store.showToast('上传「' + file.name + '」失败：' + (error as Error).message, true)
    }
  }
  await store.refreshConversation()
  store.showToast('已把 ' + files.length + ' 份文件添加为待审核资料，可在「人物资料」里一键处理。')
}
</script>

<template>
  <aside class="people-panel" :class="{ 'mobile-open': store.sidebarOpen }">
    <div class="sidebar-brand">
      <span class="brand-wordmark">PCFM</span><span class="brand-sub">对话式人物模拟</span>
      <button class="ss-new" title="新建人物" aria-label="新建人物" @click="store.openCreatePerson()">＋</button>
    </div>

    <label class="sidebar-search">
      <svg viewBox="0 0 20 20" width="16" height="16" aria-hidden="true">
        <circle cx="9" cy="9" r="6" fill="none" stroke="currentColor" stroke-width="1.6" />
        <line x1="13.6" y1="13.6" x2="18" y2="18" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
      </svg>
      <input v-model="personSearch" type="search" placeholder="搜索人物或最近对话" />
    </label>

    <div class="people-list">
      <article class="person-card" :class="{ active: store.person?.person_id === 'assistant' }">
        <button class="person-select" @click="selectPerson('assistant')">
          <span class="assistant-emoji">🤖</span>
          <span><strong>AI 助手</strong><small class="recent">建人物·加材料·搜索·归档</small></span>
        </button>
      </article>

      <article
        v-for="person in visiblePeople"
        :key="person.person_id"
        class="person-card"
        :class="{ active: store.person?.person_id === person.person_id, dragging: draggingId === person.person_id }"
        draggable="true"
        @dragstart="onDragStart($event, person.person_id)"
        @dragend="onDragEnd"
        @dragover="onCardDragOver"
        @dragleave="onCardDragLeave"
        @drop.prevent="onCardDrop($event, person.person_id)"
      >
        <button class="person-select" @click="selectPerson(person.person_id)">
          <img :src="person.avatar || '/default-person-avatar.png'" :alt="person.name" @error="onAvatarError" />
          <span>
            <strong>{{ person.name }}<em v-if="person.is_demo" class="demo-badge">演示</em></strong>
            <small class="recent">{{ person.last_message || '开始对话' }}</small>
          </span>
        </button>
        <button class="person-more" :aria-label="person.name + '的更多操作'" @click.stop="toggleMenu(person.person_id)">⋯</button>
        <div class="person-menu" v-if="openMenus === person.person_id">
          <button @click="editPerson(person.person_id)">编辑人物</button>
          <a :href="'/api/people/' + encodeURIComponent(person.person_id) + '/export'">导出备份</a>
          <button @click="requestArchive(person.person_id)">移入归档</button>
        </div>
      </article>
    </div>
    <div class="people-empty" v-if="!visiblePeople.length">还没有人物。新建人物后，为他添加原始资料即可开始。</div>

    <div class="sidebar-sessions" v-if="!store.isAssistant">
      <div class="sidebar-sessions-head">
        <strong>会话</strong>
        <button class="ss-new" title="新对话" @click="store.startNewConversation()">＋</button>
      </div>
      <input v-model="sessionSearch" class="session-search" type="search" placeholder="搜索会话" />
      <div id="sidebar-sessions-list">
        <div v-for="s in visibleSessions" :key="s.session_id" class="sidebar-session" :class="{ active: s.active }">
          <button class="ss-main" @click="store.switchSession(s.session_id)">
            <span class="ss-title">{{ s.title || '新对话' }}</span>
            <span class="ss-meta">{{ s.message_count }} 条 · {{ shortTime(s.updated_at) }}</span>
          </button>
          <span class="ss-actions">
            <button class="ss-btn" title="重命名" @click.stop="renameSession(s)">✎</button>
            <button class="ss-btn" title="删除" @click.stop="deleteSession(s)">✕</button>
          </span>
        </div>
        <p class="people-empty" v-if="!visibleSessions.length">{{ sessionSearch ? '无匹配会话' : '暂无会话' }}</p>
      </div>
    </div>

    <div class="people-footer">
      <div
        class="archive-drop-zone"
        :class="{ 'drag-ready': !!draggingId, 'drop-active': zoneActive }"
        @dragover="onZoneDragOver"
        @dragleave="onZoneDragLeave"
        @drop="onZoneDrop"
      >
        <button class="footer-link" @click="openArchive">人物归档 <span class="count-badge">{{ archiveCount }}</span></button>
        <small>{{ draggingId ? '放到这里，移入归档' : '也可以把人物卡片拖到这里' }}</small>
      </div>
      <button class="footer-link" @click="fileInput?.click()">加载人物备份</button>
      <input ref="fileInput" type="file" accept="application/json" hidden @change="onImportFile" />
      <p class="footer-note">数据仅保存在本机 · 人物间资料完全隔离</p>
    </div>
  </aside>
</template>
