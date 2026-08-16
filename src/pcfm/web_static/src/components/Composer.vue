<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import { useAppStore } from '../stores/app'

const store = useAppStore()
const textareaEl = ref<HTMLTextAreaElement>()
const sending = ref(false)
const statusMessage = ref('')
const lookup = ref(false)

const personKey = computed(() => store.person?.person_id || '')

const text = computed({
  get: () => store.drafts[personKey.value] || '',
  set: (v: string) => {
    if (store.person) store.drafts[store.person.person_id] = v
  },
})

const modelPickerLabel = computed(() => (store.isAssistant ? '助手模型' : store.currentModelLabel))

watch(personKey, () => {
  sending.value = false
  statusMessage.value = ''
  lookup.value = false
})

watch(
  () => store.composerFocusRequest,
  () => {
    nextTick(() => textareaEl.value?.focus())
  },
)

function onKeydown(event: KeyboardEvent) {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    submit()
  }
}

function openModelPicker() {
  store.loadModelServices().then(() => store.openDialog('model'))
}

async function submit() {
  const value = text.value.trim()
  if (!value || !store.person) return
  statusMessage.value = ''
  sending.value = true
  try {
    await store.sendMessage(value, lookup.value)
    text.value = ''
    lookup.value = false
  } catch (error) {
    statusMessage.value = '发送失败：' + (error as Error).message + ' 输入内容已保留。'
    store.showToast((error as Error).message, true)
  } finally {
    sending.value = false
  }
}
</script>

<template>
  <footer class="composer-wrap">
    <form class="composer" @submit.prevent="submit">
      <div class="composer-row">
        <textarea
          ref="textareaEl"
          v-model="text"
          rows="1"
          placeholder="直接对这个人物说话（Enter 发送，Shift+Enter 换行）"
          @keydown="onKeydown"
        ></textarea>
        <details class="composer-tools">
          <summary aria-label="更多选项" title="更多选项">＋</summary>
          <div class="composer-tools-body">
            <button type="button" class="tool-item" @click="store.openDialog('sources')">添加资料</button>
            <button type="button" class="tool-item model-picker-button" @click="openModelPicker">{{ modelPickerLabel }}</button>
            <label class="lookup-switch">
              <input v-model="lookup" type="checkbox" />
              <span>本次查找并对照现实回答</span>
              <small>可能增加耗时和费用</small>
            </label>
          </div>
        </details>
        <button class="send-button" type="submit" :disabled="sending">{{ sending ? '生成中…' : '发送' }}</button>
      </div>
    </form>
    <p class="composer-status" role="status" v-if="statusMessage">{{ statusMessage }}</p>
    <p class="composer-note">模型可能出错；请结合证据、适用范围和不确定性判断。</p>
  </footer>
</template>
