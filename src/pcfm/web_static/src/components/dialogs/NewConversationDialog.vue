<script setup lang="ts">
import { ref } from 'vue'
import { useDialog } from '../../composables/useDialog'
import { useAppStore } from '../../stores/app'

const { el, close, onClose } = useDialog('newConversation')
const store = useAppStore()
const busy = ref(false)

async function confirm() {
  busy.value = true
  try {
    await store.startNewConversation()
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <dialog ref="el" @close="onClose">
    <div class="dialog-card">
      <div class="dialog-head">
        <div><p class="kicker">保留旧记录</p><h2>开始新对话</h2></div>
        <button type="button" class="close-button" @click="close">关闭</button>
      </div>
      <p>当前消息会保存到本机归档，新对话不再把旧消息作为上下文。人物资料、人物模型和所选对话模型不变。</p>
      <div class="dialog-actions">
        <button type="button" class="button quiet" @click="close">取消</button>
        <button type="button" class="button primary" :disabled="busy" @click="confirm">{{ busy ? '处理中…' : '开始新对话' }}</button>
      </div>
    </div>
  </dialog>
</template>
