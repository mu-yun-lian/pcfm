<script setup lang="ts">
import { ref } from 'vue'
import { useDialog } from '../../composables/useDialog'
import { useAppStore } from '../../stores/app'

const { el, close, onClose } = useDialog('archiveConfirm')
const store = useAppStore()
const busy = ref(false)

async function confirm() {
  busy.value = true
  try {
    await store.archiveSelectedPerson()
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
        <div><p class="kicker">可恢复操作</p><h2>移入归档</h2></div>
        <button type="button" class="close-button" @click="close">关闭</button>
      </div>
      <p>人物将从人物库隐藏；资料、对话、模型版本和优化记录都会保留，之后可以恢复。</p>
      <div class="dialog-actions">
        <button type="button" class="button quiet" @click="close">取消</button>
        <button type="button" class="button primary" :disabled="busy" @click="confirm">{{ busy ? '处理中…' : '确认移入归档' }}</button>
      </div>
    </div>
  </dialog>
</template>
