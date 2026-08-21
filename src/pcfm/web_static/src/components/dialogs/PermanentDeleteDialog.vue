<script setup lang="ts">
import { ref, computed } from 'vue'
import { useDialog } from '../../composables/useDialog'
import { useAppStore } from '../../stores/app'

const { el, close, onClose } = useDialog('permanentDelete')
const store = useAppStore()
const nameInput = ref('')
const busy = ref(false)

const target = computed(() => store.permanentDeleteTarget)
const nameMatches = computed(() => !!target.value && nameInput.value.trim() === target.value.name)

async function confirm() {
  if (!target.value) return
  busy.value = true
  try {
    await store.permanentlyDelete(target.value.person_id, nameInput.value)
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
        <div><p class="kicker">不可恢复操作</p><h2>永久删除人物</h2></div>
        <button type="button" class="close-button" @click="close">关闭</button>
      </div>
      <p>永久删除后，人物资料、对话、模型版本和优化记录都无法恢复。</p>
      <label>请输入人物名称以确认<input v-model="nameInput" autocomplete="off" /></label>
      <div class="dialog-actions">
        <button type="button" class="button quiet" @click="close">取消</button>
        <button type="button" class="button primary danger" :disabled="busy || !nameMatches" @click="confirm">{{ busy ? '处理中…' : '永久删除' }}</button>
      </div>
    </div>
  </dialog>
</template>
