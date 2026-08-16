<script setup lang="ts">
import { ref, computed } from 'vue'
import { useDialog } from '../../composables/useDialog'
import { useAppStore } from '../../stores/app'
import { humanStatus } from '../../lib/labels'

const { el, close, onClose } = useDialog('versions')
const store = useAppStore()
const busy = ref(0)

const versions = computed(() => [...(store.conversation?.versions || [])].reverse())
const active = computed(() => store.conversation?.active_version)

async function rollback(v: number) {
  busy.value = v
  try {
    await store.rollbackVersion(v)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    busy.value = 0
  }
}
</script>

<template>
  <dialog ref="el" @close="onClose">
    <div class="dialog-card">
      <div class="dialog-head">
        <div><p class="kicker">可追溯与可回退</p><h2>人物模型版本</h2></div>
        <button type="button" class="close-button" @click="close">关闭</button>
      </div>
      <div class="versions-list">
        <article v-for="version in versions" :key="version.version" class="version-item">
          <header>
            <strong>版本 v{{ version.version }}</strong>
            <span class="tag" :class="{ active: version.version === active }">{{ version.version === active ? '当前' : '历史' }}</span>
          </header>
          <p>{{ version.reason }} · {{ version.created_at }}</p>
          <p>内容：{{ humanStatus(version.content_update_status) }} · 风格：{{ humanStatus(version.style_update_status) }} · 准确性：{{ humanStatus(version.response_accuracy_status) }}</p>
          <p v-if="version.validation_status === 'invalidated_evidence_contract'" class="warning-box">此版本的证据契约不合格，不能恢复为当前版本。</p>
          <button v-else-if="version.version !== active" class="mini-button" :disabled="busy === version.version" @click="rollback(version.version)">回退到此版本</button>
        </article>
        <p v-if="!versions.length" class="people-empty">尚未形成版本。确认至少一份可训练的本人逐字回答后会建立探索性 v1。</p>
      </div>
    </div>
  </dialog>
</template>
