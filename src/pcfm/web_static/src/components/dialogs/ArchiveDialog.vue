<script setup lang="ts">
import { ref, computed } from 'vue'
import { useDialog } from '../../composables/useDialog'
import { useAppStore } from '../../stores/app'
import type { Person } from '../../types'

const { el, close, onClose } = useDialog('archive')
const store = useAppStore()
const busy = ref('')

const people = computed(() => store.archivedPeople)

async function restore(personId: string) {
  busy.value = personId
  try {
    await store.restoreArchived(personId)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    busy.value = ''
  }
}

function destroy(person: Person) {
  store.permanentDeleteTarget = person
  store.openDialog('permanentDelete')
}
</script>

<template>
  <dialog ref="el" @close="onClose">
    <div class="dialog-card">
      <div class="dialog-head">
        <div><p class="kicker">可恢复删除</p><h2>人物归档</h2></div>
        <button type="button" class="close-button" @click="close">关闭</button>
      </div>
      <p class="drawer-intro">普通删除只移入归档。只有这里允许永久删除。</p>
      <div class="versions-list">
        <article v-for="person in people" :key="person.person_id" class="version-item">
          <header><strong>{{ person.name }}</strong><span class="tag">已归档</span></header>
          <p>归档时间：{{ person.archived_at || '未记录' }}</p>
          <p>{{ person.source_count }} 份资料 · {{ person.message_count }} 条消息 · {{ person.version_count }} 个模型版本</p>
          <div class="item-actions">
            <button class="mini-button confirm" :disabled="busy === person.person_id" @click="restore(person.person_id)">恢复人物</button>
            <button class="mini-button" @click="destroy(person)">永久删除</button>
          </div>
        </article>
        <p v-if="!people.length" class="people-empty">暂无归档人物。可以在人物卡片的更多菜单中将人物移入归档。</p>
      </div>
    </div>
  </dialog>
</template>
