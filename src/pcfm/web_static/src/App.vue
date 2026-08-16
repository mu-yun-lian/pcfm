<script setup lang="ts">
import { onMounted } from 'vue'
import { useAppStore } from './stores/app'
import PeopleSidebar from './components/PeopleSidebar.vue'
import ComparisonDrawer from './components/ComparisonDrawer.vue'
import PersonDialog from './components/dialogs/PersonDialog.vue'
import ModelServicesDialog from './components/dialogs/ModelServicesDialog.vue'
import NewConversationDialog from './components/dialogs/NewConversationDialog.vue'
import SourcesDialog from './components/dialogs/SourcesDialog.vue'
import VersionsDialog from './components/dialogs/VersionsDialog.vue'
import ArchiveDialog from './components/dialogs/ArchiveDialog.vue'
import ArchiveConfirmDialog from './components/dialogs/ArchiveConfirmDialog.vue'
import PermanentDeleteDialog from './components/dialogs/PermanentDeleteDialog.vue'
import AdvancedDialog from './components/dialogs/AdvancedDialog.vue'

const store = useAppStore()

onMounted(() => store.bootstrap())

function reloadPage() {
  window.location.reload()
}

async function undoToast() {
  const action = store.toast?.action
  store.hideToast()
  if (action) await action()
}
</script>

<template>
  <div id="version-banner" class="version-banner" v-if="store.versionMismatch">
    页面与本地服务版本不一致，请重启本地服务。
    <button type="button" @click="reloadPage">重新检查</button>
  </div>

  <div class="app-shell" :class="{ 'drawer-open': store.comparison }">
    <PeopleSidebar />
    <main class="chat-panel">
      <router-view />
    </main>
    <ComparisonDrawer v-if="store.comparison" />
  </div>

  <PersonDialog v-if="store.activeDialog === 'person'" />
  <ModelServicesDialog v-if="store.activeDialog === 'model'" />
  <NewConversationDialog v-if="store.activeDialog === 'newConversation'" />
  <SourcesDialog v-if="store.activeDialog === 'sources'" />
  <VersionsDialog v-if="store.activeDialog === 'versions'" />
  <ArchiveDialog v-if="store.activeDialog === 'archive'" />
  <ArchiveConfirmDialog v-if="store.activeDialog === 'archiveConfirm'" />
  <PermanentDeleteDialog v-if="store.activeDialog === 'permanentDelete'" />
  <AdvancedDialog v-if="store.activeDialog === 'advanced'" />

  <div class="toast" :class="{ error: store.toast?.error }" v-if="store.toast">
    {{ store.toast.message }}
    <button v-if="store.toast.action" type="button" @click="undoToast">撤销</button>
  </div>
</template>
