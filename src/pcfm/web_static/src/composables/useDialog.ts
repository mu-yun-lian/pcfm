import { onMounted, ref } from 'vue'
import { useAppStore } from '../stores/app'

export function useDialog(name: string) {
  const store = useAppStore()
  const el = ref<HTMLDialogElement>()

  onMounted(() => {
    el.value?.showModal()
  })

  function close() {
    if (el.value?.open) el.value?.close()
  }

  function onClose() {
    if (store.activeDialog === name) store.activeDialog = ''
  }

  return { el, close, onClose }
}
