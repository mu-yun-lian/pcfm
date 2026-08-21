<script setup lang="ts">
import { ref, computed } from 'vue'
import { useDialog } from '../../composables/useDialog'
import { useAppStore } from '../../stores/app'
import { fileToDataUrl } from '../../lib/file'

const { el, close, onClose } = useDialog('person')
const store = useAppStore()

const busy = ref(false)
const avatarPreview = ref(store.person?.avatar || '/default-person-avatar.png')
const avatarFileInput = ref<HTMLInputElement>()
const avatarDropZone = ref<HTMLElement>()

const isCreate = computed(() => store.creatingPerson)
const name = ref(isCreate.value ? '' : store.person?.name || '')
const description = ref(isCreate.value ? '' : store.person?.description || '')
const identity_note = ref(isCreate.value ? '' : store.person?.identity_note || '')
const aliases = ref(isCreate.value ? '' : (store.conversation?.profile?.aliases || []).join(', '))
const language = ref(isCreate.value ? 'zh' : store.conversation?.profile?.language || 'zh')
const focus_domain = ref(isCreate.value ? '' : store.person?.focus_domain || '')

async function submit() {
  busy.value = true
  const body = {
    name: name.value,
    description: description.value,
    aliases: aliases.value.split(/[,，]/).map((x) => x.trim()).filter(Boolean),
    language: language.value,
    identity_note: identity_note.value,
    focus_domain: focus_domain.value,
  }
  try {
    if (isCreate.value) {
      await store.createPerson(body)
    } else {
      await store.submitPerson(body)
    }
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    busy.value = false
  }
}

function onAvatarFile(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) uploadAvatar(file)
  input.value = ''
}

async function uploadAvatar(file: File) {
  if (!/^image\//.test(file.type)) {
    store.showToast('请选择图片文件。', true)
    return
  }
  if (file.size > 2 * 1024 * 1024) {
    store.showToast('头像图片不能超过 2 MB。', true)
    return
  }
  try {
    const dataUrl = await fileToDataUrl(file)
    await store.uploadAvatar(dataUrl)
    avatarPreview.value = dataUrl
  } catch (error) {
    store.showToast((error as Error).message, true)
  }
}

async function removeAvatar() {
  try {
    await store.removeAvatar()
    avatarPreview.value = '/default-person-avatar.png'
  } catch (error) {
    store.showToast((error as Error).message, true)
  }
}

function onAvatarDragOver(event: DragEvent) {
  event.preventDefault()
  avatarDropZone.value?.classList.add('dragover')
}
function onAvatarDragLeave() {
  avatarDropZone.value?.classList.remove('dragover')
}
function onAvatarDrop(event: DragEvent) {
  event.preventDefault()
  avatarDropZone.value?.classList.remove('dragover')
  const file = event.dataTransfer?.files[0]
  if (file) uploadAvatar(file)
}
</script>

<template>
  <dialog ref="el" @close="onClose">
    <form class="dialog-card" @submit.prevent="submit">
      <div class="dialog-head">
        <div><p class="kicker">人物资料</p><h2>{{ isCreate ? '新建人物' : '编辑人物' }}</h2></div>
        <button type="button" class="close-button" @click="close">关闭</button>
      </div>
      <div
        v-if="!isCreate"
        ref="avatarDropZone"
        class="avatar-row"
        @dragover="onAvatarDragOver"
        @dragleave="onAvatarDragLeave"
        @drop.prevent="onAvatarDrop"
      >
        <img class="person-avatar-preview" :src="avatarPreview" alt="头像" />
        <div class="avatar-actions">
          <button type="button" class="button secondary" @click="avatarFileInput?.click()">上传头像</button>
          <button type="button" class="button quiet" @click="removeAvatar">移除</button>
          <small>拖图片到这里也能换；PNG/JPG/WebP，≤2MB</small>
          <input ref="avatarFileInput" type="file" accept="image/png,image/jpeg,image/webp" hidden @change="onAvatarFile" />
        </div>
      </div>
      <p v-if="isCreate" class="plain-notice" style="margin:0 0 14px">头像可在创建完成后通过「编辑人物」上传。</p>
      <label>人物名称（必填）<input v-model="name" required placeholder="例如：Steve Jobs" /></label>
      <details class="advanced-fields">
        <summary>更多选项（可后改）</summary>
        <label>身份或消歧说明（选填）<input v-model="identity_note" placeholder="例如：Apple 联合创始人，1955—2011" /></label>
        <label>别名（选填，逗号分隔）<input v-model="aliases" placeholder="史蒂夫·乔布斯, Jobs" /></label>
        <div class="two-cols">
          <label>主要语言
            <select v-model="language">
              <option value="zh">中文</option>
              <option value="en">English</option>
              <option value="mixed">中英混合</option>
            </select>
          </label>
          <label>重点领域（选填）<input v-model="focus_domain" placeholder="例如：产品发布与设计" /></label>
        </div>
        <label>说明（选填）<input v-model="description" placeholder="材料范围或研究用途" /></label>
      </details>
      <div class="dialog-actions">
        <button type="button" class="button quiet" @click="close">取消</button>
        <button class="button primary" type="submit" :disabled="busy">{{ busy ? '保存中…' : '保存' }}</button>
      </div>
    </form>
  </dialog>
</template>
