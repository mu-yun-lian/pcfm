<script setup lang="ts">
import { ref, reactive, computed, nextTick } from 'vue'
import { useDialog } from '../../composables/useDialog'
import { useAppStore } from '../../stores/app'
import { api } from '../../api/client'
import type { ModelService } from '../../types'

const { el, close, onClose } = useDialog('model')
const store = useAppStore()

const MODEL_PRESETS = [
  { key: 'deepseek', label: 'DeepSeek', display_name: 'DeepSeek', protocol: 'openai_compatible', base_url: 'https://api.deepseek.com', provider: 'DeepSeek', hint: '保存后读取服务端当前模型列表，再验证实际调用。' },
  { key: 'openai', label: 'OpenAI', display_name: 'OpenAI', protocol: 'openai_native', base_url: 'https://api.openai.com/v1', provider: 'OpenAI', hint: '保存后读取服务端当前模型列表，再验证实际调用。' },
  { key: 'anthropic', label: 'Anthropic Claude', display_name: 'Anthropic', protocol: 'anthropic', base_url: 'https://api.anthropic.com', provider: 'Anthropic', hint: '保存后读取服务端当前模型列表，再验证实际调用。' },
  { key: 'gemini', label: 'Google Gemini', display_name: 'Gemini', protocol: 'gemini', base_url: 'https://generativelanguage.googleapis.com/v1beta', provider: 'Google', hint: '保存后读取服务端当前模型列表，再验证实际调用。' },
  { key: 'kimi', label: 'Kimi（Moonshot）', display_name: 'Kimi (Moonshot)', protocol: 'openai_compatible', base_url: 'https://api.moonshot.cn/v1', provider: 'Moonshot', hint: '保存后读取服务端当前模型列表，再验证实际调用。' },
  { key: 'qwen', label: '通义千问（DashScope）', display_name: '通义千问', protocol: 'openai_compatible', base_url: 'https://dashscope.aliyuncs.com/compatible-mode/v1', provider: 'Alibaba', hint: '保存后读取服务端当前模型列表，再验证实际调用。' },
  { key: 'glm', label: '智谱 GLM', display_name: '智谱 GLM', protocol: 'openai_compatible', base_url: 'https://open.bigmodel.cn/api/paas/v4', provider: 'Zhipu', hint: '保存后读取服务端当前模型列表，再验证实际调用。' },
  { key: 'ollama', label: 'Ollama（本地）', display_name: 'Ollama 本机', protocol: 'ollama', base_url: 'http://127.0.0.1:11434', provider: 'Ollama', hint: '无需 API Key；需先安装 Ollama 并拉取模型。' },
]

const settingsEl = ref<HTMLDetailsElement>()
const formEl = ref<HTMLElement>()
const formHeading = ref('添加供应商')
const preset = ref('')
const keyVisible = ref(false)
const busyAction = ref('')
const formBusy = ref(false)

const form = reactive({
  service_id: '',
  display_name: '',
  protocol: 'openai_compatible',
  provider: '',
  base_url: '',
  api_key: '',
  environment_key: '',
  models: '',
  timeout_seconds: 30,
  structured_output: true,
  enabled: true,
})

const selectedRef = computed(() => store.conversation?.dialogue_model_ref || '')
const selectedOption = computed(() => store.enabledModelOptions.find((o) => o.ref === selectedRef.value))
const readyOptions = computed(() => store.enabledModelOptions.filter((o) => o.ready))

const currentStatusText = computed(() => {
  const sel = selectedOption.value
  if (!sel) return '当前人物：未选择对话模型；历史事件与公开倾向检索仍可工作'
  const status = sel.ready ? '调用已验证' : '需要重新验证'
  return '当前人物：' + sel.label + '（' + status + '）'
})

const roleSelects = [
  { role: 'assistant', label: '助手模型', key: 'assistant' },
  { role: 'material_processing', label: '资料处理模型', key: 'material_processing' },
  { role: 'validation', label: '候选校验模型', key: 'validation' },
  { role: 'dialogue', label: '人物对话模型', key: 'default_dialogue' },
]

function serviceStatus(service: ModelService): string {
  if (service.call_readiness === 'ready') return '真实调用已验证'
  if (service.connection_status === 'unavailable') return '调用失败'
  if (service.connection_status === 'models_loaded') return '已读取列表，待验证调用'
  return '尚未验证'
}

function serviceModels(service: ModelService): string[] {
  return (service.enabled_models || service.models || []) as string[]
}

function isReady(service: ModelService, modelId: string): boolean {
  return service.call_readiness === 'ready' && service.last_probe_model === modelId
}

function resetForm() {
  form.service_id = ''
  form.display_name = ''
  form.protocol = 'openai_compatible'
  form.provider = ''
  form.base_url = ''
  form.api_key = ''
  form.environment_key = ''
  form.models = ''
  form.timeout_seconds = 30
  form.structured_output = true
  form.enabled = true
  preset.value = ''
  keyVisible.value = false
  formHeading.value = '添加供应商'
}

function addProvider() {
  resetForm()
  if (settingsEl.value) settingsEl.value.open = true
  nextTick(() => formEl.value?.scrollIntoView({ behavior: 'smooth', block: 'center' }))
}

function editService(serviceId: string) {
  const service = store.modelServices.services.find((s) => s.service_id === serviceId)
  if (!service) return
  if (settingsEl.value) settingsEl.value.open = true
  form.service_id = service.service_id
  form.display_name = service.display_name || ''
  form.protocol = service.protocol || 'openai_compatible'
  form.provider = service.provider || ''
  form.base_url = service.base_url || ''
  form.api_key = ''
  form.environment_key = service.environment_key || ''
  form.models = (service.models || []).join(', ')
  form.timeout_seconds = service.timeout_seconds || 30
  form.structured_output = service.capabilities?.structured_output !== false
  form.enabled = service.enabled !== false
  preset.value = ''
  keyVisible.value = false
  formHeading.value = '编辑服务（API Key 留空则保留原密钥）'
  nextTick(() => formEl.value?.scrollIntoView({ behavior: 'smooth', block: 'center' }))
}

function applyPreset(key: string) {
  const p = MODEL_PRESETS.find((x) => x.key === key)
  if (!p) return
  form.display_name = p.display_name
  form.protocol = p.protocol
  form.provider = p.provider
  form.base_url = p.base_url
  form.models = ''
  form.structured_output = true
  if (p.protocol === 'ollama') {
    form.api_key = ''
    keyVisible.value = false
  }
  if (p.hint) store.showToast(p.hint)
}

async function toggleKeyVisibility() {
  if (!keyVisible.value) {
    if (form.service_id) {
      try {
        const data = await api<{ key: string }>('/api/model-services/' + encodeURIComponent(form.service_id) + '/key')
        form.api_key = data.key || ''
      } catch (error) {
        store.showToast((error as Error).message, true)
      }
    }
    keyVisible.value = true
  } else {
    form.api_key = ''
    keyVisible.value = false
  }
}

async function submitService() {
  formBusy.value = true
  try {
    const models = form.models.split(',').map((x) => x.trim()).filter(Boolean)
    await store.submitModelService({
      service_id: form.service_id,
      display_name: form.display_name,
      protocol: form.protocol,
      provider: form.provider,
      base_url: form.base_url,
      api_key: form.api_key,
      environment_key: form.environment_key,
      models,
      enabled_models: models,
      timeout_seconds: Number(form.timeout_seconds) || 30,
      enabled: form.enabled,
      capabilities: { structured_output: form.structured_output },
    })
    resetForm()
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    formBusy.value = false
  }
}

async function selectModel(ref: string) {
  if (!store.person) {
    store.showToast('请先选择一个人物。', true)
    return
  }
  try {
    await store.setDialogueModel(ref)
  } catch (error) {
    store.showToast((error as Error).message, true)
  }
}

async function clearDialogueModel() {
  try {
    await store.clearDialogueModel()
  } catch (error) {
    store.showToast((error as Error).message, true)
  }
}

async function testModel(serviceId: string, modelId: string) {
  busyAction.value = serviceId + ':' + modelId
  try {
    await store.modelServiceAction(serviceId, 'test', { model_id: modelId })
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    busyAction.value = ''
  }
}

async function refreshService(serviceId: string) {
  busyAction.value = serviceId + ':refresh'
  try {
    await store.modelServiceAction(serviceId, 'refresh-models', {})
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    busyAction.value = ''
  }
}

async function deleteService(serviceId: string) {
  try {
    await api('/api/model-services/' + encodeURIComponent(serviceId), { method: 'DELETE', body: '{}' })
    await store.loadModelServices()
  } catch (error) {
    store.showToast((error as Error).message, true)
  }
}

async function setRole(role: string, event: Event) {
  try {
    await store.setModelRole(role, (event.target as HTMLSelectElement).value)
  } catch (error) {
    store.showToast((error as Error).message, true)
  }
}
</script>

<template>
  <dialog id="model-services-dialog" ref="el" class="wide-dialog" @close="onClose">
    <div class="dialog-card">
      <div class="dialog-head">
        <div><p class="kicker">每个人物独立选择</p><h2>选择对话模型</h2></div>
        <button type="button" class="close-button" @click="close">关闭</button>
      </div>
      <p class="plain-notice">只有通过真实对话调用验证的模型才可选择。添加供应商时选择类型即可自动填好地址与协议，只需填 API Key。</p>

      <div class="model-current-row">
        <span>{{ currentStatusText }}</span>
        <button class="button quiet" type="button" :disabled="!store.person || !selectedRef" @click="clearDialogueModel">本人物不使用对话模型</button>
      </div>

      <div class="provider-list-head"><button class="button secondary" type="button" @click="addProvider">＋ 添加供应商</button></div>

      <div class="source-list">
        <article v-for="service in store.modelServices.services" :key="service.service_id" class="model-service-item">
          <header><strong>{{ service.display_name }}</strong><span>{{ serviceStatus(service) }}</span></header>
          <p>{{ service.protocol }} · {{ service.base_url }} · 密钥：{{ service.api_key_configured ? '已配置' : '未配置' }}</p>
          <p v-if="service.last_error" class="warning-box">{{ service.last_error }}</p>
          <div class="model-choice-list">
            <template v-if="serviceModels(service).length">
              <div v-for="modelId in serviceModels(service)" :key="modelId" class="model-choice">
                <span><strong>{{ modelId }}</strong><small>{{ isReady(service, modelId) ? '真实调用已验证' : '尚未验证' }}</small></span>
                <button v-if="isReady(service, modelId)" type="button" :class="{ selected: (service.service_id + ':' + modelId) === selectedRef }" @click="selectModel(service.service_id + ':' + modelId)">
                  {{ (service.service_id + ':' + modelId) === selectedRef ? '当前使用' : '使用此模型' }}
                </button>
                <button v-else type="button" :disabled="busyAction === service.service_id + ':' + modelId" @click="testModel(service.service_id, modelId)">验证并使用</button>
              </div>
            </template>
            <p v-else class="people-empty">尚无模型列表。请先刷新列表，或在下方配置中填写模型 ID。</p>
          </div>
          <div class="model-service-actions">
            <button type="button" @click="refreshService(service.service_id)">刷新模型列表</button>
            <button type="button" @click="editService(service.service_id)">编辑配置</button>
            <button type="button" @click="deleteService(service.service_id)">删除</button>
          </div>
        </article>
        <p v-if="!store.modelServices.services.length" class="people-empty">尚未配置模型服务。没有模型时仍可检索人物历史事件和公开倾向；需要通用知识时会明确提示选择模型。</p>
      </div>

      <details ref="settingsEl" class="model-settings">
        <summary>配置模型服务</summary>
        <p>选择预设类型会自动填好地址与协议，只需填 API Key；Key 只存在本机服务端，点眼睛可查看。</p>
        <form ref="formEl" class="source-form" @submit.prevent="submitService">
          <h3>{{ formHeading }}</h3>
          <label>选择预设服务（只填地址，不预设可能过期的模型 ID）
            <select v-model="preset" @change="applyPreset(preset)">
              <option value="">— 手动配置 —</option>
              <option v-for="p in MODEL_PRESETS" :key="p.key" :value="p.key">{{ p.label }}</option>
            </select>
          </label>
          <div class="two-cols">
            <label>显示名称<input v-model="form.display_name" required placeholder="例如：本地 LM Studio" /></label>
            <label>协议
              <select v-model="form.protocol">
                <option value="openai_native">OpenAI 原生</option>
                <option value="openai_compatible">OpenAI 兼容</option>
                <option value="anthropic">Anthropic</option>
                <option value="gemini">Gemini</option>
                <option value="ollama">Ollama 本地</option>
                <option value="custom_compatible">自定义兼容</option>
              </select>
            </label>
          </div>
          <div class="two-cols">
            <label>供应商<input v-model="form.provider" placeholder="DeepSeek / Qwen / OpenRouter" /></label>
            <label>Base URL<input v-model="form.base_url" type="url" placeholder="留空将按协议自动填充" /></label>
          </div>
          <div class="two-cols">
            <label>API Key
              <span class="key-wrap">
                <input v-model="form.api_key" :type="keyVisible ? 'text' : 'password'" autocomplete="new-password" placeholder="粘贴 API Key" />
                <button type="button" class="key-eye" title="显示/隐藏" aria-label="显示或隐藏 API Key" @click="toggleKeyVisibility">{{ keyVisible ? '🙈' : '👁' }}</button>
              </span>
            </label>
            <label>环境变量名（可选）<input v-model="form.environment_key" placeholder="OPENAI_API_KEY" /></label>
          </div>
          <div class="two-cols">
            <label>手动模型 ID（选填，逗号分隔）<input v-model="form.models" placeholder="留空则验证时自动读取" /></label>
            <label>超时秒数<input v-model.number="form.timeout_seconds" type="number" min="2" max="300" /></label>
          </div>
          <label><input v-model="form.structured_output" type="checkbox" /> 模型支持严格 JSON 输出</label>
          <label><input v-model="form.enabled" type="checkbox" /> 启用此服务</label>
          <button class="button primary" type="submit" :disabled="formBusy">{{ formBusy ? '保存中…' : '只保存配置' }}</button>
        </form>
        <div class="model-role-grid">
          <label v-for="rs in roleSelects" :key="rs.role">
            {{ rs.label }}
            <select :value="store.modelServices.roles[rs.key] || ''" @change="setRole(rs.role, $event)">
              <option value="">未配置</option>
              <option v-for="o in readyOptions" :key="o.ref" :value="o.ref">{{ o.label }}</option>
            </select>
          </label>
        </div>
      </details>
    </div>
  </dialog>
</template>
