<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { useDialog } from '../../composables/useDialog'
import { useAppStore } from '../../stores/app'
import { humanStatus } from '../../lib/labels'
import { fileToBase64 } from '../../lib/file'
import type { Source, CollectionState } from '../../types'

const { el, close, onClose } = useDialog('sources')
const store = useAppStore()

const textForm = reactive({
  title: '',
  speaker: '',
  text: '',
  content_authenticity: 'unverified_material',
  source_locator: '',
  source_url: '',
  source_context: '',
  translation_of: '',
  speaker_scope: 'single_speaker_entire_document',
  source_date: '',
  dataset_role: 'model_source',
})

const fileForm = reactive({
  file: null as File | null,
  speaker: '',
  speaker_scope: 'single_speaker_entire_document',
  content_authenticity: 'unverified_material',
  source_locator: '',
  source_context: '',
  dataset_role: 'model_source',
})

const urlForm = reactive({
  url: '',
  speaker: '',
  speaker_scope: 'single_speaker_entire_document',
  content_authenticity: 'unverified_material',
  source_locator: '',
  source_context: '',
  dataset_role: 'model_source',
})

const textBusy = ref(false)
const fileBusy = ref(false)
const urlBusy = ref(false)
const processBusy = ref(false)
const itemBusy = ref('')

textForm.speaker = store.person?.name || ''
fileForm.speaker = store.person?.name || ''
urlForm.speaker = store.person?.name || ''

const sources = computed(() => store.conversation?.sources || [])
const counts = computed(() => store.conversation?.source_counts || { confirmed: 0, pending: 0, final_holdout: 0 })
const sourceCountSummary = computed(() => counts.value.confirmed + ' 已确认 · ' + counts.value.pending + ' 待审核 · ' + counts.value.final_holdout + ' 最终留出')
const optimizationCandidates = computed(() => [...(store.conversation?.optimization_candidates || [])].reverse())

const networkNoteText = computed(() => {
  const collection = (store.person?.collection || store.conversation?.profile?.collection || {}) as CollectionState
  if (collection.mode === 'system_search') {
    return collection.message || '搜索服务已配置；结果只进入待审核候选资料。'
  }
  return '系统搜索未配置时，请自行提供资料；所有自动提取结果都必须先审核。'
})

function speakerScopeText(source: Source): string {
  return source.speaker_scope === 'mixed_speakers' ? '多人混合，需逐段确认' : '整份材料主要说话人'
}

function hasVerbatim(source: Source): boolean {
  return !!(source.response_events || []).some((item) => item.label_status === 'confirmed_response_weak_semantic_labels')
}

async function submitText() {
  textBusy.value = true
  try {
    await store.submitTextSource({ ...textForm })
    textForm.text = ''
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    textBusy.value = false
  }
}

function onFileSelected(event: Event) {
  const input = event.target as HTMLInputElement
  fileForm.file = input.files?.[0] || null
}

async function submitFile() {
  const file = fileForm.file
  if (!file) {
    store.showToast('请选择文件。', true)
    return
  }
  fileBusy.value = true
  try {
    const content_base64 = await fileToBase64(file)
    await store.submitFileSource({
      filename: file.name,
      content_base64,
      speaker: fileForm.speaker,
      source_date: '',
      dataset_role: fileForm.dataset_role,
      content_authenticity: fileForm.content_authenticity,
      source_locator: fileForm.source_locator,
      source_context: fileForm.source_context,
      speaker_scope: fileForm.speaker_scope,
    })
    fileForm.file = null
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    fileBusy.value = false
  }
}

async function submitUrl() {
  urlBusy.value = true
  try {
    await store.submitUrlSource({
      url: urlForm.url,
      speaker: urlForm.speaker,
      source_date: '',
      dataset_role: urlForm.dataset_role,
      content_authenticity: urlForm.content_authenticity,
      source_locator: urlForm.source_locator,
      source_context: urlForm.source_context,
      speaker_scope: urlForm.speaker_scope,
    })
    urlForm.url = ''
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    urlBusy.value = false
  }
}

async function processAll() {
  processBusy.value = true
  try {
    await store.processAllMaterials()
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    processBusy.value = false
  }
}

async function reviewSource(sourceId: string, decision: string) {
  itemBusy.value = sourceId
  try {
    await store.reviewSource(sourceId, decision)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    itemBusy.value = ''
  }
}

async function extractSource(sourceId: string) {
  itemBusy.value = sourceId
  try {
    await store.extractSource(sourceId)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    itemBusy.value = ''
  }
}

async function reviewCandidate(sourceId: string, candidateId: string, decision: string) {
  itemBusy.value = candidateId
  try {
    await store.reviewEventCandidate(sourceId, candidateId, decision)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    itemBusy.value = ''
  }
}

async function reviewOptimization(candidateId: string, decision: string) {
  itemBusy.value = candidateId
  try {
    await store.reviewOptimization(candidateId, decision)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    itemBusy.value = ''
  }
}

async function reviewOptimizationStyle(candidateId: string, decision: string) {
  itemBusy.value = candidateId
  try {
    await store.reviewOptimizationStyle(candidateId, decision)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    itemBusy.value = ''
  }
}
</script>

<template>
  <dialog ref="el" class="wide-dialog" @close="onClose">
    <div class="dialog-card">
      <div class="dialog-head">
        <div><p class="kicker">原始资料与角色隔离</p><h2>管理人物资料</h2></div>
        <button type="button" class="close-button" @click="close">关闭</button>
      </div>

      <div class="source-entry-grid">
        <form class="source-form" @submit.prevent="submitText">
          <h3>粘贴文本</h3>
          <label>标题<input v-model="textForm.title" required placeholder="访谈、演讲或文章名称" /></label>
          <label>说话人<input v-model="textForm.speaker" required /></label>
          <label>原始内容<textarea v-model="textForm.text" rows="6" required placeholder="可粘贴完整文章、访谈、问答或其他原始材料"></textarea></label>
          <details class="advanced-fields">
            <summary>证据核验与用途（可选）</summary>
            <fieldset class="evidence-contract-fields">
              <legend>证据核验信息</legend>
              <label>内容真实性
                <select v-model="textForm.content_authenticity">
                  <option value="unverified_material">尚未核验的材料</option>
                  <option value="verbatim_transcript">已核验逐字稿</option>
                  <option value="verified_translation">有原文对应的准确翻译</option>
                  <option value="editorial_summary">编辑或研究者摘要</option>
                </select>
              </label>
              <label>原始材料位置<input v-model="textForm.source_locator" placeholder="例如：视频 12:30–13:05，或逐字稿第 10–12 段" /></label>
              <label>来源网址<input v-model="textForm.source_url" type="url" placeholder="https://..." /></label>
              <label>当时上下文<input v-model="textForm.source_context" placeholder="公开访谈、演讲或其他场合" /></label>
              <label>翻译对应的原文位置<input v-model="textForm.translation_of" placeholder="仅准确翻译需要填写" /></label>
            </fieldset>
            <label>说话人范围
              <select v-model="textForm.speaker_scope">
                <option value="single_speaker_entire_document">整份材料主要由该人物表达</option>
                <option value="mixed_speakers">多人混合，逐段确认后再采用</option>
              </select>
            </label>
            <label>日期<input v-model="textForm.source_date" type="date" /></label>
            <label>数据用途
              <select v-model="textForm.dataset_role">
                <option value="model_source">参数训练</option>
                <option value="applicability_reference">适用性校准</option>
                <option value="feature_discovery">特征发现</option>
                <option value="candidate_selection">候选模型选择</option>
                <option value="final_holdout">封存最终验证</option>
                <option value="post_deployment_monitoring">上线后监测</option>
                <option value="reference_only">外部现实回答对照</option>
              </select>
            </label>
          </details>
          <button class="button primary" type="submit" :disabled="textBusy">{{ textBusy ? '保存中…' : '保存为待审核资料' }}</button>
        </form>

        <div class="other-source-forms">
          <form class="source-form compact-source" @submit.prevent="submitFile">
            <h3>上传文件</h3>
            <p>支持 TXT、Markdown、HTML、SRT/VTT 字幕、JSON、CSV 和带文字层 PDF；音视频转写与扫描件 OCR 尚未配置。</p>
            <label>文件<input type="file" accept=".txt,.md,.markdown,.html,.htm,.srt,.vtt,.json,.csv,.pdf" required @change="onFileSelected" /></label>
            <label>说话人<input v-model="fileForm.speaker" required /></label>
            <details class="advanced-fields">
              <summary>证据核验与用途（可选）</summary>
              <label>说话人范围
                <select v-model="fileForm.speaker_scope">
                  <option value="single_speaker_entire_document">整份材料主要由该人物表达</option>
                  <option value="mixed_speakers">多人混合，逐段确认后再采用</option>
                </select>
              </label>
              <label>内容真实性
                <select v-model="fileForm.content_authenticity">
                  <option value="unverified_material">尚未核验的材料</option>
                  <option value="verbatim_transcript">已核验逐字稿或本人原文</option>
                  <option value="verified_translation">有原文对应的准确翻译</option>
                  <option value="editorial_summary">编辑或研究者摘要</option>
                </select>
              </label>
              <label>原始材料位置<input v-model="fileForm.source_locator" placeholder="页码、段落或时间码" /></label>
              <label>当时上下文<input v-model="fileForm.source_context" placeholder="演讲、文章、访谈等" /></label>
              <label>数据用途
                <select v-model="fileForm.dataset_role">
                  <option value="model_source">参数训练</option>
                  <option value="applicability_reference">适用性校准</option>
                  <option value="feature_discovery">特征发现</option>
                  <option value="candidate_selection">候选模型选择</option>
                  <option value="final_holdout">封存最终验证</option>
                  <option value="post_deployment_monitoring">上线后监测</option>
                  <option value="reference_only">外部现实回答对照</option>
                </select>
              </label>
            </details>
            <button class="button secondary" type="submit" :disabled="fileBusy">{{ fileBusy ? '提取中…' : '上传并提取' }}</button>
          </form>

          <form class="source-form compact-source" @submit.prevent="submitUrl">
            <h3>网页地址</h3>
            <label>网址<input v-model="urlForm.url" type="url" required placeholder="https://..." /></label>
            <label>说话人<input v-model="urlForm.speaker" required /></label>
            <details class="advanced-fields">
              <summary>证据核验与用途（可选）</summary>
              <label>说话人范围
                <select v-model="urlForm.speaker_scope">
                  <option value="single_speaker_entire_document">整份材料主要由该人物表达</option>
                  <option value="mixed_speakers">多人混合，逐段确认后再采用</option>
                </select>
              </label>
              <label>内容真实性
                <select v-model="urlForm.content_authenticity">
                  <option value="unverified_material">尚未核验的材料</option>
                  <option value="verbatim_transcript">已核验逐字稿或本人原文</option>
                  <option value="verified_translation">有原文对应的准确翻译</option>
                  <option value="editorial_summary">编辑或研究者摘要</option>
                </select>
              </label>
              <label>原始材料位置<input v-model="urlForm.source_locator" placeholder="网页段落、章节或时间码" /></label>
              <label>当时上下文<input v-model="urlForm.source_context" placeholder="演讲、文章、访谈等" /></label>
              <label>数据用途
                <select v-model="urlForm.dataset_role">
                  <option value="model_source">参数训练</option>
                  <option value="applicability_reference">适用性校准</option>
                  <option value="feature_discovery">特征发现</option>
                  <option value="candidate_selection">候选模型选择</option>
                  <option value="final_holdout">封存最终验证</option>
                  <option value="post_deployment_monitoring">上线后监测</option>
                  <option value="reference_only">外部现实回答对照</option>
                </select>
              </label>
            </details>
            <button class="button secondary" type="submit" :disabled="urlBusy">{{ urlBusy ? '抓取中…' : '抓取网页快照' }}</button>
          </form>

          <div class="network-note"><strong>系统联网收集</strong><p>{{ networkNoteText }}</p></div>
        </div>
      </div>

      <div class="source-list-head">
        <h3>资料与审核队列</h3>
        <span>{{ sourceCountSummary }}</span>
        <button class="mini-button confirm" type="button" :disabled="processBusy" @click="processAll">{{ processBusy ? '处理中…' : '一键处理全部材料' }}</button>
      </div>
      <div class="processing-progress" v-if="store.processing.visible">
        <span>{{ store.processing.text }}</span>
        <div class="progress-track"><div class="progress-fill" :style="{ width: store.processing.percent + '%' }"></div></div>
      </div>

      <div class="sources-list">
        <article v-for="source in sources" :key="source.source_id" class="source-item">
          <header>
            <strong>{{ source.title }}</strong>
            <span class="tag" :class="source.review_status">{{ humanStatus(source.review_status) }}</span>
          </header>
          <p>{{ source.text_preview }}</p>
          <p>说话人：{{ source.speaker || '未记录' }} · 范围：{{ speakerScopeText(source) }} · 格式：{{ source.format }} · 数据用途：{{ humanStatus(source.dataset_role) }} · 事件包：{{ source.response_events?.length || 0 }}</p>
          <p>{{ hasVerbatim(source) ? '包含可追溯的本人公开回应；系统已按事件、条件倾向和公开使用的知识主张整理。' : '尚无可进入人物模型的公开回应；材料仍作为待核验或参考资料保留。' }}</p>
          <p v-if="source.llm_response_event_candidates?.length">资料处理模型提出 {{ source.llm_response_event_candidates.length }} 条待审核响应事件候选；尚未进入训练。</p>

          <div v-for="candidate in source.llm_response_event_candidates || []" :key="candidate.candidate_id" class="candidate-box">
            <p><strong>{{ candidate.trigger || '公开回应候选' }}</strong> · {{ candidate.source_locator || '未标注位置' }}</p>
            <blockquote>{{ candidate.actual_response || '' }}</blockquote>
            <small>说话人：{{ candidate.speaker || '未识别' }} · {{ candidate.review_status || 'pending' }}</small>
            <div v-if="source.review_status === 'confirmed' && (candidate.review_status || 'pending') === 'pending'" class="item-actions">
              <button class="mini-button confirm" :disabled="itemBusy === candidate.candidate_id" @click="reviewCandidate(source.source_id, candidate.candidate_id, 'confirmed')">确认逐字位置并采用</button>
              <button class="mini-button" :disabled="itemBusy === candidate.candidate_id" @click="reviewCandidate(source.source_id, candidate.candidate_id, 'rejected')">拒绝候选</button>
            </div>
          </div>

          <div v-if="source.review_status === 'pending'" class="item-actions">
            <button class="mini-button" :disabled="itemBusy === source.source_id" @click="extractSource(source.source_id)">用资料处理模型提取候选</button>
            <button class="mini-button confirm" :disabled="itemBusy === source.source_id" @click="reviewSource(source.source_id, 'confirmed')">{{ source.speaker_scope === 'mixed_speakers' ? '确认来源，逐段说话人另审' : '确认来源与整份材料说话人' }}</button>
            <button class="mini-button" :disabled="itemBusy === source.source_id" @click="reviewSource(source.source_id, 'rejected')">拒绝</button>
          </div>
        </article>
        <div v-if="!sources.length" class="source-item"><p>还没有资料。粘贴文本、上传文件或输入网页地址后，系统会先生成待审核候选。</p></div>
      </div>

      <div class="optimization-section">
        <h3>待优化资料</h3>
        <p>确认前不会改变当前版本；缺少独立留出资料时，升级会被拒绝。</p>
        <div id="optimization-list">
          <article v-for="item in optimizationCandidates" :key="item.candidate_id" class="optimization-item">
            <header><strong>优化候选</strong><span class="tag" :class="item.status">{{ humanStatus(item.status) }}</span></header>
            <p>创建时版本：{{ item.active_version_before ? 'v' + item.active_version_before : '尚无版本' }}</p>
            <p>内容：{{ humanStatus(item.status) }} · 表达：{{ humanStatus(item.surface_extraction?.status || 'not_applied') }}</p>
            <p v-if="item.validation_reasons?.length">未通过原因：{{ item.validation_reasons.map((r) => humanStatus(r)).join('、') }}</p>
            <div v-if="item.status === 'pending'" class="item-actions">
              <button class="mini-button confirm" :disabled="itemBusy === item.candidate_id" @click="reviewOptimization(item.candidate_id, 'confirmed')">确认内容并运行升级门禁</button>
              <button class="mini-button" :disabled="itemBusy === item.candidate_id" @click="reviewOptimization(item.candidate_id, 'reference_only')">仅作参考</button>
              <button class="mini-button" :disabled="itemBusy === item.candidate_id" @click="reviewOptimization(item.candidate_id, 'not_same_question')">问题不相同</button>
            </div>
            <div v-if="item.status === 'accepted_exploratory' && item.surface_extraction?.status === 'pending_separate_style_review'" class="item-actions">
              <button class="mini-button confirm" :disabled="itemBusy === item.candidate_id" @click="reviewOptimizationStyle(item.candidate_id, 'confirmed')">单独审核并更新表达</button>
              <button class="mini-button" :disabled="itemBusy === item.candidate_id" @click="reviewOptimizationStyle(item.candidate_id, 'rejected')">不用于表达</button>
            </div>
          </article>
          <p v-if="!optimizationCandidates.length" class="people-empty">还没有待优化资料。</p>
        </div>
      </div>
    </div>
  </dialog>
</template>
