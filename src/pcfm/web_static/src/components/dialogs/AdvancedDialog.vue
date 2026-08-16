<script setup lang="ts">
import { computed } from 'vue'
import { useDialog } from '../../composables/useDialog'
import { useAppStore } from '../../stores/app'
import { humanStatus } from '../../lib/labels'

const { el, close, onClose } = useDialog('advanced')
const store = useAppStore()

const metrics = computed(() => store.conversation?.metrics || {})

const labels: Record<string, string> = {
  content_holdout_agreement: '内容与真实留出回答一致度',
  correct_person_uplift: '正确人物相对基线增益',
  confidence_calibration: '置信度校准',
  fact_source_support: '事实与经历来源支持率',
  style_blind_test: '风格盲测',
  style_semantic_preservation: '风格化语义保持',
  out_of_scope_handling: '新领域处理',
}

function formatMetric(value: unknown): string {
  if (typeof value === 'number') return value.toFixed(3)
  if (value && typeof value === 'object') return JSON.stringify(value)
  return humanStatus(value as string)
}

const exportUrl = computed(() =>
  store.person && !store.isAssistant ? '/api/people/' + encodeURIComponent(store.person.person_id) + '/export' : '#',
)
</script>

<template>
  <dialog ref="el" @close="onClose">
    <div class="dialog-card">
      <div class="dialog-head">
        <div><p class="kicker">不进入聊天主流程</p><h2>高级诊断与研究工具</h2></div>
        <button type="button" class="close-button" @click="close">关闭</button>
      </div>
      <p>旧行为 Logistic、认知模型卡和实验候选仍作为研究工具保留。当前聊天统一调用多头人物响应内核；只有 active 组件能影响正式预测。</p>
      <div class="metrics-box">
        <div v-for="(value, key) in metrics" :key="key" class="metric-line">
          <span>{{ labels[key] || key }}</span><strong>{{ formatMetric(value) }}</strong>
        </div>
      </div>
      <a class="button secondary link-button" :href="exportUrl">导出当前人物完整备份</a>
    </div>
  </dialog>
</template>
