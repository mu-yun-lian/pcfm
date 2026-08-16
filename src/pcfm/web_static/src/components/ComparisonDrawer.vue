<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAppStore } from '../stores/app'

const store = useAppStore()
const busy = ref('')

const comparison = computed(() => store.comparison)
const candidates = computed(() => comparison.value?.reality_candidates || [])
const selectedId = computed(() => comparison.value?.selected_candidate_id)

function selectCandidate(id: string) {
  if (store.comparison) store.comparison.selected_candidate_id = id
}

async function runAction(action: string) {
  busy.value = action
  try {
    await store.handleComparisonAction(action)
  } catch (error) {
    store.showToast((error as Error).message, true)
  } finally {
    busy.value = ''
  }
}
</script>

<template>
  <aside class="comparison-drawer">
    <div class="drawer-head">
      <div><p class="kicker">可核验材料</p><h2>现实回答对照</h2></div>
      <button class="close-button" aria-label="关闭" @click="store.closeComparison()">关闭</button>
    </div>
    <p class="drawer-intro">对照只创建审核候选，不会自动覆盖当前人物模型。</p>

    <section class="compare-step">
      <h3>1 当前模型预测回答</h3>
      <div class="compare-box"><blockquote>{{ comparison?.predicted_answer }}</blockquote></div>
    </section>

    <section v-if="candidates.length" class="compare-step">
      <h3>2 请选择要审核的现实回答候选</h3>
      <button
        v-for="item in candidates"
        :key="item.comparison_candidate_id"
        type="button"
        class="compare-box"
        style="display:block;width:100%;text-align:left"
        @click="selectCandidate(item.comparison_candidate_id)"
      >
        <strong>{{ selectedId === item.comparison_candidate_id ? '已选择 · ' : '' }}相似度 {{ Number(item.score).toFixed(2) }}</strong>
        <p>{{ item.question }}</p>
        <blockquote>{{ item.answer }}</blockquote>
        <small>{{ item.source_title }} · {{ item.speaker }} · {{ item.source_date || '未记录' }} · {{ item.locator }}</small>
      </button>
    </section>

    <section class="compare-step">
      <h3>3 匹配与差异</h3>
      <div class="compare-box">
        <p>系统只提供候选，不替用户认定问题相同。</p>
        <p>语境检查：{{ comparison?.context_consistency }}</p>
        <p>主要一致点：{{ (comparison?.agreements || []).join('、') }}</p>
        <p>差异：{{ (comparison?.differences || []).join('；') }}</p>
      </div>
    </section>

    <div class="warning-box">
      {{ comparison?.notice }} 加入待优化资料后仍需来源、身份、去重、时间、数据角色和独立留出检查。
    </div>

    <div class="compare-actions">
      <button class="button primary" :disabled="!!busy" @click="runAction('candidate')">
        {{ busy === 'candidate' ? '处理中…' : '加入待优化资料' }}
      </button>
      <button class="button quiet" :disabled="!!busy" @click="runAction('reference')">仅保存为参考</button>
      <button class="button quiet" :disabled="!!busy" @click="runAction('not-same')">不是同一个问题</button>
    </div>
  </aside>
</template>
