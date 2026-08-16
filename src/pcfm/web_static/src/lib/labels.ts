// Reused verbatim from the legacy app.js so Chinese label wording stays identical.

const STATUS_LABELS: Record<string, string> = {
  exploratory: '探索性人物模拟',
  insufficient_evidence: '尚未建立人物模型',
  answered: '已回答',
  needs_model: '需要选择对话模型',
  refused: '已拒绝强行预测',
  clarification: '需要澄清',
  ordinary_dialogue: '普通对话',
  direct_answer: '历史直接依据',
  similar_event_evidence_answer: '相似历史事件依据',
  preference_structure_answer: '公开取舍结构推断',
  orientation_projection_answer: '结合上下文的公开取向预测',
  general_assisted: '通用知识回答（非人物预测）',
  object_evaluation: '人物对象评价',
  self_evaluation: '人物自我评价',
  policy_stance: '人物政策立场',
  factual: '人物事实判断',
  identity: '身份介绍',
  direct_historical: '历史直接依据',
  clarification_needed: '需要澄清',
  content_contract_gate_failed_bounded_anchor: '内容合同检查未通过，已返回中性回答',
  generated_from_frozen_v5_content_plan: '已按冻结内容计划生成回答',
  ordinary_dialogue_content_free: '普通对话（无内容立场）',
  bounded_anchor_no_dialogue_model: '有界锚点回答（未选择对话模型）',
  unified_gate_failed: '统一守门未通过',
  source_verbatim_person_style: '源文逐字人物风格',
  not_run_no_person_prediction: '未运行（无人物预测）',
  not_run_refused: '未运行（已拒绝）',
}

const HUMAN_STATUS_LABELS: Record<string, string> = {
  not_assessed: '尚未验证',
  not_applied: '当前未启用',
  structural_gate_only: '仅完成结构检查',
  implemented_not_independently_measured: '已实现，但缺少独立测试',
  general_assisted_without_person_stance: '无合格人物依据时转通用回答，不补写人物立场',
  applied_exploratory: '探索性内容模型已更新',
  applied_structural_only: '已生成表层风格，并通过结构守门',
  rendering_enabled_exploratory: '人物风格渲染已启用（探索性）',
  style_material_ready_rendering_not_enabled: '风格资料已建立，渲染未启用',
  person_style_applied: '人物风格已应用',
  neutral_expression: '中性表达',
  neutral_fallback: '风格检查失败，已返回中性表达',
  unchanged_separate_review_required: '内容已更新；风格等待独立审核',
  pending_separate_style_review: '表达样本等待独立审核',
  rejected_separately: '表达样本已单独拒绝',
  exploratory_source_integrity_passed_accuracy_not_assessed: '证据结构通过；真实准确性尚未验证',
  invalidated_evidence_contract: '证据契约不合格，版本已失效',
  pending: '待审核',
  confirmed: '已确认',
  rejected: '已拒绝',
  model_source: '参数训练',
  reference_only: '仅参考',
  final_holdout: '封存最终验证',
  accepted_exploratory: '探索性版本已建立',
  failed_validation: '优化未通过',
  source_verbatim_person_style: '源文逐字人物风格',
  not_run_no_person_prediction: '未运行（无人物预测）',
  not_run_refused: '未运行（已拒绝）',
  unified_gate_failed: '统一守门未通过',
  generated_from_frozen_v5_content_plan: '已按冻结内容计划生成回答',
  ordinary_dialogue_content_free: '普通对话（无内容立场）',
  bounded_anchor_no_dialogue_model: '有界锚点回答（未选择对话模型）',
  content_contract_gate_failed_bounded_anchor: '内容合同检查未通过，已返回中性回答',
}

export function statusLabel(value?: string | null): string {
  if (!value) return '未记录'
  return STATUS_LABELS[value] || '未知状态'
}

export function humanStatus(value?: string | null): string {
  if (!value) return '未记录'
  return HUMAN_STATUS_LABELS[value] || '未知状态'
}

export function shortTime(value?: string | null): string {
  if (!value) return ''
  try {
    return new Date(value).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  } catch {
    return ''
  }
}

const HTML_ESCAPES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  "'": '&#39;',
  '"': '&quot;',
}

export function escapeHtml(value: unknown): string {
  return String(value ?? '').replace(/[&<>'"]/g, (ch) => HTML_ESCAPES[ch])
}
