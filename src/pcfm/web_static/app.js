const APP_VERSION = "0.10.0-simulation-v5";
const MODEL_PRESETS = [
  { key:"deepseek", label:"DeepSeek", display_name:"DeepSeek", protocol:"openai_compatible", base_url:"https://api.deepseek.com", provider:"DeepSeek", models:[], hint:"保存后读取服务端当前模型列表，再验证实际调用。" },
  { key:"openai", label:"OpenAI", display_name:"OpenAI", protocol:"openai_native", base_url:"https://api.openai.com/v1", provider:"OpenAI", models:[], hint:"保存后读取服务端当前模型列表，再验证实际调用。" },
  { key:"anthropic", label:"Anthropic Claude", display_name:"Anthropic", protocol:"anthropic", base_url:"https://api.anthropic.com", provider:"Anthropic", models:[], hint:"保存后读取服务端当前模型列表，再验证实际调用。" },
  { key:"gemini", label:"Google Gemini", display_name:"Gemini", protocol:"gemini", base_url:"https://generativelanguage.googleapis.com/v1beta", provider:"Google", models:[], hint:"保存后读取服务端当前模型列表，再验证实际调用。" },
  { key:"kimi", label:"Kimi（Moonshot）", display_name:"Kimi (Moonshot)", protocol:"openai_compatible", base_url:"https://api.moonshot.cn/v1", provider:"Moonshot", models:[], hint:"保存后读取服务端当前模型列表，再验证实际调用。" },
  { key:"qwen", label:"通义千问（DashScope）", display_name:"通义千问", protocol:"openai_compatible", base_url:"https://dashscope.aliyuncs.com/compatible-mode/v1", provider:"Alibaba", models:[], hint:"保存后读取服务端当前模型列表，再验证实际调用。" },
  { key:"glm", label:"智谱 GLM", display_name:"智谱 GLM", protocol:"openai_compatible", base_url:"https://open.bigmodel.cn/api/paas/v4", provider:"Zhipu", models:[], hint:"保存后读取服务端当前模型列表，再验证实际调用。" },
  { key:"ollama", label:"Ollama（本地）", display_name:"Ollama 本机", protocol:"ollama", base_url:"http://127.0.0.1:11434", provider:"Ollama", models:[], hint:"无需 API Key；需先安装 Ollama 并拉取模型。" },
];
const state = { people: [], person: null, conversation: null, comparison: null, editingPerson: false, archiveTarget: null, permanentDeleteTarget: null, draggedPersonId: null, pointerDragId: null, capabilities: {}, modelServices: {services:[],roles:{}} };
const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];

async function api(path, options = {}) {
  const response = await fetch(path, {headers:{"Content-Type":"application/json",...(options.headers||{})},...options});
  const raw = await response.text();
  let data;
  try { data = raw ? JSON.parse(raw) : {}; }
  catch { throw new Error(`本地服务返回了无法读取的响应（HTTP ${response.status}）。请重启服务后重试。`); }
  if (!response.ok || data.ok === false) throw new Error(data.message || "操作失败");
  return data;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>'"]/g, ch => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"})[ch]);
}

function toast(message, error = false, action = null) {
  const box = $("#toast");
  box.textContent = message; box.className = `toast${error ? " error" : ""}`; box.hidden = false;
  if (action) {
    const undo = document.createElement("button"); undo.type = "button"; undo.textContent = "撤销";
    undo.onclick = async () => { box.hidden = true; await action(); };
    box.appendChild(undo);
  }
  clearTimeout(toast.timer); toast.timer = setTimeout(() => box.hidden = true, action ? 8000 : 3600);
}

function busy(button, on, label = "处理中…") {
  if (!button) return;
  if (on) { button.dataset.old = button.textContent; button.textContent = label; button.disabled = true; }
  else { button.textContent = button.dataset.old || button.textContent; button.disabled = false; }
}

function personById(id) { return state.people.find(item => item.person_id === id); }
function currentName() { return state.person?.name || personById(state.person?.person_id)?.name || "当前人物"; }
function shortTime(value) { try { return new Date(value).toLocaleTimeString("zh-CN",{hour:"2-digit",minute:"2-digit"}); } catch { return ""; } }
function statusLabel(value) { return ({exploratory:"探索性人物模拟",insufficient_evidence:"尚未建立人物模型",answered:"已回答",needs_model:"需要选择对话模型",refused:"已拒绝强行预测",clarification:"需要澄清",ordinary_dialogue:"普通对话",direct_answer:"历史直接依据",similar_event_evidence_answer:"相似历史事件依据",preference_structure_answer:"公开取舍结构推断",orientation_projection_answer:"结合上下文的公开取向预测",general_assisted:"通用知识回答（非人物预测）",object_evaluation:"人物对象评价",self_evaluation:"人物自我评价",policy_stance:"人物政策立场",factual:"人物事实判断",identity:"身份介绍",direct_historical:"历史直接依据",clarification_needed:"需要澄清"})[value] || value; }

function humanStatus(value) {
  return ({
    not_assessed:"尚未验证", not_applied:"当前未启用", structural_gate_only:"仅完成结构检查",
    implemented_not_independently_measured:"已实现，但缺少独立测试",
    general_assisted_without_person_stance:"无合格人物依据时转通用回答，不补写人物立场",
    applied_exploratory:"探索性内容模型已更新",
    applied_structural_only:"已生成表层风格，并通过结构守门",
    rendering_enabled_exploratory:"人物风格渲染已启用（探索性）",
    style_material_ready_rendering_not_enabled:"风格资料已建立，渲染未启用",
    person_style_applied:"人物风格已应用",
    neutral_expression:"中性表达",
    neutral_fallback:"风格检查失败，已返回中性表达",
    unchanged_separate_review_required:"内容已更新；风格等待独立审核",
    pending_separate_style_review:"表达样本等待独立审核",
    rejected_separately:"表达样本已单独拒绝",
    exploratory_source_integrity_passed_accuracy_not_assessed:"证据结构通过；真实准确性尚未验证",
    invalidated_evidence_contract:"证据契约不合格，版本已失效",
    pending:"待审核", confirmed:"已确认", rejected:"已拒绝",
    model_source:"参数训练", reference_only:"仅参考", final_holdout:"封存最终验证",
    accepted_exploratory:"探索性版本已建立", failed_validation:"优化未通过",
  })[value] || value;
}

function enabledModelOptions() {
  return state.modelServices.services.flatMap(service => (service.enabled_models || []).map(modelId => ({
    ref:`${service.service_id}:${modelId}`,
    label:`${service.display_name} · ${modelId}`,
    service,
    modelId,
    ready: service.call_readiness === "ready" && service.last_probe_model === modelId,
  })));
}

function currentModelLabel() {
  const ref = state.conversation?.dialogue_model_ref || "";
  const option = enabledModelOptions().find(item => item.ref === ref);
  return option ? `${option.modelId}${option.ready ? "" : "（需验证）"} ▾` : "模型 ▾";
}

async function loadModelServices() {
  const data = await api("/api/model-services");
  state.modelServices = data.model_services;
  renderModelServices();
  if (state.conversation) $("#open-model-picker").textContent = currentModelLabel();
}

function roleOptions(selected) {
  return `<option value="">未配置</option>${enabledModelOptions().filter(item => item.ready).map(item => `<option value="${escapeHtml(item.ref)}" ${item.ref===selected?"selected":""}>${escapeHtml(item.label)}</option>`).join("")}`;
}

function renderModelServices() {
  const list = $("#model-services-list");
  if (!list) return;
  const selectedRef = state.conversation?.dialogue_model_ref || "";
  const selected = enabledModelOptions().find(item => item.ref === selectedRef);
  const selectedStatus = selected?.ready ? "调用已验证" : "需要重新验证";
  $("#current-dialogue-model-status").textContent = selected
    ? `当前人物：${selected.label}（${selectedStatus}）`
    : "当前人物：未选择对话模型；历史事件与公开倾向检索仍可工作";
  $("#clear-dialogue-model").disabled = !state.person || !selectedRef;
  list.innerHTML = state.modelServices.services.length ? state.modelServices.services.map(service => {
    const serviceModels = (service.enabled_models || service.models || []);
    const status = service.call_readiness === "ready" ? "真实调用已验证" : service.connection_status === "unavailable" ? "调用失败" : service.connection_status === "models_loaded" ? "已读取列表，待验证调用" : "尚未验证";
    const models = serviceModels.length ? serviceModels.map(modelId => {
      const ref = `${service.service_id}:${modelId}`;
      const ready = service.call_readiness === "ready" && service.last_probe_model === modelId;
      return `<div class="model-choice"><span><strong>${escapeHtml(modelId)}</strong><small>${ready ? "真实调用已验证" : "尚未验证"}</small></span>${ready ? `<button type="button" class="${ref===selectedRef?"selected":""}" data-select-model-ref="${escapeHtml(ref)}">${ref===selectedRef?"当前使用":"使用此模型"}</button>` : `<button type="button" data-test-model="${escapeHtml(modelId)}" data-service-id="${escapeHtml(service.service_id)}">验证并使用</button>`}</div>`;
    }).join("") : '<p class="people-empty">尚无模型列表。请先刷新列表，或在下方配置中填写模型 ID。</p>';
    return `<article class="model-service-item"><header><strong>${escapeHtml(service.display_name)}</strong><span>${escapeHtml(status)}</span></header><p>${escapeHtml(service.protocol)} · ${escapeHtml(service.base_url)} · 密钥：${service.api_key_configured ? "已配置" : "未配置"}</p>${service.last_error ? `<p class="warning-box">${escapeHtml(service.last_error)}</p>` : ""}<div class="model-choice-list">${models}</div><div class="model-service-actions"><button type="button" data-refresh-service="${service.service_id}">刷新模型列表</button><button type="button" data-edit-service="${service.service_id}">编辑配置</button><button type="button" data-delete-service="${service.service_id}">删除</button></div></article>`;
  }).join("") : '<p class="people-empty">尚未配置模型服务。没有模型时仍可检索人物历史事件和公开倾向；需要通用知识时会明确提示选择模型。</p>';
  $("#material-model-role").innerHTML = roleOptions(state.modelServices.roles.material_processing || "");
  $$('[data-select-model-ref]').forEach(button => button.onclick = async () => {
    if (!state.person) return toast("请先选择一个人物。", true);
    const ref = button.dataset.selectModelRef;
    await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/model`, {method:"POST",body:JSON.stringify({model_ref:ref})});
    await refreshConversation(); await loadModelServices();
    toast("已切换此人物的对话模型；历史回答和人物版本没有改变。");
  });
  $$('[data-test-model]').forEach(button => button.onclick = () => modelServiceAction(button.dataset.serviceId,"test",button,{model_id:button.dataset.testModel}));
  $$('[data-edit-service]').forEach(button => button.onclick = () => editModelService(button.dataset.editService));
  $$('[data-refresh-service]').forEach(button => button.onclick = () => modelServiceAction(button.dataset.refreshService,"refresh-models",button));
  $$('[data-delete-service]').forEach(button => button.onclick = async () => { try { await api(`/api/model-services/${encodeURIComponent(button.dataset.deleteService)}`,{method:"DELETE",body:"{}"}); await loadModelServices(); } catch(error){toast(error.message,true);} });
}

async function clearDialogueModel() {
  if (!state.person) return;
  try {
    await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/model`, {method:"POST",body:JSON.stringify({model_ref:""})});
    await refreshConversation();
    await loadModelServices();
    toast("此人物已改为不使用对话模型；历史回答和人物版本没有改变。");
  } catch(error) { toast(error.message, true); }
}

async function modelServiceAction(serviceId, action, button, payload = {}) {
  busy(button,true);
  try {
    const data=await api(`/api/model-services/${encodeURIComponent(serviceId)}/${action}`,{method:"POST",body:JSON.stringify(payload)});
    const verifiedForPerson = action === "test" && data.result?.status === "connected" && state.person && payload.model_id;
    if (verifiedForPerson) {
      const modelRef = `${serviceId}:${payload.model_id}`;
      await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/model`, {method:"POST",body:JSON.stringify({model_ref:modelRef})});
      await refreshConversation();
    }
    await loadModelServices();
    const failed = data.result?.status === "unavailable";
    toast(verifiedForPerson ? "真实调用验证成功，已用于当前人物。" : data.result?.message || "模型列表已刷新。", failed);
  }
  catch(error){toast(error.message,true);} finally {busy(button,false);}
}

async function submitModelService(event) {
  event.preventDefault();
  const form=event.currentTarget, values=Object.fromEntries(new FormData(form));
  const body={...values,enabled:form.elements.enabled.checked,models:String(values.models||"").split(",").map(v=>v.trim()).filter(Boolean),enabled_models:String(values.models||"").split(",").map(v=>v.trim()).filter(Boolean),timeout_seconds:Number(values.timeout_seconds||30),capabilities:{structured_output:form.elements.structured_output.checked}};
  const submitBtn = form.querySelector('button[type="submit"]');
  try {
    busy(submitBtn, true, "保存中…");
    await api("/api/model-services",{method:"POST",body:JSON.stringify(body)});
    form.reset(); form.elements.timeout_seconds.value="30"; form.elements.enabled.checked=true; form.elements.structured_output.checked=true; $("#model-preset-select").value="";
    await loadModelServices();
    toast("配置已保存，尚未调用模型。请先刷新模型列表，再验证要使用的模型。");
  }
  catch(error){toast(error.message,true);}
  finally { busy(submitBtn, false); }
}

function editModelService(serviceId) {
  const service = state.modelServices.services.find(item => item.service_id === serviceId);
  if (!service) return;
  const form = $("#model-service-form");
  form.closest("details").open = true;
  form.elements.service_id.value = service.service_id;
  form.elements.display_name.value = service.display_name || "";
  form.elements.protocol.value = service.protocol || "openai_compatible";
  form.elements.provider.value = service.provider || "";
  form.elements.base_url.value = service.base_url || "";
  form.elements.api_key.value = "";
  form.elements.api_key.placeholder = service.api_key_configured ? "已配置（留空则保留原密钥）" : "粘贴 API Key（保存后不再显示）";
  form.elements.environment_key.value = service.environment_key || "";
  form.elements.models.value = (service.models || []).join(", ");
  form.elements.timeout_seconds.value = service.timeout_seconds || 30;
  form.elements.structured_output.checked = service.capabilities?.structured_output !== false;
  form.elements.enabled.checked = service.enabled !== false;
  $("#model-preset-select").value = "";
  form.querySelector("h3").textContent = "编辑服务（API Key 留空则保留原密钥）";
  form.scrollIntoView({behavior:"smooth", block:"center"});
}

function applyPreset(presetKey) {
  const preset = MODEL_PRESETS.find(item => item.key === presetKey);
  if (!preset) return;
  const form = $("#model-service-form");
  form.elements.display_name.value = preset.display_name;
  form.elements.protocol.value = preset.protocol;
  form.elements.provider.value = preset.provider;
  form.elements.base_url.value = preset.base_url;
  form.elements.models.value = preset.models.join(", ");
  form.elements.structured_output.checked = true;
  if (preset.protocol === "ollama") {
    form.elements.api_key.value = "";
    form.elements.api_key.placeholder = "Ollama 无需 API Key";
  } else {
    form.elements.api_key.placeholder = "粘贴 API Key（保存后不再显示）";
  }
  if (preset.hint) toast(preset.hint);
}

async function toggleKeyVisibility() {
  const input = $("#model-service-form [name=api_key]");
  const serviceId = $("#model-service-form [name=service_id]").value;
  const eye = $("#key-eye");
  if (input.type === "password") {
    if (serviceId) {
      try {
        const data = await api(`/api/model-services/${encodeURIComponent(serviceId)}/key`);
        input.value = data.key || "";
      } catch (error) { toast(error.message, true); }
    }
    input.type = "text";
    eye.textContent = "🙈";
  } else {
    input.type = "password";
    input.value = "";
    eye.textContent = "👁";
  }
}

async function setModelRole(role, modelRef) {
  try { await api("/api/model-roles",{method:"POST",body:JSON.stringify({role,model_ref:modelRef})}); await loadModelServices(); }
  catch(error){toast(error.message,true);}
}

async function loadPeople(selectId = state.person?.person_id) {
  const data = await api("/api/people");
  state.people = data.people;
  renderPeople();
  if (selectId && personById(selectId)) await selectPerson(selectId);
  else if (!state.person && state.people.length) await selectPerson(state.people[0].person_id);
  else if (!state.people.length) showEmptyWorkspace();
}

function showEmptyWorkspace() {
  state.person = null; state.conversation = null;
  $("#empty-chat").hidden = false; $("#chat-workspace").hidden = true;
}

async function selectPerson(personId) {
  const [personData, conversationData] = await Promise.all([
    api(`/api/people/${encodeURIComponent(personId)}`),
    api(`/api/people/${encodeURIComponent(personId)}/conversation`),
  ]);
  state.person = personData.person;
  state.conversation = conversationData.conversation;
  state.comparison = null;
  closeDrawer(); renderPeople(); renderWorkspace(); await loadSessions();
}

function isAssistant() { return state.person?.person_id === "assistant"; }

async function selectAssistant() {
  const data = await api("/api/assistant/conversation");
  state.person = { person_id: "assistant", name: "AI 助手", avatar: "/default-person-avatar.png" };
  state.conversation = data.conversation;
  state.comparison = null;
  closeDrawer(); renderPeople(); renderWorkspace();
  if (!state.conversation.messages.length) {
    state.conversation.messages = [{ message_id: "assistant-greeting", role: "assistant", text: "我是 AI 助手，帮你操作这个系统。\n· 建人物\n· 加材料\n· 搜索\n· 归档 / 恢复 / 永久删除\n· 列出人物 / 归档列表", status: "answered", answer_status: "assistant" }];
  }
  renderMessages();
}

async function refreshAssistant() {
  const data = await api("/api/assistant/conversation");
  state.conversation = data.conversation;
  renderMessages();
}

async function refreshConversation() {
  if (!state.person) return;
  const data = await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation`);
  state.conversation = data.conversation;
  const summary = personById(state.person.person_id);
  if (summary) {
    summary.conversation_version = state.conversation.active_version;
    summary.source_count = state.conversation.source_counts.confirmed;
    summary.message_count = state.conversation.messages.length;
    summary.last_message = state.conversation.messages.at(-1)?.text || "";
    summary.conversation_status = state.conversation.status;
    summary.conversation_status_text = state.conversation.status_text;
  }
  renderPeople(); renderWorkspace(); await loadSessions();
}

function renderWorkspace() {
  if (!state.person || !state.conversation) return showEmptyWorkspace();
  $("#empty-chat").hidden = true; $("#chat-workspace").hidden = false;
  if (isAssistant()) {
    $("#chat-person-name").textContent = "AI 助手";
    $("#chat-avatar").src = "/default-person-avatar.png";
    $("#chat-version").textContent = "操作员助手";
    $("#chat-source-count").textContent = "建人物 · 加材料 · 搜索 · 归档";
    $("#collection-status").textContent = "说个意图，我列步骤帮你操作。";
    const badge = $("#chat-model-state");
    badge.textContent = "AI 助手（工具调用）";
    badge.className = "model-state";
    $("#open-model-picker").textContent = "助手模型";
    renderMessages();
    return;
  }
  $("#chat-person-name").textContent = state.person.name;
  $("#chat-avatar").src = state.person.avatar || "/default-person-avatar.png";
  $("#chat-avatar").onerror = () => { $("#chat-avatar").src = "/default-person-avatar.png"; };
  $("#chat-version").textContent = state.conversation.active_version ? `模型版本 v${state.conversation.active_version}` : "未建立模型版本";
  $("#chat-source-count").textContent = `${state.conversation.source_counts.confirmed} 份已确认资料`;
  const collection = state.person.collection || state.conversation.profile.collection || {};
  const collectionMessages = {
    candidates_found:`系统已找到 ${collection.candidate_count || 0} 条公开资料候选；核验原文前不会用于训练。`,
    no_candidates:"系统已完成公开搜索，但没有找到可用候选。",
    temporarily_unavailable:"公开资料搜索暂时不可用，可以稍后重试或自行提供资料。",
    search_ready:"搜索服务已配置；结果只进入待审核候选资料。",
    awaiting_user_materials:"等待用户提供原始资料；系统会自动提取待审核响应事件。",
    verified_demo_materials_loaded:"已载入可追溯的一手演示资料；预测仍属探索性，准确性尚未验证。",
  };
  const modelView = state.conversation.public_response_model || {};
  $("#collection-status").textContent = `${collectionMessages[collection.status] || collection.message || "资料状态尚未记录。"} 当前模拟层：${modelView.event_frame_count || 0} 个事件原子、${modelView.value_atom_count || 0} 个单事件公开取向原子、${modelView.value_orientation_count || 0} 个聚合公开取向、${modelView.preference_structure_count || 0} 个明确取舍结构、${modelView.knowledge_claim_count || 0} 条人物公开使用的知识主张。`;
  const networkNote = $(".network-note p");
  if (networkNote) networkNote.textContent = collection.mode === "system_search"
    ? $("#collection-status").textContent
    : "系统搜索未配置时，请自行提供资料；所有自动提取结果都必须先审核。";
  const badge = $("#chat-model-state");
  const modelRef = state.conversation.dialogue_model_ref || "";
  const modelService = modelRef ? state.modelServices.services.find(s => modelRef.startsWith(s.service_id + ":")) : null;
  const selectedModelId = modelRef.split(":").slice(1).join(":");
  const modelUnavailable = modelService && !(modelService.call_readiness === "ready" && modelService.last_probe_model === selectedModelId);
  if (modelUnavailable) {
    badge.innerHTML = `对话模型需验证：<strong>${escapeHtml(modelService.display_name)}</strong> <button type="button" class="text-button" id="fix-model-btn">处理</button>`;
    badge.className = "model-state insufficient";
    $("#fix-model-btn").onclick = async () => { await loadModelServices(); $("#model-services-dialog").showModal(); };
  } else {
    badge.textContent = state.conversation.status_text;
    badge.className = `model-state ${state.conversation.status === "insufficient_evidence" ? "insufficient" : ""}`;
  }
  $("#open-model-picker").textContent = currentModelLabel();
  $("#export-person").href = `/api/people/${encodeURIComponent(state.person.person_id)}/export`;
  renderMessages(); renderSources(); renderVersions(); renderMetrics(); renderSessionBar();
  $("#paste-source-form [name=speaker]").value = state.person.name;
  $("#file-source-form [name=speaker]").value = state.person.name;
  $("#url-source-form [name=speaker]").value = state.person.name;
}

function evidenceHtml(message) {
  if (!message.evidence?.length) return `<p>本次没有可用于支持预测内容的直接证据。</p>`;
  return message.evidence.map(item => `<p><strong>${escapeHtml(item.title)}</strong><br>响应事件：${escapeHtml(item.event_id || "未记录")} · 说话人：${escapeHtml(item.speaker || "未记录")} · 日期：${escapeHtml(item.date || "未记录")} · 位置：${escapeHtml(item.locator)}<br>候选采用分数：${Number(item.support_score).toFixed(2)}${item.url ? ` · <a href="${escapeHtml(item.url)}" target="_blank" rel="noreferrer">打开来源</a>` : ""}</p>`).join("");
}

function renderMessages() {
  const container = $("#messages");
  const messages = state.conversation.messages;
  if (isAssistant()) {
    if (!messages.length) {
      container.innerHTML = `<div class="messages-empty"><div><strong>AI 助手</strong><p>说个意图，我列步骤帮你操作：建人物 / 加材料 / 搜索 / 归档 / 恢复 / 永久删除。</p></div></div>`;
      return;
    }
    container.innerHTML = messages.map(message => {
      if (message.role === "user") return `<article class="message-row user"><div class="user-bubble">${escapeHtml(message.text)}</div></article>`;
      return `<article class="message-row assistant"><img class="assistant-avatar" src="/default-person-avatar.png" alt=""><div class="assistant-body"><div class="answer">${escapeHtml(message.text).replace(/\n/g, "<br>")}</div></div></article>`;
    }).join("");
    requestAnimationFrame(() => container.scrollTop = container.scrollHeight);
    return;
  }
  if (!messages.length) {
    const suggestions = (state.person.recommended_questions || []).map(item => `<button type="button" class="suggestion-chip" data-suggested-question="${escapeHtml(item.text)}"><span>${escapeHtml(item.label)}</span>${escapeHtml(item.text)}</button>`).join("");
    container.innerHTML = `<div class="messages-empty"><div><strong>现在可以直接开始对话</strong><p>${state.conversation.active_version ? "系统会把当前消息作为完整会话状态的增量，再结合历史事件、人物公开取向和外部知识组织回答。" : "尚未建立人物模型；选择对话模型后仍可正常回答，但会标记为通用知识而非人物预测。"}</p>${suggestions ? `<div class="suggestion-list"><p>推荐测试问题</p>${suggestions}</div>` : ""}</div></div>`;
    $$('[data-suggested-question]').forEach(button => button.onclick = () => {
      const input = $('#message-form [name="text"]');
      input.value = button.dataset.suggestedQuestion;
      input.focus();
    });
    return;
  }
  container.innerHTML = messages.map(message => {
    if (message.role === "user") return `<article class="message-row user"><div class="user-bubble">${escapeHtml(message.text)}</div></article>`;
    const comparisonReady = message.comparison?.status === "candidate_found";
    const usage = message.model_usage || {};
    const modelUsageText = usage.total_calls > 0
      ? `本次大模型调用 ${usage.total_calls} 次`
      : usage.status === "selected_but_not_needed" ? "本次未调用大模型" : "本次未调用大模型";
    return `<article class="message-row assistant ${message.status === "refused" ? "refused" : ""}">
      <img class="assistant-avatar" src="${escapeHtml(state.person.avatar || "/default-person-avatar.png")}" alt="">
      <div class="assistant-body">
        <div class="answer">${escapeHtml(message.text)}</div>
        ${message.uncertainties?.length ? `<div class="plain-notice">不确定项：${message.uncertainties.map(escapeHtml).join("；")}</div>` : ""}
        <div class="message-actions">
          <button data-action="evidence" data-id="${message.message_id}">依据</button>
          <button data-action="reality" data-id="${message.message_id}">现实回答</button>
          <button data-action="feedback" data-id="${message.message_id}">${message.feedback ? "已反馈" : "反馈"}</button>
          ${comparisonReady ? `<button data-action="open-comparison" data-id="${message.message_id}">发现可核验回答</button>` : ""}
        </div>
        <div class="evidence-details" id="evidence-${message.message_id}" hidden>
          <p class="evidence-meta">${statusLabel(message.answer_status || message.status)} · ${message.person_prediction_status && message.person_prediction_status !== "not_available" ? `证据支持 ${Number(message.confidence).toFixed(2)}（非准确率）` : "非人物预测"}</p>
          ${evidenceHtml(message)}
          <p>回应动作：${escapeHtml(message.structured_prediction?.speech_act?.label || "未输出")}；立场：${escapeHtml(message.structured_prediction?.stance?.label || "未输出")}；回答路径：${escapeHtml(statusLabel(message.answer_status || message.status))}；知识来源：${escapeHtml(message.knowledge_source || "none")}</p>
          <p>内容模型：${escapeHtml(message.model_kind)}；对话模型：${escapeHtml(message.dialogue_model_provider && message.dialogue_model_id ? `${message.dialogue_model_provider} · ${message.dialogue_model_id}` : "未选择")}；${escapeHtml(modelUsageText)}；表达状态：${escapeHtml(humanStatus(message.style_status))}；准确性：${escapeHtml(humanStatus(message.response_accuracy_status))}</p>
        </div>
      </div>
    </article>`;
  }).join("");
  $$('[data-action="evidence"]').forEach(button => button.onclick = () => { const box = $(`#evidence-${CSS.escape(button.dataset.id)}`); box.hidden = !box.hidden; });
  $$('[data-action="reality"]').forEach(button => button.onclick = () => runRealityLookup(button.dataset.id, button));
  $$('[data-action="open-comparison"]').forEach(button => button.onclick = () => openComparison(state.conversation.messages.find(item => item.message_id === button.dataset.id).comparison));
  $$('[data-action="feedback"]').forEach(button => button.onclick = () => sendFeedback(button.dataset.id, button));
  requestAnimationFrame(() => container.scrollTop = container.scrollHeight);
}

async function sendMessage(event) {
  event.preventDefault();
  if (!state.person) return;
  const form = event.currentTarget, button = event.submitter, text = form.elements.text.value.trim();
  if (!text) return;
  const lookup = form.elements.reality_lookup.checked;
  const status = $("#composer-status");
  status.hidden = true;
  busy(button, true, "生成中…");
  try {
    if (isAssistant()) {
      await api("/api/assistant/message", {method:"POST", body: JSON.stringify({text})});
      form.elements.text.value = "";
      await refreshAssistant();
      await loadPeople();
      return;
    }
    const data = await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/messages`, {method:"POST",body:JSON.stringify({text,reality_lookup_requested:lookup,dialogue_model_ref:state.conversation.dialogue_model_ref || ""})});
    form.elements.text.value = "";
    await refreshConversation();
    if (lookup) setTimeout(() => runRealityLookup(data.message.message_id), 60);
  } catch (error) {
    status.textContent = `发送失败：${error.message} 输入内容已保留。`;
    status.hidden = false;
    toast(error.message, true);
  }
  finally { busy(button, false); }
}

async function startNewConversation(button) {
  if (!state.person) return;
  busy(button, true, "处理中…");
  try {
    await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/new`, {method:"POST", body:"{}"});
    $("#new-conversation-dialog").close();
    await refreshConversation();
    toast("新对话已开始。");
  } catch(error) { toast(error.message, true); }
  finally { busy(button, false); }
}

async function runRealityLookup(messageId, button = null) {
  if (!state.person) return;
  busy(button, true, "查找中…");
  try {
    const data = await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/messages/${encodeURIComponent(messageId)}/reality`, {method:"POST",body:"{}"});
    await refreshConversation();
    if (data.comparison.status === "candidate_found") openComparison(data.comparison);
    else toast("未找到可核验的现实回答。系统没有生成伪造对照。");
  } catch (error) { toast(error.message, true); }
  finally { busy(button, false); }
}

function openComparison(comparison) {
  if (!comparison || comparison.status !== "candidate_found") return;
  state.comparison = comparison;
  const realityCandidates = comparison.reality_candidates || [];
  const candidateList = realityCandidates.length ? `<section class="compare-step"><h3>2 请选择要审核的现实回答候选</h3>${realityCandidates.map(item => `<button type="button" class="compare-box" data-reality-candidate="${escapeHtml(item.comparison_candidate_id)}"><strong>${comparison.selected_candidate_id===item.comparison_candidate_id?"已选择 · ":""}相似度 ${Number(item.score).toFixed(2)}</strong><p>${escapeHtml(item.question)}</p><blockquote>${escapeHtml(item.answer)}</blockquote><small>${escapeHtml(item.source_title)} · ${escapeHtml(item.speaker)} · ${escapeHtml(item.source_date || "未记录")} · ${escapeHtml(item.locator)}</small></button>`).join("")}</section>` : "";
  $("#comparison-content").innerHTML = `
    <section class="compare-step"><h3>1 当前模型预测回答</h3><div class="compare-box"><blockquote>${escapeHtml(comparison.predicted_answer)}</blockquote></div></section>
    ${candidateList}
    <section class="compare-step"><h3>3 匹配与差异</h3><div class="compare-box"><p>系统只提供候选，不替用户认定问题相同。</p><p>语境检查：${escapeHtml(comparison.context_consistency)}</p><p>主要一致点：${comparison.agreements.map(escapeHtml).join("、")}</p><p>差异：${comparison.differences.map(escapeHtml).join("；")}</p></div></section>
    <div class="warning-box">${escapeHtml(comparison.notice)} 加入待优化资料后仍需来源、身份、去重、时间、数据角色和独立留出检查。</div>
    <div class="compare-actions"><button class="button primary" data-compare-action="candidate">加入待优化资料</button><button class="button quiet" data-compare-action="reference">仅保存为参考</button><button class="button quiet" data-compare-action="not-same">不是同一个问题</button></div>`;
  $("#comparison-drawer").hidden = false; $(".app-shell").classList.add("drawer-open");
  $$('[data-reality-candidate]').forEach(button => button.onclick = () => { comparison.selected_candidate_id=button.dataset.realityCandidate; openComparison(comparison); });
  $$('[data-compare-action]').forEach(button => button.onclick = () => handleComparisonAction(button.dataset.compareAction, button));
}

function closeDrawer() { $("#comparison-drawer").hidden = true; $(".app-shell").classList.remove("drawer-open"); state.comparison = null; }

async function handleComparisonAction(action, button) {
  if (!state.person || !state.comparison) return;
  busy(button, true);
  try {
    const created = await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/messages/${encodeURIComponent(state.comparison.message_id)}/optimization`, {method:"POST",body:JSON.stringify({comparison_candidate_id:state.comparison.selected_candidate_id || ""})});
    if (action === "candidate") toast("已加入待优化资料；当前人物版本没有改变。");
    else {
      const decision = action === "reference" ? "reference_only" : "not_same_question";
      await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/optimization/${encodeURIComponent(created.candidate.candidate_id)}/review`, {method:"POST",body:JSON.stringify({decision})});
      toast(action === "reference" ? "已仅保存为参考，未修改模型。" : "已标记为不是同一个问题。");
    }
    closeDrawer(); await refreshConversation(); $("#sources-dialog").showModal();
  } catch (error) { toast(error.message, true); }
  finally { busy(button, false); }
}

async function sendFeedback(messageId, button) {
  if (button.textContent === "已反馈") return;
  try {
    await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/messages/${encodeURIComponent(messageId)}/feedback`, {method:"POST",body:JSON.stringify({value:"helpful"})});
    toast("已保存反馈。后续可以扩展为更细的错误类型。"); await refreshConversation();
  } catch (error) { toast(error.message, true); }
}

async function reviewSource(sourceId, decision, button) {
  busy(button, true);
  try { await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/sources/${encodeURIComponent(sourceId)}/review`, {method:"POST",body:JSON.stringify({decision})}); toast(decision === "confirmed" ? "资料已确认；只有参数训练资料会形成新的探索性版本。" : "资料已拒绝。"); await refreshConversation(); }
  catch (error) { toast(error.message, true); }
  finally { busy(button, false); }
}

async function reviewOptimization(candidateId, decision, button) {
  busy(button, true, "校验中…");
  try {
    const data = await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/optimization/${encodeURIComponent(candidateId)}/review`, {method:"POST",body:JSON.stringify({decision})});
    if (data.candidate.status === "accepted_exploratory") toast(`已生成探索性版本 v${data.candidate.new_version}；人物响应准确性仍未验证。`);
    else if (data.candidate.status === "failed_validation") toast(`优化未通过：${data.candidate.validation_reasons.join("、")}`, true);
    else toast("候选已处理，当前版本未改变。");
    await refreshConversation();
  } catch (error) { toast(error.message, true); }
  finally { busy(button, false); }
}

async function reviewOptimizationStyle(candidateId, decision, button) {
  busy(button, true, "校验表达样本…");
  try {
    const data = await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/optimization/${encodeURIComponent(candidateId)}/style-review`, {method:"POST",body:JSON.stringify({decision})});
    const status = data.candidate.surface_extraction?.status;
    toast(status === "accepted_exploratory" ? "表达样本已单独验证并形成新风格版本；内容模型未改变。" : "表达样本已单独拒绝，内容模型不受影响。", status === "failed_validation");
    await refreshConversation();
  } catch (error) { toast(error.message, true); }
  finally { busy(button, false); }
}

async function rollbackVersion(version, button) {
  busy(button, true);
  try { await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/versions/${version}/rollback`, {method:"POST",body:"{}"}); toast(`已回退到版本 v${version}。`); await refreshConversation(); }
  catch (error) { toast(error.message, true); }
  finally { busy(button, false); }
}

function renderMetrics() {
  if (!state.conversation) return;
  const labels = {content_holdout_agreement:"内容与真实留出回答一致度",correct_person_uplift:"正确人物相对基线增益",confidence_calibration:"置信度校准",fact_source_support:"事实与经历来源支持率",style_blind_test:"风格盲测",style_semantic_preservation:"风格化语义保持",out_of_scope_handling:"新领域处理"};
  const formatMetric = value => typeof value === "number" ? value.toFixed(3) : (value && typeof value === "object" ? JSON.stringify(value) : humanStatus(value));
  $("#accuracy-metrics").innerHTML = Object.entries(state.conversation.metrics).map(([key,value]) => `<div class="metric-line"><span>${labels[key] || key}</span><strong>${escapeHtml(formatMetric(value))}</strong></div>`).join("");
}

async function submitTextSource(event) {
  event.preventDefault(); const form = event.currentTarget, button = event.submitter; busy(button,true);
  try { const body = Object.fromEntries(new FormData(form)); await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/sources/text`,{method:"POST",body:JSON.stringify(body)}); form.elements.text.value=""; toast("资料已保存为待审核，尚未进入人物版本。"); await refreshConversation(); }
  catch(error){toast(error.message,true)} finally{busy(button,false)}
}

function fileToBase64(file) { return new Promise((resolve,reject)=>{ const reader=new FileReader(); reader.onload=()=>resolve(String(reader.result).split(",")[1]); reader.onerror=()=>reject(reader.error); reader.readAsDataURL(file); }); }
async function submitFileSource(event) {
  event.preventDefault(); const form=event.currentTarget,button=event.submitter,file=form.elements.file.files[0]; busy(button,true,"提取中…");
  try { const content_base64=await fileToBase64(file); const fields=Object.fromEntries(new FormData(form)); await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/sources/file`,{method:"POST",body:JSON.stringify({...fields,filename:file.name,content_base64,source_date:""})}); form.elements.file.value=""; toast("文件已按事件候选整理并进入待审核队列。"); await refreshConversation(); }
  catch(error){toast(error.message,true)} finally{busy(button,false)}
}

async function submitUrlSource(event) {
  event.preventDefault(); const form=event.currentTarget,button=event.submitter; busy(button,true,"抓取中…");
  try { const fields=Object.fromEntries(new FormData(form)); await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/sources/url`,{method:"POST",body:JSON.stringify({...fields,source_date:""})}); form.elements.url.value=""; toast("网页快照已按事件候选整理并进入待审核队列。"); await refreshConversation(); }
  catch(error){toast(error.message,true)} finally{busy(button,false)}
}

function openPersonDialog(editing = false) {
  state.editingPerson = editing;
  const form=$("#person-form"); form.reset();
  if (editing && state.person) {
    form.elements.name.value=state.person.name; form.elements.description.value=state.person.description||"";
    form.elements.language.value=state.conversation.profile.language; form.elements.aliases.value=(state.conversation.profile.aliases||[]).join(", ");
    form.elements.source_mode.value=state.person.collection?.mode||"user_provided"; form.elements.identity_note.value=state.person.identity_note||"";
    form.elements.focus_domain.value=state.person.focus_domain||""; form.elements.avatar.value=state.person.avatar||""; form.elements.notes.value=state.person.notes||"";
  }
  $("#person-dialog h2").textContent=editing?"编辑人物":"新建人物"; $("#person-dialog").showModal();
}

function configureSearchCapability(capabilities = {}) {
  state.capabilities = capabilities;
  const search = capabilities.public_search || {available:false};
  const form = $("#person-form");
  const option = form.elements.source_mode.querySelector('option[value="system_search"]');
  option.disabled = !search.available;
  option.textContent = search.available ? "系统自动搜索公开资料" : "系统自动搜索公开资料（暂未配置）";
  if (!search.available && form.elements.source_mode.value === "system_search" && !state.editingPerson) {
    form.elements.source_mode.value = "user_provided";
  }
  $("#person-source-notice").textContent = search.available
    ? "系统搜索已配置。搜索结果只进入待审核候选资料，核验来源、身份、说话人和内容真实性后才能用于建模。"
    : "系统自动搜索暂未配置。请粘贴文本、上传文件或输入网页地址，系统会自动整理为待审核资料。";
}

async function submitPerson(event) {
  event.preventDefault(); const form=event.currentTarget,button=event.submitter; busy(button,true);
  try {
    const body={name:form.elements.name.value,description:form.elements.description.value,aliases:form.elements.aliases.value.split(/[,，]/).map(x=>x.trim()).filter(Boolean),language:form.elements.language.value,time_start:form.elements.time_start.value,time_end:form.elements.time_end.value,source_mode:form.elements.source_mode.value,identity_note:form.elements.identity_note.value,focus_domain:form.elements.focus_domain.value,avatar:form.elements.avatar.value,notes:form.elements.notes.value};
    let personId;
    if (state.editingPerson) { await api(`/api/people/${encodeURIComponent(state.person.person_id)}`,{method:"PUT",body:JSON.stringify(body)}); personId=state.person.person_id; }
    else { const data=await api("/api/conversation/people",{method:"POST",body:JSON.stringify(body)}); personId=data.person.person_id; }
    $("#person-dialog").close(); state.person=null; await loadPeople(personId);
    if (!state.editingPerson && body.source_mode === "system_search") {
      const collection = state.person?.collection || {};
      toast(collection.message || "公开资料搜索已结束；所有结果都需要核验后才能训练。", collection.status === "temporarily_unavailable");
      return;
    }
    if (state.editingPerson) toast("人物信息已更新。");
    else if (body.source_mode==="system_search") toast("人物已创建；当前搜索服务未配置，系统没有伪造收集结果。");
    else { toast("人物已创建；下一步请添加原始资料。"); $("#sources-dialog").showModal(); }
  } catch(error){toast(error.message,true)} finally{busy(button,false)}
}

async function importBackup(file) {
  try { const payload=JSON.parse(await file.text()); const data=await api("/api/import-product",{method:"POST",body:JSON.stringify({payload})}); await loadPeople(data.person.person_id); toast("人物备份已加载。"); }
  catch(error){toast(error.message,true)}
}

function renderPeople() {
  const query = $("#person-search").value.trim().toLowerCase();
  const visible = state.people.filter(person => `${person.name} ${person.last_message}`.toLowerCase().includes(query));
  $("#people-empty").hidden = visible.length !== 0;
  const assistantActive = state.person?.person_id === "assistant" ? "active" : "";
  $("#people-list").innerHTML = `<article class="person-card ${assistantActive}" data-person-card="assistant"><button class="person-select" data-person-id="assistant"><span class="assistant-emoji">🤖</span><span><strong>AI 助手</strong><small class="recent">建人物·加材料·搜索·归档</small></span></button></article>` + visible.map(person => `
    <article class="person-card ${state.person?.person_id === person.person_id ? "active" : ""}" draggable="true" data-person-card="${escapeHtml(person.person_id)}">
      <button class="person-select" data-person-id="${escapeHtml(person.person_id)}">
        <img src="${escapeHtml(person.avatar || "/default-person-avatar.png")}" alt="${escapeHtml(person.name)}" onerror="this.src='/default-person-avatar.png'">
        <span><strong>${escapeHtml(person.name)}${person.is_demo ? ' <em class="demo-badge">演示</em>' : ''}</strong><small class="recent">${escapeHtml(person.last_message || "开始对话")}</small></span>
      </button>
      <button class="person-more" aria-label="${escapeHtml(person.name)}的更多操作" data-person-more="${escapeHtml(person.person_id)}">⋯</button>
      <div class="person-menu" data-person-menu="${escapeHtml(person.person_id)}" hidden>
        <button data-edit-person="${escapeHtml(person.person_id)}">编辑人物</button>
        <a href="/api/people/${encodeURIComponent(person.person_id)}/export">导出备份</a>
        <button data-archive-person="${escapeHtml(person.person_id)}">移入归档</button>
      </div>
    </article>`).join("");
  $$(".person-select").forEach(button => button.onclick = () => (button.dataset.personId === "assistant" ? selectAssistant() : selectPerson(button.dataset.personId)));
  $$('[data-person-more]').forEach(button => button.onclick = event => {
    event.stopPropagation();
    const menu = document.querySelector(`[data-person-menu="${CSS.escape(button.dataset.personMore)}"]`);
    $$(".person-menu").forEach(item => { if (item !== menu) item.hidden = true; });
    menu.hidden = !menu.hidden;
  });
  $$('[data-edit-person]').forEach(button => button.onclick = async () => {
    await selectPerson(button.dataset.editPerson); openPersonDialog(true);
  });
  $$('[data-archive-person]').forEach(button => button.onclick = () => requestArchive(button.dataset.archivePerson));
  $$('[data-person-card]').forEach(card => {
    card.addEventListener("dragstart", event => {
      state.draggedPersonId = card.dataset.personCard;
      event.dataTransfer.effectAllowed = "move";
      event.dataTransfer.setData("text/plain", state.draggedPersonId);
      card.classList.add("dragging");
      setArchiveDropState("ready");
    });
    card.addEventListener("dragend", () => {
      card.classList.remove("dragging");
      state.draggedPersonId = null;
      setArchiveDropState();
    });
  });
  $$('[data-person-drag]').forEach(handle => {
    handle.addEventListener("pointerdown", event => {
      if (event.button !== 0) return;
      event.preventDefault();
      state.draggedPersonId = handle.dataset.personDrag;
      state.pointerDragId = event.pointerId;
      handle.closest(".person-card").classList.add("dragging");
      setArchiveDropState("ready");
    });
  });
}

function finishPointerArchiveDrag(event, cancelled = false) {
  if (state.pointerDragId === null) return;
  const rect = $("#archive-drop-zone").getBoundingClientRect();
  const overZone = !cancelled && event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
  const personId = state.draggedPersonId;
  const handle = document.querySelector(`[data-person-drag="${CSS.escape(personId || "")}"]`);
  handle?.closest(".person-card")?.classList.remove("dragging");
  state.draggedPersonId = null;
  state.pointerDragId = null;
  setArchiveDropState();
  if (overZone && personId) requestArchive(personId).catch(error => toast(error.message, true));
}

function setArchiveDropState(mode = "") {
  const zone = $("#archive-drop-zone");
  zone.classList.toggle("drag-ready", mode === "ready" || mode === "active");
  zone.classList.toggle("drop-active", mode === "active");
  $("#archive-drop-hint").textContent = mode ? "放到这里，移入归档" : "也可以把人物卡片拖到这里";
}

async function requestArchive(personId) {
  if (state.person?.person_id !== personId) await selectPerson(personId);
  state.archiveTarget = state.person;
  if (state.archiveTarget) $("#archive-confirm-dialog").showModal();
}

function wireArchiveDropZone() {
  const zone = $("#archive-drop-zone");
  document.addEventListener("pointermove", event => {
    if (state.pointerDragId === null) return;
    const rect = zone.getBoundingClientRect();
    const overZone = event.clientX >= rect.left && event.clientX <= rect.right && event.clientY >= rect.top && event.clientY <= rect.bottom;
    setArchiveDropState(overZone ? "active" : "ready");
  });
  document.addEventListener("pointerup", event => finishPointerArchiveDrag(event));
  document.addEventListener("pointercancel", event => finishPointerArchiveDrag(event, true));
  document.addEventListener("mouseup", event => finishPointerArchiveDrag(event));
  zone.addEventListener("dragenter", event => {
    if (!state.draggedPersonId) return;
    event.preventDefault();
    setArchiveDropState("active");
  });
  zone.addEventListener("dragover", event => {
    if (!state.draggedPersonId) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
    setArchiveDropState("active");
  });
  zone.addEventListener("dragleave", event => {
    if (!zone.contains(event.relatedTarget)) setArchiveDropState(state.draggedPersonId ? "ready" : "");
  });
  zone.addEventListener("drop", event => {
    event.preventDefault();
    const personId = event.dataTransfer.getData("text/plain") || state.draggedPersonId;
    state.draggedPersonId = null;
    setArchiveDropState();
    if (personId) requestArchive(personId).catch(error => toast(error.message, true));
  });
}

async function refreshArchiveCount() {
  const data = await api("/api/archived-people");
  $("#archive-count").textContent = String(data.people.length);
  return data.people;
}

function renderSources() {
  if (!state.conversation) return;
  const counts = state.conversation.source_counts;
  $("#source-count-summary").textContent = `${counts.confirmed} 已确认 · ${counts.pending} 待审核 · ${counts.final_holdout} 最终留出`;
  $("#sources-list").innerHTML = state.conversation.sources.length ? state.conversation.sources.map(source => `
    <article class="source-item"><header><strong>${escapeHtml(source.title)}</strong><span class="tag ${escapeHtml(source.review_status)}">${escapeHtml(humanStatus(source.review_status))}</span></header>
    <p>${escapeHtml(source.text_preview)}</p>
    <p>说话人：${escapeHtml(source.speaker || "未记录")} · 范围：${source.speaker_scope === "mixed_speakers" ? "多人混合，需逐段确认" : "整份材料主要说话人"} · 格式：${escapeHtml(source.format)} · 数据用途：${escapeHtml(humanStatus(source.dataset_role))} · 事件包：${source.response_events?.length || 0}</p>
    <p>${source.response_events?.some(item => item.label_status === "confirmed_response_weak_semantic_labels") ? "包含可追溯的本人公开回应；系统已按事件、条件倾向和公开使用的知识主张整理。" : "尚无可进入人物模型的公开回应；材料仍作为待核验或参考资料保留。"}</p>
    ${source.llm_response_event_candidates?.length ? `<p>资料处理模型提出 ${source.llm_response_event_candidates.length} 条待审核响应事件候选；尚未进入训练。</p>` : ""}
    ${(source.llm_response_event_candidates || []).map(candidate => `<div class="candidate-box"><p><strong>${escapeHtml(candidate.trigger || "公开回应候选")}</strong> · ${escapeHtml(candidate.source_locator || "未标注位置")}</p><blockquote>${escapeHtml(candidate.actual_response || "")}</blockquote><small>说话人：${escapeHtml(candidate.speaker || "未识别")} · ${escapeHtml(candidate.review_status || "pending")}</small>${source.review_status === "confirmed" && (candidate.review_status || "pending") === "pending" ? `<div class="item-actions"><button class="mini-button confirm" data-event-candidate-review="confirmed" data-source-id="${source.source_id}" data-candidate-id="${candidate.candidate_id}">确认逐字位置并采用</button><button class="mini-button" data-event-candidate-review="rejected" data-source-id="${source.source_id}" data-candidate-id="${candidate.candidate_id}">拒绝候选</button></div>` : ""}</div>`).join("")}
    ${source.review_status === "pending" ? `<div class="item-actions"><button class="mini-button" data-extract-source="${source.source_id}">用资料处理模型提取候选</button><button class="mini-button confirm" data-source-review="confirmed" data-id="${source.source_id}">${source.speaker_scope === "mixed_speakers" ? "确认来源，逐段说话人另审" : "确认来源与整份材料说话人"}</button><button class="mini-button" data-source-review="rejected" data-id="${source.source_id}">拒绝</button></div>` : ""}</article>`).join("") : '<div class="source-item"><p>还没有资料。粘贴文本、上传文件或输入网页地址后，系统会先生成待审核候选。</p></div>';
  $$('[data-source-review]').forEach(button => button.onclick = () => reviewSource(button.dataset.id, button.dataset.sourceReview, button));
  $$('[data-extract-source]').forEach(button => button.onclick = async () => { busy(button,true); try { await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/sources/${encodeURIComponent(button.dataset.extractSource)}/extract-candidates`,{method:"POST",body:"{}"}); await refreshConversation(); toast("候选已生成，仍需逐条核对原文位置后才能进入模型。"); } catch(error){toast(error.message,true);} finally{busy(button,false);} });
  $$('[data-event-candidate-review]').forEach(button => button.onclick = async () => { busy(button,true); try { await api(`/api/people/${encodeURIComponent(state.person.person_id)}/conversation/sources/${encodeURIComponent(button.dataset.sourceId)}/candidates/${encodeURIComponent(button.dataset.candidateId)}/review`,{method:"POST",body:JSON.stringify({decision:button.dataset.eventCandidateReview})}); await refreshConversation(); toast(button.dataset.eventCandidateReview === "confirmed" ? "候选已与原文逐字核对并形成新人物版本。" : "候选已拒绝。", false); } catch(error){toast(error.message,true);} finally{busy(button,false);} });
  const candidates = state.conversation.optimization_candidates;
  $("#optimization-list").innerHTML = candidates.length ? [...candidates].reverse().map(item => `<article class="optimization-item"><header><strong>优化候选</strong><span class="tag ${escapeHtml(item.status)}">${escapeHtml(humanStatus(item.status))}</span></header><p>创建时版本：${item.active_version_before ? `v${item.active_version_before}` : "尚无版本"}</p><p>内容：${escapeHtml(humanStatus(item.status))} · 表达：${escapeHtml(humanStatus(item.surface_extraction?.status || "not_applied"))}</p>${item.validation_reasons?.length ? `<p>未通过原因：${item.validation_reasons.map(humanStatus).map(escapeHtml).join("、")}</p>` : ""}${item.status === "pending" ? `<div class="item-actions"><button class="mini-button confirm" data-opt-review="confirmed" data-id="${item.candidate_id}">确认内容并运行升级门禁</button><button class="mini-button" data-opt-review="reference_only" data-id="${item.candidate_id}">仅作参考</button><button class="mini-button" data-opt-review="not_same_question" data-id="${item.candidate_id}">问题不相同</button></div>` : ""}${item.status === "accepted_exploratory" && item.surface_extraction?.status === "pending_separate_style_review" ? `<div class="item-actions"><button class="mini-button confirm" data-style-review="confirmed" data-id="${item.candidate_id}">单独审核并更新表达</button><button class="mini-button" data-style-review="rejected" data-id="${item.candidate_id}">不用于表达</button></div>` : ""}</article>`).join("") : '<p class="people-empty">还没有待优化资料。</p>';
  $$('[data-opt-review]').forEach(button => button.onclick = () => reviewOptimization(button.dataset.id, button.dataset.optReview, button));
  $$('[data-style-review]').forEach(button => button.onclick = () => reviewOptimizationStyle(button.dataset.id, button.dataset.styleReview, button));
}

function renderVersions() {
  if (!state.conversation) return;
  const active = state.conversation.active_version;
  $("#versions-list").innerHTML = state.conversation.versions.length ? [...state.conversation.versions].reverse().map(version => `<article class="version-item"><header><strong>版本 v${version.version}</strong><span class="tag ${version.version === active ? "active" : ""}">${version.version === active ? "当前" : "历史"}</span></header><p>${escapeHtml(version.reason)} · ${escapeHtml(version.created_at)}</p><p>内容：${escapeHtml(humanStatus(version.content_update_status))} · 风格：${escapeHtml(humanStatus(version.style_update_status))} · 准确性：${escapeHtml(humanStatus(version.response_accuracy_status))}</p>${version.validation_status === "invalidated_evidence_contract" ? '<p class="warning-box">此版本的证据契约不合格，不能恢复为当前版本。</p>' : (version.version !== active ? `<button class="mini-button" data-rollback="${version.version}">回退到此版本</button>` : "")}</article>`).join("") : '<p class="people-empty">尚未形成版本。确认至少一份可训练的本人逐字回答后会建立探索性 v1。</p>';
  $$('[data-rollback]').forEach(button => button.onclick = () => rollbackVersion(Number(button.dataset.rollback), button));
}

async function loadArchive() {
  try {
    const people = await refreshArchiveCount();
    $("#archive-list").innerHTML = people.length ? people.map(person => `
      <article class="version-item">
        <header><strong>${escapeHtml(person.name)}</strong><span class="tag">已归档</span></header>
        <p>归档时间：${escapeHtml(person.archived_at || "未记录")}</p>
        <p>${person.source_count} 份资料 · ${person.message_count} 条消息 · ${person.version_count} 个模型版本</p>
        <div class="item-actions"><button class="mini-button confirm" data-restore="${escapeHtml(person.person_id)}">恢复人物</button><button class="mini-button" data-destroy="${escapeHtml(person.person_id)}">永久删除</button></div>
      </article>`).join("") : '<p class="people-empty">暂无归档人物。可以在人物卡片的更多菜单中将人物移入归档。</p>';
    $$('[data-restore]').forEach(button => button.onclick = async () => {
      await api(`/api/archived-people/${encodeURIComponent(button.dataset.restore)}/restore`, {method:"POST", body:"{}"});
      $("#archive-dialog").close(); await loadPeople(button.dataset.restore); await refreshArchiveCount();
      toast("人物及其资料、对话和版本已恢复。");
    });
    $$('[data-destroy]').forEach(button => button.onclick = () => {
      state.permanentDeleteTarget = people.find(person => person.person_id === button.dataset.destroy);
      $("#permanent-delete-name").value = ""; $("#permanent-delete-dialog").showModal();
    });
  } catch (error) { toast(error.message, true); }
}

async function archiveSelectedPerson() {
  const target = state.archiveTarget;
  if (!target) return;
  await api(`/api/people/${encodeURIComponent(target.person_id)}`, {method:"DELETE"});
  $("#archive-confirm-dialog").close();
  state.archiveTarget = null; state.person = null; state.conversation = null;
  await loadPeople(); await refreshArchiveCount();
  toast(`“${target.name}”已移入归档。`, false, async () => {
    await api(`/api/archived-people/${encodeURIComponent(target.person_id)}/restore`, {method:"POST", body:"{}"});
    await loadPeople(target.person_id); await refreshArchiveCount();
  });
}

function renderSessionBar() {
  $("#session-title").textContent = state.conversation?.session_title || "新对话";
  $("#session-count").textContent = (state.conversation?.messages?.length || 0) + " 条消息";
}

async function loadSessions() {
  if (!state.person) return;
  const data = await api('/api/people/' + encodeURIComponent(state.person.person_id) + '/conversation/sessions');
  state.sessions = data.sessions;
  $("#sidebar-sessions").hidden = false;
  renderSessionList();
}

function renderSessionList() {
  const query = ($("#session-search")?.value || "").trim().toLowerCase();
  const sessions = (state.sessions || []).filter(s => (s.title || '新对话').toLowerCase().includes(query));
  $("#sidebar-sessions-list").innerHTML = sessions.length ? sessions.map(s => {
    return '<div class="sidebar-session' + (s.active ? ' active' : '') + '" data-session-id="' + escapeHtml(s.session_id) + '">' +
      '<button class="ss-main" data-session-id="' + escapeHtml(s.session_id) + '">' +
        '<span class="ss-title">' + escapeHtml(s.title || '新对话') + '</span>' +
        '<span class="ss-meta">' + s.message_count + ' 条 · ' + escapeHtml(shortTime(s.updated_at)) + '</span>' +
      '</button>' +
      '<span class="ss-actions">' +
        '<button class="ss-btn" data-rename-session="' + escapeHtml(s.session_id) + '" title="重命名">✎</button>' +
        '<button class="ss-btn" data-delete-session="' + escapeHtml(s.session_id) + '" title="删除">✕</button>' +
      '</span>' +
    '</div>';
  }).join("") : '<p class="people-empty">' + (query ? '无匹配会话' : '暂无会话') + '</p>';
  $$(".ss-main").forEach(button => button.onclick = async () => {
    await api('/api/people/' + encodeURIComponent(state.person.person_id) + '/conversation/sessions/' + encodeURIComponent(button.dataset.sessionId) + '/switch', {method:'POST', body:'{}'});
    await refreshConversation();
  });
  $$("[data-rename-session]").forEach(button => button.onclick = async (event) => {
    event.stopPropagation();
    const current = button.closest('.sidebar-session')?.querySelector('.ss-title')?.textContent || '';
    const title = prompt('新标题：', current);
    if (title === null) return;
    await api('/api/people/' + encodeURIComponent(state.person.person_id) + '/conversation/sessions/' + encodeURIComponent(button.dataset.renameSession) + '/rename', {method:'POST', body:JSON.stringify({title})});
    await loadSessions(); await refreshConversation();
  });
  $$("[data-delete-session]").forEach(button => button.onclick = async (event) => {
    event.stopPropagation();
    if (!confirm('删除这个会话？消息将无法恢复。')) return;
    await api('/api/people/' + encodeURIComponent(state.person.person_id) + '/conversation/sessions/' + encodeURIComponent(button.dataset.deleteSession), {method:'DELETE', body:'{}'});
    await loadSessions(); await refreshConversation();
  });
}

function wire() {
  $("#person-search").oninput=renderPeople;
  $("#session-search").oninput=renderSessionList;
  $("#empty-create").onclick=()=>selectAssistant();
  $("#message-form").onsubmit=sendMessage;
  $("#message-form textarea").onkeydown=event=>{ if(event.key==="Enter"&&!event.shiftKey){event.preventDefault();$("#message-form").requestSubmit($("#message-form .send-button"));} };
  $("#open-sources").onclick=()=>$("#sources-dialog").showModal(); $("#composer-add-source").onclick=()=>$("#sources-dialog").showModal();
  $("#open-versions").onclick=()=>$("#versions-dialog").showModal(); $("#open-advanced").onclick=()=>{ $$(".person-menu").forEach(item=>item.hidden=true); $("#advanced-dialog").showModal(); };
  $("#new-conversation").onclick=()=>$("#new-conversation-dialog").showModal();
  $("#sidebar-new-conversation").onclick=()=>startNewConversation(null);
  $("#open-archive").onclick=async()=>{await loadArchive();$("#archive-dialog").showModal();};
  $("#open-model-picker").onclick=async()=>{await loadModelServices();$("#model-services-dialog").showModal();};
  $("#clear-dialogue-model").onclick=clearDialogueModel;
  $("#model-service-form").onsubmit=submitModelService;
  $("#model-preset-select").onchange=event=>applyPreset(event.target.value);
  $("#add-provider").onclick=()=>{ const form=$("#model-service-form"); form.closest("details").open=true; form.reset(); form.elements.service_id.value=""; form.elements.timeout_seconds.value="30"; form.elements.enabled.checked=true; form.elements.structured_output.checked=true; form.elements.api_key.type="password"; form.elements.api_key.placeholder="粘贴 API Key"; $("#key-eye").textContent="👁"; form.querySelector("h3").textContent="添加供应商"; form.elements.display_name.focus(); form.scrollIntoView({behavior:"smooth",block:"center"}); };
  $("#key-eye").onclick=toggleKeyVisibility;
  $("#material-model-role").onchange=event=>setModelRole("material_processing",event.target.value);
  $("#close-drawer").onclick=closeDrawer;
  $$('[data-close]').forEach(button=>button.onclick=()=>$("#"+button.dataset.close).close());
  $("#person-form").onsubmit=submitPerson; $("#paste-source-form").onsubmit=submitTextSource; $("#file-source-form").onsubmit=submitFileSource; $("#url-source-form").onsubmit=submitUrlSource;
  $("#import-backup").onclick=()=>$("#backup-file").click(); $("#backup-file").onchange=event=>event.target.files[0]&&importBackup(event.target.files[0]);
}

async function checkAppVersion() {
  try {
    const response = await fetch("/api/health", {cache:"no-store"});
    const data = await response.json();
    $("#version-banner").hidden = data.app_version === APP_VERSION;
    configureSearchCapability(data.capabilities || {});
  } catch { /* regular API errors remain visible through normal actions */ }
}

async function init() {
  wire();
  wireArchiveDropZone();
  const presetSelect = $("#model-preset-select");
  if (presetSelect) presetSelect.innerHTML = '<option value="">— 手动配置 —</option>' + MODEL_PRESETS.map(p => `<option value="${p.key}">${p.label}</option>`).join("");
  $("#confirm-archive").onclick = () => archiveSelectedPerson().catch(error => toast(error.message, true));
  $("#confirm-new-conversation").onclick = event => startNewConversation(event.currentTarget);
  $("#confirm-permanent-delete").onclick = async () => {
    const target = state.permanentDeleteTarget;
    if (!target) return;
    try {
      await api(`/api/archived-people/${encodeURIComponent(target.person_id)}`, {
        method:"DELETE",
        body:JSON.stringify({expected_name:$("#permanent-delete-name").value}),
      });
      $("#permanent-delete-dialog").close(); state.permanentDeleteTarget = null;
      await loadArchive(); toast("归档人物已永久删除，无法恢复。");
    } catch (error) { toast(error.message, true); }
  };
  $("#refresh-page").onclick = () => location.reload();
  try { await Promise.all([loadModelServices(), loadPeople(), refreshArchiveCount(), checkAppVersion()]); }
  catch(error) { toast(error.message, true); }
  setInterval(checkAppVersion, 60000);
}
init();
