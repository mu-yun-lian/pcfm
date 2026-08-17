# PCFM 写路径全量审计（主线第 1 步）

> 目标：为「SQLite 读真相源」盘点所有写 JSON 的路径，逐条标注同步状态，找出镜像会陈旧的漏网路径。
> 结论：SQLite 是「写后同步的镜像」，JSON 是真相源。同步已覆盖绝大多数服务层写路径，但有 **4 处内部直写路径未同步**（见第四节），这正是之前「切读路径读到旧数据」的根因。

---

## 一、镜像范围（6 张表 ↔ 6 类文件）

| SQLite 表 | JSON 真相文件 | 同步函数 |
|---|---|---|
| person | person.json | person_repo.upsert / mark_archived / delete |
| source | conversation_sources.json | _sync_sources_to_sqlite |
| version | conversation_versions.json | _sync_versions_to_sqlite |
| conversation_state | conversation_state.json | _sync_versions_to_sqlite（同事务） |
| session | conversation_sessions/*.json | _sync_sessions_to_sqlite |
| message | 活动会话 messages | _sync_messages_to_sqlite |

注：history.json / predictions.json / versions.json(旧) / cognitive_*.json / evidence.json / scenarios.json 等**不在 SQLite schema 内**，不属于本次镜像范围。

---
## 二、服务层同步覆盖（已落地）

| 服务方法 | 同步调用 |
|---|---|
| start_new_conversation / create_session / switch_session / rename_session / delete_session | _sync_sessions_to_sqlite |
| send_conversation_message / find_conversation_reality_answer / review_conversation_feedback | _sync_messages_to_sqlite |
| add_conversation_text_source / add_conversation_file_source / add_conversation_url_source | _sync_sources_to_sqlite |
| review_conversation_source | _sync_sources_to_sqlite + _sync_versions_to_sqlite |
| review_conversation_response_candidate / review_conversation_optimization_candidate / review_conversation_style_candidate | _sync_versions_to_sqlite |
| rollback_conversation_version | _sync_versions_to_sqlite |
| set_avatar / update_person / delete_person / restore_person / permanently_delete | person_repo.upsert / mark_archived / delete（支线 1 已补） |

---
## 三、内部写路径清单（conversation/*.py 直写 JSON）

### conversation_sources.json（→ source 表）
- extraction.py:201 / 229 / 402 — 提取候选写入
- optimization.py:443 — 优化流程回写来源
- sources.py:105 — add_text_source 落盘
- sources.py:314 — migrate_evidence_contract（**未同步**，见下）
- sources.py:349 — merge_source_entity_aliases（**未同步**，见下）
- sources.py:403 / 431 / 456 — review_source 与回写

### conversation_versions.json（→ version 表）
- version_builder.py:244 / 273 / 427 / 543 / 589 — 版本创建/回滚/重写
- sources.py:315 — migrate_evidence_contract（**未同步**）

### conversation_state.json（→ conversation_state 表）
- conversation_mvp.py:122 — 状态写
- session_store.py:108 / 124 / 176 / 201 / 208 / 234 — 会话/活动版本切换
- sources.py:316 — migrate_evidence_contract（**未同步**）
- version_builder.py:546 / 591 — 版本创建后写状态
- optimization.py:534 — 优化后写状态

### 会话文件（→ session 表）
- session_store.py:52 — 写单个会话文件

### 消息（→ message 表，存于会话文件内）
- session_store.py:81 _save_active_messages — 消息落盘（message_pipeline.py:71/274/590、reality_lookup.py:69/106、optimization.py:542 调用）

---
## 四、镜像会陈旧的漏网路径（核心结论）

以下 4 处内部直写 JSON 后**没有**调用对应 _sync_*，导致 SQLite 镜像落后：

### 1. migrate_evidence_contract（启动迁移）
- 位置：conversation/sources.py:255-317；调用点 application.py:108-115。
- 写：conversation_sources.json + conversation_versions.json + conversation_state.json。
- 影响：证据契约迁移把版本 validation_status 改为 invalidated_evidence_contract、把 active_version 置空，但 version/state/source 表不更新。
- 复现：这正是之前「切读路径读到旧 validation_status」的那条路径。

### 2. merge_source_entity_aliases（启动 demo 刷新）
- 位置：conversation/sources.py:330-364；调用点 application.py:157-162。
- 写：conversation_sources.json +（_create_version 时）conversation_versions.json + state。
- 影响：source/version/state 表不更新。

### 3. collect_public_sources（公开搜索任务）
- 位置：services/source_service.py:18-90；调用点 webapp.py:385-389（POST /conversation/search 任务）。
- 写：person.json(collection) + conversation_profile.json + 通过 conversation.add_text_source 直写 conversation_sources.json。
- 影响：搜索出的候选资料写入了 JSON，但 source 表不更新（列表聚合 source_counts 会少算 pending 候选）。

### 4. _refresh_demo_sources 的 add_text_source（启动 demo 刷新）
- 位置：application.py:164-174。
- 写：conversation_sources.json（内置 demo 资料）。
- 影响：source 表缺内置 demo 资料。

---
## 五、结论与后续步骤

1. **第 2 步（统一写入口）**：ConversationWorkbench 内部保持纯写 JSON 不动；服务层补「写后同步」即可，不必大改内核。
2. **第 3 步（补齐内部直写同步）**：为第 4 节 4 处漏网路径在调用点补 _sync_*（启动迁移后按人同步、搜索任务后同步 source）。
3. **第 4 步（一致性校验 CLI）**：新增 python -m pcfm.consistency-check，逐人物比对 6 表 ↔ 6 文件。
4. **第 5 步（切读路径）**：仅当前 4 步完成且校验全绿后，才把版本/会话/消息读路径切到 SQLite（JSON 回退 + 哈希校验）。
