# 会话一等实体重构 — 设计文档

日期：2026-08-15
状态：已获用户批准（方案 B + 重命名/删除进第一版 + 顶部会话条/历史抽屉布局）
范围：PCFM conversation-mvp（src/pcfm/conversation_mvp.py、product_service.py、webapp.py、web_static/）

## 1. 背景与根因

用户报告：「人物内部开始新对话后，切换不回旧对话」。

根因不是前端 bug，而是「会话」在数据模型里不是一等实体：

- 每个 person 只有一个 conversation_messages.json 保存活跃对话。
- 点「新对话」时，start_new_conversation 把当前消息写入 conversation_archives/{archive_id}.json，再清空 conversation_messages.json。
- 归档是只进不出：后端没有任何 API 能列出、读取、或切换回归档会话；前端也没有任何入口看到它们。

因此旧对话被塞进一个前端不可见、后端无读取入口的目录，用户自然切不回去。

## 2. 现状数据模型（问题所在）

每个 person 目录下，状态被拆散在两类文件里：

- conversation_state.json：schema_version、active_version、rollback_history、dialogue_model_ref、dialogue_state —— 混装：前四项是人物级，dialogue_state 是会话级。
- conversation_messages.json：活跃会话消息数组（会话级）。
- conversation_archives/{archive_id}.json：archive_id、person_id、archived_at、active_version、messages（会话级，但只进不出、无 dialogue_state、无标题）。

关键矛盾：dialogue_state（话题线程、当前话题、上下文摘要、承诺等）存在人物级文件里，但它是会话级状态。切换会话时它必须随会话走，否则换回来的旧会话会丢失话题上下文。

## 3. 目标

1. 把「会话」扶正为一等实体：一个会话 = 一个文件，含消息 + 会话级对话状态 + 标题/时间/消息数元数据。
2. 切换会话 = 改一个人物级 active_session_id 指针（O(1)），不再搬消息。
3. 支持：列会话、新建、切换（双向）、重命名、删除。
4. 人物级状态（active_version 人物模型版本、dialogue_model_ref 对话模型选择、rollback_history）保持人物级，跨会话共享——换会话只清空上下文，人设/知识/模型版本不变。

## 4. 目标数据模型

每个 person 目录：

- conversation_sessions/{session_id}.json —— 一个会话一个文件
- conversation_state.json —— 只留人物级状态 + active_session_id 指针

### 4.1 会话文件 conversation_sessions/{session_id}.json

字段：schema_version、session_id、person_id、title、created_at、updated_at、message_count、dialogue_state、messages。

- title：首条用户消息截前 24 字；空会话为「新对话」；用户可 rename 覆盖。
- dialogue_state：从人物级 conversation_state.json 迁入，此后只随会话走。
- message_count：len(messages)（冗余字段，供列表页 O(1) 展示）。

### 4.2 人物级状态 conversation_state.json

保留：schema_version、active_version、rollback_history、dialogue_model_ref、dialogue_model_selected_at。
移除：dialogue_state。
新增：active_session_id。

切换会话 = 把 active_session_id 改成目标会话，O(1)。人物级字段跨会话共享，切换不回写。

## 5. 后端 API

| 方法 | 路由 | 行为 |
|---|---|---|
| GET | /api/people/{id}/conversation/sessions | 返回 sessions 列表，updated_at 倒序 |
| POST | /api/people/{id}/conversation/new | 改造现有：新建会话文件，active_session_id 指到新会话 |
| POST | /api/people/{id}/conversation/sessions/{sid}/switch | active_session_id = sid |
| POST | /api/people/{id}/conversation/sessions/{sid}/rename | body {title}，更新会话 title |
| DELETE | /api/people/{id}/conversation/sessions/{sid} | 删会话；活跃会话被删则切到最近一个或新建空会话，始终至少保留一个 |

约束：

- summary() 返回新增 session_id、session_title、active_session_id 字段（供前端渲染会话条与状态）。
- summary()、send_message()、list_people()、get_person() 全部改为读写活跃会话文件（由 active_session_id 定位），而非 conversation_messages.json。
- list_people() 的 message_count / last_message 反映活跃会话（非全部会话），与会话切换后保持一致。
- send_message 写消息后更新会话 updated_at 与 message_count，并写回 dialogue_state。
- active_version / dialogue_model_ref 仍读 conversation_state.json，不随会话变。
- 并发：沿用 ProductService._lock 串行化。

### 5.1 路由接线（webapp.py）

- GET 分支新增 sessions 列表路由。
- POST 分支：conversation/new 改签名；新增 switch、rename。
- DELETE 分支新增 conversation/sessions/{sid}。

## 6. 前端 UI

### 6.1 顶部会话条

- 显示当前会话标题 + 消息数。
- 右侧：「＋ 新对话」按钮（复用 #new-conversation）+「历史会话」按钮（#open-sessions）。

### 6.2 历史会话抽屉（dialog，仿现有 sources-dialog）

- 时间倒序列表：标题、消息数、更新时间；当前项高亮。
- hover 出「重命名 / 删除」。
- 点列表项 = 切换（无确认）；删除 = 二次确认。

### 6.3 状态与渲染

- state.conversation 增加 session_id、session_title 字段（后端 summary() 提供）。
- selectPerson / refreshConversation 加载后渲染会话条。
- 切换/新建/重命名/删除后调用 refreshConversation() + 刷新会话列表。

## 7. 数据迁移（一次性、无损、向后兼容读）

在 ProductService 初始化时执行幂等迁移：

1. 若 conversation_state.json 已含 active_session_id，跳过。
2. 收集候选：conversation_archives/*.json 每个归档一个会话；当前 conversation_messages.json 若非空一个会话。
3. 生成 session_id、title（首条用户消息前 24 字）、created_at、updated_at、message_count。
4. dialogue_state：当前活跃会话从 conversation_state.json 迁入并清除人物级该字段；归档会话重建空 dialogue_state（降级）。
5. active_session_id = 当前活跃会话；若当前活跃为空，指向最新归档，或新建空会话。
6. 迁移完成后保留旧目录/文件不删除，便于回滚。

迁移测试覆盖：空 person、单归档、多归档 + 活跃、重复执行幂等。

## 8. 测试计划（新增 tests/test_conversation_sessions.py）

- 新建会话：首条消息标题自动生成；空会话标题「新对话」。
- 切换会话：A↔B 双向，消息/dialogue_state 随会话隔离。
- 切换后 active_version 不变。
- 重命名：持久化 + 列表反映。
- 删除非活跃：文件消失、活跃不变。
- 删除活跃（还有别的）：自动切到最近一个。
- 删除最后一个：自动新建空会话。
- 迁移：旧目录无损失迁移、幂等。
- 回归：tests.test_conversation_mvp 继续通过。

## 9. 范围外（本设计不包含）

- 会话搜索、置顶/收藏（方案 C）。
- 会话跨人物共享 / 多人物聚合视图。
- 云端同步、多端。
- 归档人物的会话（archived_people 另有一套，不动）。

## 10. 风险与对策

- 迁移丢消息：幂等 + 保留旧文件不删 + 测试覆盖。
- dialogue_state 恢复不精确：旧归档接受降级为重建。
- 并发竞态：沿用 ProductService._lock。
- 前端会话条与人物库混淆：顶部横条 + 抽屉，不嵌进左侧人物面板。
