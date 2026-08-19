# 主线第 5 步 SQLite 读路径 · 完成报告

> 依据 `docs/主线第5步SQLite读路径流程设计方案.md` 实施第 5.1(影子读取) 与第 5.2(灰度开关)。
> 结论：影子 + 灰度已落地；默认影子模式(PCFM_SQLITE_READ_PRIMARY=0)，全量回归 251 OK(skipped=12)。

---

## 一、交付内容

| 步骤 | 内容 | 结果 |
|---|---|---|
| 5.1 影子读取 | 恢复 list_full_by_person + 注入 repo + ReadPathMixin + 读路径替换 | ✅ |
| 5.2 灰度开关 | PCFM_SQLITE_READ_PRIMARY + SQLite 优先 + 回退 + 自愈 | ✅ |
| 5.3 默认开启 | 灰度期验收通过后默认改 1 | ✅ |
| 5.4 message 读路径 | 单独立项 | ⏸ 不纳入本轮 |

---

## 二、关键实现

### 影子读取（默认）
- `version_repo.list_full_by_person`：从 `data` 列读完整版本（剥离同步注入的 `person_id`）。
- `conversation/read_path.py` 新增 `ReadPathMixin`：
  - `_read_versions` / `_read_state` / `_read_sessions` 三个混合读函数；
  - 影子模式：读 JSON + 对比 SQLite，不一致仅告警，返回值始终 JSON；
  - 影子模式且镜像库未打开（未发生同步）时跳过对比，避免无谓打开连接（零副作用）。
- `ConversationWorkbench` 混入 `ReadPathMixin`；`summary` 与 `list_sessions` 的读路径替换为混合读。
- `application.py` 注入 `_session_repo`/`_state_repo`/`_message_repo` + `_sync_callback` 自愈回调。

### 灰度开关（PCFM_SQLITE_READ_PRIMARY=1）
- `version` 表有完整 `data` 列 → 一致时返回 SQLite 数据；
- `state`/`session` 表只存关键字段（缺 rollback_history / message_count / dialogue_state / messages）→ **仍返回 JSON**，但检测到不一致时触发自愈；
- 自愈走服务层 `_sync_*`，失败只告警、不阻断读。

### 发现并修复的两个实现问题
1. **session 自愈递归**：`_sync_sessions_to_sqlite` 原先经 `list_sessions`(现走混合读) 读数据，混合读在灰度空库时又触发自愈 → 无限递归。改为用纯 JSON 读 `_list_sessions` 构建镜像。
2. **读路径副作用致测试文件锁**：混合读让 `get_person`/`summary` 打开 db 连接，未 close 的测试服务泄漏 `pcfm.db`。修复：影子模式库未打开时跳过对比 + 补 `test_cognitive_workbench` 的 `restored.close()`。

---

## 三、验证结果

| 项 | 结果 |
|---|---|
| 后端全量回归 | **251 OK（skipped=12）** |
| 新增单测 | test_read_path.py 6/6（影子不一致返回JSON+告警 / 灰度一致返回SQLite / 灰度空回退 / 灰度不一致回退+自愈 / state+session返回JSON / message仍JSON） |
| consistency-check | 0 不一致（test_sqlite_sync 覆盖） |
| Playwright E2E | 1/1 |
| 前端 build / vitest | 无改动，仍 exit 0 / 15/15 |

---

## 四、5.3 默认开启验收结果

- 全量回归：影子 3 轮 + 默认开启 2 轮，均 251 OK(skipped=12)。
- consistency-check：连续 10/10 次 0 不一致（含 version.data 全量对比）。
- 人工不一致演练：制造不一致 → consistency-check 捕获 1 处 → 影子/灰度告警 + JSON 回退 → 自愈 → 恢复 0。
- 性能：100 人列表 829.9ms(≤1.0s)。
- 默认值已改 `1`；设 `PCFM_SQLITE_READ_PRIMARY=0` 可立即回退影子模式。

## 五、剩余（按设计）

- **第 5.4 步 message 读路径**：需先让 `_sync_messages_to_sqlite` 覆盖所有会话并给 message 表加完整 `data` 列，单独立项。

---

## 六、回滚方式

- 灰度误开：`PCFM_SQLITE_READ_PRIMARY=0`（默认）即回退影子模式；
- 彻底回退：删除 `read_path.py` 并把 `summary`/`list_sessions` 恢复为 `_list`/`_list_sessions`；
- 任意阶段可回退上一个 commit/tag。
