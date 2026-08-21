# Changelog

All notable changes to PCFM are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.11.0] - 2026-08-21

### Added
- 5 个初始人物模型全部就绪：Steve Jobs、Donald Trump、Warren Buffett、王阳明、鲁迅
- 领域画像模块 `domain_profile.py` 及测试
- 叙述模块 `profile_narration.py` 及测试
- 前端 UI 改进：AI 助手不绑定人物对话模型、资料处理模型引导、头像静态资源回退

### Fixed
- 对话检索召回：英文/口语提问因领域解析缺失导致相关事件检不到 → 领域缺失时按 trigger 相似度 + 内容词双门槛检索
- 统一推导门禁契约：prompt 与门禁的倾向 ID 空间对齐，推导类问题不再恒失败
- Windows 归档恢复 PermissionError
- 中文领域画像中英混杂
- 中文叙述倾向类型误拒
- `atomic_write_json` 并发竞态：唯一临时文件名 + 失败清理
- `test_jobs` 轮询在高负载下偶发失败：死线 15s 替代 50x50ms

### Security
- 移除历史会话记录（`talk_histroy/`，曾含 API key）并加入 `.gitignore`

## [0.10.0] - 2026-08-19

### Added
- SQLite 优先读路径（影子对比 + 灰度开关 + 回退自愈）
- 一致性校验 CLI：version 表 data 列全量对比
- 写路径审计 + 内部直写同步补齐 + schema 迁移
- T1-T6 SQLite 镜像一致性验收用例（会话 CRUD 同步/删会话清消息/消息+状态同步/迁移回滚清空六表）
- 非 Windows 平台 `secret_storage=environment_only`，前端禁用 API Key 并提示用环境变量

### Fixed
- 现实回答同步不可达
- 会话 CRUD 同步
- 迁移回滚补全
- 同步事务化
- 创建人物搜索异步化
- E2E 现实对照竞态

## [0.9.0] - 2026-08-17

### Added
- `list_people` 性能优化：基线报告短路 + light summary，100 人列表 4.26s → 1.x s
- 性能基准脚本 `benchmarks/perf_check.py`

### Changed
- 倾向原子增加 `tendency_type` 8 类封闭词表（LLM 候选 + 代码校验，流入模型工件）

## [0.8.0] - 2026-08-15

### Added
- 人物对话 MVP v0.3：浏览器验收、对话抽屉、版本对比
- 受约束表达渲染层 v1（Steve Jobs 风格档案）
- 材料分块提取
- DeepSeek 模型服务集成

### Fixed
- 禁用 DeepSeek 思考模式，解决结构化输出 content 为空导致倾向原子无法提取

## [0.1.0] - 2026-08-14

### Added
- 初始仓库：源码、测试、文档、报告、对话记录
- PCFM 人物认知模型工作台 v0.2
- 证据约束的窄域人物推演
- Logistic 行为基线模型
- 研究候选模块：Decision-Context-Rationale、Person-Issue Relational Core、Joint Person Core、Reality Bridge、Support-set HyperNetwork、Anisotropic Empirical-Bayes Adapter、Person-choice Benchmark、Prospective Single-Person Pilot

[0.11.0]: https://github.com/mu-yun-lian/pcfm/releases/tag/v0.11.0
[0.10.0]: https://github.com/mu-yun-lian/pcfm/releases/tag/v0.10.0
[0.9.0]: https://github.com/mu-yun-lian/pcfm/releases/tag/v0.9.0
[0.8.0]: https://github.com/mu-yun-lian/pcfm/releases/tag/v0.8.0
[0.1.0]: https://github.com/mu-yun-lian/pcfm/releases/tag/v0.1.0
