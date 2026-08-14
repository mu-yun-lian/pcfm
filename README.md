# PCFM 人物对话系统

## 对话式现实人物模拟 MVP / 会话条件响应版 0.10

当前首页是人物库与多轮对话工作区。人物之间的资料、V5 会话条件响应模型、纯表达风格、消息和优化候选分别保存。V5 将 V4 冻结为证据子模型：原始问答可作为直接历史证据；无结构长文本的固定宽度分段只用于原文导航，不能自动成为人物回应事件。事件类型、条件、知识片段和“优先 A、接受 B”只能先由材料处理模型提出带逐字位置的候选，再经确认进入事件原子。事件原子保留触发窗口、回应片段、时间、对象、场合、已知信息和缺失字段；多个独立来源谱系的伴随原子才形成可追溯的公开取向、反证、时间范围和角色范围。这里模拟的是人物可观察的对外表现，不宣称恢复真实内心。

对话输入不再只处理最新一句。系统从原始消息重建话题线程、当前话题、被引用消息、人物此前生成的对话承诺、关系、场合和时间范围，把当前消息作为状态增量。历史问题相同或近乎相同时返回直接公开回答；复合问题可展示多个相关历史事件，但不伪装成新立场；新问题可由所选模型提出受限的情境—价值影响映射，代码再用人物的重复公开取向、角色、领域、时间和反证门禁决定方向。模型不能提交人物立场字段，不能把生成对话变成人物证据。内容计划冻结后，模型才可补充通用知识并组织自然回答，最后进入独立的纯表面风格层。

0.10 的网页调用顺序是“确定性证据优先，语义模型补足”：历史直接依据和已充分解析的取向问题不调用语义模型；没有形成可用人物路线、存在歧义或需要外部知识时才调用。页面会在每条回答的“查看依据”中显示本次规划、内容补充和验证调用次数。保存模型配置不会自动产生模型调用；读取模型列表也不等于可用。每个具体模型必须通过一次真实结构化调用验证后才能选择。旧对话可用“新对话”在本机归档后清空上下文，人物资料、人物模型和所选模型不变。

Windows 用户可双击 `启动PCFM人物对话系统.bat`。启动后会自动打开浏览器，也可以手动访问：

```text
http://127.0.0.1:8765
```

可用主流程：创建人物（系统搜索或自行提供资料）→ 系统从原始材料整理事件候选 → 审核原文真实性、位置和说话人 → 直接多轮交流 → 按需查找相近的现实回答 → 用户选择一条现实事件进入优化候选 → 通过角色隔离和留出非退化门禁后生成探索性新版本 → 查看或回退版本 → 归档、恢复或名称确认后永久删除。多人材料不会整份自动归因，必须逐条确认说话人与逐字位置。支持粘贴文本、TXT、Markdown、HTML、PDF、SRT/VTT、JSON、CSV 和网页地址；按人物姓名的公开资料搜索可通过 Bing RSS 后端启用，结果只能成为待核验候选，不能直接训练。OCR 与音视频转写尚未配置。

证据硬门槛：只有可追溯的逐字稿，或能对应原文位置的准确翻译，才可能成为人物回答训练标签。编辑摘要、模型生成内容、搜索摘要和第三方转述只能保留为参考或待核验候选。现有 Steve Jobs 中文编辑摘要已按此规则降级，依赖它的旧 v1 已失效，因此当前 Steve Jobs 显示“证据不足”是正确行为。

当前 V5 模拟内核可运行但尚未验证现实人物准确性。页面中的数值是证据支持度，不是准确率或人物选择概率。准确性验收必须使用按时间和来源谱系隔离的完整多轮对话留出集，并与只看当前消息、错误人物、打乱历史、纯检索和通用模型基线比较；在这类数据建立前结果固定为 `not_assessed_full_conversation_holdout_required`。软件回归通过、回答更流畅或大模型参与都不会改变这一证据状态。

以下 v0.2、表达层和研究候选保留为历史与高级诊断资料，不再是首页主流程。

## 受约束表达渲染层 v1

网页已接入独立的表达渲染闭环：PCFM 上游先冻结主张、理由、允许使用的记忆和不确定性，表达层再生成轻度、标准、强辨识度三档候选。表达层不能读取原始用户问题、人物认知库、事实库或完整人物 Skill，也不能改变立场、事实、否定、模态词、数字、日期、引语和置信度；全部候选失败时返回中性版本。

首个内部包位于 `src/pcfm/expression_profiles/steve_jobs_v1/`，它不是 `SKILL.md`，不能独立回答问题。实现、审计、测试范围和诚实完成状态见 `EXPRESSION_RENDERER_V1_REPORT.md`。风格辨识度和人物响应预测准确性仍是 `not_assessed`。

# PCFM 人物认知模型工作台 MVP v0.2

## 本地网页产品：证据约束的窄域人物推演

MVP v0.2 保留原有网页框架、人物管理、导入导出、本地存储、版本、结果回填、指标、
适用域检测和启动方式，同时把原 Logistic 核心明确降级为“行为基线模型”。它不再被单独
称为人物思维模型，也不会为真实人物自动注入合成人群来伪造可用性。

Windows 用户可双击 `启动PCFM人物认知模型工作台.bat`（旧入口仍兼容）。应用默认自动
打开浏览器，也可手动访问：

```text
http://127.0.0.1:8765
```

主流程限定为“一个人物、一个领域、一类决策”：审核带原文/摘要、来源、日期、情境、选择、
同期理由、证据角色、领域、冲突、可信度和原始位置的材料；生成逐项绑定证据的认知模型卡；
把自然语言新情境先转换为可编辑结构，再经用户确认后运行同一个确定性评分内核；展示选项
概率、证据化驱动因素、行为基线对照、翻转条件、未知项和适用范围；最后回填外部真实选择与
理由并生成新版本。大模型候选在用户确认前始终是 `pending`，语言解释不能改变已计算结果。

首次启动还保留“林澄（合成数据）”供行为基线工程测试；Josh Hawley 的内置公开材料案例只
用于 Section 230 平台责任这一窄域纵向流程。初始材料截至 2022-12-07，2023-12-13 的 AI
相关行动只在预测完成后回填。该案例证明产品闭环可运行，不证明恢复了人物私有思想，也不
证明人物特异性认知建模有效。

普通历史数据不会被 Decision Evidence v1 阻塞。该证据合同仍保留为网页中的高级导入入口，
经签名验证后单独保存，不会自动获得训练资格。

当前认知评分概率尚未统计校准，只是证据覆盖、模型项权重与未知项收缩后的探索性分数。
只有在留出或未来情境中，正确人物持续优于错误人物、内容检索、行为基线和打乱理由基线，
才可称为“人物特异性认知建模获得初步证据”；MVP v0.2 的状态固定为 `not_assessed`。

实现、案例来源、自动化与浏览器验收见 `COGNITIVE_WORKBENCH_V02_REPORT.md`。下方 v0.10
研究记录属于历史背景，不代表这些候选已经进入 v0.2 产品核心。

## Decision-Context-Rationale Evidence Contract v1

当前主线已经从“继续更换人物编码器”转为建立更窄的真实证据入口。新合同要求
每条记录同时绑定：人物选择、选择前已经公开的完整题干和选项、选择来源，以及
同一人物在前 24 小时至后 168 小时内的同期理由，或同一事件的可验证行动后果。

工件会离线重算来源文本 SHA-256、引用片段、人物归属、时间方向、角色预分配、
跨角色重放、近重复、汇总计数、工件身份和 HMAC 签名。`sealed_confirmation`
必须在选择发生前分配，并绑定外部登记摘要；但本地签名仍不能证明远程时间戳、
材料完备性或理由真诚性，这些需要外部档案或登记服务。

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path

python -m pcfm decision-evidence-build-v1 `
  --input <decision-evidence-input.json> `
  --keys <verification-keys.json> `
  --verifier-id <evidence-verifier> `
  --created-at <ISO-8601> `
  --output <decision-evidence-bundle.json>

python -m pcfm decision-evidence-verify-v1 `
  --bundle <decision-evidence-bundle.json> `
  --keys <verification-keys.json>
```

该阶段状态为 `implemented_unintegrated`：验证、保存、重载和 CLI 闭环已经实现，
但尚未采集真实证据包，也不会自动进入 `fit`、`update`、动态状态、机制或语义
模块。`training_authorized` 固定为 `false`，语义声明列表固定为空。详细合同见
`DECISION_EVIDENCE_V1.md`，阶段结果见 `DECISION_EVIDENCE_V1_REPORT.md`。

## Person-Issue Relational Core v1

冻结六人新时间段诊断结果为
`person_issue_relational_candidate_not_supported`。固定哈希与训练期 SVD 构造的
非 LLM 人物—议题剖面仅在 `2/6` 人上优于同党动态基线，六人等权 NLL 提升为
`-0.000784`，聚类自助法下界为 `-0.019430`；历史结果在人物和届次内打乱后，
打乱剖面反而平均优于正确剖面 `0.155233 NLL`，并有 56 次适用域拒答。1,800
条最终概率已从原始数据和重载工件完整重算一致。候选未进入生产核心，语义模块
继续阻塞。详见 `PERSON_ISSUE_RELATIONAL_CORE_V1_REPORT.md`。

## Joint Person Core Candidate v1

冻结六人历史诊断结果为 `joint_core_candidate_not_supported`。完整联合模型在
`0/6` 人上优于最佳非人物对照，六人等权 NLL 提升为 `-0.037527`，聚类自助法
下界为 `-0.044140`，并出现 170 次适用域拒答。共享参数、按各人既往结果在线
更新的动态群体对照反而在六个人上全部更好。因此当前证据只支持近期动态具有
预测价值，不支持现有表示能够识别稳定人物差异。候选保留在 `work/`，未修改
生产核心；语义模块继续阻塞。详见 `JOINT_PERSON_CORE_CANDIDATE_V1_REPORT.md`。

## Reality Bridge v1

当前现实桥接状态为 `core_entry_not_established`。六名按冻结哈希规则、
两党平衡选出的现任参议员中，`0/6` 通过现有稳定人物模型门禁；正确人物
模型在第 117 届验证期的 NLL 全部差于群体模型，并全部被判定时间不稳定。
动态状态因此被基础资格门禁阻断，机制比较也因个体化与时间稳定性未验证而
拒绝。完整结果见 `REALITY_BRIDGE_V1_REPORT.md`。

该结果是历史反事实重放，不是前瞻确认，也没有检验进入合格状态后的动态或
机制算法现实效用。它目前不支持恢复信念、价值、目标、记忆、社会或自我等
预留语义模块。下一核心任务是重新定义能够联合检验稳定人物差异、可观测环境
与动态状态的现实入口，同时保留现有数据隔离和拒绝门禁。

## Tyler historical corpus v1

历史语料汇编层只接收本地官方快照及其 `tyler-source-v1` 工件，逐个重放验证后执行跨页去重和固定时间角色分离：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pcfm tyler-corpus-v1 `
  --manifest artifacts/tyler_corpus_v1/input-manifest.json `
  --created-at 2026-08-01T01:00:00+08:00 `
  --output artifacts/tyler_corpus_v1/tyler-corpus-2026-07-31.json
```

当前真实工件只有一个 RSS 来源、13 篇帖子，全部属于 `retrospective_diagnostic`，标注前训练资格为 0。该模块尚未与多来源双人标注包整合，因此状态为 `implemented_unintegrated`，不能进入模型训练。

## Person-choice Benchmark v1

当前开发范围已收缩为一个可检验目标：在固定的五维二选一任务中，只用测试人物的支持集，预测该人物在未见场景与较晚时间场景中的选择概率。它不是“思想复制”或“人格恢复”的完成版。

运行完整轻量基准：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pcfm benchmark-v1 --seed 7301
```

若只检查结构化 Logistic 基线，可加入 `--skip-neural`。默认烟雾配置生成 3,408 条合成记录，仅依赖 NumPy、使用 CPU，小型神经基线少于 2,000 个可训练标量。

固定种子 `7301` 的主要结果见 `PERSON_CHOICE_BENCHMARK_V1_REPORT.md`。64 条支持样本时，个体 MAP Logistic 在场景测试上的 NLL 为 `0.466396`，优于群体 Logistic 的 `0.628307` 和错配人物 Logistic 的 `0.661764`；人物嵌入 MLP 为 `0.678970`，没有超过简单个体模型。因此 v1 只证明实现能在受控合成任务上测出人物特异性，不证明神经表示具有心理语义，也不证明可迁移到真人。

## Support-set HyperNetwork v1

已实现 36 参数的 rank-3 support-set HyperNetwork，并用固定五种子审计：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pcfm hypernetwork-v1
```

最终状态为 `rejected_candidate`：`0/5` 种子通过预注册门槛。它在五个种子上均超过人物嵌入 MLP，但均未超过个体 MAP Logistic，时间测试也全部超过允许劣化线。因此该候选保留为可复现实验，不进入人物模型主路径。详见 `HYPERNETWORK_V1_REPORT.md`。

## Anisotropic Empirical-Bayes Adapter v1

已实现噪声校正的人物间协方差估计与各向异性 MAP 适配：

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pcfm empirical-bayes-v1
```

本阶段使用此前未查看的 `8101–8105` 作为固定审计种子。结果为 `3/5` 通过，五种子平均 NLL 改善 `0.017130`，但未达到至少 `4/5` 的稳定性门，因此状态仍为 `rejected_candidate`。它明显比 HyperNetwork 更接近目标，但不替代当前各向同性个体 MAP。详见 `EMPIRICAL_BAYES_V1_REPORT.md`。

## Prospective Single-Person Pilot v1

下一阶段不再添加认知标签，而是运行一个人、一个领域的前瞻盲测。工具会在真人答案产生前，把 100 道以上题目的完整题干和四套概率一起冻结：

- PCFM 个体模型；
- 群体模型；
- 该人物历史常数概率；
- “LLM＋人物资料”基线。

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path

python -m pcfm pilot-create `
  --person-id <person-id> `
  --scenarios <scenarios.json> `
  --forecasts <forecasts.json> `
  --keys <verification-keys.json> `
  --verifier-id <study-author> `
  --created-at <ISO-8601> `
  --collection-end <ISO-8601> `
  --output <pilot-plan.json>

python -m pcfm pilot-register `
  --plan <pilot-plan.json> `
  --keys <verification-keys.json> `
  --registry-verifier-id <independent-registry> `
  --registered-at <ISO-8601> `
  --output <registry-receipt.json>

python -m pcfm pilot-score `
  --plan <pilot-plan.json> `
  --receipt <registry-receipt.json> `
  --outcomes <future-human-outcomes.jsonl> `
  --keys <verification-keys.json> `
  --output <pilot-report.json>
```

四个 forecast 的 `model_reference` 必须是对应模型、提示词、人物资料或历史快照的 SHA-256 摘要。所有题目必须包含完整 `context.question_text`。评分采用 NLL、校准误差，以及按真实回答时间排序的 Newey–West 配对区间。当前只完成了工具和合成行为测试，尚未采集任何真人结果，不能据此声称人物模拟有效。详细合同见 `PROSPECTIVE_PILOT_V1.md`。

PCFM 是一个可证伪的人物特定决策模型研究原型。它从经验证的结构化行为记录中估计人物参数，在独立数据上验证预测增益，并在证据不足、超出适用域或时间状态不可复核时拒绝输出。

它目前不是完整的思想、信念或意识模型。它提供的是后续人物认知模型所需的统计测量、证据谱系和防自欺闭环。

v0.9.1 修复复合阶段审计发现的功能缺口：主动候选和更新后验证集不得重放机制证据；稳定、动态、机制和主动规划共用同一 Logistic-normal 部署核；一次公共操作只重算一次机制；复合实验结果现在可以更新基础模型，并明确使旧复合工件和机制报告失效。

## 数据闭环

稳定人物模型使用三份互不重叠的签名账本：

```text
训练账本
  └─ 估计人群先验和人物参数

适用性校准账本
  └─ 建立输入支持域和 OOD 阈值

最终验证账本
  └─ 检验人物增益、校准、机制错设和时间稳定性
```

三份账本的事件 ID 和“人物 ID + 场景 ID”必须两两不重叠。bundle v7 还把忽略非设计时间元数据的场景内容指纹及主动实验计划 ID 写入模型身份。动态监测或主动实验不能靠更换事件 ID、场景 ID 或 `prediction_at` 重放已用于建模的设计。

## 稳定模型门禁

- 全局边界：收缩协方差下的平方马氏距离。
- 局部边界：校准点到参考点的第 5 近邻马氏距离。
- 时间变化：比较早期和后期的预测残差 score，并对候选切分点和特征维度校正。
- 验证覆盖与适用性覆盖彼此独立，所有覆盖都会记入 `Prediction.gate_overrides` 并撤销概率区间。

硬拒绝包括 `feature_distribution_shift`、`local_support_gap`、`prediction_time_required`、`prediction_precedes_reference_data` 和 `stale_model`。

未见过的领域、选项或上下文会触发跨域外推警告。稳定模型仍可在明确覆盖后给出诊断点预测，但模型形式不确定性记为未量化。

## 主动实验规划

主动规划器只接受不含人物答案的 `Scenario` 候选池。令当前参数近似为 \(\theta\sim\mathcal N(\mu,\Sigma)\)，候选场景的 logit 为 \(z=x^\top\theta\)，则：

\[
z\sim\mathcal N(x^\top\mu,x^\top\Sigma x)
\]

系统使用确定性的 Gauss–Hermite 积分计算真实的后验预测互信息：

\[
I(y;\theta\mid x)
=H_b\!\left(\mathbb E[\sigma(z)]\right)
-\mathbb E\!\left[H_b(\sigma(z))\right]
\]

每选一题，规划器用 \(\operatorname{Cov}(\theta,y)\) 的矩匹配更新预期协方差，再对剩余题目重算互信息。相同候选池与模型始终产生相同结果，输入顺序不影响计划。单题计划标记为 `adaptive_single_step`；多题计划明确标记为 `outcome_blind_batch_approximation`。要获得真正的逐题自适应序列，必须每次只规划一题，吸收真实结果后再规划下一题。

硬约束：

- 基础模型必须已通过独立验证；
- 候选必须是 `Scenario`，不能传入带答案的 `Observation`；
- 场景 ID 和场景设计均不得重复；
- 不能复用模型谱系中的旧试验；
- 所有候选必须在已验证特征域、领域、选项和上下文内；
- 参数协方差必须正定；
- 预期信息增益必须达到配置下限。

`pcfm-active-experiment-v3` 计划额外绑定实际用于选题的预测模型身份、版本和参数维度。基础模型与复合模型均可驱动同一规划内核；计划验证会从原候选池和对应预测组件重新运行选题，而不是只检查签名。

执行结果必须是签名账本，题目集合和每道题的完整内容必须与计划一致，且采集时间晚于计划创建时间。少题、加题、换题、修改特征和倒签时间都会被拒绝。`apply-experiment` 会按采集顺序吸收结果、把计划 ID 绑定到每个后继模型，并要求重新验证所用封存账本的每条记录均晚于所有实验结果。

## 竞争性机制比较

机制模块不从自然语言总结“这个人相信什么”，而是比较预先声明的数值结构项。目前支持：

- 截距；
- 单特征线性项；
- 两特征交互项；
- 单特征绝对值项；
- 单特征二次项。

每个候选假设都是这些项的有限集合。它们只表示可计算的预测结构，不带心理语义。计划必须在数据出现前固定候选、配置和三段严格有序的证据窗口：

```text
发现账本
  └─ 拟合每个候选的残差参数

选择账本
  └─ 只按独立 NLL 选择一个候选

确认账本
  └─ 不再选模，只检验已选候选相对基础模型的 NLL 增益
```

确认门禁要求最小 NLL 增益不低于 `0.01`、95% 区间下界大于零且确认集 ECE 不高于 `0.15`；这些硬边界只能配置得更严格。确认事件时间唯一时使用时间有序 HAC 标准误；同一时间批次内没有可识别顺序时使用与事件 ID 无关的 IID 标准误。确认结果无论通过与否都固定为 `predictive_structure_only`，并同时标记 `causal_interpretation_not_identified` 和 `temporal_vs_structural_not_identified`。

`pcfm-mechanism-plan-v2` 与 `pcfm-mechanism-report-v2` 均签名。报告额外绑定三段证据共同覆盖的领域、选项、上下文和所选机制项数值范围；跨域、局部项超界或报告超过 180 天都会拒绝。用于预测时，系统必须重新读取三份原始签名账本并重跑拟合、选模和确认；只修改报告、重算哈希或持密钥重签报告不能改变原证据推导。

预测区间包含基础参数与已选修正项参数的条件高斯近似，但不包含“候选集合是否完整”和“选择了哪个候选”的模型选择不确定性，因此结果明确标记为 `candidate_selection_not_quantified`。

## 已验证复合预测视图

`pcfm-composite-model-v1` 不复制一份不可追踪的“人物总结”，而是绑定基础模型 ID、机制计划/报告 ID、三段原始证据摘要、所选候选、创建时间、有效期和验证者。每次预测仍会从三份签名账本重跑机制比较，并要求结果与复合工件逐字段一致。

联合预测的均值由基础参数和机制修正参数拼接，协方差暂采用分块对角近似。正式预测和复合主动实验规划使用同一 Logistic-normal 概率近似；主动计划额外绑定复合模型 ID。旧的基础模型更新器仍会拒绝这种计划；复合专用更新入口会核验计划、结果和更晚的独立验证账本，更新基础模型并返回 `base_updated_composite_invalidated`。旧复合工件与机制报告随后不可继续使用，必须针对新基础模型重新预注册、比较和确认机制。

当前动态状态 v2 是相对基础模型残差推断的，直接与机制修正相加会重复解释残差。因此复合创建接口会拒绝现有动态工件；后续需要实现相对复合预测核重新推断的动态状态版本。

## 动态残差状态

固定稳定人物参数 \(\theta\) 后，动态层为：

```text
y_t ~ Bernoulli(sigmoid(x_t · θ + z_t))
z_t = ρ(Δt) z_(t-1) + ε_t
ρ(Δt) = exp(-log(2) Δt / half_life)
```

`z_t` 是一维、无心理标签的 logit 残差偏移。它不等同于情绪、疲劳、信念变化或任何已识别的心理原因。

每个动态概率都在读取当前事件结果前产生。系统保存静态概率、逐事件动态概率和真实选择，并由这些原始量重新计算：

- 静态与动态 prequential NLL；
- 平均 NLL 提升及描述性的 HAC 区间；
- 静态预测与动态预测之间的逐事件似然比；
- 最终及历史最大 log e-value；
- 连续状态检出长度和最终状态证据。

`prequential_residual_signal` 必须同时满足：

- 达到预注册的最小样本数；
- 平均 NLL 提升达到阈值；
- 最大 e-value 达到 `1 / sequential_alpha` 的序贯证据门槛；
- 状态效应和连续检出长度达到阈值。

HAC 区间保留为描述性诊断，不再承担可被反复查看数据破坏的序贯决策门禁。

状态名称只有 `not_assessed`、`no_prequential_residual_signal` 和 `prequential_residual_signal`。这些名称有意不使用“认知状态已验证”或“人物改变已证实”。

## 签名预注册与可复核产物

动态推断必须先创建签名计划。计划固定基础模型、人物、注册时间、监测窗口、预期事件数、完整配置和签名验证者。

注册时间必须早于监测开始。推断时，事件数量、首尾时间、模型、人物和配置必须与计划完全一致，因此不能在看到结果后截取有利窗口或调整阈值。

动态报告 v2 包含真实选择和全部派生量，并由验证机构签名。加载报告会检查内部重算和签名；用于预测时还必须提供原始签名监测账本，系统会重新执行完整推断，并要求结果与报告逐字段一致。单独重算自哈希，甚至拿验证密钥给篡改后的报告重新签名，都不能绕过原始证据重算。

当前签名机制为 HMAC-SHA256，适合单一受信验证机构的原型。它不能防止控制共享密钥且同时能篡改原始账本的恶意机构；生产系统应替换为分权的非对称签名、透明日志或外部时间戳服务。

## 时间与跨域限制

- 第一条动态先验从基础模型最后参考时间传播到第一条监测事件，不再假设间隔为零。
- 监测时间戳必须严格递增，且全部晚于基础模型证据。
- 状态预测必须晚于最后监测事件，并处于预注册的最大传播间隔内。
- 动态状态默认不能迁移到未验证的领域、选项或上下文。
- `--state-override` 只允许诊断使用，会记录原因并撤销概率区间。

## 命令行

拟合稳定模型：

```powershell
python -m pcfm fit `
  --input <training-ledger.jsonl> `
  --applicability-ledger <applicability-ledger.jsonl> `
  --validation-ledger <validation-ledger.jsonl> `
  --verification-keys <verification-keys.json> `
  --person-id <person-id> `
  --feature-names reward,risk,control `
  --output <person-model.json>
```

生成主动实验计划：

```powershell
python -m pcfm plan-experiment `
  --model <person-model.json> `
  --candidates <candidate-scenarios.json> `
  --verification-keys <verification-keys.json> `
  --verifier-id <verifier-id> `
  --created-at 2026-09-02T00:00:00Z `
  --selection-count 12 `
  --output <active-experiment-plan.json>
```

候选文件是只包含场景、不包含选择结果的 JSON 数组。执行试验并生成签名结果账本后，可同时重算计划和核验结果：

```powershell
python -m pcfm verify-experiment `
  --model <person-model.json> `
  --candidates <candidate-scenarios.json> `
  --plan <active-experiment-plan.json> `
  --input <active-experiment-results.jsonl> `
  --verification-keys <verification-keys.json>
```

核验后吸收结果并用更晚的封存数据重新验证：

```powershell
python -m pcfm apply-experiment `
  --model <person-model.json> `
  --ledger <training-ledger.jsonl> `
  --applicability-ledger <applicability-ledger.jsonl> `
  --future-validation-ledger <future-validation-ledger.jsonl> `
  --candidates <candidate-scenarios.json> `
  --plan <active-experiment-plan.json> `
  --input <active-experiment-results.jsonl> `
  --verification-keys <verification-keys.json> `
  --output <updated-person-model.json> `
  --output-ledger <updated-training-ledger.jsonl>
```

逐题自适应运行时，将 `plan-experiment --selection-count` 设为 `1`，每次完成 `apply-experiment` 后从剩余候选池重新规划。主动实验最小信息增益有不可关闭的 `1e-12` 硬下限。

机制假设文件是 JSON 数组，每个假设由显式项组成：

```json
[
  {
    "hypothesis_id": "reward-control-interaction",
    "terms": [
      {
        "term_id": "reward-x-control",
        "kind": "interaction",
        "feature_names": ["reward_gain", "control"]
      }
    ]
  }
]
```

使用 `plan-mechanisms` 在采集前固定发现、选择和确认三段窗口及事件数；完整参数可通过 `python -m pcfm plan-mechanisms --help` 查看。采集完成后比较：

```powershell
python -m pcfm compare-mechanisms `
  --model <person-model.json> `
  --plan <mechanism-plan.json> `
  --discovery-ledger <discovery.jsonl> `
  --selection-ledger <selection.jsonl> `
  --confirmation-ledger <confirmation.jsonl> `
  --verification-keys <verification-keys.json> `
  --output <mechanism-report.json>
```

只有报告状态为 `supported_candidate` 才允许调用 `predict-mechanism`；该命令仍要求同时提供计划、报告和三份原始账本以重新推导结果。

创建并使用复合预测工件：

```powershell
python -m pcfm create-composite `
  --model <person-model.json> `
  --mechanism-plan <mechanism-plan.json> `
  --mechanism-report <mechanism-report.json> `
  --discovery-ledger <discovery.jsonl> `
  --selection-ledger <selection.jsonl> `
  --confirmation-ledger <confirmation.jsonl> `
  --verification-keys <verification-keys.json> `
  --verifier-id <verifier-id> `
  --created-at 2026-10-10T23:59:59Z `
  --output <composite-model.json>

python -m pcfm predict-composite `
  --model <person-model.json> `
  --composite <composite-model.json> `
  --mechanism-plan <mechanism-plan.json> `
  --mechanism-report <mechanism-report.json> `
  --discovery-ledger <discovery.jsonl> `
  --selection-ledger <selection.jsonl> `
  --confirmation-ledger <confirmation.jsonl> `
  --verification-keys <verification-keys.json> `
  --scenario <scenario.json> `
  --prediction-at 2026-10-15T00:00:00Z
```

复合主动实验使用 `plan-composite-experiment`。它会排除基础模型谱系以及发现、选择、确认三份机制账本中已经出现的场景和设计。采集签名结果并准备更晚、与机制证据不重叠的验证账本后，使用 `apply-composite-experiment`；该命令输出更新后的基础模型和训练账本，并明确报告旧复合身份失效。完整参数可通过两个命令的 `--help` 查看。

在收集监测结果前注册动态计划：

```powershell
python -m pcfm plan-state `
  --model <person-model.json> `
  --verification-keys <verification-keys.json> `
  --verifier-id <verifier-id> `
  --registered-at 2026-09-01T00:00:00Z `
  --monitoring-start-at 2026-09-02T00:00:00Z `
  --monitoring-end-at 2026-09-24T09:00:00Z `
  --expected-event-count 180 `
  --output <dynamic-state-plan.json>
```

自定义配置必须通过 `plan-state --config <config.json>` 在监测前固定。

推断签名动态报告：

```powershell
python -m pcfm infer-state `
  --model <person-model.json> `
  --input <future-monitoring-ledger.jsonl> `
  --verification-keys <verification-keys.json> `
  --plan <dynamic-state-plan.json> `
  --output <dynamic-state.json>
```

使用动态状态预测：

```powershell
python -m pcfm predict-state `
  --model <person-model.json> `
  --state <dynamic-state.json> `
  --plan <dynamic-state-plan.json> `
  --state-ledger <future-monitoring-ledger.jsonl> `
  --verification-keys <verification-keys.json> `
  --scenario <scenario.json> `
  --prediction-at 2026-09-25T00:00:00Z
```

## 版本兼容性

- 包版本：`0.9.1`
- bundle：`pcfm-bundle-v7`
- 稳定模型代码身份：`pcfm-mvp-0.8.0`
- 主动实验计划：`pcfm-active-experiment-v3`
- 主动实验模块：`gaussian-mutual-information-v2`
- 机制比较计划：`pcfm-mechanism-plan-v2`
- 机制比较报告：`pcfm-mechanism-report-v2`
- 机制比较模块：`preregistered-mechanism-comparison-v2`
- 复合模型工件：`pcfm-composite-model-v1`
- 复合预测视图：`composite-predictive-view-v1`
- 动态计划：`pcfm-dynamic-state-plan-v1`
- 动态报告：`pcfm-dynamic-state-v2`
- 动态模块：`continuous-time-logit-state-v2`

旧 bundle、主动实验 v2、动态报告和机制 v1 工件缺少现行合同字段，必须从原始签名账本重新拟合、规划、推断或比较，不做静默升级。

## 验证结果

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

当前全量回归为 246/246。其中原有研究回归 240 项保持通过，产品服务层与网页核心流程新增 6 项测试；Decision-Context-Rationale Evidence Contract v1 有 11 项聚焦测试，Prospective Single-Person Pilot v1 有 15 项聚焦测试，Anisotropic Empirical-Bayes Adapter v1 有 17 项聚焦测试，Support-set HyperNetwork v1 有 15 项聚焦测试，Person-choice Benchmark v1 有 13 项聚焦测试；阶段 6.5 原有 14 项测试，修复审计再增加 6 项，覆盖机制证据重放与改名重放、未来验证证据隔离、单次重算、信息增益硬下限、复合结果吸收/失效和 CLI 闭环。

20 个独立合成随机种子压力测试结果：

- 静态序列误报：0/20；
- logit `+2.2` 暂时偏移检出：20/20；
- logit `+2.5` 暂时偏移检出：20/20；
- `+2.2` 偏移在恢复期末回落：20/20。

这些数字只描述当前合成生成器，不是现实人物数据上的灵敏度或特异度。

内部目标测试验证规划器确实优化了声明的互信息目标。独立于该目标的固定合成基准中，主动选择吸收 10 个真实结果后的留出 NLL 为 `0.513229`，100 个等量随机批次的平均值为 `0.526107`，主动批次胜过其中 94 个。该结果只证明这一固定合成基准，不是现实访谈效率结论。

固定的阶段六非线性合成基准中，系统从三个候选中选择预设的交互—绝对值结构；确认集基础 NLL 为 `0.597100`，候选 NLL 为 `0.329305`，改善 `0.267796`，IID 区间为 `[0.189821, 0.345770]`。在线性空数据对照中，最佳候选改善 `0.009278`，区间下界 `-0.022255`，因此状态为 `no_supported_candidate`。这些仍是实现回归，不是现实人物机制恢复率。

## 仍然存在的功能边界

- 只支持固定数值特征下的结构化二元选择。
- 一维残差只能表示总体选择倾向，不能表达多维记忆、信念、目标或价值状态。
- 无外部状态测量或随机干预时，残差原因在统计上不可识别。
- 内容指纹能识别改 ID 和非设计时间元数据后的重放；若上游同时修改真实设计内容并提供合法签名，本原型仍缺少外部事实来源判断它是否为新试验。
- 当前 e-process 比较的是已固定的静态与动态预测器；它控制该预注册比较中的序贯误报风险，不证明模型结构正确。
- 当前主动规划器只降低已有线性人物参数的不确定性，不能发现候选模型中没有的信念、价值或因果机制。
- 机制模块只能在预先列出的数学结构中选择，不能证明候选集合完整；结构项也不等同于信念、价值、目标或因果解释。
- 机制预测区间是选定候选条件下的高斯参数近似，不包含候选选择和模型集合不确定性。
- 复合模型只整合一个已确认机制；其分块对角协方差不包含基础参数与修正参数的后验相关。
- 现有动态状态相对旧基础预测核推断，尚不能与复合模型联合使用。
- 复合更新会安全更新基础模型并使旧复合工件失效；由于独立确认数据不能由程序凭空生成，重新确认机制和创建新复合工件仍是后续必需步骤。
- 候选题池仍需由外部实验设计者产生；如果候选池没有区分关键机制的题目，信息增益排序无法补救。
- 多题计划使用无结果矩匹配近似，不是逐题观察结果后的精确自适应策略，也不保证全局最优批次；严格自适应运行必须采用单题规划—结果吸收—重新规划。
- 半衰期、过程方差、阈值和有效期仍只在合成任务中测试，真实项目必须先做外部预注册和校准。
- 通过全部门禁只说明在定义任务上的可重复预测价值，不说明已经还原人物思想。
