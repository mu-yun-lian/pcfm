# Person Choice Benchmark v1

## 目的

本基准只回答一个问题：

> 给定一个新人物在单一结构化权衡任务中的少量历史选择，模型能否在未见场景和更晚时间的场景上，比人群模型和错误人物模型更准确地预测该人物的二元选择概率？

它不是思想、信念、价值、人格、语言风格或意识模型。合成数据成功也不代表真实人物有效。

## 固定任务

每个场景由五个有名称的数值因素组成：

1. `reward_gain`
2. `loss_risk`
3. `delay`
4. `control`
5. `fairness`

输出为 `P(choice=1)`。第一版不接收自由文本，不生成自然语言，不跨任务或领域迁移。

## 人物与数据角色

人物集合在生成前固定并互斥：

- `meta_train_person`：学习人群规律和共享模型；
- `validation_person`：预留给后续工程参数选择；v1 的参数预先固定，本次结果不读取该组答案；
- `test_person`：只用于最终少样本适应和评估。

每个测试人物的记录进一步互斥：

- `support`：允许模型适应该人物；
- `scenario_test`：未见过的场景组合；
- `temporal_test`：时间晚于全部 support；
- `ood_test`：超出训练支持域，只检查拒绝或退化，不计入主增益。

记录不得通过修改人物 ID、场景 ID 或时间戳跨角色重放。所有主结果必须按人物配对计算。

## 少样本曲线

固定报告以下 support 大小：

```text
0, 16, 32, 64
```

冒烟配置只生成足以支持 64 条适应记录的数据。扩大到 128 或 256 条必须作为后续显式版本变更。

## 第一批模型

按顺序建立：

1. `population_logistic`：只使用 meta-train 人物；
2. `personal_map_logistic`：以人群模型为先验，只用测试人物 support；
3. `person_embedding_mlp`：共享场景网络和人物嵌入；面对新人物时只用 support 优化新嵌入。

HyperNetwork 不属于本阶段。只有前三个基线、数据隔离和评估闭环稳定后，才增加“由 support 集合生成低秩适配器”的模型。

## 主要指标

- Negative log likelihood（主指标）；
- Brier score；
- Expected calibration error；
- 正确人物相对人群模型的配对 NLL 增益；
- 正确人物相对错误人物模型的配对 NLL 增益；
- scenario 与 temporal 测试分别报告；
- 每个 support 大小分别报告。

## v1 通过条件

在固定合成生成器和固定随机种子集合上：

1. `personal_map_logistic` 在 `scenario_test` 上优于 population；
2. 正确人物参数优于循环错配人物参数；
3. support 增加时，跨人物平均 NLL 不出现预注册容差以外的系统恶化；
4. 人物、场景、内容和时间角色隔离测试全部通过；
5. `person_embedding_mlp` 能在安全资源预算内完成训练和新人物适应，并输出有限概率；
6. 所有结论明确标记为 `synthetic_benchmark_only`。

本阶段不要求 MLP 超过个人 MAP Logistic。它是未来 HyperNetwork 必须击败的较强神经基线。

## 安全资源配置

默认冒烟配置：

```text
meta-train 人物：20
validation 人物：4
test 人物：6
每个 meta-train 人物：96 条
每个 test 人物：64 support + 48 scenario + 48 temporal + 24 OOD
MLP 隐层：16
人物嵌入：8
训练轮数上限：80
batch：64
```

预计记录少于 4,000 条，模型少于 2,000 个可训练标量。实现只依赖 NumPy，不启用 GPU，不产生并行数据加载进程。

## 后续升级门槛

只有 v1 通过后才进入 HyperNetwork v1。HyperNetwork 首版只能生成预测头或低秩适配器，不生成完整网络。其验收条件必须是相对 `person_embedding_mlp` 和 `personal_map_logistic` 的独立人物测试增益，而不是训练集拟合效果。
