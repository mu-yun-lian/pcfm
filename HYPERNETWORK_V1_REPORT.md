# Support-set HyperNetwork v1 实验报告

## 结论先行

候选状态：`rejected_candidate`。

实现闭环和工程门禁通过，但预注册的模型比较门槛失败。该 HyperNetwork 不应进入人物模型主路径，也不能替代 `personal_map_logistic`。

## 实现

HyperNetwork 接收一个测试人物的无序 support 集，计算群体 Logistic 下的平均 score，再通过两个小矩阵生成 Logistic 预测头的低秩增量：

\[
z=Vs(S),\qquad \Delta w=Uz,\qquad w(S)=w_0+\Delta w.
\]

- 预测头维度：6；
- rank：3；
- 可训练标量：36；
- 最大轮数：80；
- 每轮确定性选择 4 个元训练人物；
- 每 4 轮在 4 个 validation 人物上选择检查点；
- 测试人物不执行梯度优化；
- 生成头使用与其他 Logistic 基线相同的概率部署核；
- NumPy、CPU、单进程，无 GPU。

## 五种子结果

主指标为 64 条 support 后的 `scenario_test` NLL。

| 种子 | 群体 Logistic | 个体 MAP | 人物嵌入 MLP | HyperNetwork | 选中轮次 | 单种子状态 |
|---:|---:|---:|---:|---:|---:|---|
| 7301 | 0.628307 | 0.466396 | 0.678970 | 0.541601 | 40 | fail |
| 7302 | 0.593972 | 0.462772 | 0.787075 | 0.594532 | 48 | fail |
| 7303 | 0.516951 | 0.407369 | 0.470266 | 0.460164 | 76 | fail |
| 7304 | 0.621799 | 0.473722 | 0.674811 | 0.532374 | 48 | fail |
| 7305 | 0.617050 | 0.439184 | 0.597776 | 0.497053 | 80 | fail |

汇总：

- `0/5` 种子通过全部预注册门槛，要求至少 `4/5`；
- 5 个种子均未超过个体 MAP；
- 5 个种子均超过人物嵌入 MLP；
- 4 个种子超过群体 Logistic，种子 7302 略差于群体；
- 5 个种子的时间测试 NLL 都比个体 MAP 高出超过 `0.02`；
- 零异质性 NLL 绝对差为 `0.000260`，通过 `0.04` 上限。

## 为什么必须淘汰

该模型确实能从 support 中提取部分人物特异性信号，因此不是完全无效；但在当前线性人物生成器下，直接对测试人物 support 做正则化 MAP 估计更准确、更稳定、更简单。

HyperNetwork 的主要失败不是算力不足，而是归纳偏置不合适：rank-3 线性生成器把六维人物差异压缩到三个方向，并依赖群体层面的 score 到参数增量映射。这种摊销适配节省了测试时优化，却损失了个体 MAP 对当前人物数据的直接利用。

不能因为它名为 HyperNetwork、结构更前沿，就把“优于较弱神经基线”解释成胜利。预注册目标是同时超过两个基线，它没有做到。

## 已通过的工程门禁

- 封存测试答案翻转不改变拟合工件；
- OOD 答案不进入接受判定；
- support 顺序与场景 ID 改名不改变生成权重；
- 跨域 support 被拒绝；
- 空 support 精确退化为群体头；
- 生成增量范数有硬上限；
- rank、学习率、null 种子和比较阈值不可调弱；
- 工件覆盖训练证据摘要、配置、群体头、矩阵、选中轮次和 validation NLL；
- 工件可序列化、可检出篡改，并可从训练证据确定性重算；
- CLI 运行固定五种子审计。

## 仍未覆盖

- 稳定模型族错设；
- 真人数据与真人级置信区间；
- 外部签名、可信时间戳和试验完整性注册；
- 真实时间过期门禁；
- 局部 OOD 支持与近重复语义检测；
- 生成权重的任何心理或因果含义。

因此模块工程状态只能是 `implemented_exploratory`，模型候选状态是 `rejected_candidate`。

最终验证：HyperNetwork 聚焦测试 `15/15`，包含旧模块的全量回归 `166/166`，全量用时 `175.520` 秒。认知模块门禁检查器仍拒绝 `implemented_confirmatory`，原因是缺少真实未来、完整适用性/过期门和稳定模型族错设证据；该拒绝与本报告的探索性状态一致。

## 复现

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
python -m pcfm hypernetwork-v1
python -m unittest discover -s tests -p "test_hypernetwork_v1.py" -v
python -m unittest discover -s tests -v
```
