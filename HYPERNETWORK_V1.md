# Support-set HyperNetwork v1 合同

## 可检验问题

本阶段只检验：

> 对完全未参与元训练和模型选择的合成测试人物，一个由 support 集直接生成低秩 Logistic 预测头增量的 HyperNetwork，能否在 64 条 support 后的未见场景上，同时超过个体 MAP Logistic 与人物嵌入 MLP？

输入是一个人物的 `support` 记录集合；输出是六参数 Logistic 预测头及其二元选择概率。结果只能解释为 `synthetic_support_conditioned_weight_generation`，不能解释为思想、人格、价值观、心理机制或意识恢复。

## 固定替代解释

1. 群体 Logistic 已经足够，不需要人物适配。
2. 个体 MAP Logistic 已经是充分且更稳定的适配器。
3. 人物嵌入 MLP 已经捕获了神经模型能够利用的信息。
4. 表面增益来自测试答案泄漏、人物 ID 记忆、support 顺序或场景改名。
5. 单一生成器中的成功不能迁移到真人或模型族错设情形。

## 数据角色

- `meta_train`：训练群体 Logistic 和 HyperNetwork 参数；
- `validation`：只选择训练轮次，不选择 rank、阈值、特征或候选结构；
- `support`：为最终测试人物生成预测头；
- `scenario_test`：唯一主验收集；
- `temporal_test`：次要稳定性检查；
- `ood_test`：只报告退化，不参与接受判定。

测试人物的 `scenario_test`、`temporal_test` 和 `ood_test` 答案不得进入预测头生成、训练轮次选择、停止条件或任何参数更新。

## 固定计算

群体模型权重记为 \(w_0\)。对 support 集 \(S\)，先计算群体模型下的无序平均 score：

\[
s(S)=\sqrt{\frac{n}{n+16}}\frac{1}{n}
\sum_{(x,y)\in S}x\left(y-\sigma(x^\top w_0)\right).
\]

当 \(n=0\) 时，\(s(S)=0\)。

HyperNetwork 只有两个矩阵：

\[
z=Vs(S),\qquad \Delta w=Uz,\qquad w(S)=w_0+\Delta w,
\]

其中 \(U\in\mathbb R^{6\times3}\)、\(V\in\mathbb R^{3\times6}\)。因此它生成的是 rank 不超过 3 的 Logistic 预测头增量，不生成完整神经网络，也不在测试人物上进行梯度优化。

`w(S)` 进入与群体模型和个体 MAP 完全相同的 `LogisticChoiceModel.probabilities` 部署核。

## 训练与选择

- rank 固定为 3，不搜索；
- 最多 80 轮；
- 只依赖 NumPy 和 CPU；
- 元训练采用 support/query 分离的确定性 episode，并直接最小化 query NLL；
- validation 人物只选择最佳轮次；
- 所有分割由内容稳定摘要和固定种子决定，输入顺序及无意义 ID 不影响结果；
- 最大可训练参数为 36，硬上限为 128；
- 不根据最终测试结果修改模型、阈值、rank 或训练轮数。

## 预注册接受与淘汰条件

主判定只使用 `scenario_test`、support 大小 64：

1. HyperNetwork NLL 至少比 `personal_map_logistic` 低 `0.01`；
2. HyperNetwork NLL 至少比 `person_embedding_mlp` 低 `0.01`；
3. HyperNetwork 的主场景 NLL 不得高于 `0.60`；
4. 五个固定种子中至少四个同时满足前三项；
5. `temporal_test` 上不得比个体 MAP 高超过 `0.02`；
6. 零异质性对照中不得制造超过 `0.04` 的绝对 NLL 差；
7. 泄漏、改名、重排、篡改、资源上限和复算测试全部通过。

只要任一主条件失败，状态就是 `rejected_candidate`。代码能运行不等于模型被接受。若工程闭环完成但证据不足，使用 `implemented_exploratory`，不得进入人物模型主路径。

## 资源界限

默认仍使用 3,408 条合成记录。HyperNetwork 最多 36 个训练参数、80 轮、单 Python 进程、CPU 执行，不使用 GPU、Torch、TensorFlow 或并行数据加载。

## 无法由本阶段解决

- 合成数据到真人的外部效度；
- 生成权重与心理概念之间的语义对应；
- 观察等价的不同内部过程；
- 上游是否隐瞒真实试验；
- 单一线性生成器之外的稳定模型族错设。
