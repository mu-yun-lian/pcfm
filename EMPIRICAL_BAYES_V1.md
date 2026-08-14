# Anisotropic Empirical-Bayes Adapter v1 合同

## 可检验问题

本阶段只检验：

> 在未检查过的固定合成种子上，从元训练人物估计六维人物参数差异的各向异性协方差，再以该协方差作为新人物 Logistic MAP 适配的先验，能否稳定超过使用固定各向同性精度的 `personal_map_logistic`？

输入是元训练人物、validation 人物和一个测试人物的 support 集；输出是该测试人物的六参数 Logistic 预测头及其选择概率。

允许的解释仅为 `synthetic_anisotropic_population_prior_adaptation`。协方差的方向和大小不具有思想、价值观、人格或因果心理机制语义。

## 为什么换确认种子

`7301–7305` 已在 Person-choice 和 HyperNetwork 阶段查看，不能继续充当新候选的封存确认集。本阶段预注册：

```text
主审计：8101, 8102, 8103, 8104, 8105
零异质性：8110
稳定错设：8120
```

在合同、候选集合和门槛固定前，不读取这些种子的最终测试结果。

## 替代解释

1. 固定各向同性 MAP 已经充分；
2. 协方差估计只是 20 个元训练人物上的采样噪声；
3. 增益来自 validation 或测试答案泄漏；
4. 增益来自错误人物 support、输入顺序或 ID；
5. 线性生成器中的结果不能迁移到模型族错设或真人。

## 数据角色

- `meta_train`：拟合群体头、每人物参数和人物间协方差；
- `validation`：从固定协方差收缩候选中选择一个；
- `support`：最终测试人物的 MAP 适配；
- `scenario_test`：唯一主确认集；
- `temporal_test`：次要稳定性门；
- `ood_test`：只报告，不参与接受判定。

任何测试人物的 scenario、temporal 或 OOD 答案均不得影响先验、候选选择、适配参数或门槛。

## 固定估计器

1. 用全部 `meta_train` 记录拟合群体 Logistic 权重 \(w_0\)；
2. 对每个元训练人物，以弱固定先验拟合 \(\hat w_i\) 和后验协方差 \(C_i\)；
3. 计算噪声校正人物间协方差：

\[
\Sigma_{\text{between}}
=\operatorname{Cov}(\hat w_i)
-\frac{1}{m}\sum_i C_i;
\]

4. 对称化并把特征值截到固定下限 `0.02`；
5. validation 只从以下预注册收缩系数中选择：

```text
0.00, 0.25, 0.50, 0.75, 1.00
```

\[
\Sigma_\alpha
=(1-\alpha)\Sigma_{\text{between}}
+\alpha\operatorname{diag}(\Sigma_{\text{between}}).
\]

6. 新人物只用 support 做 MAP：

\[
\hat w_{\text{new}}
=\arg\max_w
\log p(y_S\mid x_S,w)
-\frac12(w-w_0)^\top\Sigma_\alpha^{-1}(w-w_0).
\]

部署时仍使用共享的 `LogisticChoiceModel.probabilities`。

## 固定资源与数值门

- 维度固定为 6；
- 元训练人物不得少于 8，validation 不得少于 2；
- 协方差必须对称、正定、有限；
- 特征值下限固定为 `0.02`；
- 人物头相对群体头的范数最多为 `6.0`；
- support 只允许 `0、16、32、64`；
- support 内特征距离不高于 `1e-6` 的改名近重复记录被拒绝；
- NumPy、CPU、单进程，不使用 GPU；
- 不搜索网络、特征、阈值、种子或候选集合。

## 预注册接受条件

每个主种子在 support 64、`scenario_test` 上必须同时满足：

1. 相对 `personal_map_logistic` 的 NLL 改善至少 `0.005`；
2. 相对错误人物 EB support 的 NLL 改善至少 `0.01`；
3. 主 NLL 不高于 `0.58`；
4. `temporal_test` 不得比 `personal_map_logistic` 高超过 `0.02`。

最终接受还要求：

5. 五个主种子至少四个通过；
6. 五种子平均 NLL 改善至少 `0.005`；
7. 零异质性对照相对群体模型的绝对 NLL 差不超过 `0.04`；
8. 稳定错设对照不得获得确认状态；
9. 泄漏、顺序、改名、跨域、篡改、重算和 CLI 测试全部通过。

任一条件失败即为 `rejected_candidate`。预测增益通过也只支持合成任务中的先验适配，不构成真人或心理语义证据。

## 无法由本阶段解决

- 真人外部效度与真人级统计功效；
- 上游是否隐瞒试验或伪造时间；
- 观察等价的不同内部过程；
- 协方差方向的心理含义；
- 单一五维二选一任务之外的迁移。
