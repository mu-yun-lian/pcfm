# Contributing to PCFM

感谢你对 PCFM 的兴趣！本文档说明如何参与贡献。

## 开始之前

请先阅读 [ARCHITECTURE.md](ARCHITECTURE.md) 了解项目的核心机制、数据模型和三条铁律。PCFM 是一个**证据约束**的研究原型，所有设计决策都围绕"可证伪、可追溯、不伪造"展开。

## 开发环境

```bash
# 克隆仓库
git clone https://github.com/mu-yun-lian/pcfm.git
cd pcfm

# 安装 Python 依赖（可编辑模式）
pip install -e .

# 安装前端依赖
cd src/pcfm/web_static
npm install
cd ../../..
```

## 运行测试

```bash
# Python 全量回归
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v

# 前端类型检查与单测
cd src/pcfm/web_static
npm run typecheck
npm run test
```

提交前请确保所有测试通过。

## 提交规范

- 使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：
  - `feat:` 新功能
  - `fix:` 缺陷修复
  - `docs:` 文档变更
  - `refactor:` 重构（不改变功能）
  - `test:` 测试相关
  - `chore:` 构建/工具/依赖
- 一次提交只做一件事
- 提交信息用中文或英文均可，但需清晰说明变更内容

##  Pull Request 流程

1. Fork 仓库并创建特性分支
2. 确保测试通过
3. 提交 PR，描述清楚：
   - 变更了什么
   - 为什么这样做
   - 测试覆盖情况
4. 等待 review

## 代码规范

- **内容与风格分离**：新增模块时，立场/事实/数字在内容层锁定，风格层只改措辞
- **LLM 只出候选**：任何涉及人物立场的字段，必须由代码验证/门禁，不能由 LLM 直接提交
- **证据可追溯**：新增的人物材料必须能逐字溯源到原文位置
- **不静默升级**：工件格式变更时，旧工件必须从原始数据重新生成，不做自动转换

## 报告问题

- Bug 报告请使用 [Bug Report 模板](.github/ISSUE_TEMPLATE/bug_report.md)
- 功能建议请使用 [Feature Request 模板](.github/ISSUE_TEMPLATE/feature_request.md)
- 报告时请包含复现步骤、预期行为、实际行为和环境信息

## 许可证

提交代码即表示你同意你的贡献在 [MIT License](LICENSE) 下发布。
