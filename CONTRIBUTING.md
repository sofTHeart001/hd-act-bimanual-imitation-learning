# Contributing / 参与与反馈

*(English below / 中文在前)*

本仓库是 **TronCamp · ACT 四任务套餐** 的选手包(已发布的比赛工具包)。请先区分两类
"贡献":

## 比赛提交 ≠ 代码 PR

- **比赛作品**通过 `submit.py` 唯一通道提交(带 token,进后端评测队列),**不要**用本仓库的
  Issue / Pull Request 提交你的模型或代码。参赛说明见
  [赛题官网](https://limx-troncamp.github.io/troncamp-web-mani/)。
- 本仓库的 Issue / PR **仅用于选手包工具本身**(安装 / 自检脚本 `setup/`、评测内核 `recipes/eval/`、
  自评脚本 `starter/`、提交脚本 `submit/`、文档)的 bug 反馈与改进。

## 提 Issue

- 用 `.github/ISSUE_TEMPLATE/` 里的 **Bug report** 或 **Feature request** 模板。
- Bug 请附:复现步骤、安装 / 运行的相关命令与输出、conda 环境 / GPU / CUDA 版本、
  完整报错栈。
- **安全问题不要开公开 Issue** —— 见 [`SECURITY.md`](SECURITY.md)。

## 提 PR

- 一个 PR 只做一件事;附清晰说明与复现/验证方式(见 `.github/PULL_REQUEST_TEMPLATE.md` 清单)。
- **不要修改比赛计分口径、阈值或评测内核的判定逻辑**(这些由主办方后端掌握,选手侧改动
  不会影响官方成绩,只会造成本地自评与官方榜不一致)。
- 不要提交任何权重、数据集、私有 seed 或含密钥/内部地址的文件。

## 贡献边界

本仓库不接受对**第三方组件**(`external/robotwin_local/` 内的 RoboTwin / ACT / DETR /
cuRobo 等)的改动 PR —— 请到其各自上游仓库提交。第三方组件各自遵循其许可证,见
[`NOTICE`](NOTICE)。

---

## English

This repository is the released contestant kit for **TronCamp · ACT Four-Task
Suite**. Please distinguish two kinds of "contribution":

- **Competition entries** are submitted only through `submit.py` (token-gated, into
  the backend evaluation queue). Do **not** submit your model or code via GitHub
  Issues/PRs. See the [competition site](https://limx-troncamp.github.io/troncamp-web-mani/).
- Issues/PRs on **this** repository are for the **kit tooling only** (setup scripts,
  evaluation kernel `recipes/eval/`, self-eval `starter/`, `submit/`, docs).

**Filing issues:** use the Bug report / Feature request templates in
`.github/ISSUE_TEMPLATE/`. For security issues, do **not** open a public issue —
see [`SECURITY.md`](SECURITY.md).

**Pull requests:** keep them focused; fill in the checklist in
`.github/PULL_REQUEST_TEMPLATE.md`. Do not modify scoring, thresholds, or the evaluation
kernel's decision logic (owned by the organizers' backend). Do not commit weights,
datasets, private seeds, or any secrets/internal addresses.

**Boundary:** changes to third-party components under `external/robotwin_local/`
(RoboTwin / ACT / DETR / cuRobo, etc.) belong upstream, not here. Each third-party
component keeps its own license — see [`NOTICE`](NOTICE).
