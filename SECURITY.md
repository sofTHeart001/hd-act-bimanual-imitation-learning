# Security Policy / 安全策略

## 报告安全问题 / Reporting a Vulnerability

**请不要在公开 Issue / PR / 讨论区披露安全问题。**
**Please do NOT disclose security issues in public Issues / PRs / discussions.**

如果你发现了本选手包工具(安装脚本、评测内核、自评或提交脚本)或比赛基础设施中的安全
问题(例如可绕过提交鉴权、可读取他人 token / 私有 seed、可在评测机上越权执行等),请
通过**私下渠道**报告给 TronCamp 主办方:

- 通过[赛题官网](https://limx-troncamp.github.io/troncamp-web-mani/)上公布的官方联系方式
  私下联系主办方;
- 报告中请包含:影响范围、复现步骤、相关版本 / 环境、可能的影响评估。

If you discover a security issue in this contestant kit (install script, evaluation
kernel, self-eval or submission scripts) or in the competition infrastructure
(e.g. bypassing submission authentication, reading another team's token / private
seeds, privilege escalation on the evaluation host), please report it **privately**
to the TronCamp organizers via the official contact channel listed on the
[competition website](https://limx-troncamp.github.io/troncamp-web-mani/). Include
scope, reproduction steps, affected version/environment, and an impact assessment.

我们会尽快确认并处理;在修复发布前请勿公开细节。
We will acknowledge and address the report promptly; please refrain from public
disclosure until a fix has been released.

## 范围 / Scope

- **本仓库范围内**:选手包自带的脚本与工具(`setup/`、`recipes/eval/`、`starter/`、
  `submit/`)。
- **第三方组件**(`external/robotwin_local/` 内的 RoboTwin / ACT / DETR / cuRobo 等)的
  漏洞请报告到其**各自上游项目**;本仓库仅原样再分发,见 [`NOTICE`](NOTICE)。
