# TronCamp Mani T1-T4：机器人模仿学习综合项目

这是我围绕 TronCamp Mani 四个机器人操作任务做的综合项目记录。整体目标是在 RoboTwin 双臂仿真环境中，逐步完成 T1-T4 任务的数据采集、ACT 策略训练、本地评估、策略部署演示和官方提交复盘。

当前仓库已经完成第一阶段 T1 `adjust_bottle`，并推进到第二阶段 T2 `grab_roller`。T2 已完成 400 条成功演示数据采集、ACT baseline 训练和公开 seed 本地评估，正在继续做数据增强与 InterACT 结构验证。后续会继续补充 T3 `stack_bowls_two` 和 T4 `stack_bowls_three` 的实验记录、部署视频和失败分析。

这个仓库面向项目展示和复盘，不直接包含大体积训练产物。采集数据、processed data、checkpoint、评估日志和提交 token 都已排除。

## 当前进度

| 阶段 | 任务 | 当前状态 |
|---|---|---|
| T1 | `adjust_bottle` | 已完成采集、训练、本地评估、策略部署演示和官方提交 |
| T2 | `grab_roller` | 已完成 400 条采集、ACT baseline 训练和公开 seed 评估；增强训练进行中 |
| T3 | `stack_bowls_two` | 计划中 |
| T4 | `stack_bowls_three` | 计划中，后续作为综合得分任务重点验证 |

## T1 阶段结果

| 项目 | 结果 |
|---|---|
| 方法 | ACT imitation learning |
| 本地演示数据 | 200 episodes |
| 训练 seed | 0 |
| 最佳验证损失 | 0.028757，epoch 5125 |
| 本地公开 100 seed 评估 | `sr = 0.52` |
| 策略部署演示 | seed `20260631`，138 步成功 |
| 官方提交 | 已提交，queue id `#70` |

本地评估结果保存在 [records/t1_public_eval.json](records/t1_public_eval.json)，完整记录见 [records/t1_record.md](records/t1_record.md)。

## T2 阶段进展

| 项目 | 结果 |
|---|---|
| 方法 | ACT imitation learning baseline |
| 本地演示数据 | 400 episodes |
| 训练 seed | 0 |
| 最佳验证损失 | 0.029596，epoch 2457 |
| 本地公开 100 seed 评估 | `sr = 0.53` |
| 当前优化 | 只对训练集做轻量视觉增强，继续冲击 60% 目标 |

完整记录见 [records/t2_record.md](records/t2_record.md)。

下面是一次 T2 `grab_roller` 成功专家/数据采集样例，用于展示第二个任务的双臂抓举形态。

![T2 collect success](media/t2_collect_success_grab_roller_episode1.gif)

原始 MP4：[`media/t2_collect_success_grab_roller_episode1.mp4`](media/t2_collect_success_grab_roller_episode1.mp4)

> 这个视频是专家/采集成功示例，不是当前 ACT policy 的闭环部署视频。T2 policy rollout 会在模型稳定超过目标阈值后补充。

## InterACT 算法改造

除了 ACT baseline，本仓库新增了一个独立的 InterACT 风格算法目录：

```text
policies/inter-act/
```

这个版本保留当前 ACT 的 HDF5 数据格式、三相机 RGB 输入和 16 维双臂动作接口，同时新增层次注意力编码器和 multi-arm decoder。它不会覆盖原始 ACT baseline，主要用于 T3/T4 长序列、多阶段协同任务的后续对比实验。

设计说明见 [docs/interact_design.md](docs/interact_design.md)。

## 策略部署演示

下面是训练后的 ACT policy 在 T1 环境中的一次成功 rollout。该视频不是专家演示，而是加载训练得到的 `policy_best.ckpt` 后，由策略闭环控制机器人完成任务。

![T1 policy rollout success](media/t1_policy_rollout_success_seed_20260631.gif)

原始 MP4：[`media/t1_policy_rollout_success_seed_20260631.mp4`](media/t1_policy_rollout_success_seed_20260631.mp4)

这次 rollout 使用公开 seed `20260631`，在 138 个策略步内触发成功条件。

## 数据采集样例

下面是一次 T1 专家/数据采集样例，用于对比任务形态和演示数据来源。

![T1 collect demo](media/t1_collect_demo_episode43.gif)

原始 MP4：[`media/t1_collect_demo_episode43.mp4`](media/t1_collect_demo_episode43.mp4)

## 综合技术路线

每个任务都会复用同一套工程流程，并在任务难度提升时补充新的实验设计：

```text
T1-T4 RoboTwin 仿真任务
        |
        v
任务配置与数据采集
        |
        v
ACT 数据预处理
        |
        v
训练 ACT policy
        |
        v
公开 100 seed 本地评估
        |
        v
策略部署视频与失败分析
        |
        v
官方提交与复盘
```

主要技术栈：

- Python / PyTorch
- ACT policy
- RoboTwin 双臂机器人仿真环境
- CUDA 单卡训练与评估
- 本地评估脚本和官方提交脚本

## 我完成的工作

- 修复本地 GPU / CUDA 运行环境，保证训练和评估能使用显卡。
- 基于 T1 `adjust_bottle` 任务采集 200 条本地演示数据。
- 完成 ACT 数据处理和模型训练。
- 对 `policy_best.ckpt` 做公开 100 seed 本地评估。
- 录制成功的策略部署 rollout，用于展示训练后模型的闭环执行效果。
- 将模型提交到官方评测队列。
- 基于 T2 `grab_roller` 任务采集 400 条成功演示数据，完成 ACT baseline 训练和公开 seed 本地评估。
- 基于 InterACT 思路重写一个独立策略目录，保留 ACT 数据接口，同时加入层次注意力编码器和 multi-arm decoder。
- 整理项目记录仓库，排除大体积数据、权重和敏感凭据。

## 阶段性观察

- T1 当前 `sr = 0.52` 只比 0.50 略高，说明这是一个能跑通完整流程的 baseline，但不是稳定高分模型。
- T2 当前 ACT baseline `sr = 0.53`，说明双臂抓举任务已经能部分泛化，但距离 60% 目标还需要数据增强或更多轨迹。
- `policy_best.ckpt` 是按验证 loss 选出的，不一定等价于 rollout 成功率最高的 checkpoint。
- 机器人闭环任务对早期动作误差比较敏感，轻微偏差可能在后续放大为失败。
- 本地公开 seed 和官方私有 seed 同分布但不相同，公开分数只能作为官方成绩的估计。

## 仓库结构

```text
records/
  t1_record.md             # T1 训练、评估、提交记录
  t2_record.md             # T2 采集、训练、评估和优化记录
  t1_public_eval.json      # 本地公开 seed 评估结果
media/
  t1_policy_rollout_success_seed_20260631.gif
  t1_policy_rollout_success_seed_20260631.mp4
  t1_collect_demo_episode43.gif
  t1_collect_demo_episode43.mp4
  t2_collect_success_grab_roller_episode1.gif
  t2_collect_success_grab_roller_episode1.mp4
policies/
  inter-act/               # 独立 InterACT 风格算法改造，不覆盖 ACT baseline
recipes/
  eval/                    # 本地评估相关脚本
  train/                   # ACT 训练相关脚本
scripts/                   # 一键采集、处理、训练、评估、提交入口
starter/                   # 本地评估和可视化入口
  record_policy_rollout.py # 单 seed/搜索成功 seed 的策略部署录制脚本
submit/                    # 官方提交脚本
```

## 不包含的内容

这些内容没有放进公开仓库：

- `external/robotwin_local/`
- 采集得到的 `.hdf5` 演示数据
- ACT `processed_data/`
- `.ckpt` checkpoint
- 本地评估日志
- 官方提交 token 或其他凭据

这样做是为了避免仓库体积过大，也避免把比赛提交凭据或本地生成产物公开。

## T1-T4 后续验证计划

我后续会把这个项目继续做成一个覆盖 T1-T4 的机器人模仿学习实验记录，重点验证：

- checkpoint 选择：对比 `policy_best`、`policy_last` 和不同 epoch checkpoint 的 rollout 成功率。
- 训练随机性：用不同训练 seed 复现实验，观察成功率波动。
- 数据规模：比较 200、更多演示数据下的 ACT 表现。
- 失败分析：继续录制失败 rollout，对比瓶子姿态、夹爪动作和关键时刻偏差。
- 评估稳定性：多次 repeat 和不同公开 seed 子集下的成功率置信区间。
- 多任务推进：按 T1 -> T2 -> T3 -> T4 逐步解锁，记录每个任务的数据、训练、评估和部署差异。
- T4 综合得分：重点分析三碗堆叠任务的分层得分、失败模式和末态空间误差。

## 复现实验说明

这个仓库是项目记录版，不是完整的 clone-and-run 发布版。原始实验依赖 TronCamp Mani starter package，并在本地安装了 `external/robotwin_local` 作为 RoboTwin 运行环境。

本地完整流程使用过的入口形式如下：

```bash
make collect TRACK=T1 GPU=0
make process TRACK=T1
make train TRACK=T1 SEED=0 GPU=0
python starter/eval_local.py --track T1 --ckpt-dir <ACT_CKPT_DIR>
python starter/record_policy_rollout.py --track T1 --ckpt-dir <ACT_CKPT_DIR> --seed 20260631
make submit TRACK=T1
```

提交时使用 token 文件或环境变量，不在代码或 README 中保存 token。
