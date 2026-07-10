# TronCamp Mani T1-T4：机器人模仿学习综合项目

这是一个围绕 TronCamp Mani 四个机器人操作任务构建的机器人模仿学习项目。项目基于 RoboTwin 双臂仿真环境，完成从任务配置、专家轨迹采集、ACT 策略训练、本地评估到可视化展示的完整流程，并在此基础上扩展 InterACT 风格的双臂协同策略结构。

当前已经完成 T1 `adjust_bottle` 和 T2 `grab_roller` 的阶段性实验，并准备继续推进 T3 `stack_bowls_two` 与 T4 `stack_bowls_three`。

## 视频展示

### T1：ACT 策略闭环执行

训练后的 ACT policy 在 T1 环境中完成瓶子姿态调整。

![T1 policy rollout success](media/t1_policy_rollout_success_seed_20260631.gif)

原始 MP4：[`media/t1_policy_rollout_success_seed_20260631.mp4`](media/t1_policy_rollout_success_seed_20260631.mp4)

### T2：双臂抓举滚筒成功示例

T2 `grab_roller` 任务中，双臂协同完成滚筒抓取和举升。

![T2 collect success](media/t2_collect_success_grab_roller_episode1.gif)

原始 MP4：[`media/t2_collect_success_grab_roller_episode1.mp4`](media/t2_collect_success_grab_roller_episode1.mp4)

## 当前进度

| 阶段 | 任务 | 当前状态 |
|---|---|---|
| T1 | `adjust_bottle` | 已完成数据采集、ACT 训练、本地评估、策略部署演示和官方提交 |
| T2 | `grab_roller` | 已完成 400 条成功轨迹采集、ACT baseline 训练和成功示例展示 |
| T3 | `stack_bowls_two` | 准备中 |
| T4 | `stack_bowls_three` | 准备中，后续作为综合任务重点验证 |

## 项目亮点

- 打通 RoboTwin 双臂仿真任务的完整模仿学习流程。
- 完成 T1/T2 的专家轨迹采集、ACT 数据预处理和策略训练。
- 录制可展示的任务成功视频，便于直观看到策略和任务效果。
- 在 ACT baseline 旁新增独立的 `policies/inter-act/` 算法目录，准备验证更适合双臂协同和长序列任务的 InterACT 风格结构。
- 对训练产物、checkpoint、HDF5 数据和 token 做公开仓库隔离，保留可展示代码和记录。

## InterACT 算法改造

本仓库新增了一个独立的 InterACT 风格策略目录：

```text
policies/inter-act/
```

这个版本保留当前 ACT 的 HDF5 数据格式、三相机 RGB 输入和 16 维双臂动作接口，同时加入：

- 层次注意力编码器
- 左右臂 segment 建模
- 图像 segment 融合
- multi-arm decoder
- 独立训练、评估和 checkpoint 目录

设计说明见 [docs/interact_design.md](docs/interact_design.md)。

## 技术路线

```text
RoboTwin T1-T4 双臂操作任务
        |
        v
任务配置与专家轨迹采集
        |
        v
ACT / InterACT 数据预处理
        |
        v
策略训练与 checkpoint 选择
        |
        v
公开 seed 本地评估
        |
        v
成功示例录制与提交复盘
```

主要技术栈：

- Python / PyTorch
- ACT imitation learning
- InterACT-style hierarchical attention
- RoboTwin 双臂机器人仿真
- CUDA 单卡训练与评估
- GitHub 项目记录与视频展示

## 仓库结构

```text
records/
  t1_record.md             # T1 阶段记录
  t2_record.md             # T2 阶段记录
  t1_public_eval.json      # T1 本地评估结果归档
media/
  t1_policy_rollout_success_seed_20260631.gif
  t1_policy_rollout_success_seed_20260631.mp4
  t1_collect_demo_episode43.gif
  t1_collect_demo_episode43.mp4
  t2_collect_success_grab_roller_episode1.gif
  t2_collect_success_grab_roller_episode1.mp4
policies/
  inter-act/               # 独立 InterACT 风格算法改造
docs/
  interact_design.md       # InterACT 架构说明
recipes/
  eval/                    # 本地评估相关脚本
  train/                   # ACT 训练相关脚本
scripts/                   # 一键采集、处理、训练、评估、提交入口
starter/                   # 本地评估和可视化入口
submit/                    # 官方提交脚本
```

## 不包含的内容

公开仓库不包含：

- 采集得到的 `.hdf5` 演示数据
- ACT processed data
- `.ckpt` checkpoint
- 本地训练/评估日志
- 官方提交 token 或其他凭据
- 本地完整 RoboTwin 运行环境副本

## 后续计划

- 补充 T2 policy rollout 视频。
- 推进 T3/T4 长序列双臂协同任务。
- 对比 ACT、ACT + 数据增强、InterACT 三组策略。
- 继续整理可展示的视频、训练记录和阶段成果。
