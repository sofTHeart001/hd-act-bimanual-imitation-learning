# T2 阶段记录：grab_roller

这是 TronCamp Mani T1-T4 综合项目中的第二阶段记录，任务是 T2 `grab_roller`。该阶段开始从单臂任务进入双臂协同操作，重点验证 ACT 在更长动作序列、双腕相机和轻量随机化数据上的泛化能力。

## 任务信息

- 赛道：T2
- 任务：`grab_roller`
- 策略：ACT baseline，后续准备对比 InterACT
- 本地演示数据：400 episodes
- baseline 训练 seed：0
- 增强训练 seed：1
- 训练配置：`chunk_size=50`、`hidden_dim=512`、`dim_feedforward=3200`、`kl_weight=10`、`lr=1e-5`
- baseline 最佳验证损失：0.029596，出现在 epoch 2457
- 增强训练最佳验证损失：0.029514，出现在 epoch 2514

## Baseline 本地公开 seed 评估

评估使用官方本地评估入口，在公开 100 seed 上执行，`repeats=1`。

```json
{
  "sr": 0.53,
  "n_repeats": 1,
  "n_episodes": 100,
  "per_repeat": [0.53],
  "track": "T2"
}
```

## 最新 checkpoint 本地公开 seed 评估

最新选择的是增强训练目录下的 `policy_last.ckpt`。在公开 100 seed 上，本地评估结果为：

```json
{
  "sr": 0.64,
  "n_repeats": 1,
  "n_episodes": 100,
  "per_repeat": [0.64],
  "track": "T2"
}
```

该 checkpoint 已在 2026-07-15 作为 T2 最新版本提交官方队列，提交编号 `#380`。公开仓库只记录结果和复现方式，不提交 `.ckpt` 权重文件。

结论：400 条轨迹训练出的 ACT 已经具备完成 T2 双臂抓举任务的能力。轻量视觉增强和 checkpoint 筛选对公开 seed 稳定性有明显帮助。

## T2 Policy 自主执行示例

下面是增强训练后的 ACT checkpoint 在 T2 `grab_roller` 上的闭环执行结果，使用公开 seed `20260630`，99 步完成任务。

![T2 policy rollout success](../media/t2_policy_rollout_success_seed_20260630.gif)

原始 MP4：[`media/t2_policy_rollout_success_seed_20260630.mp4`](../media/t2_policy_rollout_success_seed_20260630.mp4)

该视频来自本地 policy rollout，用于展示 T2 双臂协同抓举的策略效果，不是专家采集回放。

## 当前优化

当前追加了一轮轻量视觉增强训练，用于提升模型对光照和背景变化的适应能力：

- 只对 train dataset 做亮度、对比度、饱和度扰动。
- validation dataset 保持干净，避免 best checkpoint 选择被随机增强噪声污染。
- 不把 checkpoint、processed data 或训练日志放入 GitHub。

下一步会继续对比增强训练、更多演示轨迹和 InterACT 结构在 T2/T3/T4 上的表现。

## 未公开的本地文件

以下内容没有放入 GitHub：

- 采集到的 `.hdf5` 演示数据
- ACT processed data
- `.ckpt` checkpoint
- 本地训练和评估日志
- 官方提交 token 或其他凭据
