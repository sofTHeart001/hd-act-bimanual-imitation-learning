# T3 阶段记录：stack_bowls_two

这是 TronCamp Mani T1-T4 综合项目中的第三阶段记录，任务是 T3 `stack_bowls_two`。相比 T2 抓举滚筒，T3 更强调双臂长序列协同、物体放置精度和动作 chunk 的稳定性。

## 任务信息

- 赛道：T3
- 任务：`stack_bowls_two`
- 策略：ACT baseline
- 本地演示数据：600 episodes
- 训练 seed：0
- 训练配置：`chunk_size=100`、`hidden_dim=512`、`dim_feedforward=3200`、`kl_weight=10`
- 优化配置：`batch_size=24`、`lr=2e-5`、`lr_backbone=3e-6`、`weight_decay=1e-4`
- 训练技巧：`warmup_steps=500`、`grad_clip=0.1`、`val_ratio=0.1`、`ACT_AUG=1`
- 训练轮数：4000 epochs
- 最佳验证损失：0.017439，出现在 epoch 3286

## 本地公开 seed 评估

评估使用官方本地评估入口，在公开 100 seed 上执行，`repeats=1`。

```json
{
  "sr": 0.72,
  "n_repeats": 1,
  "n_episodes": 100,
  "per_repeat": [0.72],
  "track": "T3"
}
```

T3 使用了 `chunk_size=100`，因此评估时必须使用与训练结构匹配的部署配置：

```text
configs/deploy_t3.yml
```

脚本 `make eval-local TRACK=T3` 会优先把该配置同步到：

```text
external/robotwin_local/policy/ACT/deploy_t3.yml
```

并传入：

```bash
--deploy-config policy/ACT/deploy_t3.yml
```

## T3 Policy 自主执行示例

下面是训练出的 ACT policy 在 T3 `stack_bowls_two` 上的闭环执行样例。该视频不是专家采集轨迹，而是 checkpoint 在 clean eval 配置下自主 rollout，公开 seed 为 `20260629`，最终成功，耗时 `668` steps。

![T3 policy rollout success](../media/t3_policy_rollout_success_seed_20260629.gif)

原始 MP4：[`media/t3_policy_rollout_success_seed_20260629.mp4`](../media/t3_policy_rollout_success_seed_20260629.mp4)

## 复盘

T3 的主要变化是把轨迹数量扩展到 600 条，并把 ACT 的 `chunk_size` 提升到 100，以适配更长的连续操作。训练中保留轻量视觉增强，用于提升对光照和背景扰动的鲁棒性；评估端保持确定性配置，避免引入额外随机性。

公开仓库不包含 T3 的 HDF5 数据、processed data、checkpoint 和本地完整日志，只保留可复现配置、评估结果和视频记录。
