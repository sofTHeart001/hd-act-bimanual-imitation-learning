# T1 阶段记录：adjust_bottle

这是 TronCamp Mani T1-T4 综合项目中的第一阶段记录，任务是 T1 `adjust_bottle`。该阶段目标是先打通数据采集、ACT 训练、本地评估、策略部署演示和官方提交流程，为后续 T2-T4 任务迁移做基线。

## 任务信息

- 赛道：T1
- 任务：`adjust_bottle`
- 策略：ACT
- 本地演示数据：200 episodes
- 训练 seed：0
- 最佳验证损失：0.028757，出现在 epoch 5125

## 本地公开 seed 评估

评估使用官方本地评估入口，在公开 100 seed 上执行，`repeats=1`。

```json
{
  "sr": 0.52,
  "n_repeats": 1,
  "n_episodes": 100,
  "per_repeat": [0.52],
  "track": "T1"
}
```

结论：当前 checkpoint 已经跑通完整训练、评估、部署演示和提交流程，为后续 T2-T4 迁移提供了可复用 baseline。

## 策略部署演示

已录制一次成功的策略 rollout：

- seed：`20260631`
- 结果：成功
- 步数：138
- GIF：[`media/t1_policy_rollout_success_seed_20260631.gif`](../media/t1_policy_rollout_success_seed_20260631.gif)
- MP4：[`media/t1_policy_rollout_success_seed_20260631.mp4`](../media/t1_policy_rollout_success_seed_20260631.mp4)

该视频是加载训练后的 ACT checkpoint 后由策略闭环执行得到，不是专家采集视频。

## 官方提交

- 提交赛道：T1
- 提交 checkpoint：`policy_best.ckpt`
- 官方队列编号：`#70`
- 提交日期：2026-07-08

## 后续计划

- 对比不同 epoch checkpoint 的 rollout 成功率。
- 用不同训练 seed 复现实验。
- 增加更多成功 rollout 视频。
- 扩展数据量和数据覆盖度。
- 将流程迁移到 T2/T3/T4 任务。

## 未公开的本地文件

以下内容没有放入 GitHub：

- 采集到的 `.hdf5` 演示数据
- ACT processed data
- `.ckpt` checkpoint
- 本地日志
- 官方提交 token 或其他凭据
