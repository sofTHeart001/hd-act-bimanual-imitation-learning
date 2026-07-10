# InterACT 改造说明

本项目在原 ACT 训练框架旁新增了独立的 `policies/inter-act/` 目录，用于验证 InterACT 风格的多段注意力结构。该目录不覆盖原始 ACT baseline，方便在 T3/T4 长序列任务上做算法对比。

## 设计目标

- 保留原 ACT 的 HDF5 数据格式、三相机输入和 16 维双臂动作接口。
- 新增 InterACT 风格的层次注意力编码器和 multi-arm decoder。
- 保持训练、评估和 checkpoint 目录与 ACT baseline 隔离。
- 先实现 RGB 版本，暂不加入点云、SAC 或 PPO 后训练。

## 核心结构

InterACT 主干位于：

```text
policies/inter-act/detr/models/interact_vae.py
```

当前实现包含：

- Arm tokenization：把 16 维状态/动作按左右臂拆成 8+8，并构造左右臂 segment。
- Image segment：复用三路 RGB camera backbone 特征，加入 image CLS token 和 camera embedding。
- Hierarchical Attention Encoder：先做 segment 内注意力，再对 CLS token 做跨 segment 融合。
- Multi-Arm Decoder：左右臂分别 pre-decode，中间用 sync self-attention 交换信息，再分别 post-decode。
- Action heads：输出左右臂动作后拼回 16 维动作序列。

## 和 ACT baseline 的关系

ACT baseline 使用 CVAE/KL 训练，适合作为第一条稳定基线。InterACT 目前按论文/开源实现中的 active path 对齐，不额外引入 CVAE latent；如果后续 rollout 显示明显多模态平均动作，再把 `InterACT-CVAE` 作为独立 ablation，而不是混入 baseline。

## 适配状态

- 已对齐当前 ACT 的 HDF5 数据读取和三相机输入。
- 已保持独立 checkpoint 目录：`inter_act_ckpt/`。
- 已保留独立训练入口和部署配置。
- 已做过小 batch smoke test，确认真实 T2 HDF5 样本可前向并计算 loss。

## 后续验证计划

- 在 T2 上先做小规模训练，确认训练速度、loss 曲线和 rollout 能正常闭环。
- 如果 T3/T4 的 ACT 出现长序列协同瓶颈，再把 InterACT 作为主要候选。
- 后续记录中会对比 ACT baseline、ACT + 增强数据、InterACT 三组设置。
