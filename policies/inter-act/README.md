# inter-act

This folder is the isolated preparation area for an InterACT-style policy built on top of the bundled ACT implementation.

It intentionally does not replace `policy/ACT`. The first version keeps the ACT data format, training loop, and deployment contract, then adds InterACT-specific code incrementally.

## Current status

- Baseline ACT files were copied without checkpoints, processed data, or cache files.
- `InterACTPolicy` now builds an RGB InterACT-style model with hierarchical attention and a multi-arm decoder.
- Training and eval scripts point to `inter_act_ckpt/` so experiments do not overwrite ACT checkpoints.
- The module is kept as a standalone policy folder so it can be copied back under RoboTwin `policy/` for training and evaluation.

## Migration plan

1. Keep ACT data processing unchanged.
2. Train/evaluate RGB InterACT on T3/T4.
3. Compare ACT vs InterACT failure cases.
4. Add point-cloud token support after RGB InterACT is validated.
5. Keep point cloud and RL residual out of scope unless the project direction changes.

## Tron2 shape contract

```text
qpos/action: 16
left arm + gripper:  0:8
right arm + gripper: 8:16
```

The official InterACT-LeRobot implementation assumes ALOHA `14 = 7 + 7`, so all imported ideas must be parameterized with `arm_dim = state_dim // 2`.
