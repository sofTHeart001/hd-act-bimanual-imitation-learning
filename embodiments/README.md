# Tron2 Embodiments

本目录放置 Tron2 接入 RoboTwin 的本体资产、运行配置和资产需求文档。

顶层只保留当前可被运行代码直接使用的 embodiment。历史本体统一放入 `archiv/`，运行脚本原则上不得读取 `archiv/` 内部内容。

## Active Maintained Version

**今后本项目的 Tron2 v5 DACH 资产 / URDF 讨论默认使用：**

```text
tron2_v5_DACH_validing/
```

该目录是当前维护中的 v5 DACH asset package，版本见 `tron2_v5_DACH_validing/VERSION.md`，机器可读语义见 `tron2_v5_DACH_validing/asset_manifest.yml`。

已确认吸收的 redesign：

- R1：grasper 拆分为 base + left/right jaw，jaw 使用显式 `prismatic` joints；URDF 不使用 `<mimic>`，RoboTwin coupling 在 config 层表达。
- R2：collision 改为 `meshes_collision/` multi-convex / cleaned jaw collision。
- R3：5 deg shoulder pitch 斜轴用 fixed-frame sandwich 表达，Curobo 看到纯 Y revolute axis。
- R7：新增 `asset_manifest.yml` 和 `validation/tron2_pd_hold.py`。

## Current Status

`tron2_v5_DACH_validing/` 当前是 **RoboTwin runtime candidate**。资产验证已经完成到可作为维护基准；runtime 文件已经 materialize，下一步是在服务器跑 `collect_data` smoke。检查结果：

- `robot.urdf` XML parse 通过。
- URDF 中 mesh / collision mesh 引用均能在该目录内找到。
- URDF joint 结构包含 18 个 revolute、4 个 prismatic、12 个 fixed joint。
- 目录内有 `robot.urdf`、`robot_orig.urdf`、`VERSION.md`、`asset_manifest.yml`、`validation/tron2_pd_hold.py`。
- 目录内已有 RoboTwin runtime candidate 文件：`config.yml`、`curobo_left.yml`、`curobo_right.yml`、`collision_tron2_left.yml`、`collision_tron2_right.yml`。
- 目录内也缺少 manifest 中提到的 optional files：`robot.xml`、`cameras.yml`、`homestate_factory.yml`。

因此：

- 资产 / URDF / mesh / manifest 审查：使用 `tron2_v5_DACH_validing/`。
- RoboTwin `collect_data.sh` 运行：使用 `recipes/rollout/register_embodiment.sh` 注册后跑 `tron2_smoke`。
- 不再用 `archiv/tron2_v5_DACH/robot.urdf` 作为新设计判断依据；它只保留 requirements docs 和历史 runtime 配置。

## Directory Roles

| Directory | Role | Use For New Work |
|---|---|---|
| `tron2_v5_DACH_validing/` | current maintained v5 DACH asset package | Yes, asset source of truth |
| `archiv/` | archived historical embodiments | No, do not use as runtime input |

Archived contents:

- `archiv/tron2/`: v4 DA_TRON2A historical baseline.
- `archiv/tron2_v5_DA/`: v5 DA_TRON2A without head.
- `archiv/tron2_v5_DACH/`: historical v5 DACH extraction plus formal asset requirements.

## Runtime Candidate

Runtime files now present in `tron2_v5_DACH_validing/`:

```text
config.yml
curobo_left.yml
curobo_right.yml
collision_tron2_left.yml
collision_tron2_right.yml
```

These files must follow `tron2_v5_DACH_validing/asset_manifest.yml`:

- `move_group`: `tcp_L_Link`, `tcp_R_Link`
- `ee_joints`: `tcp_L_Joint`, `tcp_R_Joint`
- `gripper_name.base`: `grasper_*_jaw_left_Joint`
- `gripper_name.mimic`: `grasper_*_jaw_right_Joint`
- `gripper_scale`: `[-0.0045, 0.0375]`
- Curobo `ee_link`: `tcp_*_Link`

Pending before 50-episode collection:

- Run `validation/tron2_pd_hold.py` in the server RoboTwin/SAPIEN env.
- Run `collect_data.sh pick_dual_bottles tron2_smoke 0`.
- If Curobo reports IK failure, debug home pose / wrist pitch limit / task contact point before scaling.
