# tron2_v5_DACH — Asset Package Version

## Current: 0.5.0 (2026-07-03)

**URDF source**: LimX robot-description `develop@9c8f175`（`76711a0 feat(DACH): grasper linkage`）→ `tron2/DACH_TRON2A/`；非夹爪部分沿用 `feat-tron2_v5_ros2` (2026-04-24)，逐字节不变
**Form factor**: 双臂 7DOF + 头部 2DOF + 双臂 L4 混合夹爪（物理=双 jaw prismatic 直驱 · 视觉=单驱动连杆机构 crank/bar/nut 纯跟随）
**Package role**: RoboTwin runtime；4 任务同 seed 配对专家回归 vs 旧爪 ≤5pp 过门

## Package Completeness

当前目录已包含：

- `robot.urdf`
- `robot_orig.urdf`
- `meshes/`
- `meshes_collision/`
- `asset_manifest.yml`
- `validation/tron2_pd_hold.py`
- `config.yml`
- `curobo_left.yml`
- `curobo_right.yml`
- `collision_tron2_left.yml`
- `collision_tron2_right.yml`

当前目录尚未包含 optional runtime / reference files：

- `robot.xml`
- `cameras.yml`
- `homestate_factory.yml`

因此本目录目前作为资产 / URDF / collision / manifest / runtime candidate 维护基准使用；下一步是在服务器侧注册到 RoboTwin 并跑 1 episode smoke。

## Changelog

下方 `config.yml` / `curobo_*.yml` 条目记录的是本包已经 materialize 的 runtime candidate。

### 0.5.0 → L4 连杆化夹爪迁移 (2026-07-03)

从本地简化的悬空双 prismatic 夹爪（R1，jaw 悬空平移、视觉失真）迁移到上游 robot-description `develop@9c8f175` 的连杆机构外形，采用 **L4 混合**方案（控制层"物理走连杆"三采样点均不过门 → 用户拍板走 L4）：

- **物理层 = 旧 prismatic（proven）**：`grasper_*_jaw_left/right_Joint` 恢复为旧双 jaw prismatic（parent=grasper_base、axis `0 ±1 0`、limits `[-0.0045, 0.0375]`、effort 20、k1000/c200 直驱），保留旧爪已验证的夹持物理（clamp ≥5N、190g 物体保持）。
- **视觉外形 = 新 jaw mesh**：物理 jaw link 挂上游重切 jaw STL（visual+collision），origin 加装配偏移 `O=[0, ±13.64, −116.06] mm`，把 K_B 铰点系新 mesh 摆到连杆弧中高位。
- **视觉层 = 连杆纯跟随**：crank×2 / bar×2 / nut（5 件/侧）保留为**无碰撞、近零质量（0.001 kg）**被动 link，Python 耦合列表按上游 mimic 系数（drive/crank_right/bar = 20.4115、nut = 0.4082）随 base 开合转动。
- **删除**：tip×2、jaw ref×2 关节（旧名回归物理 prismatic）；tendon / μ 补偿 patch 全部废弃（物理=旧爪，不需补偿）。
- **上游 mesh 完整性**：16 个夹爪 STL（10 新 crank/bar/nut + 6 覆盖 base/jaw）逐一 `git hash-object` == 上游 blob SHA，无透明加密污染。

**接口影响（下游）**：
- `config.yml`：夹爪耦合 = 旧 2 prismatic（base + jaw_right mimic）+ 5 视觉跟随关节；`gripper_scale [-0.0045, 0.0375]` / `gripper_bias 0.2` 不变；obs/action 16-D 归一化夹爪维零漂移（readback 2e-8）。
- `robot.py`：`gripper_name` 按"有碰撞 child"过滤 = 4 物理 jaw（视觉连杆无碰撞被排除，防 ref link 骗过接触检测）；耦合列表泛化任意长。
- `curobo_*.yml`：`lock_joints` 回旧 3/侧（base + jaw_left + jaw_right）；视觉连杆无碰撞被 cuRobo 剪枝，列出会 `KeyError`。
- `_base_task.py`：两处 `set_mass(1)` 循环加豁免谓词 `"grasper" in name and not link.get_collision_shapes()`（只命中 10 个视觉连杆件，保持 0.001 kg；jaw/base 有碰撞仍置 1 kg）——**下游若用不同 `_base_task` 需自带此豁免**。

**已知偏差**：
- 连杆-jaw 接缝穿帮：理论峰值 2.82 mm（抓取典型态 1.73 mm），弧-线本质约束；720p 渲染实测不可见（阈下）。
- 全身惯量仍沿用旧交付值（过时，未修，超本轮范围）。
- `head_yaw_Joint` / `head_pitch_Link` 惯量占位 bug 仍在（未修，超本轮范围）。
- 场景随 cuRobo 构造 numpy RNG 消耗移位 → 同 seed 场景与旧爪不同；τ / seed 若做标定须在本版重做。
- 新夹爪 visual 无 `<material>`（上游 visual 无 material）→ 渲染偏白，纯外观。

### 0.4.2 → RoboTwin smoke validated (2026-05-01)
- **RoboTwin config**:
  - `robot_pose.z` 固化为 **1.18**。`1.10` 在 SAPIEN/RoboTwin replay 中会让夹爪/手腕接触桌面并造成大 qerr；`1.21` 解决桌面接触但使左臂 place 更容易不可达。
- **Runtime patch**:
  - Tron2 `pick_dual_bottles` 默认使用 side grasp，避免左臂在较高站姿下选择 top grasp 后 place 不可达。
  - recorded replay 不再使用段末 qpos snap；只保留 terminal hold，确保视频/HDF5 是连续 controller 运动。
- **Server smoke**:
  - `pick_dual_bottles tron2_smoke_z118_side`, seed `0`, 153 frames, collector saved HDF5/video/instructions without `Collect Error`.

### 0.4.1 → runtime candidate materialized (2026-04-29)
- **Runtime files**:
  - 新增 `config.yml`
  - 新增 `curobo_left.yml` / `curobo_right.yml`
  - 新增 `collision_tron2_left.yml` / `collision_tron2_right.yml`
- **RoboTwin config**:
  - `ee_joints` / `move_group` 使用 `tcp_L/R_Joint` / `tcp_L/R_Link`
  - gripper 使用 jaw prismatic joints：base=`grasper_*_jaw_left_Joint`，mimic=`grasper_*_jaw_right_Joint`
  - `gripper_scale=[-0.0045, 0.0375]`
  - `robot_pose=[0, -0.45, 1.21, 0.707, 0, 0, 0.707]`
  - `rotate_lim=[-1.5, 1.5]`
- **Curobo config**:
  - `ee_link=tcp_*_Link`
  - `lock_joints={}`
  - `retract_config` 与 `config.yml.homestate` 一致

### 0.4.0 → snapshot for delivery (2026-04-29)
- **URDF**:
  - 添加 `tcp_L_Link` / `tcp_R_Link` 虚拟 fixed-frame ee link（rpy=`(0, π/2, 0)`），让 +x = 抓取 approach 方向，匹配 RoboTwin `get_grasp_pose` / `_trans_endpose` 硬编码约定
  - `meshes_collision/grasper_*_jaw_*_Link.STL` 用 trimesh.convex_hull 重建：单 component / watertight / ~2284 面
- **config.yml**:
  - `ee_joints` / `move_group` → `tcp_L/R_Joint` / `tcp_L/R_Link`（与 curobo `ee_link` 全链对齐）
  - `robot_pose.y` -0.65 → **-0.45**（shoulder→hammer 距离 0.768m → 0.59m，进入工作半径）
  - `rotate_lim` [0, 1] → **[-1.5, 1.5]**（给 IK 更多朝向候选）
  - `joint_stiffness=1000`, `joint_damping=200` 保持与 `tron2_pd_hold.py` baseline 一致；`10000/1000` 只作为 sim-tightness 调参候选，不作为默认交付值
  - `static_camera_list` 重对齐到 `robot.y=-0.45`、`z=1.21` 站立机器人
- **curobo_left/right.yml** (via `_tmp.yml`):
  - `ee_link`: `wrist_roll_*_Link` → `tcp_*_Link`
  - `lock_joints`: same-arm gripper subtree only（grasper_base + two jaw prismatic joints），保持 Curobo arm planner 为 7DOF
  - `retract_config` = homestate（IK seed 与启动 qpos 一致）
  - `self_collision_ignore` 扩展：`base_Link` ↔ `proximal_roll_*_Link` / `proximal_yaw_*_Link`（base 球 r=0.15 与肩关节几何重叠的 false-positive）；同侧 jaw-left/jaw-right 也互相忽略（jaw spheres 在锁定 qpos 下重合）
  - 全 ASCII（curobo loads with ascii codec）

### 0.3.0 ← previous baseline (2026-04-27)
- R1 grasper 拆 base + 双 jaw prismatic（替代旧单 revolute）
- R2 collision 改 multi-convex per link（B013 实测消除 PD shaking 根因）
- R3 5° pitch 真实斜轴用 fixed-frame sandwich 重构（FK 误差 < 0.001 mm，Curobo 看到纯 Y 轴）
- 删除误导的 `base_joints_name` 字段（base_2/3 是 fixed 不是 actuator）

## Local Modifications vs LimX Original Zip

- mesh path: `../meshes/<X>.STL` → `meshes/<X>.STL`
- 拆 grasper 为 base + jaw_left + jaw_right（R1）
- collision 用 `meshes_collision/_p<i>.STL` multi-convex（R2）
- proximal_pitch 用 R3 fixed-frame sandwich
- 加 `left_camera` / `right_camera` 虚拟 mount link（C1，HANDOFF_TO_LIMX）
- 加 `tcp_L/R_Link` 虚拟 ee frame（0.4.0）

## Inertial Properties (核心 link)

| link | mass (kg) | ixx (kg·m²) |
|------|----------|-------------|
| wrist_yaw_L_Link | 1.15 | 0.00313 |
| proximal_yaw_L_Link | 1.15 | 0.00994 |
| elbow_L_Link | 1.11 | 0.00127 |
| head_pitch_Link | 10.94 | 0.108 |

## PD 稳定性 (SAPIEN, Δt=1/250)

| 配置 | PD K/D | 1000 步 elbow drift | 备注 |
|------|--------|-------------------|------|
| 当前交付（与 aloha 同 / 与真机量级一致） | 1000/200 | ~0.50 rad | pd_hold baseline (max_range 0.46 rad) |

> 调参提示：弯肘 home 下重力漂移可通过提升 K/D 抑制（实验过 10000/1000，drift 显著降低），但属于 sim-only 调高值，迁移真机时需还原物理 PD（通常 100-500 N·m/rad）。

## MJCF

`robot.xml` 当前未随本 runtime candidate materialize。若 LimX 同步交付 MJCF，可作为 MuJoCo workflow 参考；RoboTwin 当前不依赖它。

## 下游集成快速门槛

切到本版本，需在下游：
1. 给 `head_yaw_Joint` / `head_pitch_Joint` 设 drive target / PD（RoboTwin 默认不驱动头部）
2. 决定 `head_camera` 挂点（`head_pitch_Link` 或外挂）
3. 真机部署时确认 `gripper_open_close_direction` 与 sim 是否一致（manifest 标 TBD）

---

## 验证脚本：`tron2_pd_hold.py`

本目录下随包附带的独立脚本，**不依赖 RoboTwin 任务流程**，仅做 SAPIEN PD-hold smoke 验证。来源 `recipes/tron2_pd_hold.py`，复制到此目录便于交付包独立运行。

### 用途

- ✅ 验证 URDF 在 SAPIEN 下能正确加载（无 `cleanupVertices` warning / 无 broken collision shape）
- ✅ 验证 home pose 下 PD 能稳定 hold（无 1.5+ rad 异常振荡）
- ✅ 复现 R2 collision fix 设计文档里的 B012 baseline / B013 strip-collision 对照
- ✅ 提供交互式 viewer，可手动施力测试 PD 韧性

### 用法

```bash
# 默认（无可视化、20000 物理步、K=1000 D=200 dt=1/250）
python embodiments/tron2_v5_DACH_validing/validation/tron2_pd_hold.py \
       embodiments/tron2_v5_DACH_validing/robot.urdf

# 带 viewer 实时观察
python embodiments/tron2_v5_DACH_validing/validation/tron2_pd_hold.py \
       embodiments/tron2_v5_DACH_validing/robot.urdf --view

# viewer + 实时节流 + 测试结束后保持窗口
python embodiments/tron2_v5_DACH_validing/validation/tron2_pd_hold.py \
       embodiments/tron2_v5_DACH_validing/robot.urdf --view --realtime --keep-open

# MuJoCo 风格鼠标施力测试（左键选中 link + Shift+拖拽施力）
python embodiments/tron2_v5_DACH_validing/validation/tron2_pd_hold.py \
       embodiments/tron2_v5_DACH_validing/robot.urdf --view --push-mode --push-scale 200
```

### 命令行参数

| 参数 | 默认 | 含义 |
|------|------|------|
| `urdf` (positional) | 必填 | URDF 路径 |
| `--steps` | 20000 | 物理步数（dt=1/250 → 默认 80 秒仿真） |
| `--view` | False | 启动 SAPIEN viewer 可视化 |
| `--realtime` | False | 物理按 dt 实时节流，肉眼能看到振荡 |
| `--keep-open` | False | 物理结束后 viewer 保持打开直到手动关闭 |
| `--push-mode` | False | 启用鼠标施力交互（需 `--view`） |
| `--push-scale` | 200 | 拖拽刚度 K (N/m), `F = K · |drag - anchor|` |

### 内置默认值（脚本内常量）

- **PD gain**: `K=1000.0`, `D=200.0`（与 RoboTwin `config.yml.joint_stiffness/joint_damping` 一致）
- **物理时间步**: `dt = 1/250 s`
- **Home pose**（`ARM_HOME_L/R`）：与 `config.yml.homestate` 一致
- **Gripper**: `grasper_base_L/R_Joint = 0`，4 个 jaw prismatic = 0（STL 中位）
- **Head**: `head_yaw=0`, `head_pitch=0.5743`

如需在脚本内调 K/D，修改 `main(K=..., D=...)` 默认值即可。

### 期望输出

- **正常情况（当前交付包）**：
  - SAPIEN 加载零 warning
  - 14 个 arm joint 在 4000 步内 `range < 0.5 rad`（PD baseline 验收标准）
  - `[RESULT]` 行打印 `max_range`, `zero_crossings` 等汇总指标

- **broken collision 情况（用 LimX 原 mesh 复现）**：
  - SAPIEN stderr 报 `Less than four valid vertices` / `multiple convex collision meshes ... unsupported`
  - arm joint range 1.0+ rad，可观察到 PD 大幅振荡

### 与 RoboTwin 任务的差异

`tron2_pd_hold.py` 是**直接 sapien 调用**，不走 `envs/_base_task.py` / `envs/robot/robot.py` 的封装：

| 维度 | tron2_pd_hold.py | collect_data.sh |
|------|------------------|-----------------|
| 物理引擎 | SAPIEN 直调 | RoboTwin → SAPIEN |
| 加载流程 | `URDFLoader.load` | `Robot._init_robot_` |
| PD 设置 | 脚本里 `set_drive_property(K, D)` | 走 `config.yml.joint_stiffness/damping` |
| 重力补偿 | `compute_passive_force` 每步调用 | 同 |
| 其他驱动 | 仅 `set_drive_target` | 加 `set_drive_velocity_target` + 轨迹规划 |
| 任务对象 | 无 | 加载 actor / hammer / block |
| Curobo IK | 无 | 走 `envs/robot/planner.py` |

→ **诊断时**：先跑 pd_hold 隔离 URDF / mesh / PD 问题；通过后再上 collect_data 排查 IK / 任务规划侧问题。
