#!/usr/bin/env python3
"""PD-hold smoke test for tron2 — 复现 R2_collision_fix_design.md 的 B012 baseline / B013 strip 对照.

期望 (broken collision):
    - SAPIEN stderr 报 "Less than four valid vertices" / "multiple convex"
    - arm joint range 0.1-1.7 rad，zero-crossings > 100

期望 (strip collision):
    - 无 warning
    - arm joint range ~ 0，zero-crossings = 0

用法：
    python recipes/tron2_pd_hold.py assets/embodiments/dach_tron2/robot.urdf [--steps 4000] [--view]
"""
import argparse
import math
import sys
import time
import numpy as np
import sapien


# ---------------- target home pose (来自 config.yml) ----------------
ARM_ORDER = ["proximal_pitch", "proximal_roll", "proximal_yaw",
             "elbow", "wrist_yaw", "wrist_pitch", "wrist_roll"]
ARM_HOME_L = [0.8477, 0.124, -0.1424, -2.3204, 0.0, 0.0, 0.0]
ARM_HOME_R = [0.8477, -0.124, 0.1424, -2.3204, 0.0, 0.0, 0.0]

TARGETS = {}
for base, hL, hR in zip(ARM_ORDER, ARM_HOME_L, ARM_HOME_R):
    TARGETS[f"{base}_L_Joint"] = hL
    TARGETS[f"{base}_R_Joint"] = hR
TARGETS["head_yaw_Joint"] = 0.0
TARGETS["head_pitch_Joint"] = 0.5743

# 老 grasper 单 joint (revolute, parent of jaw 拆分; 重命名为 grasper_base_<X>_Joint)
TARGETS["grasper_base_L_Joint"] = 0.0
TARGETS["grasper_base_R_Joint"] = 0.0

# 新 finger prismatic (robot_finger.urdf 走这个; 老 URDF 没有这些 joint, 跑时跳过)
# 命名规则: grasper_<手>_jaw_<左|右>_Joint
#   例 grasper_L_jaw_left_Joint  = 左手臂的左 jaw
#       grasper_R_jaw_right_Joint = 右手臂的右 jaw
# limit 通常 [-0.0045, +0.0375] (STL 中位 0 = 半开, 负值闭合 / 正值张开)
for hand in "LR":                          # 哪只手臂
    for side in ("left", "right"):         # 该手臂的左/右 jaw
        TARGETS[f"grasper_{hand}_jaw_{side}_Joint"] = 0.0   # 0 = STL 中位

ARM_NAMES = [f"{b}_{s}_Joint" for b in ARM_ORDER for s in "LR"]      # 14 arm joints
FINGER_NAMES = [f"grasper_{hand}_jaw_{side}_Joint"
                for hand in "LR" for side in ("left", "right")]      # 4 prismatic
HEAD_NAMES = ["head_yaw_Joint", "head_pitch_Joint"]
GRASPER_REV_NAMES = ["grasper_base_L_Joint", "grasper_base_R_Joint"]


def _quat_to_mat(q):
    """SAPIEN quat (w, x, y, z) → 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ])


def _find_link_for_entity(robot, entity):
    if entity is None:
        return None
    for link in robot.get_links():
        if link.entity is entity:
            return link
    return None


def _screen_to_world_ray(win):
    """屏幕鼠标位置 → 世界系射线 (eye, dir)。SAPIEN viewer 相机看 -z (OpenGL 风格)。"""
    sx, sy = win.mouse_position
    W, H = win.size
    if W == 0 or H == 0:
        return None, None
    ndc_x = (sx / W) * 2.0 - 1.0
    ndc_y = 1.0 - (sy / H) * 2.0
    fovy = win.fovy
    aspect = W / H
    ty = math.tan(fovy / 2)
    tx = ty * aspect
    # camera local 看 -z (OpenGL),y 上,x 右
    ray_cam = np.array([ndc_x * tx, ndc_y * ty, -1.0])
    ray_cam /= np.linalg.norm(ray_cam)
    cam_pose = win.get_camera_pose()
    R = _quat_to_mat(cam_pose.q)
    ray_world = R @ ray_cam
    eye = np.array(cam_pose.p)
    return eye, ray_world


def _intersect_plane(eye, ray, plane_point, plane_normal):
    """射线与平面求交; plane_normal 应非零。返回交点世界坐标或 None。"""
    denom = float(ray @ plane_normal)
    if abs(denom) < 1e-6:
        return None
    t = float((plane_point - eye) @ plane_normal / denom)
    if t <= 0:
        return None
    return eye + t * ray


def _spawn_marker(scene, color, radius=0.025):
    """创建一个无物理小球 visual marker。失败返回 None。"""
    try:
        builder = scene.create_actor_builder()
        mat = sapien.render.RenderMaterial()
        mat.set_base_color(color + [1.0])
        builder.add_sphere_visual(radius=radius, material=mat)
        return builder.build_kinematic(name="push_marker")
    except Exception as e:
        print(f"[WARN] marker spawn failed: {e}", file=sys.stderr)
        return None


def _interactive_push(scene, viewer, robot, k_per_meter, drag_state):
    """
    MuJoCo-style 拖拽施力. 每物理步调一次.

    交互:
        - 左键单击 link 选中(SAPIEN viewer 默认行为)
        - 按住 Shift → 锁定当前选中 link 为锚点,在过 anchor 且垂直相机视线
          的平面上,鼠标位置 = drag_world,实时画 anchor 球(绿) + drag 球(红)
        - 力 F = (drag_world - anchor_world) * k_per_meter
        - 拖动越远力越大;松开 Shift 自动停止 + 移除 marker
    """
    win = viewer.window
    cur_shift = win.shift

    # ---------- 状态切换 ----------
    if cur_shift and not drag_state["active"]:
        sel = viewer.selected_entity
        link = _find_link_for_entity(robot, sel)
        if link is None:
            return None
        drag_state["active"] = True
        drag_state["link"] = link
        drag_state["anchor"] = np.array(link.get_pose().p)
        drag_state["anchor_marker"] = _spawn_marker(scene, [0.1, 0.9, 0.2], radius=0.025)
        drag_state["drag_marker"] = _spawn_marker(scene, [0.95, 0.15, 0.15], radius=0.03)
        if drag_state["anchor_marker"] is not None:
            drag_state["anchor_marker"].set_pose(sapien.Pose(drag_state["anchor"].tolist()))
    elif not cur_shift and drag_state["active"]:
        # 结束拖拽,清理 marker
        for key in ("anchor_marker", "drag_marker"):
            m = drag_state.get(key)
            if m is not None:
                try:
                    scene.remove_actor(m)
                except Exception:
                    pass
        drag_state.update({"active": False, "link": None, "anchor": None,
                           "anchor_marker": None, "drag_marker": None})
        return None
    elif not cur_shift:
        return None

    # ---------- 拖拽中 ----------
    link = drag_state["link"]
    anchor = drag_state["anchor"]
    eye, ray = _screen_to_world_ray(win)
    if eye is None:
        return None

    # 平面法线 = 相机 forward(看向场景方向);OpenGL 风格 forward = -R[:,2]
    cam_pose = win.get_camera_pose()
    R = _quat_to_mat(cam_pose.q)
    plane_normal = -R[:, 2]
    drag_world = _intersect_plane(eye, ray, anchor, plane_normal)
    if drag_world is None:
        return None

    if drag_state["drag_marker"] is not None:
        drag_state["drag_marker"].set_pose(sapien.Pose(drag_world.tolist()))

    F = (drag_world - anchor) * k_per_meter
    link.add_force_at_point(F.tolist(), anchor.tolist())
    return (link.name, F, np.linalg.norm(drag_world - anchor))


def main(urdf, steps=20000, K=1000.0, D=200.0, dt=1/250, view=False,
         realtime=False, keep_open=False, push_mode=False, push_scale=2.0):
    scene = sapien.Scene()
    scene.set_timestep(dt)
    scene.add_ground(altitude=0)

    # 光照（不加这个 viewer 3D 区域全黑）
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.8, 0.8, 0.8], shadow=True)
    scene.add_point_light([1.0, 1.0, 2.0], [1.0, 1.0, 1.0])
    scene.add_point_light([-1.0, -1.0, 2.0], [1.0, 1.0, 1.0])

    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    print(f"[INFO] Loading URDF: {urdf}", file=sys.stderr)
    robot = loader.load(urdf)
    robot.set_root_pose(sapien.Pose([0, 0, 1.21], [0.707, 0, 0, 0.707]))

    active = robot.get_active_joints()
    name2idx = {j.name: i for i, j in enumerate(active)}

    # 配置 PD + 设 drive target
    qpos = robot.get_qpos().copy()
    for j in active:
        if j.name in TARGETS:
            j.set_drive_property(stiffness=K, damping=D)
            j.set_drive_target(TARGETS[j.name])
            qpos[name2idx[j.name]] = TARGETS[j.name]
    robot.set_qpos(qpos)

    # 按存在与否构造 track list (老 URDF 没 finger 时自动跳过)
    track_groups = []   # [(group_label, names_list)]
    arm_present    = [n for n in ARM_NAMES        if n in name2idx]
    finger_present = [n for n in FINGER_NAMES     if n in name2idx]
    grasper_rev    = [n for n in GRASPER_REV_NAMES if n in name2idx]
    if arm_present:    track_groups.append(("ARM",     arm_present))
    if grasper_rev:    track_groups.append(("GRASPER (revolute, 老)", grasper_rev))
    if finger_present: track_groups.append(("FINGER (prismatic)", finger_present))

    track_names = [n for _, names in track_groups for n in names]
    track_idx    = [name2idx[n] for n in track_names]
    track_target = np.array([TARGETS[n] for n in track_names])
    qpos_log = np.zeros((steps, len(track_idx)))

    viewer = None
    if view:
        viewer = scene.create_viewer()
        viewer.set_camera_xyz(2, -1, 1.5)
        viewer.set_camera_rpy(0, -0.4, 1.0)

    drag_state = {"active": False, "link": None, "anchor": None,
                  "anchor_marker": None, "drag_marker": None}

    if viewer and push_mode:
        print(f"[INFO] push 模式: 左键选中 link → 按住 Shift 拖拽 → 松 Shift 停止 "
              f"(K={push_scale:.0f} N/m, 拖 0.5m ≈ {push_scale*0.5:.0f}N)",
              file=sys.stderr)

    t0 = time.time()
    for t in range(steps):
        if viewer and push_mode:
            pushed = _interactive_push(scene, viewer, robot, push_scale, drag_state)
            if pushed is not None and t % 25 == 0:
                name, F, dist = pushed
                print(f"[push] step={t:5d}  link={name:<22s} "
                      f"|F|={np.linalg.norm(F):6.1f}N  drag={dist:.3f}m",
                      file=sys.stderr)
        scene.step()
        full_qpos = robot.get_qpos()
        qpos_log[t] = full_qpos[track_idx]
        if viewer and t % 4 == 0:
            scene.update_render()
            viewer.render()
        if realtime:
            sim_t = (t + 1) * dt
            wall_t = time.time() - t0
            if sim_t > wall_t:
                time.sleep(sim_t - wall_t)

    if viewer and keep_open:
        print("[INFO] 物理已结束，viewer 保持打开。关闭窗口或 Ctrl+C 退出。", file=sys.stderr)
        if push_mode:
            print("[INFO] keep-open 阶段仍可拖拽施力,物理仍按 dt 推进",
                  file=sys.stderr)
        while not viewer.closed:
            if push_mode:
                _interactive_push(scene, viewer, robot, push_scale, drag_state)
                scene.step()
                if realtime:
                    time.sleep(dt)
            scene.update_render()
            viewer.render()
    if viewer:
        viewer.close()

    # ---------------- 统计 (按 group 分块输出) ----------------
    print()
    print(f"[RESULT] URDF: {urdf}   steps={steps}  K={K}  D={D}  dt={dt}")
    print(f"  tracked: " + " + ".join(f"{label} ({len(names)})" for label, names in track_groups))

    rng_all, zc_all = [], []
    cursor = 0
    for label, names in track_groups:
        print()
        print(f"== {label} ==")
        print(f"{'joint':<28s} {'target':>10s} {'mean':>10s} {'range':>10s} {'std':>10s} {'zero_x':>7s}")
        print("-" * 86)
        for k, name in enumerate(names):
            col_idx = cursor + k
            col = qpos_log[:, col_idx]
            tgt = track_target[col_idx]
            rng = float(col.max() - col.min())
            sign = np.sign(col[1:] - tgt) != np.sign(col[:-1] - tgt)
            zc = int(sign.sum())
            rng_all.append(rng); zc_all.append(zc)
            print(f"{name:<28s} {tgt:>10.4f} {col.mean():>10.4f} "
                  f"{rng:>10.4f} {col.std():>10.4f} {zc:>7d}")
        cursor += len(names)

    print()
    print("=" * 86)
    print(f"{'OVERALL':<28s} max_range={np.max(rng_all):.4f}  "
          f"mean_range={np.mean(rng_all):.4f}  "
          f"max_zc={int(np.max(zc_all))}  mean_zc={np.mean(zc_all):.1f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("urdf")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--view", action="store_true")
    p.add_argument("--realtime", action="store_true",
                   help="物理按 dt 实时节流，肉眼能看到振荡")
    p.add_argument("--keep-open", action="store_true",
                   help="物理结束后 viewer 保持打开直到手动关闭")
    p.add_argument("--push-mode", action="store_true",
                   help="MuJoCo-style 鼠标施力: 左键选中 link + Shift+鼠标移动施力")
    p.add_argument("--push-scale", type=float, default=200.0,
                   help="拖拽刚度 K (N/m): F = K * |drag - anchor|; 默认 200 → 拖 0.5m 出 100N")
    args = p.parse_args()
    main(args.urdf, steps=args.steps, view=args.view,
         realtime=args.realtime, keep_open=args.keep_open,
         push_mode=args.push_mode, push_scale=args.push_scale)
