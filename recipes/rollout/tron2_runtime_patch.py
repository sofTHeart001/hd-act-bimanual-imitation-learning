"""Runtime patches for Tron2 RoboTwin collection.

These patches are intentionally kept in this hand-off repo instead of editing
the external RoboTwin checkout. They address two Tron2-specific integration
issues seen in smoke:

1. PhysX robot self-collision can disturb PD tracking between pre-grasp and
   final-grasp motions.
2. RoboTwin's default grasp selector checks only pre-grasp reachability; for
   Tron2 it can choose a pre-grasp that is reachable while the final approach
   is not, causing late-stage object contact/knockover or seed failure.
3. Some tasks emit duplicate terminal place moves when pre-place distance equals
   final-place distance. Replaying the same terminal motion can disturb a held
   object at the end of the plan.
4. PD execution can stop a little away from the planned endpoint. The next
   Curobo segment is then planned from the drifted SAPIEN qpos instead of the
   planned terminal qpos, which breaks chained pre-grasp -> final-grasp moves.
"""

from __future__ import annotations

from copy import deepcopy
import os

import numpy as np
import transforms3d as t3d


def disable_robot_self_collision() -> None:
    """Patch Robot initialization to mask collisions between robot links."""

    import envs.robot.robot as robot_module

    if getattr(robot_module.Robot, "_tron2_self_collision_patch", False):
        return

    original_init = robot_module.Robot._init_robot_

    def override_tron2_robot_pose(kwargs):
        robot_file = str(kwargs.get("left_robot_file", "")) + str(kwargs.get("right_robot_file", ""))
        if "tron2" not in robot_file.lower():
            return kwargs

        pose_updates = {
            0: os.environ.get("TRON2_ROBOT_POSE_X"),
            1: os.environ.get("TRON2_ROBOT_POSE_Y"),
            2: os.environ.get("TRON2_ROBOT_POSE_Z"),
        }
        if not any(value is not None for value in pose_updates.values()):
            return kwargs

        patched = dict(kwargs)
        for key in ("left_embodiment_config", "right_embodiment_config"):
            cfg = deepcopy(patched[key])
            robot_pose = deepcopy(cfg.get("robot_pose", [[0, -0.65, 0, 1, 0, 0, 1]]))
            for index, value in pose_updates.items():
                if value is not None:
                    robot_pose[0][index] = float(value)
            cfg["robot_pose"] = robot_pose
            patched[key] = cfg
        return patched

    def patched_init_robot(self, scene, need_topp=False, **kwargs):
        kwargs = override_tron2_robot_pose(kwargs)
        original_init(self, scene, need_topp, **kwargs)

        seen = set()
        entities = []
        for entity in (
            getattr(self, "_entity", None),
            getattr(self, "left_entity", None),
            getattr(self, "right_entity", None),
        ):
            if entity is not None and id(entity) not in seen:
                seen.add(id(entity))
                entities.append(entity)

        for entity in entities:
            for link in entity.get_links():
                shapes = link.get_collision_shapes() if hasattr(link, "get_collision_shapes") else []
                for shape in shapes:
                    groups = list(shape.get_collision_groups())
                    while len(groups) < 4:
                        groups.append(0)
                    groups[2] = groups[2] | 1
                    shape.set_collision_groups(groups)

    robot_module.Robot._init_robot_ = patched_init_robot
    robot_module.Robot._tron2_self_collision_patch = True


def retune_tron2_arm_drives() -> None:
    """Patch Robot joint initialization to optionally tighten Tron2 arm drives."""

    import envs.robot.robot as robot_module

    if getattr(robot_module.Robot, "_tron2_drive_retune_patch", False):
        return

    original_init_joints = robot_module.Robot.init_joints

    def is_tron2_robot(robot) -> bool:
        paths = [
            getattr(robot, "left_urdf_path", ""),
            getattr(robot, "right_urdf_path", ""),
        ]
        return any("tron2" in str(path).lower() for path in paths)

    def read_float(name: str, fallback: float) -> float:
        value = os.environ.get(name)
        return fallback if value is None else float(value)

    def patched_init_joints(self, *args, **kwargs):
        result = original_init_joints(self, *args, **kwargs)
        if not is_tron2_robot(self):
            return result

        stiffness = read_float("TRON2_ARM_JOINT_STIFFNESS", float(self.left_joint_stiffness))
        damping = read_float("TRON2_ARM_JOINT_DAMPING", float(self.left_joint_damping))
        force_limit = read_float("TRON2_ARM_JOINT_FORCE_LIMIT", 3.4028234663852886e38)
        mode = os.environ.get("TRON2_ARM_JOINT_DRIVE_MODE", "force")

        seen = set()
        joints = []
        for joint in list(self.left_arm_joints) + list(self.right_arm_joints):
            if joint is None or id(joint) in seen:
                continue
            seen.add(id(joint))
            joints.append(joint)

        for joint in joints:
            joint.set_drive_property(
                stiffness=stiffness,
                damping=damping,
                force_limit=force_limit,
                mode=mode,
            )

        support_targets = {
            "grasper_base_L_Joint": 0.0,
            "grasper_base_R_Joint": 0.0,
            "head_yaw_Joint": 0.0,
            "head_pitch_Joint": float(os.environ.get("TRON2_HEAD_PITCH_TARGET", "0.4008")),
        }
        support_joints = []
        lock_support = os.environ.get("TRON2_LOCK_SUPPORT_JOINTS", "1") != "0"
        if lock_support:
            active_joints = list(self.left_entity.get_active_joints())
            active_names = [joint.get_name() for joint in active_joints]
            qpos = np.array(self.left_entity.get_qpos(), dtype=np.float32)
            qvel = np.array(self.left_entity.get_qvel(), dtype=np.float32)
            for joint_name, target in support_targets.items():
                if joint_name not in active_names:
                    continue
                joint = self.left_entity.find_joint_by_name(joint_name)
                joint.set_drive_property(
                    stiffness=stiffness,
                    damping=damping,
                    force_limit=force_limit,
                    mode=mode,
                )
                joint.set_drive_target(float(target))
                joint.set_drive_velocity_target(0.0)
                joint_idx = active_names.index(joint_name)
                qpos[joint_idx] = float(target)
                qvel[joint_idx] = 0.0
                support_joints.append(joint_name)
            self.left_entity.set_qpos(qpos)
            self.left_entity.set_qvel(qvel)

        self._tron2_arm_drive_retune = {
            "stiffness": stiffness,
            "damping": damping,
            "force_limit": force_limit,
            "mode": mode,
            "joints": [joint.get_name() for joint in joints],
            "support_joints": support_joints,
        }
        return result

    robot_module.Robot.init_joints = patched_init_joints
    robot_module.Robot._tron2_drive_retune_patch = True


def require_final_grasp_reachability() -> None:
    """Patch Base_Task grasp selection to validate final approach as well."""

    from envs._GLOBAL_CONFIGS import GRASP_DIRECTION_DIC
    from envs import _base_task as base_task_module
    from envs.utils import ArmTag, cal_quat_dis

    if getattr(base_task_module.Base_Task, "_tron2_final_grasp_patch", False):
        return

    def final_pose_from_pre(pre_grasp_pose, pre_grasp_dis, target_dis):
        grasp_pose = np.array(deepcopy(pre_grasp_pose))
        direction_mat = t3d.quaternions.quat2mat(grasp_pose[-4:])
        grasp_pose[:3] += [pre_grasp_dis - target_dis, 0, 0] @ np.linalg.inv(direction_mat)
        return grasp_pose.tolist()

    def same_pose(left_pose, right_pose, atol=1e-6):
        if left_pose is None or right_pose is None:
            return False
        try:
            return np.allclose(np.array(left_pose, dtype=float), np.array(right_pose, dtype=float), atol=atol)
        except (TypeError, ValueError):
            return left_pose == right_pose

    def full_qpos_from_arm_path(robot, arm_tag, arm_qpos):
        entity = robot.left_entity if arm_tag == "left" else robot.right_entity
        arm_names = robot.left_arm_joints_name if arm_tag == "left" else robot.right_arm_joints_name
        active_names = [joint.get_name() for joint in entity.get_active_joints()]
        full_qpos = np.array(entity.get_qpos(), dtype=np.float32)
        for joint_name, joint_value in zip(arm_names, arm_qpos):
            full_qpos[active_names.index(joint_name)] = joint_value
        return full_qpos

    def snap_arm_to_path_terminal(robot, arm_tag, path_result):
        if path_result is None or path_result.get("status") != "Success":
            return
        positions = path_result.get("position")
        if positions is None or len(positions) == 0:
            return

        entity = robot.left_entity if arm_tag == "left" else robot.right_entity
        arm_names = robot.left_arm_joints_name if arm_tag == "left" else robot.right_arm_joints_name
        arm_joints = robot.left_arm_joints if arm_tag == "left" else robot.right_arm_joints
        active_names = [joint.get_name() for joint in entity.get_active_joints()]
        terminal = np.array(positions[-1], dtype=np.float32)

        qpos = np.array(entity.get_qpos(), dtype=np.float32)
        qvel = np.array(entity.get_qvel(), dtype=np.float32) if hasattr(entity, "get_qvel") else None
        for joint_name, joint_value, joint in zip(arm_names, terminal, arm_joints):
            joint_idx = active_names.index(joint_name)
            qpos[joint_idx] = joint_value
            if qvel is not None:
                qvel[joint_idx] = 0.0
            joint.set_drive_target(float(joint_value))
            joint.set_drive_velocity_target(0.0)

        entity.set_qpos(qpos)
        if qvel is not None and hasattr(entity, "set_qvel"):
            entity.set_qvel(qvel)

    def should_snap_terminal(task) -> bool:
        return not getattr(task, "save_data", False)

    def record_terminal_hold_steps(task) -> int:
        if not getattr(task, "save_data", False):
            return 0
        return int(os.environ.get("TRON2_RECORD_TERMINAL_HOLD_STEPS", "120"))

    def record_motion_substeps(task) -> int:
        if not getattr(task, "save_data", False):
            return 1
        return max(1, int(os.environ.get("TRON2_RECORD_MOTION_SUBSTEPS", "1")))

    def path_with_motion_substeps(path_result, motion_substeps: int):
        if motion_substeps <= 1 or path_result is None or path_result.get("status") != "Success":
            return path_result

        positions = path_result.get("position")
        velocities = path_result.get("velocity")
        if positions is None or len(positions) == 0:
            return path_result

        slowed = deepcopy(path_result)
        repeat_idx = np.repeat(np.arange(len(positions)), motion_substeps)
        slowed["position"] = positions[repeat_idx]
        if velocities is not None:
            slowed["velocity"] = velocities[repeat_idx] / float(motion_substeps)
        return slowed

    def path_with_terminal_hold(path_result, hold_steps: int):
        if hold_steps <= 0 or path_result is None or path_result.get("status") != "Success":
            return path_result

        positions = path_result.get("position")
        velocities = path_result.get("velocity")
        if positions is None or len(positions) == 0:
            return path_result

        padded = deepcopy(path_result)
        terminal = np.array(positions[-1:], dtype=positions.dtype)
        padded["position"] = np.concatenate(
            [positions, np.repeat(terminal, hold_steps, axis=0)],
            axis=0,
        )
        if velocities is not None:
            zero_velocity = np.zeros_like(velocities[-1:])
            padded["velocity"] = np.concatenate(
                [velocities, np.repeat(zero_velocity, hold_steps, axis=0)],
                axis=0,
            )
        return padded

    def path_with_record_modifiers(path_result, hold_steps: int, motion_substeps: int):
        path_result = path_with_motion_substeps(path_result, motion_substeps)
        return path_with_terminal_hold(path_result, hold_steps)

    def control_seq_with_record_modifiers(control_seq, hold_steps: int, motion_substeps: int):
        if hold_steps <= 0 and motion_substeps <= 1:
            return control_seq
        padded = dict(control_seq)
        padded["left_arm"] = path_with_record_modifiers(
            control_seq.get("left_arm"),
            hold_steps,
            motion_substeps,
        )
        padded["right_arm"] = path_with_record_modifiers(
            control_seq.get("right_arm"),
            hold_steps,
            motion_substeps,
        )
        return padded

    def choose_grasp_pose_checked(
        self,
        actor,
        arm_tag: ArmTag,
        pre_dis=0.1,
        target_dis=0,
        contact_point_id=None,
    ):
        if not self.plan_success:
            return None, None

        arm_tag = ArmTag(arm_tag)
        if arm_tag == "left":
            plan_multi_pose = self.robot.left_plan_multi_path
            plan_pose = self.robot.left_plan_path
            top_down_key = "top_down_little_right"
            pref_index = 0
        else:
            plan_multi_pose = self.robot.right_plan_multi_path
            plan_pose = self.robot.right_plan_path
            top_down_key = "top_down_little_left"
            pref_index = 1

        pref_direction = self.robot.get_grasp_perfect_direction(arm_tag)
        if isinstance(pref_direction, (list, tuple)):
            pref_direction = pref_direction[pref_index]

        if contact_point_id is not None:
            if type(contact_point_id) != list:
                contact_point_id = [contact_point_id]
            contact_points = [(i, None) for i in contact_point_id]
        else:
            contact_points = actor.iter_contact_points()

        best_top = (1e9, None, None)
        best_side = (1e9, None, None)
        best_weighted = (1e9, None, None)
        debug = {
            "contact_points": 0,
            "pre_success": 0,
            "exact_pre_success": 0,
            "final_success": 0,
        }

        for point_id, _ in contact_points:
            debug["contact_points"] += 1
            contact_matrix = actor.get_contact_point(point_id, "matrix")
            if contact_matrix is None:
                continue

            grasp_frame = contact_matrix @ np.array(
                [[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
            )
            grasp_rot = grasp_frame[:3, :3]
            pre_pos = grasp_frame[:3, 3] + grasp_rot @ np.array([-0.12 - pre_dis, 0, 0]).T
            pre_pose = list(pre_pos) + list(t3d.quaternions.mat2quat(grasp_rot))

            rotated_pre_poses = self.robot.create_target_pose_list(
                pre_pose,
                actor.get_contact_point(point_id, "list"),
                arm_tag,
            )
            pre_paths = plan_multi_pose(rotated_pre_poses)
            statuses = pre_paths.get("status", [])

            for pose_idx, rotated_pre_pose in enumerate(rotated_pre_poses):
                if pose_idx >= len(statuses) or statuses[pose_idx] != "Success":
                    continue
                debug["pre_success"] += 1
                pre_path = plan_pose(rotated_pre_pose)
                if pre_path.get("status") != "Success":
                    continue
                debug["exact_pre_success"] += 1
                pre_terminal_qpos = pre_path["position"][-1]
                pre_terminal_qpos = full_qpos_from_arm_path(self.robot, str(arm_tag), pre_terminal_qpos)

                final_pose = final_pose_from_pre(rotated_pre_pose, pre_dis, target_dis)
                final_path = plan_pose(
                    final_pose,
                    constraint_pose=[1, 1, 1, 0, 0, 0],
                    last_qpos=pre_terminal_qpos,
                )
                if final_path.get("status") != "Success":
                    continue
                debug["final_success"] += 1

                top_dist = cal_quat_dis(final_pose[-4:], GRASP_DIRECTION_DIC[top_down_key])
                side_dist = cal_quat_dis(final_pose[-4:], GRASP_DIRECTION_DIC[pref_direction])
                weighted = 0.7 * top_dist + 0.3 * side_dist

                if top_dist < best_top[0]:
                    best_top = (top_dist, rotated_pre_pose, final_pose)
                if side_dist < best_side[0]:
                    best_side = (side_dist, rotated_pre_pose, final_pose)
                if weighted < best_weighted[0]:
                    best_weighted = (weighted, rotated_pre_pose, final_pose)

        if not hasattr(self, "_tron2_last_grasp_debug"):
            self._tron2_last_grasp_debug = {}
        grasp_preference = os.environ.get("TRON2_GRASP_PREFERENCE")
        if grasp_preference is None:
            robot = getattr(self, "robot", None)
            left_urdf = str(getattr(robot, "left_urdf_path", "")).lower()
            if getattr(self, "task_name", "") in ("pick_dual_bottles", "lift_pot") and "tron2" in left_urdf:
                grasp_preference = "side"   # both tasks use side grasps (bottle ends / pot handles)
            else:
                grasp_preference = "auto"
        if grasp_preference == "side" and best_side[1] is not None:
            debug["choice"] = "side"
            self._tron2_last_grasp_debug[str(arm_tag)] = debug
            return best_side[1], best_side[2]
        if best_top[0] < 0.15:
            debug["choice"] = "top"
            self._tron2_last_grasp_debug[str(arm_tag)] = debug
            return best_top[1], best_top[2]
        if best_side[0] < 0.15:
            debug["choice"] = "side"
            self._tron2_last_grasp_debug[str(arm_tag)] = debug
            return best_side[1], best_side[2]
        debug["choice"] = "weighted" if best_weighted[1] is not None else "none"
        self._tron2_last_grasp_debug[str(arm_tag)] = debug
        return best_weighted[1], best_weighted[2]

    original_grasp_actor = base_task_module.Base_Task.grasp_actor
    original_place_actor = base_task_module.Base_Task.place_actor
    original_take_dense_action = base_task_module.Base_Task.take_dense_action
    original_together_move_to_pose = base_task_module.Base_Task.together_move_to_pose

    def grasp_actor_guarded(self, *args, **kwargs):
        try:
            arm_tag, actions = original_grasp_actor(self, *args, **kwargs)
        except AssertionError as exc:
            if "target_pose cannot be None" not in str(exc):
                raise
            self.plan_success = False
            arm_tag = args[1] if len(args) > 1 else kwargs.get("arm_tag")
            return arm_tag, []

        if actions and any(
            getattr(action, "action", None) == "move" and getattr(action, "target_pose", None) is None
            for action in actions
        ):
            self.plan_success = False
            return arm_tag, []
        return arm_tag, actions

    def place_actor_guarded(self, *args, **kwargs):
        arm_tag, actions = original_place_actor(self, *args, **kwargs)
        if not actions:
            return arm_tag, actions

        compacted = []
        dropped = 0
        for action in actions:
            if (
                compacted
                and getattr(action, "action", None) == "move"
                and getattr(compacted[-1], "action", None) == "move"
                and same_pose(getattr(action, "target_pose", None), getattr(compacted[-1], "target_pose", None))
            ):
                dropped += 1
                continue
            compacted.append(action)
        if dropped:
            self._tron2_last_place_compaction = getattr(self, "_tron2_last_place_compaction", 0) + dropped
        return arm_tag, compacted

    def take_dense_action_snapped(self, control_seq, *args, **kwargs):
        control_seq = control_seq_with_record_modifiers(
            control_seq,
            record_terminal_hold_steps(self),
            record_motion_substeps(self),
        )
        result = original_take_dense_action(self, control_seq, *args, **kwargs)
        if should_snap_terminal(self):
            snap_arm_to_path_terminal(self.robot, "left", control_seq.get("left_arm"))
            snap_arm_to_path_terminal(self.robot, "right", control_seq.get("right_arm"))
        return result

    def together_move_to_pose_snapped(self, *args, **kwargs):
        left_path_len = len(getattr(self, "left_joint_path", []))
        right_path_len = len(getattr(self, "right_joint_path", []))
        left_cnt = getattr(self, "left_cnt", 0)
        right_cnt = getattr(self, "right_cnt", 0)

        hold_steps = record_terminal_hold_steps(self)
        motion_substeps = record_motion_substeps(self)
        restore_left = None
        restore_right = None
        if (hold_steps or motion_substeps > 1) and not getattr(self, "need_plan", True):
            if left_cnt < len(getattr(self, "left_joint_path", [])):
                restore_left = self.left_joint_path[left_cnt]
                self.left_joint_path[left_cnt] = path_with_record_modifiers(
                    restore_left,
                    hold_steps,
                    motion_substeps,
                )
            if right_cnt < len(getattr(self, "right_joint_path", [])):
                restore_right = self.right_joint_path[right_cnt]
                self.right_joint_path[right_cnt] = path_with_record_modifiers(
                    restore_right,
                    hold_steps,
                    motion_substeps,
                )

        try:
            result = original_together_move_to_pose(self, *args, **kwargs)
        finally:
            if restore_left is not None:
                self.left_joint_path[left_cnt] = restore_left
            if restore_right is not None:
                self.right_joint_path[right_cnt] = restore_right

        left_path = None
        right_path = None
        if getattr(self, "need_plan", True):
            if len(getattr(self, "left_joint_path", [])) > left_path_len:
                left_path = self.left_joint_path[-1]
            if len(getattr(self, "right_joint_path", [])) > right_path_len:
                right_path = self.right_joint_path[-1]
        else:
            if getattr(self, "left_cnt", 0) > left_cnt:
                left_path = self.left_joint_path[self.left_cnt - 1]
            if getattr(self, "right_cnt", 0) > right_cnt:
                right_path = self.right_joint_path[self.right_cnt - 1]

        if should_snap_terminal(self):
            snap_arm_to_path_terminal(self.robot, "left", left_path)
            snap_arm_to_path_terminal(self.robot, "right", right_path)
        return result

    base_task_module.Base_Task.choose_grasp_pose = choose_grasp_pose_checked
    base_task_module.Base_Task.grasp_actor = grasp_actor_guarded
    base_task_module.Base_Task.place_actor = place_actor_guarded
    base_task_module.Base_Task.take_dense_action = take_dense_action_snapped
    base_task_module.Base_Task.together_move_to_pose = together_move_to_pose_snapped
    base_task_module.Base_Task._tron2_final_grasp_patch = True


def allow_pick_dual_target_overrides() -> None:
    """Allow project-local target sweeps for pick_dual_bottles."""

    import envs.pick_dual_bottles as pick_dual_module

    task_cls = pick_dual_module.pick_dual_bottles
    if getattr(task_cls, "_tron2_target_override_patch", False):
        return

    original_load_actors = task_cls.load_actors

    def read_pose(prefix: str, default_pose: list[float]) -> list[float]:
        pose = list(default_pose)
        updates = {
            0: os.environ.get(f"{prefix}_X"),
            1: os.environ.get(f"{prefix}_Y"),
            2: os.environ.get(f"{prefix}_Z"),
        }
        for index, value in updates.items():
            if value is not None:
                pose[index] = float(value)
        return pose

    def patched_load_actors(self, *args, **kwargs):
        result = original_load_actors(self, *args, **kwargs)
        self.left_target_pose = read_pose("TRON2_LEFT_TARGET", self.left_target_pose)
        self.right_target_pose = read_pose("TRON2_RIGHT_TARGET", self.right_target_pose)
        return result

    task_cls.load_actors = patched_load_actors
    task_cls._tron2_target_override_patch = True


def apply_tron2_runtime_patches() -> None:
    disable_robot_self_collision()
    retune_tron2_arm_drives()
    require_final_grasp_reachability()
    allow_pick_dual_target_overrides()
