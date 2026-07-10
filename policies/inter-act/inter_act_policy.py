"""InterACT policy wrapper for the local RoboTwin ACT training loop."""

import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.nn import functional as F

try:
    from .detr.main import build_INTERACT_model_and_optimizer
except ImportError:
    from detr.main import build_INTERACT_model_and_optimizer


class InterACTPolicy(nn.Module):
    """Hierarchical attention + multi-arm decoder policy.

    The training loop expects the policy call to return a dict with a `loss`
    tensor. Inference expects a `[B, chunk_size, action_dim]` action sequence.
    """

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        model, optimizer = build_INTERACT_model_and_optimizer(args_override, RoboTwin_Config)
        self.model = model
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            a_hat, _is_pad_hat = self.model(qpos, image, env_state, actions, is_pad)
            valid = ~is_pad.unsqueeze(-1)
            all_l1 = F.l1_loss(actions, a_hat, reduction="none")
            all_l2 = F.mse_loss(actions, a_hat, reduction="none")
            l1 = (all_l1 * valid).mean()
            l2 = (all_l2 * valid).mean()
            return {
                "l1": l1,
                "l2": l2,
                "loss": l1,
            }

        a_hat, _is_pad_hat = self.model(qpos, image, env_state)
        return a_hat

    def configure_optimizers(self):
        return self.optimizer


class InterACT:
    """Deployment wrapper matching `act_policy.ACT` but using InterACTPolicy."""

    def __init__(self, args_override=None, RoboTwin_Config=None):
        if args_override is None:
            args_override = {
                "device": "cuda:0",
                "chunk_size": 50,
            }
        self.policy = InterACTPolicy(args_override, RoboTwin_Config)
        self.device = torch.device(args_override["device"])
        self.policy.to(self.device)
        self.policy.eval()

        self.temporal_agg = args_override.get("temporal_agg", False)
        self.num_queries = args_override["chunk_size"]
        self.state_dim = RoboTwin_Config.action_dim
        self.max_timesteps = 3000
        self.query_frequency = 1 if self.temporal_agg else self.num_queries
        if self.temporal_agg:
            self.all_time_actions = torch.zeros([
                self.max_timesteps,
                self.max_timesteps + self.num_queries,
                self.state_dim,
            ]).to(self.device)
            print(f"Temporal aggregation enabled with {self.num_queries} queries")
        self.t = 0

        ckpt_dir = args_override.get("ckpt_dir", "")
        if ckpt_dir:
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_path}")
            else:
                raise FileNotFoundError(
                    f"dataset_stats.pkl not found at {stats_path} -- refusing to eval without normalization."
                )

            ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
            if os.path.exists(ckpt_path):
                loading_status = self.policy.load_state_dict(torch.load(ckpt_path))
                print(f"Loaded InterACT weights from {ckpt_path}")
                print(f"Loading status: {loading_status}")
            else:
                raise FileNotFoundError(
                    f"policy checkpoint not found at {ckpt_path} -- refusing to eval a randomly initialised model."
                )
        else:
            self.stats = None

    def pre_process(self, qpos):
        if self.stats is not None:
            return (qpos - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos

    def post_process(self, action):
        if self.stats is not None:
            return action * self.stats["action_std"] + self.stats["action_mean"]
        return action

    def get_action(self, obs=None):
        if obs is None:
            return None

        qpos_numpy = np.array(obs["qpos"])
        qpos_normalized = self.pre_process(qpos_numpy)
        qpos = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        curr_images = []
        camera_names = ["head_cam", "right_cam", "left_cam"]
        for cam_name in camera_names:
            curr_images.append(obs[cam_name])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            if self.t % self.query_frequency == 0:
                self.all_actions = self.policy(qpos, curr_image)

            if self.temporal_agg:
                self.all_time_actions[[self.t], self.t:self.t + self.num_queries] = self.all_actions
                actions_for_curr_step = self.all_time_actions[:, self.t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).to(self.device).unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
            else:
                raw_action = self.all_actions[:, self.t % self.query_frequency]

        raw_action = raw_action.cpu().numpy()
        action = self.post_process(raw_action)
        self.t += 1
        return action
