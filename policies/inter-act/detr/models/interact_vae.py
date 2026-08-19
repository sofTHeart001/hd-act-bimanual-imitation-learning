"""InterACT-style action chunking model for Tron2.

This module adapts the InterACT-LeRobot architecture to the local RoboTwin ACT
contract. It keeps the same public forward signature as the bundled ACT model:

    forward(qpos, image, env_state, actions=None, is_pad=None)

but uses a hierarchical attention encoder and a multi-arm decoder:

    left/right qpos segments + image segment
    -> segment-wise attention
    -> cross-segment CLS attention
    -> left/right decoders
    -> synchronization self-attention
    -> left/right action heads

The official InterACT implementation assumes ALOHA 14-D actions (7+7). Tron2
uses 16-D actions, so all arm splits are parameterized by state_dim // 2.
"""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .backbone import build_backbone


def _getattr(args, name, default):
    return getattr(args, name, default)


def create_sinusoidal_pos_embedding(num_positions, dimension):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")


def reparametrize(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class InterACTEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, dim_feedforward, dropout=0.1, activation="relu", pre_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

    def forward(self, x, pos_embed=None, key_padding_mask=None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)[0]
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class InterACTDecoderLayer(nn.Module):
    def __init__(self, hidden_dim, nheads, dim_feedforward, dropout=0.1, activation="relu", pre_norm=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout)
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
        self.pre_norm = pre_norm

    @staticmethod
    def maybe_add_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, x, encoder_out, decoder_pos_embed=None, encoder_pos_embed=None):
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        x = self.self_attn(q, k, value=x)[0]
        x = skip + self.dropout1(x)

        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]
        x = skip + self.dropout2(x)

        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


class CVAEActionEncoder(nn.Module):
    """Posterior q(z | qpos, action chunk) used by ACT-style CVAE training."""

    def __init__(self, args, state_dim, num_queries):
        super().__init__()
        hidden_dim = args.hidden_dim
        nheads = args.nheads
        dim_feedforward = args.dim_feedforward
        dropout = _getattr(args, "dropout", 0.1)
        activation = _getattr(args, "feedforward_activation", "relu")
        pre_norm = _getattr(args, "pre_norm", False)
        num_layers = _getattr(args, "cvae_encoder_layers", _getattr(args, "enc_layers", 4))

        self.num_queries = num_queries
        self.latent_dim = _getattr(args, "latent_dim", 32)
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.qpos_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(state_dim, hidden_dim)
        self.layers = nn.ModuleList([
            InterACTEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation, pre_norm)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            create_sinusoidal_pos_embedding(2 + num_queries, hidden_dim),
        )

    def forward(self, qpos, actions, is_pad=None):
        batch_size = qpos.shape[0]
        actions = actions[:, :self.num_queries]
        if is_pad is None:
            is_pad = torch.zeros(batch_size, actions.shape[1], dtype=torch.bool, device=qpos.device)
        else:
            is_pad = is_pad[:, :actions.shape[1]].bool()

        cls_token = self.cls_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        qpos_token = self.qpos_proj(qpos).unsqueeze(1)
        action_tokens = self.action_proj(actions)
        encoder_input = torch.cat([cls_token, qpos_token, action_tokens], dim=1).transpose(0, 1)

        cls_qpos_pad = torch.zeros(batch_size, 2, dtype=torch.bool, device=qpos.device)
        key_padding_mask = torch.cat([cls_qpos_pad, is_pad], dim=1)
        pos = self.pos_table[:encoder_input.shape[0]].unsqueeze(1).to(
            device=qpos.device,
            dtype=encoder_input.dtype,
        )

        encoded = encoder_input
        for layer in self.layers:
            encoded = layer(encoded, pos_embed=pos, key_padding_mask=key_padding_mask)

        cls_output = self.norm(encoded[0])
        latent_info = self.latent_proj(cls_output)
        mu = latent_info[:, :self.latent_dim]
        logvar = latent_info[:, self.latent_dim:]
        return mu, logvar


class HierarchicalAttentionEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_dim = args.hidden_dim
        nheads = args.nheads
        dim_feedforward = args.dim_feedforward
        dropout = _getattr(args, "dropout", 0.1)
        activation = _getattr(args, "feedforward_activation", "relu")
        pre_norm = _getattr(args, "pre_norm", False)
        num_blocks = _getattr(args, "num_blocks", 3)

        self.num_blocks = num_blocks
        self.segment_wise_encoder = nn.ModuleList([
            InterACTEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation, pre_norm)
            for _ in range(num_blocks)
        ])
        self.cross_segment_encoder = nn.ModuleList([
            InterACTEncoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation, pre_norm)
            for _ in range(num_blocks)
        ])

        self.arm_cls = _getattr(args, "num_cls_tokens_arm", 7)
        self.image_cls = _getattr(args, "num_cls_tokens_image", 5)
    def forward(self, left_segment, right_segment, image_segment, left_pos, right_pos, image_pos, cls_pos):
        left_segment = left_segment.transpose(0, 1)
        right_segment = right_segment.transpose(0, 1)
        image_segment = image_segment.transpose(0, 1)

        left_pos = left_pos.transpose(0, 1)
        right_pos = right_pos.transpose(0, 1)
        image_pos = image_pos.transpose(0, 1)
        cls_pos = cls_pos.transpose(0, 1)

        for block_id in range(self.num_blocks):
            left_updated = self.segment_wise_encoder[block_id](left_segment, left_pos)
            right_updated = self.segment_wise_encoder[block_id](right_segment, right_pos)
            image_updated = self.segment_wise_encoder[block_id](image_segment, image_pos)

            cls_tokens = torch.cat([
                left_updated[:self.arm_cls],
                right_updated[:self.arm_cls],
                image_updated[:self.image_cls],
            ], dim=0)
            cls_tokens = self.cross_segment_encoder[block_id](cls_tokens, cls_pos)

            left_segment = torch.cat([cls_tokens[:self.arm_cls], left_updated[self.arm_cls:]], dim=0)
            right_segment = torch.cat([
                cls_tokens[self.arm_cls:2 * self.arm_cls],
                right_updated[self.arm_cls:],
            ], dim=0)
            image_segment = torch.cat([cls_tokens[2 * self.arm_cls:], image_updated[self.image_cls:]], dim=0)

        return left_segment, right_segment, image_segment, left_pos, right_pos, image_pos


class MultiArmDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_dim = args.hidden_dim
        nheads = args.nheads
        dim_feedforward = args.dim_feedforward
        dropout = _getattr(args, "dropout", 0.1)
        activation = _getattr(args, "feedforward_activation", "relu")
        pre_norm = _getattr(args, "pre_norm", False)

        pre_layers = _getattr(args, "n_pre_decoder_layers", 2)
        post_layers = _getattr(args, "n_post_decoder_layers", 2)
        sync_layers = _getattr(args, "n_sync_decoder_layers", 1)

        self.left_pre_decoder = nn.ModuleList([
            InterACTDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation, pre_norm)
            for _ in range(pre_layers)
        ])
        self.right_pre_decoder = nn.ModuleList([
            InterACTDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation, pre_norm)
            for _ in range(pre_layers)
        ])
        self.sync_block = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, nheads, dropout=dropout)
            for _ in range(sync_layers)
        ])
        self.left_post_decoder = nn.ModuleList([
            InterACTDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation, pre_norm)
            for _ in range(post_layers)
        ])
        self.right_post_decoder = nn.ModuleList([
            InterACTDecoderLayer(hidden_dim, nheads, dim_feedforward, dropout, activation, pre_norm)
            for _ in range(post_layers)
        ])
        self.left_norm = nn.LayerNorm(hidden_dim)
        self.right_norm = nn.LayerNorm(hidden_dim)

    def forward(self, decoder_input, left_context, right_context, left_pos, right_pos, decoder_pos):
        left_output = decoder_input.clone()
        right_output = decoder_input.clone()

        for layer in self.left_pre_decoder:
            left_output = layer(left_output, left_context, decoder_pos_embed=decoder_pos, encoder_pos_embed=left_pos)
        for layer in self.right_pre_decoder:
            right_output = layer(right_output, right_context, decoder_pos_embed=decoder_pos, encoder_pos_embed=right_pos)

        concatenated = torch.cat([left_output, right_output], dim=0)
        sync_pos = torch.cat([decoder_pos, decoder_pos], dim=0)
        for sync_layer in self.sync_block:
            synchronized = sync_layer(concatenated + sync_pos, concatenated + sync_pos, concatenated)[0]
            concatenated = concatenated + synchronized

        chunk_size = left_output.shape[0]
        left_output = concatenated[:chunk_size]
        right_output = concatenated[chunk_size:]

        for layer in self.left_post_decoder:
            left_output = layer(left_output, left_context, decoder_pos_embed=decoder_pos, encoder_pos_embed=left_pos)
        for layer in self.right_post_decoder:
            right_output = layer(right_output, right_context, decoder_pos_embed=decoder_pos, encoder_pos_embed=right_pos)

        return self.left_norm(left_output), self.right_norm(right_output)


class InterACTModel(nn.Module):
    def __init__(self, backbones, state_dim, num_queries, camera_names, args):
        super().__init__()
        if state_dim % 2 != 0:
            raise ValueError(f"InterACT requires an even state/action dimension, got {state_dim}.")

        self.num_queries = num_queries
        self.camera_names = camera_names
        self.state_dim = state_dim
        self.arm_dim = state_dim // 2
        self.hidden_dim = args.hidden_dim
        self.num_cls_tokens_arm = _getattr(args, "num_cls_tokens_arm", 7)
        self.num_cls_tokens_image = _getattr(args, "num_cls_tokens_image", 5)
        self.use_cvae = False
        self.latent_dim = _getattr(args, "latent_dim", 32)

        self.backbones = nn.ModuleList(backbones)
        self.image_input_proj = nn.Conv2d(backbones[0].num_channels, self.hidden_dim, kernel_size=1)
        self.camera_embed = nn.Embedding(len(camera_names), self.hidden_dim)

        self.joint_proj = nn.Linear(1, self.hidden_dim)
        self.left_cls = nn.Embedding(self.num_cls_tokens_arm, self.hidden_dim)
        self.right_cls = nn.Embedding(self.num_cls_tokens_arm, self.hidden_dim)
        self.image_cls = nn.Embedding(self.num_cls_tokens_image, self.hidden_dim)

        self.register_buffer(
            "left_pos_table",
            create_sinusoidal_pos_embedding(self.num_cls_tokens_arm + self.arm_dim, self.hidden_dim),
        )
        self.register_buffer(
            "right_pos_table",
            create_sinusoidal_pos_embedding(self.num_cls_tokens_arm + self.arm_dim, self.hidden_dim),
        )
        self.register_buffer(
            "image_cls_pos_table",
            create_sinusoidal_pos_embedding(self.num_cls_tokens_image, self.hidden_dim),
        )
        self.register_buffer(
            "cls_pos_table",
            create_sinusoidal_pos_embedding(2 * self.num_cls_tokens_arm + self.num_cls_tokens_image,
                                            self.hidden_dim),
        )

        self.encoder = HierarchicalAttentionEncoder(args)
        self.decoder = MultiArmDecoder(args)
        self.action_encoder = CVAEActionEncoder(args, state_dim, num_queries) if self.use_cvae else None
        self.latent_out_proj = nn.Linear(self.latent_dim, self.hidden_dim)
        self.decoder_pos_embed = nn.Embedding(num_queries, self.hidden_dim)
        self.left_action_head = nn.Linear(self.hidden_dim, self.arm_dim)
        self.right_action_head = nn.Linear(self.hidden_dim, self.arm_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        if self.action_encoder is not None:
            params += list(self.action_encoder.parameters())
        for p in params:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for module in [
            self.joint_proj,
            self.image_input_proj,
            self.latent_out_proj,
            self.left_action_head,
            self.right_action_head,
        ]:
            if hasattr(module, "weight") and module.weight is not None:
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, "bias") and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _make_arm_segment(self, qpos, side):
        cls_embed = self.left_cls if side == "left" else self.right_cls
        cls_tokens = cls_embed.weight.unsqueeze(0).repeat(qpos.shape[0], 1, 1)
        joint_tokens = self.joint_proj(qpos.unsqueeze(-1))
        return torch.cat([cls_tokens, joint_tokens], dim=1)

    def _make_image_segment(self, image):
        batch_size = image.shape[0]
        all_features = []
        all_pos = []

        for cam_id, _cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[0](image[:, cam_id])
            features = features[0]
            pos = pos[0].expand(features.shape[0], -1, -1, -1)
            features = self.image_input_proj(features)
            features = features + self.camera_embed.weight[cam_id].view(1, -1, 1, 1)
            all_features.append(features)
            all_pos.append(pos)

        image_features = torch.cat(all_features, dim=3)
        image_pos = torch.cat(all_pos, dim=3)
        image_tokens = image_features.flatten(2).transpose(1, 2)
        image_pos_tokens = image_pos.flatten(2).transpose(1, 2)

        cls_tokens = self.image_cls.weight.unsqueeze(0).repeat(batch_size, 1, 1)
        cls_pos = self.image_cls_pos_table.unsqueeze(0).repeat(batch_size, 1, 1).to(image.device)
        return torch.cat([cls_tokens, image_tokens], dim=1), torch.cat([cls_pos, image_pos_tokens], dim=1)

    def forward(self, qpos, image, env_state=None, actions=None, is_pad=None):
        batch_size = qpos.shape[0]
        mu = logvar = None
        latent_input = None
        if self.use_cvae:
            if actions is not None:
                mu, logvar = self.action_encoder(qpos, actions, is_pad)
                latent_sample = reparametrize(mu, logvar)
            else:
                latent_sample = torch.zeros(
                    batch_size,
                    self.latent_dim,
                    dtype=qpos.dtype,
                    device=qpos.device,
                )
            latent_input = self.latent_out_proj(latent_sample)

        left_qpos = qpos[:, :self.arm_dim]
        right_qpos = qpos[:, self.arm_dim:]

        left_segment = self._make_arm_segment(left_qpos, "left")
        right_segment = self._make_arm_segment(right_qpos, "right")
        image_segment, image_pos = self._make_image_segment(image)

        left_pos = self.left_pos_table.unsqueeze(0).repeat(batch_size, 1, 1).to(qpos.device)
        right_pos = self.right_pos_table.unsqueeze(0).repeat(batch_size, 1, 1).to(qpos.device)
        cls_pos = self.cls_pos_table.unsqueeze(0).repeat(batch_size, 1, 1).to(qpos.device)

        left_out, right_out, image_out, left_pos, right_pos, image_pos = self.encoder(
            left_segment,
            right_segment,
            image_segment,
            left_pos,
            right_pos,
            image_pos,
            cls_pos,
        )

        left_cls = left_out[:self.num_cls_tokens_arm]
        right_cls = right_out[:self.num_cls_tokens_arm]
        left_cls_pos = left_pos[:self.num_cls_tokens_arm]
        right_cls_pos = right_pos[:self.num_cls_tokens_arm]

        left_context = torch.cat([left_out, right_cls, image_out], dim=0)
        right_context = torch.cat([left_cls, right_out, image_out], dim=0)
        left_context_pos = torch.cat([left_pos, right_cls_pos, image_pos], dim=0)
        right_context_pos = torch.cat([left_cls_pos, right_pos, image_pos], dim=0)

        decoder_input = torch.zeros(
            self.num_queries,
            batch_size,
            self.hidden_dim,
            dtype=qpos.dtype,
            device=qpos.device,
        )
        if latent_input is not None:
            decoder_input = decoder_input + latent_input.unsqueeze(0)
        decoder_pos = self.decoder_pos_embed.weight.unsqueeze(1)
        left_decoded, right_decoded = self.decoder(
            decoder_input,
            left_context,
            right_context,
            left_context_pos,
            right_context_pos,
            decoder_pos,
        )

        left_decoded = left_decoded.transpose(0, 1)
        right_decoded = right_decoded.transpose(0, 1)
        left_actions = self.left_action_head(left_decoded)
        right_actions = self.right_action_head(right_decoded)
        actions_hat = torch.cat([left_actions, right_actions], dim=-1)
        return actions_hat, None, (mu, logvar)


def build(args):
    state_dim = _getattr(args, "state_dim", _getattr(args, "action_dim", 16))
    num_queries = _getattr(args, "chunk_size", _getattr(args, "num_queries", 50))
    camera_names = _getattr(args, "camera_names", ["cam_high", "cam_right_wrist", "cam_left_wrist"])

    backbone = build_backbone(args)
    model = InterACTModel(
        [backbone],
        state_dim=state_dim,
        num_queries=num_queries,
        camera_names=camera_names,
        args=args,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("InterACT number of parameters: %.2fM" % (n_parameters / 1e6, ))
    return model
