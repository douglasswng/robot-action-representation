"""ARBeT (Auto-Regressive Behaviour Transformer) policy implementation.

Converts integer (x,y) push actions into BPE-compressed chain-code token sequences
(via ScribeTokenizer) and predicts them autoregressively with cross-entropy loss,
using an AdaLN transformer decoder conditioned on vision and proprioceptive features.

Architecture:
    - Vision encoder: DiffusionRgbEncoder (ResNet18-based, shared with DiTFlow)
    - Conditioning: image features + proprioceptive state → projected to hidden_dim
    - Token embedding: nn.Embedding(vocab_size, hidden_dim)
    - Decoder: causal AdaLNTransformerDecoder (shift-scale modulation from conditioning)
    - Output head: AdaLNFinalLayer → logits over token vocabulary
"""

from __future__ import annotations

from collections import deque

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
)

from adaln_transformer import AdaLNFinalLayer, AdaLNTransformerDecoder
from lerobot_policy_arbet.configuration_arbet import ARBeTConfig
from scribe_tokenizer import ScribeTokenizer


class ARBeTPolicy(PreTrainedPolicy):
    """ARBeT policy: autoregressive token prediction over ScribeToken sequences.

    Training: teacher-forced next-token prediction with cross-entropy loss.
    Inference: autoregressive sampling → detokenize to (x,y) action coordinates.
    """

    config_class = ARBeTConfig  # pyright: ignore[reportAssignmentType] — parent declares as None
    name = "ARBeT"  # pyright: ignore[reportAssignmentType] — parent declares as None

    def __init__(
        self,
        config: ARBeTConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        **kwargs,
    ):
        super().__init__(config)
        config.validate_features()
        self.config = config

        self._queues: dict[str, deque] = {}
        self.arbet = ARBeTModel(config)
        self.tokenizer = ScribeTokenizer(bpe_vocab_size=config.bpe_vocab_size)

        # Store state min/max stats as buffers for unnormalizing state back to raw
        # pixel coords when using it as the starting position for trajectory decoding.
        if dataset_stats and OBS_STATE in dataset_stats:
            state_min = dataset_stats[OBS_STATE]["min"]
            state_max = dataset_stats[OBS_STATE]["max"]
        else:
            state_min = torch.zeros(2)
            state_max = torch.ones(2)
        self.state_min: torch.Tensor
        self.state_max: torch.Tensor
        self.register_buffer("state_min", torch.as_tensor(state_min, dtype=torch.float32))
        self.register_buffer("state_max", torch.as_tensor(state_max, dtype=torch.float32))

        self.reset()

    def get_optim_params(self) -> dict:
        return self.arbet.parameters()  # pyright: ignore[reportReturnType] — parent annotation is wrong; callers expect Iterator[Parameter]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`."""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    def unnormalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Reverse MIN_MAX normalization: [-1, 1] → [min, max] in raw pixel coords."""
        return (state + 1) / 2 * (self.state_max - self.state_min) + self.state_min

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Generate an action chunk from queued observations.

        Returns:
            (B, n_action_steps, 2) tensor of action waypoints.
        """
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        # Recover raw pixel-space state for use as the trajectory origin.
        raw_state = self.unnormalize_state(batch[OBS_STATE][:, -1, :2])
        start_positions = raw_state.round().long()
        actions = self.arbet.generate_actions(
            batch,
            self.tokenizer,
            self.config.temperature,
            start_positions,
        )
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs) -> torch.Tensor:
        """Select a single action given environment observations.

        Generates n_action_steps tokens worth of waypoints and queues them.
        Actions are popped one at a time; a new chunk is generated when the queue is empty.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            # actions: (B, n_action_steps, 2) → queue expects (n_action_steps, B, 2)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.arbet.compute_loss(batch)
        return loss, None


class ARBeTModel(nn.Module):
    """Core ARBeT model: vision encoder + causal AdaLN transformer + token output head."""

    def __init__(self, config: ARBeTConfig):
        super().__init__()
        self.config = config

        # ---- Observation encoders (matched with DiTFlow) ----
        if config.use_proprioceptive:
            assert config.robot_state_feature is not None, "robot_state_feature required when use_proprioceptive=True"
            global_cond_dim = config.robot_state_feature.shape[0]
        else:
            global_cond_dim = 0

        if config.image_features:
            num_images = len(config.image_features)
            # DiffusionRgbEncoder's annotation requires DiffusionConfig but only reads
            # vision-related fields that ARBeTConfig also declares (both extend PreTrainedConfig).
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]  # pyright: ignore[reportArgumentType]
                self.rgb_encoder: nn.ModuleList | DiffusionRgbEncoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)  # pyright: ignore[reportArgumentType]
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        self.global_cond_dim = global_cond_dim

        # ---- Conditioning projection ----
        self.cond_proj = nn.Linear(global_cond_dim * config.n_obs_steps, config.hidden_dim)

        # ---- Token embedding ----
        # ScribeTokenizer's actual vocab_size may differ slightly from bpe_vocab_size
        # (due to remapping), so we use bpe_vocab_size as an upper bound.
        self.token_embedding = nn.Embedding(config.bpe_vocab_size, config.hidden_dim)

        # ---- Causal transformer decoder ----
        self.transformer = AdaLNTransformerDecoder(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            num_layers=config.num_blocks,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            causal=True,
            max_seq_len=config.horizon,
        )

        # ---- Output head: logits over token vocabulary ----
        self.output_head = AdaLNFinalLayer(config.hidden_dim, config.bpe_vocab_size)

        print(f"Number of ARBeT params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode image features and concatenate with state vector."""
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                assert isinstance(self.rgb_encoder, nn.ModuleList)
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)]
                )
                img_features = einops.rearrange(
                    img_features_list,
                    "(n b s) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            else:
                img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
                img_features = einops.rearrange(
                    img_features,
                    "(b s n) ... -> b s (n ...)",
                    b=batch_size,
                    s=n_obs_steps,
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        # Concatenate features then flatten to (B, global_cond_dim * n_obs_steps).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute cross-entropy loss for next-token prediction.

        Expects batch to contain:
            - observation.state: (B, n_obs_steps, state_dim)
            - observation.images: (B, n_obs_steps, num_cameras, C, H, W) and/or observation.environment_state
            - action_tokens: (B, horizon) int64 — [BOS, ...tokens..., EOS/PAD...]
            - action_tokens_mask: (B, horizon) bool — True for real tokens, False for PAD
        """
        assert set(batch).issuperset({"observation.state", "action_tokens", "action_tokens_mask"})
        assert "observation.images" in batch or "observation.environment_state" in batch

        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim * n_obs_steps)
        cond = self.cond_proj(global_cond)  # (B, hidden_dim)

        tokens = batch["action_tokens"]  # (B, H)
        mask = batch["action_tokens_mask"]  # (B, H)

        # Teacher-forced next-token prediction:
        #   input:  [BOS, t1, t2, ..., t_{H-2}]  (all but last)
        #   target: [t1,  t2, t3, ..., t_{H-1}]  (all but first)
        input_tokens = tokens[:, :-1]  # (B, H-1)
        target_tokens = tokens[:, 1:]  # (B, H-1)
        target_mask = mask[:, 1:]  # (B, H-1)

        # Embed tokens: (B, H-1) → (B, H-1, D) → (H-1, B, D) for T-first transformer
        x = self.token_embedding(input_tokens).transpose(0, 1)

        # Padding mask for attention: True = ignore position
        input_mask = mask[:, :-1]
        key_padding_mask = ~input_mask

        hidden = self.transformer(x, cond, key_padding_mask=key_padding_mask)  # (H-1, B, D)
        logits = self.output_head(hidden, cond)  # (H-1, B, vocab_size)
        logits = logits.transpose(0, 1)  # (B, H-1, vocab_size)

        # Cross-entropy loss
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_tokens.reshape(-1),
            reduction="none",
        )
        loss = loss.view(target_tokens.shape)  # (B, H-1)

        if self.config.do_mask_loss_for_padding:
            loss = loss * target_mask.float()
            loss = loss.sum() / target_mask.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def generate_actions(
        self,
        batch: dict[str, torch.Tensor],
        tokenizer: ScribeTokenizer,
        temperature: float = 1.0,
        start_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressively generate tokens, then decode to (x,y) action coordinates.

        Args:
            batch: Observation batch with stacked obs history.
            tokenizer: ScribeTokenizer for encoding/decoding.
            temperature: Sampling temperature (1.0 = unmodified logits).
            start_positions: (B, 2) int64 raw pixel-space starting positions for trajectory
                decoding. Required when state is normalized (MIN_MAX). If None, falls back
                to extracting directly from batch (assumes unnormalized state).

        Returns:
            (B, n_action_steps, 2) absolute (x,y) action tensor.
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        batch_size = batch[OBS_STATE].shape[0]
        global_cond = self._prepare_global_conditioning(batch)
        cond = self.cond_proj(global_cond)  # (B, hidden_dim)

        # Start with BOS token
        generated = torch.full((batch_size, 1), tokenizer.BOS_ID, dtype=torch.long, device=device)

        n_token_steps = self.config.n_token_steps
        for _ in range(n_token_steps):
            x = self.token_embedding(generated).transpose(0, 1)  # (T, B, D)
            hidden = self.transformer(x, cond)  # (T, B, D)
            logits = self.output_head(hidden, cond)  # (T, B, vocab_size)

            # Sample from last position
            next_logits = logits[-1]  # (B, vocab_size)
            if temperature != 1.0:
                next_logits = next_logits / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

        # Decode content tokens into a dense (x,y) path,
        # then evenly subsample to n_action_steps waypoints for execution.
        if start_positions is None:
            state = batch[OBS_STATE][:, -1, :2]  # (B, 2) — last observed agent position
            start_positions = state.round().long()

        n_action_steps = self.config.n_action_steps
        all_actions = []

        for b in range(batch_size):
            token_ids = generated[b].tolist()

            # Truncate at EOS if present.
            if tokenizer.EOS_ID in token_ids:
                token_ids = token_ids[: token_ids.index(tokenizer.EOS_ID) + 1]

            try:
                trajectory = tokenizer.decode(token_ids)  # origin-relative dense (x, y) points
            except Exception:
                trajectory = []

            origin = start_positions[b].float()

            if len(trajectory) <= 1:
                actions = origin.unsqueeze(0).expand(n_action_steps, -1)
            else:
                abs_traj = torch.tensor(trajectory[1:], dtype=dtype, device=device) + origin  # (L, 2)
                if len(abs_traj) <= n_action_steps:
                    pad = abs_traj[-1:].expand(n_action_steps - len(abs_traj), -1)
                    actions = torch.cat([abs_traj, pad], dim=0)
                else:
                    # Left-exclusive subsample: skip near-origin points, include endpoint.
                    # For 3 from 9: indices at positions 3, 6, 9 (i.e. --*--*--*)
                    L = len(abs_traj)
                    step = L / n_action_steps
                    indices = (torch.arange(1, n_action_steps + 1) * step - 1).long()
                    indices = indices.clamp(0, L - 1)
                    actions = abs_traj[indices]

            all_actions.append(actions)

        return torch.stack(all_actions, dim=0)  # (B, n_action_steps, 2)
