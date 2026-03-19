"""ARBeT: Auto-Regressive Behaviour Transformer.

Predicts robot actions as sequences of BPE-compressed chain-code tokens,
decoded autoregressively with cross-entropy loss. Mirrors the structure of
DiTFlowPolicy / DiTFlowModel but replaces the diffusion noise net with a
causal AdaLN transformer over discrete ScribeTokens.
"""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from lerobot.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import populate_queues
from lerobot.utils.constants import (
    ACTION,
    OBS_ENV_STATE,
    OBS_IMAGES,
    OBS_STATE,
)

from adaln_transformer import AdaLNFinalLayer, AdaLNTransformerDecoder
from lerobot_policy_arbet.configuration_arbet import ARBeTConfig
from scribe_tokenizer import ScribeTokenizer

if TYPE_CHECKING:
    from typing import Unpack

    from lerobot.policies.pretrained import ActionSelectKwargs


class ARBeTPolicy(PreTrainedPolicy):
    """Auto-Regressive Behaviour Transformer policy.

    Converts (x,y) action trajectories into BPE-compressed chain-code token sequences
    and predicts them autoregressively using a causal transformer with AdaLN conditioning.
    """

    config_class = ARBeTConfig  # type: ignore[assignment]
    name = "ARBeT"  # type: ignore[assignment]

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
        self.reset()

    def get_optim_params(self) -> dict:
        return self.arbet.parameters()  # type: ignore[return-value]

    def reset(self):
        """Clear observation and action queues. Called on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues[OBS_ENV_STATE] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, torch.Tensor], **kwargs: Unpack[ActionSelectKwargs]
    ) -> torch.Tensor:
        """Generate a chunk of actions from buffered observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.arbet.generate_actions(batch)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> torch.Tensor:
        """Select a single action, managing the observation history and action queue.

        Same caching logic as DiTFlowPolicy: predicts horizon steps, executes n_action_steps,
        then re-predicts when the action queue is depleted.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict | None]:
        """Run the batch through the model and compute the cross-entropy loss."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack([batch[key] for key in self.config.image_features], dim=-4)
        loss = self.arbet.compute_loss(batch)
        return loss, None


class ARBeTModel(nn.Module):
    """Inner model: vision encoder + ScribeTokenizer + causal AdaLN transformer.

    Mirrors DiTFlowModel but replaces the continuous noise net with discrete
    autoregressive token prediction.
    """

    def __init__(self, config: ARBeTConfig):
        super().__init__()
        self.config = config

        # ---- Tokenizer (not a nn.Module, lives on CPU) ----
        self.tokenizer = ScribeTokenizer(bpe_vocab_size=config.bpe_vocab_size)

        # ---- Observation encoders (identical to DiTFlow) ----
        global_cond_dim = 0
        if config.use_proprioceptive:
            assert config.robot_state_feature is not None
            global_cond_dim = config.robot_state_feature.shape[0]

        if config.image_features:
            num_images = len(config.image_features)
            if config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]  # type: ignore[arg-type]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)  # type: ignore[arg-type]
                global_cond_dim += self.rgb_encoder.feature_dim * num_images

        if config.env_state_feature:
            global_cond_dim += config.env_state_feature.shape[0]

        self.global_cond_dim = global_cond_dim

        # ---- Projection: flat global conditioning → transformer hidden dim ----
        self.cond_proj = nn.Linear(global_cond_dim * config.n_obs_steps, config.hidden_dim)

        # ---- Token embedding ----
        self.token_embedding = nn.Embedding(self.tokenizer.vocab_size, config.hidden_dim)

        # ---- Causal transformer with AdaLN conditioning ----
        self.transformer = AdaLNTransformerDecoder(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            num_layers=config.num_blocks,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            causal=True,
        )

        # ---- Output head: AdaLN + linear → vocab logits ----
        self.output_head = AdaLNFinalLayer(config.hidden_dim, self.tokenizer.vocab_size)

        print(f"Number of ARBeT params: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M")

    # ------------------------------------------------------------------
    # Observation encoding (verbatim from DiTFlowModel)
    # ------------------------------------------------------------------

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode image features and concatenate them with the state vector.

        Returns:
            (B, global_cond_dim * n_obs_steps) flat conditioning vector.
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        global_cond_feats = [batch[OBS_STATE]]

        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(batch[OBS_IMAGES], "b s n ... -> n (b s) ...")
                assert isinstance(self.rgb_encoder, nn.ModuleList)
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                assert isinstance(self.rgb_encoder, DiffusionRgbEncoder)
                img_features = self.rgb_encoder(einops.rearrange(batch[OBS_IMAGES], "b s n ... -> (b s n) ..."))
                img_features = einops.rearrange(img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps)
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV_STATE])

        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    # ------------------------------------------------------------------
    # Tokenization helpers
    # ------------------------------------------------------------------

    def _tokenize_actions(
        self,
        actions: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert (B, horizon, 2) float actions to padded token-ID tensors.

        When ``action_is_pad`` is provided, only non-padded (in-episode) action
        steps are tokenized so the model never trains on meaningless padding tokens.

        Returns:
            token_ids:    (B, S) long tensor, PAD_ID in unused positions.
            padding_mask: (B, S) bool tensor — True where token_ids is padding.
        """
        batch_size, horizon, _ = actions.shape
        actions_int = actions.round().long()

        all_tokens: list[list[int]] = []
        for i in range(batch_size):
            valid_len = horizon if action_is_pad is None else int((~action_is_pad[i]).sum().item())
            trajectory = [(int(actions_int[i, t, 0]), int(actions_int[i, t, 1])) for t in range(valid_len)]
            all_tokens.append(self.tokenizer.encode(trajectory))

        max_len = max(len(t) for t in all_tokens)
        device = actions.device

        token_ids = torch.full((batch_size, max_len), self.tokenizer.PAD_ID, dtype=torch.long, device=device)
        padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)

        for i, tokens in enumerate(all_tokens):
            token_ids[i, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)
            padding_mask[i, : len(tokens)] = False

        return token_ids, padding_mask

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def compute_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Teacher-forced cross-entropy loss over autoregressive token prediction.

        Expects batch keys: observation.state, action (B, horizon, 2),
        action_is_pad (B, horizon), and at least one of observation.images /
        observation.environment_state.
        """
        assert set(batch).issuperset({OBS_STATE, ACTION, "action_is_pad"})
        assert OBS_IMAGES in batch or OBS_ENV_STATE in batch
        n_obs_steps = batch[OBS_STATE].shape[1]
        horizon = batch[ACTION].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode observations → conditioning vector.
        global_cond = self._prepare_global_conditioning(batch)
        cond = self.cond_proj(global_cond)  # (B, D)

        # Tokenize actions (only non-padded steps when masking is enabled).
        action_is_pad = batch["action_is_pad"] if self.config.do_mask_loss_for_padding else None
        token_ids, _ = self._tokenize_actions(batch[ACTION], action_is_pad)
        # token_ids: (B, S) including [BOS, ..., EOS, PAD...]

        # Teacher forcing: input = tokens[:-1], target = tokens[1:].
        input_tokens = token_ids[:, :-1]  # (B, S-1)
        target_tokens = token_ids[:, 1:]  # (B, S-1)
        input_pad = input_tokens == self.tokenizer.PAD_ID  # (B, S-1)

        # Embed tokens → (T, B, D) for the transformer.
        x = self.token_embedding(input_tokens).transpose(0, 1)

        # Causal transformer forward.
        hidden = self.transformer(x, cond, key_padding_mask=input_pad)  # (T, B, D)

        # Project to vocab logits → (B, V, T) for cross_entropy.
        logits = self.output_head(hidden, cond)  # (T, B, V)
        logits = logits.permute(1, 2, 0)  # (B, V, T)

        # Cross-entropy, ignoring PAD positions in the target.
        return F.cross_entropy(logits, target_tokens, ignore_index=self.tokenizer.PAD_ID)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_actions(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Autoregressively sample token sequences, decode to (x,y) actions.

        Returns:
            (B, n_action_steps, 2) float action tensor.
        """
        batch_size, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)
        cond = self.cond_proj(global_cond)
        device = cond.device

        # Start every sequence with BOS.
        generated = torch.full((batch_size, 1), self.tokenizer.BOS_ID, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        max_new_tokens = 1024  # safety cap
        for _ in range(max_new_tokens):
            x = self.token_embedding(generated).transpose(0, 1)  # (T, B, D)
            hidden = self.transformer(x, cond)  # (T, B, D)
            logits = self.output_head(hidden, cond)  # (T, B, V)

            next_logits = logits[-1]  # (B, V) — last position
            if self.config.temperature != 1.0:
                next_logits = next_logits / self.config.temperature

            next_token = torch.multinomial(F.softmax(next_logits, dim=-1), num_samples=1)  # (B, 1)

            # Force already-finished sequences to emit PAD.
            next_token[finished] = self.tokenizer.PAD_ID
            generated = torch.cat([generated, next_token], dim=1)

            finished = finished | (next_token.squeeze(1) == self.tokenizer.EOS_ID)
            if finished.all():
                break

        # Decode each token sequence back to (x, y) coordinates and resample.
        all_actions: list[torch.Tensor] = []
        for i in range(batch_size):
            tokens = generated[i].tolist()
            if self.tokenizer.EOS_ID in tokens:
                tokens = tokens[: tokens.index(self.tokenizer.EOS_ID) + 1]

            coords = self.tokenizer.decode(tokens)

            if len(coords) < 2:
                pt = coords[0] if coords else (0, 0)
                actions_i = torch.tensor([pt] * self.config.horizon, dtype=torch.float32, device=device)
            else:
                # Uniformly resample the Bresenham-dense decoded path to `horizon` points.
                coords_t = torch.tensor(coords, dtype=torch.float32, device=device)
                indices = torch.linspace(0, len(coords) - 1, self.config.horizon).round().long()
                actions_i = coords_t[indices]

            all_actions.append(actions_i)

        actions = torch.stack(all_actions)  # (B, horizon, 2)

        # Extract the n_action_steps slice (same convention as DiTFlow).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        return actions[:, start:end]
