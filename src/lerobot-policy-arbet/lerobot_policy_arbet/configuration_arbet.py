from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("arbet")
@dataclass
class ARBeTConfig(PreTrainedConfig):
    """Configuration class for ARBeTPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    ARBeT (Auto-Regressive Behaviour Transformer) uses ScribeTokens to convert integer (x,y) actions into
    sequences of 8-directional chain-code tokens (via Bresenham/Freeman), optionally compressed with BPE,
    and predicts them autoregressively with cross-entropy loss.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - At least one key starting with "observation.image" is required as an input,
          AND/OR the key "observation.environment_state" is required as input.
        - "action" is required as an output key.
    """

    # ---- ARBeT-specific ----

    # ScribeToken vocabulary.
    bpe_vocab_size: int = 256  # Total vocab size after BPE merges (includes 8 base directions + merged tokens)

    # Token chunking.
    n_obs_steps: int = 2
    horizon: int = 32  # total tokens predicted per step (including BOS; model generates horizon-1 content tokens)
    n_token_steps: int = 8  # number of generated content tokens to actually decode for execution (≤ horizon-1)
    n_action_steps: int = 2  # number of waypoints to subsample the decoded dense path into for execution

    # Autoregressive inference sampling.
    temperature: float = 0.7

    # Transformer.
    hidden_dim: int = 256
    num_blocks: int = 8
    num_heads: int = 4
    dropout: float = 0.1
    dim_feedforward: int = 1024
    activation: str = "gelu"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    drop_n_last_frames: int = 0

    # ---- Shared (matched with VQ-BeT / DiTFlow) ----

    use_proprioceptive: bool = True

    # Vision backbone.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    resize_shape: tuple[int, int] | None = None

    # Loss computation.
    # True for ARBeT because padded positions have meaningless token IDs (unlike DiTFlow where
    # padded actions are copies of the last real action and are at least plausible targets).
    do_mask_loss_for_padding: bool = True

    # Training presets.
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

        if self.n_token_steps > self.horizon - 1:
            raise ValueError(
                f"n_token_steps ({self.n_token_steps}) must be <= horizon - 1 ({self.horizon - 1}), "
                "since horizon includes BOS."
            )

        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(f"`vision_backbone` must be a ResNet variant. Got {self.vision_backbone}.")

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the image shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for `{key}`."
                    )

        first_image_key, first_image_ft = next(iter(self.image_features.items()))
        for key, image_ft in self.image_features.items():
            if image_ft.shape != first_image_ft.shape:
                raise ValueError(
                    f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(0, self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
