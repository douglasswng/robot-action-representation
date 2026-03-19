from dataclasses import replace

import pytest
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_STATE
from lerobot_policy_arbet.configuration_arbet import N_BASE_DIRECTIONS, ARBeTConfig

_BASE_CONFIG = ARBeTConfig(
    input_features={
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    },
    output_features={
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    },
)


def _make_config(**overrides) -> ARBeTConfig:
    """Create an ARBeTConfig with PushT-like features pre-populated."""
    return replace(_BASE_CONFIG, **overrides)


# ---------------------------------------------------------------------------
# Defaults and basic construction
# ---------------------------------------------------------------------------


class TestDefaults:
    def test_default_construction(self):
        cfg = _make_config()
        assert cfg.horizon == 16
        assert cfg.n_action_steps == 8
        assert cfg.n_obs_steps == 2
        assert cfg.bpe_vocab_size == 8192

    def test_hidden_dim(self):
        cfg = _make_config()
        assert cfg.hidden_dim == 512
        assert cfg.num_blocks == 6
        assert cfg.num_heads == 16

    def test_do_mask_loss_for_padding_default_true(self):
        cfg = _make_config()
        assert cfg.do_mask_loss_for_padding is True


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_non_resnet_backbone_raises(self):
        with pytest.raises(ValueError, match="ResNet"):
            _make_config(vision_backbone="vit_base")

    def test_resnet_variants_accepted(self):
        for name in ("resnet18", "resnet34", "resnet50"):
            cfg = _make_config(vision_backbone=name)
            assert cfg.vision_backbone == name

    def test_bpe_vocab_size_too_small_raises(self):
        with pytest.raises(ValueError, match="bpe_vocab_size"):
            _make_config(bpe_vocab_size=N_BASE_DIRECTIONS - 1)

    def test_bpe_vocab_size_at_minimum(self):
        cfg = _make_config(bpe_vocab_size=N_BASE_DIRECTIONS)
        assert cfg.bpe_vocab_size == N_BASE_DIRECTIONS


# ---------------------------------------------------------------------------
# validate_features()
# ---------------------------------------------------------------------------


class TestValidateFeatures:
    def test_no_image_no_env_state_raises(self):
        cfg = _make_config(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
            },
        )
        with pytest.raises(ValueError, match="at least one image"):
            cfg.validate_features()

    def test_crop_larger_than_image_raises(self):
        cfg = _make_config(crop_shape=(128, 128))
        with pytest.raises(ValueError, match="crop_shape"):
            cfg.validate_features()

    def test_mismatched_image_shapes_raises(self):
        cfg = _make_config(
            crop_shape=None,
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
                "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
            },
        )
        with pytest.raises(ValueError, match="does not match"):
            cfg.validate_features()

    def test_valid_features_pass(self):
        cfg = _make_config()
        cfg.validate_features()  # should not raise


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_observation_delta_indices(self):
        cfg = _make_config(n_obs_steps=2)
        assert cfg.observation_delta_indices == [-1, 0]

    def test_observation_delta_indices_single_step(self):
        cfg = _make_config(n_obs_steps=1)
        assert cfg.observation_delta_indices == [0]

    def test_action_delta_indices_length(self):
        cfg = _make_config(n_obs_steps=2, horizon=16)
        indices = cfg.action_delta_indices
        assert len(indices) == 16
        assert indices[0] == -1
        assert indices[-1] == 14

    def test_reward_delta_indices_is_none(self):
        cfg = _make_config()
        assert cfg.reward_delta_indices is None


# ---------------------------------------------------------------------------
# Optimizer / scheduler presets
# ---------------------------------------------------------------------------


class TestPresets:
    def test_optimizer_preset_fields(self):
        cfg = _make_config()
        opt = cfg.get_optimizer_preset()
        assert opt.lr == cfg.optimizer_lr
        assert opt.betas == cfg.optimizer_betas
        assert opt.weight_decay == cfg.optimizer_weight_decay

    def test_scheduler_preset_fields(self):
        cfg = _make_config()
        sched = cfg.get_scheduler_preset()
        assert sched.name == cfg.scheduler_name
        assert sched.num_warmup_steps == cfg.scheduler_warmup_steps
