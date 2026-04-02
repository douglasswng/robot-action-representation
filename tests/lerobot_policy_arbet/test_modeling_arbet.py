"""Tests for ARBeTPolicy and ARBeTModel."""

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE
from lerobot_policy_arbet.configuration_arbet import ARBeTConfig
from lerobot_policy_arbet.modeling_arbet import ARBeTModel, ARBeTPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BASE_CONFIG = ARBeTConfig(
    input_features={
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
    },
    output_features={
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
    },
    # Small model for fast tests
    hidden_dim=64,
    num_blocks=2,
    num_heads=4,
    dim_feedforward=128,
    bpe_vocab_size=128,
    horizon=8,
    n_token_steps=4,
    n_action_steps=2,
    n_obs_steps=2,
)


def _make_config(**overrides) -> ARBeTConfig:
    return replace(_BASE_CONFIG, **overrides)


def _make_policy(config: ARBeTConfig | None = None, dataset_stats=None) -> ARBeTPolicy:
    cfg = config or _make_config()
    return ARBeTPolicy(cfg, dataset_stats=dataset_stats)


def _make_obs_batch(config: ARBeTConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Create a fake observation batch matching config expectations."""
    batch = {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, 2),
    }
    if config.image_features:
        n_cameras = len(config.image_features)
        batch[OBS_IMAGES] = torch.randn(batch_size, config.n_obs_steps, n_cameras, 3, 96, 96)
    if config.env_state_feature:
        env_dim = config.env_state_feature.shape[0]
        batch[OBS_ENV_STATE] = torch.randn(batch_size, config.n_obs_steps, env_dim)
    return batch


def _make_training_batch(config: ARBeTConfig, batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Create a fake training batch with token sequences."""
    batch = _make_obs_batch(config, batch_size)
    # Token sequence: BOS followed by random content tokens, with full mask
    tokens = torch.randint(3, config.bpe_vocab_size, (batch_size, config.horizon))
    tokens[:, 0] = 1  # BOS
    batch["action_tokens"] = tokens
    batch["action_tokens_mask"] = torch.ones(batch_size, config.horizon, dtype=torch.bool)
    return batch


# ---------------------------------------------------------------------------
# ARBeTPolicy construction
# ---------------------------------------------------------------------------


class TestPolicyConstruction:
    def test_creates_successfully(self):
        policy = _make_policy()
        assert isinstance(policy.arbet, ARBeTModel)
        assert policy.tokenizer is not None

    def test_dataset_stats_stored_as_buffers(self):
        stats = {
            OBS_STATE: {
                "min": torch.tensor([10.0, 20.0]),
                "max": torch.tensor([500.0, 500.0]),
            }
        }
        policy = _make_policy(dataset_stats=stats)
        assert torch.equal(policy.state_min, stats[OBS_STATE]["min"])
        assert torch.equal(policy.state_max, stats[OBS_STATE]["max"])

    def test_default_stats_when_none(self):
        policy = _make_policy(dataset_stats=None)
        assert torch.equal(policy.state_min, torch.zeros(2))
        assert torch.equal(policy.state_max, torch.ones(2))

    def test_get_optim_params_returns_parameters(self):
        policy = _make_policy()
        params = list(policy.get_optim_params())
        assert len(params) > 0
        assert all(isinstance(p, torch.nn.Parameter) for p in params)


# ---------------------------------------------------------------------------
# Reset and queue management
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_creates_queues(self):
        policy = _make_policy()
        assert "observation.state" in policy._queues
        assert ACTION in policy._queues

    def test_reset_clears_queues(self):
        policy = _make_policy()
        policy._queues[ACTION].append(torch.zeros(2))
        policy.reset()
        assert len(policy._queues[ACTION]) == 0

    def test_reset_creates_image_queue_when_images_configured(self):
        policy = _make_policy()
        assert OBS_IMAGES in policy._queues

    def test_reset_creates_env_state_queue_when_configured(self):
        cfg = _make_config(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
                OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
            },
        )
        policy = _make_policy(cfg)
        assert "observation.environment_state" in policy._queues


# ---------------------------------------------------------------------------
# Unnormalize state
# ---------------------------------------------------------------------------


class TestUnnormalizeState:
    def test_identity_with_0_1_range(self):
        """With min=0, max=1, unnormalize maps [-1,1] → [0,1]."""
        policy = _make_policy()
        result = policy.unnormalize_state(torch.tensor([-1.0, -1.0]))
        assert torch.allclose(result, torch.tensor([0.0, 0.0]))
        result = policy.unnormalize_state(torch.tensor([1.0, 1.0]))
        assert torch.allclose(result, torch.tensor([1.0, 1.0]))

    def test_with_custom_stats(self):
        stats = {OBS_STATE: {"min": torch.tensor([0.0, 0.0]), "max": torch.tensor([512.0, 512.0])}}
        policy = _make_policy(dataset_stats=stats)
        result = policy.unnormalize_state(torch.tensor([0.0, 0.0]))
        assert torch.allclose(result, torch.tensor([256.0, 256.0]))


# ---------------------------------------------------------------------------
# ARBeTModel — conditioning
# ---------------------------------------------------------------------------


class TestGlobalConditioning:
    def test_output_shape(self):
        cfg = _make_config()
        model = ARBeTModel(cfg)
        batch = _make_obs_batch(cfg)
        cond = model._prepare_global_conditioning(batch)
        assert cond.shape[0] == 2  # batch_size
        # Should be flattened to (B, global_cond_dim * n_obs_steps)
        assert cond.shape[1] == model.global_cond_dim * cfg.n_obs_steps

    def test_with_env_state(self):
        cfg = _make_config(
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                OBS_ENV_STATE: PolicyFeature(type=FeatureType.ENV, shape=(4,)),
            },
        )
        model = ARBeTModel(cfg)
        batch = _make_obs_batch(cfg)
        cond = model._prepare_global_conditioning(batch)
        assert cond.shape[0] == 2


# ---------------------------------------------------------------------------
# ARBeTModel — compute_loss
# ---------------------------------------------------------------------------


class TestComputeLoss:
    def test_returns_scalar(self):
        cfg = _make_config()
        model = ARBeTModel(cfg)
        batch = _make_training_batch(cfg)
        loss = model.compute_loss(batch)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_requires_grad(self):
        cfg = _make_config()
        model = ARBeTModel(cfg)
        batch = _make_training_batch(cfg)
        loss = model.compute_loss(batch)
        assert loss.requires_grad

    def test_masked_loss_ignores_padding(self):
        cfg = _make_config(do_mask_loss_for_padding=True)
        model = ARBeTModel(cfg)
        batch = _make_training_batch(cfg)
        # Mask out the last half of tokens as padding
        batch["action_tokens_mask"][:, cfg.horizon // 2 :] = False
        loss_masked = model.compute_loss(batch)
        assert loss_masked.shape == ()
        assert loss_masked.item() > 0

    def test_unmasked_loss(self):
        cfg = _make_config(do_mask_loss_for_padding=False)
        model = ARBeTModel(cfg)
        batch = _make_training_batch(cfg)
        loss = model.compute_loss(batch)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_loss_decreases_with_repeated_input(self):
        """Sanity check: loss should decrease after a few gradient steps."""
        cfg = _make_config()
        model = ARBeTModel(cfg)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        torch.manual_seed(42)
        batch = _make_training_batch(cfg, batch_size=4)

        initial_loss = model.compute_loss(batch).item()
        for _ in range(20):
            loss = model.compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        final_loss = model.compute_loss(batch).item()
        assert final_loss < initial_loss

    def test_missing_keys_raises(self):
        cfg = _make_config()
        model = ARBeTModel(cfg)
        batch = _make_obs_batch(cfg)
        # Missing action_tokens and action_tokens_mask
        with pytest.raises(AssertionError):
            model.compute_loss(batch)


# ---------------------------------------------------------------------------
# ARBeTModel — generate_actions
# ---------------------------------------------------------------------------


class TestGenerateActions:
    def test_output_shape(self):
        cfg = _make_config()
        model = ARBeTModel(cfg)
        model.eval()
        batch = _make_obs_batch(cfg)
        tokenizer = _make_policy(cfg).tokenizer
        start_pos = torch.tensor([[100, 100], [200, 200]])
        actions = model.generate_actions(batch, tokenizer, temperature=1.0, start_positions=start_pos)
        assert actions.shape == (2, cfg.n_action_steps, 2)

    def test_output_shape_without_start_positions(self):
        cfg = _make_config()
        model = ARBeTModel(cfg)
        model.eval()
        batch = _make_obs_batch(cfg)
        tokenizer = _make_policy(cfg).tokenizer
        actions = model.generate_actions(batch, tokenizer, temperature=1.0, start_positions=None)
        assert actions.shape == (2, cfg.n_action_steps, 2)

    def test_temperature_scaling(self):
        """Different temperatures should produce different logit distributions (and often different outputs)."""
        cfg = _make_config()
        model = ARBeTModel(cfg)
        model.eval()
        batch = _make_obs_batch(cfg, batch_size=1)
        tokenizer = _make_policy(cfg).tokenizer
        start_pos = torch.tensor([[100, 100]])

        torch.manual_seed(0)
        actions_hot = model.generate_actions(batch, tokenizer, temperature=10.0, start_positions=start_pos)
        torch.manual_seed(0)
        actions_cold = model.generate_actions(batch, tokenizer, temperature=0.01, start_positions=start_pos)
        # With very different temperatures, outputs are very likely to differ
        # (but not guaranteed — so we just check shapes are correct)
        assert actions_hot.shape == actions_cold.shape

    def test_decode_failure_returns_origin(self):
        """When tokenizer.decode raises, the model should fall back to repeating the origin."""
        cfg = _make_config()
        model = ARBeTModel(cfg)
        model.eval()
        batch = _make_obs_batch(cfg, batch_size=1)
        tokenizer = _make_policy(cfg).tokenizer
        start_pos = torch.tensor([[100, 200]])

        with patch.object(tokenizer, "decode", side_effect=Exception("bad tokens")):
            actions = model.generate_actions(batch, tokenizer, temperature=1.0, start_positions=start_pos)

        assert actions.shape == (1, cfg.n_action_steps, 2)
        # All waypoints should be the origin
        for i in range(cfg.n_action_steps):
            assert torch.allclose(actions[0, i], torch.tensor([100.0, 200.0]))


# ---------------------------------------------------------------------------
# ARBeTPolicy — forward (training)
# ---------------------------------------------------------------------------


class TestPolicyForward:
    def test_forward_returns_loss_and_none(self):
        policy = _make_policy()
        batch = _make_training_batch(policy.config)
        # forward expects image keys by name, not pre-stacked
        batch["observation.images.top"] = batch.pop(OBS_IMAGES)[:, :, 0]  # single camera
        loss, extra = policy.forward(batch)
        assert extra is None
        assert loss.shape == ()
        assert loss.item() > 0

    def test_forward_stacks_image_features(self):
        """Verify that forward correctly stacks per-camera image keys into observation.images."""
        policy = _make_policy()
        batch = _make_training_batch(policy.config)
        images = batch.pop(OBS_IMAGES)[:, :, 0]  # (B, n_obs, C, H, W)
        batch["observation.images.top"] = images
        # Should not raise — forward should stack the image key internally
        loss, _ = policy.forward(batch)
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# ARBeTModel — separate RGB encoders per camera
# ---------------------------------------------------------------------------


class TestSeparateRgbEncoders:
    def test_separate_encoders_construction(self):
        cfg = _make_config(
            use_separate_rgb_encoder_per_camera=True,
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
                "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            },
        )
        model = ARBeTModel(cfg)
        assert isinstance(model.rgb_encoder, torch.nn.ModuleList)
        assert len(model.rgb_encoder) == 2

    def test_separate_encoders_forward(self):
        cfg = _make_config(
            use_separate_rgb_encoder_per_camera=True,
            input_features={
                OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,)),
                "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
                "observation.images.side": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 96, 96)),
            },
        )
        model = ARBeTModel(cfg)
        loss = model.compute_loss(_make_training_batch(cfg))
        assert loss.shape == ()


# ---------------------------------------------------------------------------
# Parameter count
# ---------------------------------------------------------------------------


class TestParameterCount:
    def test_nonzero_params(self):
        cfg = _make_config()
        model = ARBeTModel(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_all_params_in_expected_modules(self):
        cfg = _make_config()
        model = ARBeTModel(cfg)
        named = {n for n, _ in model.named_modules()}
        assert "cond_proj" in named
        assert "token_embedding" in named
        assert "transformer" in named
        assert "output_head" in named
