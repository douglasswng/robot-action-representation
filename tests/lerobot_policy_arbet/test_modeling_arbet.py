from dataclasses import replace

import pytest
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from lerobot_policy_arbet.configuration_arbet import ARBeTConfig
from lerobot_policy_arbet.modeling_arbet import ARBeTModel, ARBeTPolicy

# Use small dimensions for fast tests.
B, H, W, C = 2, 96, 96, 3
STATE_DIM = 2
N_OBS_STEPS = 2
HORIZON = 16
N_ACTION_STEPS = 8

_BASE_CONFIG = ARBeTConfig(
    input_features={
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(C, H, W)),
    },
    output_features={
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(STATE_DIM,)),
    },
    # Shrink model for test speed.
    hidden_dim=64,
    num_blocks=2,
    num_heads=4,
    dim_feedforward=128,
    bpe_vocab_size=64,
    spatial_softmax_num_keypoints=8,
    n_obs_steps=N_OBS_STEPS,
    horizon=HORIZON,
    n_action_steps=N_ACTION_STEPS,
)


def _make_config(**overrides) -> ARBeTConfig:
    return replace(_BASE_CONFIG, **overrides)


def _make_batch(config: ARBeTConfig, batch_size: int = B) -> dict[str, torch.Tensor]:
    """Create a minimal training batch matching the config."""
    return {
        OBS_STATE: torch.randn(batch_size, config.n_obs_steps, STATE_DIM),
        OBS_IMAGES: torch.randn(batch_size, config.n_obs_steps, 1, C, H, W),
        ACTION: torch.randint(0, 50, (batch_size, config.horizon, 2)).float(),
        "action_is_pad": torch.zeros(batch_size, config.horizon, dtype=torch.bool),
    }


# ---------------------------------------------------------------------------
# ARBeTModel
# ---------------------------------------------------------------------------


class TestARBeTModel:
    @pytest.fixture
    def config(self):
        return _make_config()

    @pytest.fixture
    def model(self, config):
        return ARBeTModel(config)

    def test_construction(self, model):
        assert model.tokenizer is not None
        assert model.token_embedding is not None
        assert model.transformer is not None
        assert model.output_head is not None

    def test_global_cond_dim_positive(self, model):
        assert model.global_cond_dim > 0

    # --- Tokenization ---

    def test_tokenize_actions_shape(self, model, config):
        actions = torch.randint(0, 50, (B, config.horizon, 2)).float()
        token_ids, padding_mask = model._tokenize_actions(actions)
        assert token_ids.shape[0] == B
        assert padding_mask.shape == token_ids.shape
        assert token_ids.dtype == torch.long

    def test_tokenize_actions_starts_bos_ends_eos(self, model, config):
        actions = torch.tensor([[[0, 0], [1, 0], [2, 0]]], dtype=torch.float)
        token_ids, _ = model._tokenize_actions(actions)
        assert token_ids[0, 0].item() == model.tokenizer.BOS_ID
        # Find last non-pad token.
        non_pad = (token_ids[0] != model.tokenizer.PAD_ID).nonzero(as_tuple=True)[0]
        assert token_ids[0, non_pad[-1]].item() == model.tokenizer.EOS_ID

    def test_tokenize_actions_with_padding_mask(self, model, config):
        actions = torch.randint(0, 50, (B, config.horizon, 2)).float()
        action_is_pad = torch.zeros(B, config.horizon, dtype=torch.bool)
        action_is_pad[:, -4:] = True  # Last 4 steps are padding.
        token_ids_masked, _ = model._tokenize_actions(actions, action_is_pad)
        token_ids_full, _ = model._tokenize_actions(actions, None)
        # Masked version should have fewer or equal non-pad tokens.
        n_real_masked = (token_ids_masked != model.tokenizer.PAD_ID).sum()
        n_real_full = (token_ids_full != model.tokenizer.PAD_ID).sum()
        assert n_real_masked <= n_real_full

    # --- Loss ---

    def test_compute_loss_returns_scalar(self, model, config):
        batch = _make_batch(config)
        loss = model.compute_loss(batch)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_compute_loss_gradients_flow(self, model, config):
        batch = _make_batch(config)
        loss = model.compute_loss(batch)
        loss.backward()
        # At least the token embedding should have gradients.
        assert model.token_embedding.weight.grad is not None
        assert model.token_embedding.weight.grad.abs().sum() > 0

    # --- Generation ---

    def test_generate_actions_shape(self, model, config):
        batch = _make_batch(config, batch_size=1)
        model.eval()
        actions = model.generate_actions(batch)
        assert actions.shape == (1, config.n_action_steps, 2)

    def test_generate_actions_finite(self, model, config):
        batch = _make_batch(config, batch_size=1)
        model.eval()
        actions = model.generate_actions(batch)
        assert torch.isfinite(actions).all()


# ---------------------------------------------------------------------------
# ARBeTPolicy
# ---------------------------------------------------------------------------


class TestARBeTPolicy:
    @pytest.fixture
    def config(self):
        return _make_config()

    @pytest.fixture
    def policy(self, config):
        return ARBeTPolicy(config)

    def test_construction(self, policy):
        assert policy.arbet is not None

    def test_reset_clears_queues(self, policy):
        policy.reset()
        assert len(policy._queues[ACTION]) == 0
        assert len(policy._queues[OBS_STATE]) == 0

    # --- Forward (training) ---

    def test_forward_returns_loss_and_none(self, policy, config):
        batch = _make_batch(config)
        # forward expects image keys, not OBS_IMAGES directly.
        batch["observation.images.top"] = batch.pop(OBS_IMAGES).squeeze(-4)
        loss, info = policy.forward(batch)
        assert loss.shape == ()
        assert info is None

    # --- select_action ---

    def test_select_action_returns_single_action(self, policy, config):
        policy.eval()
        policy.reset()

        # Fill observation queue with preceding steps.
        for _ in range(config.n_obs_steps - 1):
            policy.select_action(
                {
                    OBS_STATE: torch.randn(1, STATE_DIM),
                    "observation.images.top": torch.randn(1, C, H, W),
                }
            )

        # Final step — the one we actually assert on.
        action = policy.select_action(
            {
                OBS_STATE: torch.randn(1, STATE_DIM),
                "observation.images.top": torch.randn(1, C, H, W),
            }
        )
        assert action.shape == (1, STATE_DIM)

    def test_select_action_drains_queue(self, policy, config):
        """Calling select_action n_action_steps times should drain the queue."""
        policy.eval()
        policy.reset()

        # Fill observation queues first.
        for _ in range(config.n_obs_steps):
            batch = {
                OBS_STATE: torch.randn(1, STATE_DIM),
                "observation.images.top": torch.randn(1, C, H, W),
            }
            policy.select_action(batch)

        # After filling, the action queue should have been populated.
        # Keep calling until the queue is empty (will be refilled on next call).
        remaining = len(policy._queues[ACTION])
        for _ in range(remaining):
            batch = {
                OBS_STATE: torch.randn(1, STATE_DIM),
                "observation.images.top": torch.randn(1, C, H, W),
            }
            policy.select_action(batch)

        # Queue should now be empty (just drained).
        assert len(policy._queues[ACTION]) == 0

    def test_get_optim_params(self, policy):
        params = policy.get_optim_params()
        param_list = list(params)
        assert len(param_list) > 0
