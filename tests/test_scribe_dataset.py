import pytest
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scribe_dataset import ScribeTokenDataset
from scribe_tokenizer import ScribeTokenizer

HORIZON = 16


@pytest.fixture(scope="module")
def base_dataset():
    return LeRobotDataset("lerobot/pusht")


@pytest.fixture(scope="module")
def tokenizer():
    return ScribeTokenizer(bpe_vocab_size=256)


@pytest.fixture(scope="module")
def dataset(base_dataset, tokenizer):
    return ScribeTokenDataset(base=base_dataset, tokenizer=tokenizer, horizon=HORIZON)


class TestInit:
    def test_length_matches_base(self, dataset, base_dataset):
        assert len(dataset) == len(base_dataset)

    def test_episode_boundaries_populated(self, dataset):
        assert len(dataset._ep_start) > 0
        assert len(dataset._ep_end) > 0

    def test_forwards_base_attributes(self, dataset, base_dataset):
        assert dataset.fps == base_dataset.fps
        assert dataset.num_episodes == base_dataset.num_episodes


class TestGetItem:
    def test_keys(self, dataset):
        item = dataset[0]
        assert "action_tokens" in item
        assert "action_tokens_mask" in item
        assert "action_start" in item
        assert "observation.state" in item
        assert "action" not in item
        assert "action_is_pad" not in item

    def test_shapes(self, dataset):
        item = dataset[0]
        assert item["action_tokens"].shape == (HORIZON,)
        assert item["action_tokens_mask"].shape == (HORIZON,)
        assert item["action_start"].shape == (2,)

    def test_dtypes(self, dataset):
        item = dataset[0]
        assert item["action_tokens"].dtype == torch.long
        assert item["action_tokens_mask"].dtype == torch.bool
        assert item["action_start"].dtype == torch.float32

    def test_starts_with_bos(self, dataset, tokenizer):
        item = dataset[0]
        assert item["action_tokens"][0].item() == tokenizer.BOS_ID

    def test_token_ids_in_range(self, dataset, tokenizer):
        item = dataset[0]
        mask = item["action_tokens_mask"]
        tokens = item["action_tokens"][mask]
        assert (tokens >= 0).all()
        assert (tokens < tokenizer.vocab_size).all()

    def test_pad_positions_are_pad_id(self, dataset, tokenizer):
        item = dataset[0]
        mask = item["action_tokens_mask"]
        if not mask.all():
            pad_tokens = item["action_tokens"][~mask]
            assert (pad_tokens == tokenizer.PAD_ID).all()

    def test_mask_is_contiguous(self, dataset):
        """Real tokens should come first, then padding (no gaps)."""
        item = dataset[0]
        mask = item["action_tokens_mask"]
        n_real = mask.sum().item()
        assert mask[:n_real].all()
        if n_real < HORIZON:
            assert not mask[n_real:].any()


class TestEpisodeBoundary:
    def test_last_frame_is_short(self, dataset, tokenizer):
        """Last frame of an episode has only one action left, so the tokenized
        sequence is short — fewer real tokens than HORIZON, thus padded."""
        ep_end = dataset._ep_end[0]
        item = dataset[ep_end - 1]
        tokens = item["action_tokens"]
        mask = item["action_tokens_mask"]
        n_real = mask.sum().item()
        assert n_real < HORIZON
        assert tokens[0].item() == tokenizer.BOS_ID

    def test_first_frame_fills_horizon(self, dataset):
        """First frame of an episode should have enough remaining trajectory to fill the horizon."""
        item = dataset[0]
        n_real = item["action_tokens_mask"].sum().item()
        assert n_real == HORIZON

    def test_action_start_matches_obs_state(self, dataset, base_dataset):
        """action_start should be the observation state (trajectory origin for decoding)."""
        idx = 10
        base_item = base_dataset[idx]
        scribe_item = dataset[idx]
        obs_state = base_item["observation.state"]
        expected = (obs_state[:2] if obs_state.ndim == 1 else obs_state[-1, :2]).round()
        assert torch.allclose(scribe_item["action_start"], expected, atol=1.0)


class TestConsistency:
    def test_same_frame_is_deterministic(self, dataset):
        """Indexing the same frame twice should produce identical tokens."""
        item_a = dataset[0]
        item_b = dataset[0]
        assert torch.equal(item_a["action_tokens"], item_b["action_tokens"])
