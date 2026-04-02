"""Custom training entry point for ARBeT that wraps the dataset with ScribeTokenDataset.

Monkey-patches make_dataset so the standard LeRobot training loop receives a
ScribeTokenDataset (which pre-tokenizes actions into scribe token IDs) instead
of the raw LeRobotDataset.

Usage (same CLI args as lerobot_train):
    uv run python scripts/train_arbet.py --policy.type=arbet --dataset.repo_id=lerobot/pusht ...
"""

from lerobot.datasets.factory import make_dataset as _make_dataset
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.scripts import lerobot_train
from lerobot.utils.import_utils import register_third_party_plugins

from scribe_dataset import ScribeTokenDataset
from scribe_tokenizer import ScribeTokenizer


def make_scribe_dataset(cfg):
    dataset = _make_dataset(cfg)
    if not isinstance(dataset, LeRobotDataset):
        raise TypeError(f"ScribeTokenDataset requires a LeRobotDataset, got {type(dataset).__name__}")
    tokenizer = ScribeTokenizer(bpe_vocab_size=cfg.policy.bpe_vocab_size)
    return ScribeTokenDataset(
        base=dataset,
        tokenizer=tokenizer,
        horizon=cfg.policy.horizon,
    )


lerobot_train.make_dataset = make_scribe_dataset


if __name__ == "__main__":
    register_third_party_plugins()
    lerobot_train.main()
