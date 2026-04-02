"""Dataset wrapper that tokenizes (x, y) action trajectories into ScribeToken sequences.

Wraps a LeRobotDataset so that each item contains:
    - observation.image:  (C, H, W)  - unchanged
    - observation.state:  (2,)       - unchanged (agent xy position)
    - action_tokens:      (horizon,) int64 - first `horizon` tokens from the tokenized
                                              remaining trajectory [BOS, ...tokens..., EOS?, PAD...]
    - action_tokens_mask: (horizon,) bool  - True for real tokens, False for PAD
    - action_start:       (2,) float32 - absolute (x, y) of the first action point (needed to
                                          decode origin-relative token sequence back to absolute coords)

All other keys from the base dataset are passed through unchanged (episode_index, frame_index, etc.),
except the raw "action" key which is removed.
"""

from __future__ import annotations

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from scribe_tokenizer import ScribeTokenizer


class ScribeTokenDataset(torch.utils.data.Dataset):
    """Wraps a LeRobotDataset, replacing (x, y) action chunks with tokenized scribe sequences.

    For each frame, tokenizes the remaining trajectory from that frame to the end of the
    episode, then takes the first ``horizon`` tokens. Padding (with PAD_ID) only occurs
    when the tokenized remainder is shorter than ``horizon`` (i.e. near episode ends where
    EOS arrives before the horizon is filled).
    """

    def __init__(
        self,
        base: LeRobotDataset,
        tokenizer: ScribeTokenizer,
        horizon: int,
    ) -> None:
        self.base = base
        self.tokenizer = tokenizer
        self.horizon = horizon

        # Pre-compute episode boundaries: episode_index -> (start, end) absolute frame indices.
        hf = base.hf_dataset
        ep_indices = torch.tensor(hf["episode_index"])
        changes = torch.where(ep_indices[1:] != ep_indices[:-1])[0] + 1
        starts = torch.cat([torch.tensor([0]), changes])
        ends = torch.cat([changes, torch.tensor([len(ep_indices)])])
        self._ep_start = {}
        self._ep_end = {}
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            ep = hf["episode_index"][s]
            ep = ep.item() if isinstance(ep, torch.Tensor) else ep
            self._ep_start[ep] = s
            self._ep_end[ep] = e

        # Cache all actions as a flat tensor for fast slicing.
        self._all_actions = torch.stack(list(hf["action"]))  # (total_frames, 2)

    # ------------------------------------------------------------------
    # Forward LeRobotDataset attributes so the training loop sees a
    # duck-typed dataset (meta, features, fps, etc.).
    # ------------------------------------------------------------------

    def __getattr__(self, name: str):
        return getattr(self.base, name)

    def __len__(self) -> int:
        return len(self.base)

    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.base[idx]

        # Remove chunked action keys (we replace them with tokens).
        for key in list(item.keys()):
            if key == "action" or key == "action_is_pad":
                item.pop(key)

        abs_idx = item["index"].item()
        ep_idx = item["episode_index"].item()
        ep_end = self._ep_end[ep_idx]

        # Remaining trajectory: from current frame to end of episode.
        remaining_actions = self._all_actions[abs_idx:ep_end]  # (n_remaining, 2)

        # Round to integer coords and make origin-relative to last observed state.
        int_actions = remaining_actions.round().long()
        obs_state = item["observation.state"]
        # (2,) last obs position
        obs_origin = (obs_state[-1, :2] if obs_state.ndim >= 2 else obs_state[:2]).round().long()
        action_start = obs_origin.float()  # (2,) absolute origin used for encoding
        origin_relative = int_actions - obs_origin.unsqueeze(0)  # relative to obs state

        trajectory = [(0, 0)] + [(int(x), int(y)) for x, y in origin_relative.tolist()]
        token_ids = self.tokenizer.encode(trajectory)  # [BOS, ..., EOS]

        # Take first `horizon` tokens; pad only if sequence is shorter.
        n_tokens = min(len(token_ids), self.horizon)
        padded = torch.full((self.horizon,), self.tokenizer.PAD_ID, dtype=torch.long)
        padded[:n_tokens] = torch.tensor(token_ids[:n_tokens], dtype=torch.long)

        mask = torch.zeros(self.horizon, dtype=torch.bool)
        mask[:n_tokens] = True

        item["action_tokens"] = padded  # (horizon,)
        item["action_tokens_mask"] = mask  # (horizon,)
        item["action_start"] = action_start  # (2,)

        return item
