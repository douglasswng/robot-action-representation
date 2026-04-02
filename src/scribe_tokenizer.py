from tokink import Tokinkizer
from tokink.ink import Ink, Point, Stroke


class ScribeTokenizer:
    """Converts (x,y) integer trajectories to/from BPE-compressed chain-code token IDs.

    Wraps tokink's Tokinkizer, removing digital-ink-specific pen state tokens (UP/DOWN)
    and remapping IDs so vocab_size reflects the true usable vocabulary:
        PAD=0, BOS=1, EOS=2, 8 direction arrows=3-10, BPE merges=11+
    """

    def __init__(self, bpe_vocab_size: int = 16000):
        if bpe_vocab_size < 11:
            raise ValueError("bpe_vocab_size must be >= 11 (3 special + 8 directions)")

        # Pen-state tokens to exclude from our vocabulary
        self._excluded = {Tokinkizer._UP, Tokinkizer._DOWN}

        # tokink produces tokink_vocab_size IDs (including its PAD at 0).
        # After excluding UP/DOWN and keeping our own PAD, we want exactly
        # bpe_vocab_size distinct IDs, so: tokink_vocab_size - len(excluded) + 1 = bpe_vocab_size.
        self._tokink = Tokinkizer.from_pretrained(vocab_size=bpe_vocab_size + len(self._excluded) - 1)

        # Build compact ID remapping, skipping excluded tokens.
        self._to_ours: dict[int, int] = {0: 0}
        self._from_ours: dict[int, int] = {0: 0}
        our_id = 1
        for tokink_id in sorted(self._tokink._reverse_vocab):
            token = self._tokink._reverse_vocab[tokink_id]
            if token in self._excluded:
                continue
            self._to_ours[tokink_id] = our_id
            self._from_ours[our_id] = tokink_id
            our_id += 1

        self.vocab_size = our_id
        self.PAD_ID = 0
        self.BOS_ID = self._to_ours[self._tokink._vocab[Tokinkizer._BOS]]
        self.EOS_ID = self._to_ours[self._tokink._vocab[Tokinkizer._EOS]]

    def encode(self, trajectory: list[tuple[int, int]]) -> list[int]:
        """Encode an origin-relative (x,y) trajectory into token IDs: [BOS, ...chain-code tokens..., EOS].

        The trajectory must start at (0, 0). Callers are responsible for
        subtracting the absolute start position before encoding and adding it
        back after decoding.
        """
        if trajectory[0] != (0, 0):
            raise ValueError(f"Trajectory must start at (0, 0), got {trajectory[0]}")
        ink = Ink(strokes=[Stroke(points=[Point(x=x, y=y) for x, y in trajectory])])
        tokens = self._tokink.tokenize(ink)
        tokens = [t for t in tokens if t not in self._excluded]
        return [self._to_ours[self._tokink.token_to_id(t)] for t in tokens]

    def decode(self, token_ids: list[int]) -> list[tuple[int, int]]:
        """Decode token IDs back into origin-relative (x,y) coordinates.

        The returned path starts at (0, 0). Callers are responsible for offsetting
        by the absolute start position.

        Note: output may be Bresenham-dense (more points than the original trajectory)
        since chain codes decompose each delta into unit-step intermediates.
        """
        # Ensure the sequence ends with EOS so tokink can finalize the stroke.
        clean_ids = [i for i in token_ids if i != self.PAD_ID]
        if not clean_ids or clean_ids[-1] != self.EOS_ID:
            clean_ids.append(self.EOS_ID)

        tokink_ids = [self._from_ours[i] for i in clean_ids]
        tokens = self._tokink.convert_ids_to_tokens(tokink_ids)

        # Reconstruct pen-state tokens for tokink's detokenize:
        # single stroke → [DOWN] right after [BOS], [UP] right before [EOS]
        reconstructed: list[str] = []
        for t in tokens:
            if t == Tokinkizer._BOS:
                reconstructed.extend([Tokinkizer._BOS, Tokinkizer._DOWN])
            elif t == Tokinkizer._EOS:
                reconstructed.extend([Tokinkizer._UP, Tokinkizer._EOS])
            else:
                reconstructed.append(t)

        ink = self._tokink.detokenize(reconstructed)
        return [(p.x, p.y) for stroke in ink.strokes for p in stroke.points]
