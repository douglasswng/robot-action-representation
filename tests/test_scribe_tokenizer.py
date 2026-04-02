import pytest

from scribe_tokenizer import ScribeTokenizer


@pytest.fixture
def tokenizer():
    return ScribeTokenizer(bpe_vocab_size=256)


class TestInit:
    def test_vocab_size(self, tokenizer):
        assert tokenizer.vocab_size == 256

    def test_special_token_ids(self, tokenizer):
        assert tokenizer.PAD_ID == 0
        assert tokenizer.BOS_ID >= 1
        assert tokenizer.EOS_ID >= 1
        assert tokenizer.BOS_ID != tokenizer.EOS_ID

    def test_vocab_too_small(self):
        with pytest.raises(ValueError, match="must be >= 11"):
            ScribeTokenizer(bpe_vocab_size=10)

    def test_id_remapping_is_compact(self, tokenizer):
        """Our IDs should be 0..N with no gaps."""
        our_ids = sorted(tokenizer._from_ours.keys())
        assert our_ids == list(range(len(our_ids)))

    def test_no_excluded_tokens_in_mapping(self, tokenizer):
        """UP and DOWN tokens should not appear in our ID space."""
        for tokink_id in tokenizer._from_ours.values():
            if tokink_id in tokenizer._tokink._reverse_vocab:
                token = tokenizer._tokink._reverse_vocab[tokink_id]
                assert token not in tokenizer._excluded


class TestEncode:
    def test_starts_with_bos_ends_with_eos(self, tokenizer):
        ids = tokenizer.encode([(0, 0), (3, 0)])
        assert ids[0] == tokenizer.BOS_ID
        assert ids[-1] == tokenizer.EOS_ID

    def test_all_ids_in_range(self, tokenizer):
        ids = tokenizer.encode([(0, 0), (5, 3), (10, 7)])
        for i in ids:
            assert 0 <= i < tokenizer.vocab_size

    def test_horizontal_move(self, tokenizer):
        ids = tokenizer.encode([(0, 0), (1, 0)])
        # BOS + one direction token + EOS
        assert len(ids) == 3

    def test_no_movement(self, tokenizer):
        ids = tokenizer.encode([(0, 0)])
        # Single point: just BOS + EOS
        assert ids[0] == tokenizer.BOS_ID
        assert ids[-1] == tokenizer.EOS_ID

    def test_different_directions_produce_different_tokens(self, tokenizer):
        right = tokenizer.encode([(0, 0), (1, 0)])
        up = tokenizer.encode([(0, 0), (0, 1)])
        # The middle token (direction) should differ
        assert right[1] != up[1]


class TestDecode:
    def test_round_trip_unit_steps(self, tokenizer):
        """Unit-step trajectories should round-trip exactly."""
        trajectory = [(0, 0), (1, 0), (2, 0), (3, 0)]
        ids = tokenizer.encode(trajectory)
        result = tokenizer.decode(ids)
        assert result == trajectory

    def test_round_trip_diagonal(self, tokenizer):
        trajectory = [(0, 0), (1, 1), (2, 2)]
        ids = tokenizer.encode(trajectory)
        result = tokenizer.decode(ids)
        assert result == trajectory

    def test_round_trip_preserves_endpoints(self, tokenizer):
        """Even with Bresenham resampling, start and end points are preserved."""
        trajectory = [(0, 0), (7, 3)]
        ids = tokenizer.encode(trajectory)
        result = tokenizer.decode(ids)
        assert result[0] == (0, 0)
        assert result[-1] == (7, 3)

    def test_bresenham_resampling_adds_points(self, tokenizer):
        """A large step decomposes into unit intermediates."""
        trajectory = [(0, 0), (5, 0)]
        ids = tokenizer.encode(trajectory)
        result = tokenizer.decode(ids)
        assert len(result) == 6  # 0,1,2,3,4,5
        assert result == [(i, 0) for i in range(6)]

    def test_pad_tokens_ignored(self, tokenizer):
        ids = tokenizer.encode([(0, 0), (2, 0)])
        padded = [tokenizer.PAD_ID, tokenizer.PAD_ID, *ids, tokenizer.PAD_ID]
        assert tokenizer.decode(padded) == tokenizer.decode(ids)

    def test_negative_coordinates(self, tokenizer):
        trajectory = [(0, 0), (-1, 0), (-2, 0)]
        ids = tokenizer.encode(trajectory)
        result = tokenizer.decode(ids)
        assert result == trajectory

    def test_mixed_directions(self, tokenizer):
        """An L-shaped path should round-trip with unit steps."""
        trajectory = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
        ids = tokenizer.encode(trajectory)
        result = tokenizer.decode(ids)
        assert result == trajectory


class TestIdMapping:
    def test_bijective(self, tokenizer):
        """to_ours and from_ours should be perfect inverses."""
        for our_id, tokink_id in tokenizer._from_ours.items():
            assert tokenizer._to_ours[tokink_id] == our_id

    def test_mapping_size_equals_vocab(self, tokenizer):
        """Mapping covers exactly vocab_size entries (PAD included)."""
        assert len(tokenizer._from_ours) == tokenizer.vocab_size
