import pytest
import torch

from adaln_transformer import (
    AdaLNFinalLayer,
    AdaLNTransformerBlock,
    AdaLNTransformerDecoder,
    GateModulation,
    ShiftScaleModulation,
    get_activation_fn,
    modulate,
)

T, B, D = 10, 4, 64  # sequence length, batch size, hidden dim


# ---------------------------------------------------------------------------
# modulate()
# ---------------------------------------------------------------------------


class TestModulate:
    def test_identity_when_shift_zero_scale_zero(self):
        x = torch.randn(T, B, D)
        shift = torch.zeros(B, D)
        scale = torch.zeros(B, D)
        out = modulate(x, shift, scale)
        torch.testing.assert_close(out, x)

    def test_scale_only(self):
        x = torch.ones(T, B, D)
        shift = torch.zeros(B, D)
        scale = torch.ones(B, D)  # (1 + 1) = 2x
        out = modulate(x, shift, scale)
        torch.testing.assert_close(out, torch.full_like(x, 2.0))

    def test_shift_only(self):
        x = torch.zeros(T, B, D)
        shift = torch.full((B, D), 3.0)
        scale = torch.zeros(B, D)
        out = modulate(x, shift, scale)
        torch.testing.assert_close(out, torch.full_like(x, 3.0))

    def test_output_shape_t_first(self):
        x = torch.randn(T, B, D)
        out = modulate(x, torch.randn(B, D), torch.randn(B, D))
        assert out.shape == (T, B, D)

    def test_output_shape_b_first(self):
        x = torch.randn(B, T, D)
        out = modulate(x, torch.randn(B, D), torch.randn(B, D))
        assert out.shape == (B, T, D)

    def test_ambiguous_when_batch_equals_seq_len(self):
        """When B == T, the heuristic cannot distinguish T-first from B-first.

        This documents the known ambiguity: modulate always picks the B-first
        (unsqueeze dim 1) path when shape[0] == shift.shape[0].
        """
        N, D_ = 4, 64  # noqa: N806
        x = torch.randn(N, N, D_)
        shift = torch.zeros(N, D_)
        scale = torch.zeros(N, D_)
        out = modulate(x, shift, scale)
        # With zero shift/scale the output is always x regardless of layout,
        # but verify the function at least runs and produces correct shape.
        assert out.shape == (N, N, D_)
        torch.testing.assert_close(out, x)


# ---------------------------------------------------------------------------
# get_activation_fn()
# ---------------------------------------------------------------------------


class TestGetActivationFn:
    def test_gelu(self):
        assert isinstance(get_activation_fn("gelu"), torch.nn.GELU)

    def test_relu(self):
        assert isinstance(get_activation_fn("relu"), torch.nn.ReLU)

    def test_unsupported_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            get_activation_fn("swish")


# ---------------------------------------------------------------------------
# ShiftScaleModulation
# ---------------------------------------------------------------------------


class TestShiftScaleModulation:
    def test_output_shape(self):
        mod = ShiftScaleModulation(D)
        x = torch.randn(T, B, D)
        c = torch.randn(B, D)
        assert mod(x, c).shape == (T, B, D)

    def test_identity_after_reset(self):
        """After reset_parameters (all zeros), output should equal input (scale=0 → 1+0=1, shift=0)."""
        mod = ShiftScaleModulation(D)
        mod.reset_parameters()
        x = torch.randn(T, B, D)
        c = torch.randn(B, D)
        # SiLU(c) is nonzero, but linear weights are zero → scale=0, shift=0
        torch.testing.assert_close(mod(x, c), x)


# ---------------------------------------------------------------------------
# GateModulation
# ---------------------------------------------------------------------------


class TestGateModulation:
    def test_output_shape(self):
        gate = GateModulation(D)
        x = torch.randn(T, B, D)
        c = torch.randn(B, D)
        assert gate(x, c).shape == (T, B, D)

    def test_zero_after_reset(self):
        """After reset_parameters (all zeros), output should be zero (gate is closed)."""
        gate = GateModulation(D)
        gate.reset_parameters()
        x = torch.randn(T, B, D)
        c = torch.randn(B, D)
        out = gate(x, c)
        torch.testing.assert_close(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# AdaLNTransformerBlock
# ---------------------------------------------------------------------------


class TestAdaLNTransformerBlock:
    @pytest.fixture
    def block(self):
        return AdaLNTransformerBlock(d_model=D, nhead=4, dim_feedforward=128, dropout=0.0)

    def test_output_shape(self, block):
        x = torch.randn(T, B, D)
        cond = torch.randn(B, D)
        out = block(x, cond)
        assert out.shape == (T, B, D)

    def test_identity_after_reset(self, block):
        """After reset_parameters, gates are zero → block is identity (residual only)."""
        block.reset_parameters()
        x = torch.randn(T, B, D)
        cond = torch.randn(B, D)
        out = block(x, cond)
        torch.testing.assert_close(out, x)

    def test_causal_mask_accepted(self, block):
        """Block should run without error when given a causal attention mask."""
        x = torch.randn(T, B, D)
        cond = torch.randn(B, D)
        mask = torch.triu(torch.full((T, T), float("-inf")), diagonal=1)
        out = block(x, cond, attn_mask=mask)
        assert out.shape == (T, B, D)

    def test_gradients_flow(self, block):
        x = torch.randn(T, B, D, requires_grad=True)
        cond = torch.randn(B, D, requires_grad=True)
        out = block(x, cond)
        out.sum().backward()
        assert x.grad is not None
        assert cond.grad is not None


# ---------------------------------------------------------------------------
# AdaLNFinalLayer
# ---------------------------------------------------------------------------


class TestAdaLNFinalLayer:
    def test_output_shape(self):
        out_dim = 32
        layer = AdaLNFinalLayer(D, out_dim)
        x = torch.randn(T, B, D)
        cond = torch.randn(B, D)
        out = layer(x, cond)
        assert out.shape == (T, B, out_dim)

    def test_zero_after_reset(self):
        """After reset_parameters (all zeros), output should be zero."""
        layer = AdaLNFinalLayer(D, 32)
        layer.reset_parameters()
        x = torch.randn(T, B, D)
        cond = torch.randn(B, D)
        out = layer(x, cond)
        torch.testing.assert_close(out, torch.zeros_like(out))


# ---------------------------------------------------------------------------
# AdaLNTransformerDecoder
# ---------------------------------------------------------------------------


class TestAdaLNTransformerDecoder:
    @pytest.fixture
    def decoder(self):
        return AdaLNTransformerDecoder(
            d_model=D,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            causal=False,
        )

    @pytest.fixture
    def causal_decoder(self):
        return AdaLNTransformerDecoder(
            d_model=D,
            nhead=4,
            num_layers=2,
            dim_feedforward=128,
            dropout=0.0,
            causal=True,
        )

    def test_output_shape(self, decoder):
        x = torch.randn(T, B, D)
        cond = torch.randn(B, D)
        out = decoder(x, cond)
        assert out.shape == (T, B, D)

    def test_causal_output_shape(self, causal_decoder):
        x = torch.randn(T, B, D)
        cond = torch.randn(B, D)
        out = causal_decoder(x, cond)
        assert out.shape == (T, B, D)

    def test_variable_sequence_length(self, decoder):
        """Decoder should handle different sequence lengths up to max_seq_len."""
        cond = torch.randn(B, D)
        for seq_len in [1, 5, T]:
            x = torch.randn(seq_len, B, D)
            out = decoder(x, cond)
            assert out.shape == (seq_len, B, D)

    def test_causal_masking_prevents_future_leakage(self, causal_decoder):
        """Changing a future token should not affect earlier positions."""
        cond = torch.randn(B, D)
        x = torch.randn(T, B, D)
        x_modified = x.clone()
        x_modified[-1] = torch.randn(B, D)  # change last token

        causal_decoder.eval()
        with torch.no_grad():
            out_original = causal_decoder(x, cond)
            out_modified = causal_decoder(x_modified, cond)

        # All positions except the last should be identical
        torch.testing.assert_close(out_original[:-1], out_modified[:-1])

    def test_gradients_flow(self, decoder):
        x = torch.randn(T, B, D, requires_grad=True)
        cond = torch.randn(B, D, requires_grad=True)
        out = decoder(x, cond)
        out.sum().backward()
        assert x.grad is not None
        assert cond.grad is not None

    def test_key_padding_mask(self, decoder):
        x = torch.randn(T, B, D)
        cond = torch.randn(B, D)
        pad_mask = torch.zeros(B, T, dtype=torch.bool)
        pad_mask[:, -3:] = True  # last 3 positions are padding
        out = decoder(x, cond, key_padding_mask=pad_mask)
        assert out.shape == (T, B, D)

    def test_key_padding_mask_ignores_padded_tokens(self, decoder):
        """Changing a masked token should not affect unmasked positions."""
        cond = torch.randn(B, D)
        x = torch.randn(T, B, D)
        x_modified = x.clone()
        x_modified[-1] = torch.randn(B, D)  # change last token

        pad_mask = torch.zeros(B, T, dtype=torch.bool)
        pad_mask[:, -1] = True  # mask the changed token

        decoder.eval()
        with torch.no_grad():
            out1 = decoder(x, cond, key_padding_mask=pad_mask)
            out2 = decoder(x_modified, cond, key_padding_mask=pad_mask)

        # Non-padded positions should be unaffected
        torch.testing.assert_close(out1[:-1], out2[:-1])

    def test_sequence_exceeding_max_seq_len_raises(self):
        """Passing a sequence longer than max_seq_len should raise an error."""
        decoder = AdaLNTransformerDecoder(
            d_model=D, nhead=4, num_layers=1, dim_feedforward=128, dropout=0.0, max_seq_len=8
        )
        x = torch.randn(16, B, D)  # 16 > max_seq_len=8
        with pytest.raises((IndexError, RuntimeError)):
            decoder(x, torch.randn(B, D))

    def test_layers_are_independent(self, decoder):
        """deepcopy should produce independent layers — mutating one doesn't affect another."""
        w0 = decoder.layers[0].norm1.weight.data.clone()
        decoder.layers[1].norm1.weight.data.fill_(999.0)
        torch.testing.assert_close(decoder.layers[0].norm1.weight.data, w0)
