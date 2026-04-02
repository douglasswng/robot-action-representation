"""Generic transformer decoder with Adaptive Layer Normalization (AdaLN) conditioning.

Provides modular building blocks for transformers that condition on external signals
(images, proprioception, time, etc.) via adaptive layer normalization — the same
shift-scale modulation used in DiT / Pi0 architectures.

The components here are policy-agnostic: ARBeT uses them for autoregressive token
prediction (causal masking, token embeddings), while DiTFlow could also be refactored
to use them for diffusion velocity prediction (no causal mask, continuous inputs).
"""

import copy

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------


def get_activation_fn(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU(approximate="tanh")
    raise ValueError(f"Unsupported activation: {name}")


# ---------------------------------------------------------------------------
# AdaLN modulation primitives
# ---------------------------------------------------------------------------


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply adaptive shift-scale: x * (1 + scale) + shift.

    Args:
        x: (T, B, D) sequence tensor (T-first layout).
        shift: (B, D) conditioning shift.
        scale: (B, D) conditioning scale.
    """
    return x * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


class ShiftScaleModulation(nn.Module):
    """Learnable shift-scale modulation conditioned on a vector c.

    Given input x and conditioning c, computes:
        x * (1 + scale(SiLU(c))) + shift(SiLU(c))

    Used before self-attention and FFN sublayers.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)
        self.shift = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = self.act(c)
        return modulate(x, self.shift(c), self.scale(c))

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.scale.bias)
        nn.init.zeros_(self.shift.bias)


class GateModulation(nn.Module):
    """Learnable scale-only gating conditioned on a vector c.

    Given input x and conditioning c, computes:
        x * scale(SiLU(c))

    Used after self-attention and FFN sublayers (zero-initialized → residual starts as identity).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.act = nn.SiLU()
        self.scale = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = self.act(c)
        s = self.scale(c).unsqueeze(0)  # (1, B, D) → broadcasts over T
        return x * s

    def reset_parameters(self):
        nn.init.zeros_(self.scale.weight)
        nn.init.zeros_(self.scale.bias)


# ---------------------------------------------------------------------------
# Transformer block with AdaLN conditioning
# ---------------------------------------------------------------------------


class AdaLNTransformerBlock(nn.Module):
    """Single transformer decoder block with AdaLN modulation.

    Architecture per block:
        1. LayerNorm → ShiftScale modulation → self-attention → Gate modulation → residual
        2. LayerNorm → ShiftScale modulation → FFN → Gate modulation → residual

    Supports optional causal masking for autoregressive decoding.

    Args:
        d_model: Hidden dimension.
        nhead: Number of attention heads.
        dim_feedforward: FFN intermediate dimension.
        dropout: Dropout rate.
        activation: Activation function name ("gelu" or "relu").
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 16,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN modulation layers
        self.attn_modulate = ShiftScaleModulation(d_model)
        self.attn_gate = GateModulation(d_model)
        self.mlp_modulate = ShiftScaleModulation(d_model)
        self.mlp_gate = GateModulation(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (T, B, D) input sequence.
            cond: (B, D) conditioning vector (e.g. image + state features).
            attn_mask: (T, T) causal or other attention mask.
            key_padding_mask: (B, T) padding mask.

        Returns:
            (T, B, D) output sequence.
        """
        # Self-attention with AdaLN
        x2 = self.attn_modulate(self.norm1(x), cond)
        x2, _ = self.self_attn(x2, x2, x2, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = x + self.attn_gate(self.dropout1(x2), cond)

        # FFN with AdaLN
        x3 = self.mlp_modulate(self.norm2(x), cond)
        x3 = self.mlp(x3)
        x3 = self.mlp_gate(x3, cond)
        return x + x3

    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for mod in (self.attn_modulate, self.attn_gate, self.mlp_modulate, self.mlp_gate):
            mod.reset_parameters()


# ---------------------------------------------------------------------------
# Final output layer with AdaLN
# ---------------------------------------------------------------------------


class AdaLNFinalLayer(nn.Module):
    """Output projection with adaptive layer normalization.

    Applies AdaLN (shift + scale from conditioning) before a linear projection.

    Args:
        hidden_dim: Input dimension.
        out_dim: Output dimension (e.g. vocab size for classification, action dim for regression).
    """

    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_dim, out_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 2 * hidden_dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (T, B, D) or (B, T, D) hidden states.
            cond: (B, D) conditioning vector.

        Returns:
            Same layout as x, with last dim replaced by out_dim.
        """
        shift, scale = self.adaLN_modulation(cond).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)

    def reset_parameters(self):
        for p in self.parameters():
            nn.init.zeros_(p)


# ---------------------------------------------------------------------------
# Full AdaLN Transformer Decoder
# ---------------------------------------------------------------------------


class AdaLNTransformerDecoder(nn.Module):
    """Stack of AdaLN transformer blocks with optional causal masking.

    This is the core reusable transformer — it takes embedded token sequences and a
    conditioning vector, and returns hidden states. The caller is responsible for
    input embedding and output projection.

    Args:
        d_model: Hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer blocks.
        dim_feedforward: FFN intermediate dimension.
        dropout: Dropout rate.
        activation: Activation function name.
        causal: If True, generates and applies a causal attention mask.
        max_seq_len: Maximum sequence length for positional encoding.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 16,
        num_layers: int = 6,
        dim_feedforward: int = 4096,
        dropout: float = 0.1,
        activation: str = "gelu",
        causal: bool = False,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.d_model = d_model
        self.causal = causal

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.empty(max_seq_len, 1, d_model))
        nn.init.xavier_uniform_(self.pos_embedding.data)

        # Transformer blocks
        base_block = AdaLNTransformerBlock(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
        )
        self.layers = nn.ModuleList([copy.deepcopy(base_block) for _ in range(num_layers)])
        self._reset_parameters()

    def _reset_parameters(self):
        for layer in self.layers:
            assert isinstance(layer, AdaLNTransformerBlock)
            layer.reset_parameters()

    @staticmethod
    def _generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate a boolean causal mask (True = position to ignore)."""
        return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (T, B, D) embedded input sequence.
            cond: (B, D) conditioning vector (image + state features).
            key_padding_mask: (B, T) True for positions to ignore.

        Returns:
            (T, B, D) hidden states.
        """
        seq_len = x.shape[0]

        # Add positional encoding
        x = x + self.pos_embedding[:seq_len]

        # Build causal mask if needed
        attn_mask = self._generate_causal_mask(seq_len, x.device) if self.causal else None

        for layer in self.layers:
            x = layer(x, cond, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        return x
