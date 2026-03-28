"""Conformer encoder: Conv + Self-Attention hybrid for speech recognition.

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", 2020.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import RelativePositionalEncoding
from .feature import ConvSubsampling


class FeedForwardModule(nn.Module):
    """Macaron-style feed forward module with swish activation."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),  # Swish
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelativeMultiHeadAttention(nn.Module):
    """Multi-head attention with relative positional encoding."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_pos = nn.Linear(d_model, d_model, bias=False)
        self.w_out = nn.Linear(d_model, d_model)

        self.pos_bias_u = nn.Parameter(torch.empty(num_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.empty(num_heads, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u.unsqueeze(0))
        nn.init.xavier_uniform_(self.pos_bias_v.unsqueeze(0))

        self.dropout = nn.Dropout(dropout)

    def _relative_shift(self, x: torch.Tensor) -> torch.Tensor:
        """Compute relative position shift for efficient attention."""
        b, h, t, _ = x.size()
        x = F.pad(x, (1, 0))  # pad left
        x = x.view(b, h, -1, t)
        x = x[:, :, 1:, :]
        x = x.view(b, h, t, -1)[:, :, :, :t]
        return x

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            pos_enc: (1, seq_len, d_model)
            mask: (batch, 1, seq_len) padding mask
        """
        b, t, _ = x.size()

        q = self.w_q(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(b, t, self.num_heads, self.d_k).transpose(1, 2)
        pos = self.w_pos(pos_enc).view(1, t, self.num_heads, self.d_k).transpose(1, 2)

        # Content-based attention
        q_with_u = q + self.pos_bias_u.unsqueeze(1)  # (b, h, t, d_k)
        content_score = torch.matmul(q_with_u, k.transpose(-2, -1))

        # Position-based attention
        q_with_v = q + self.pos_bias_v.unsqueeze(1)
        pos_score = torch.matmul(q_with_v, pos.transpose(-2, -1))
        pos_score = self._relative_shift(pos_score)

        scores = (content_score + pos_score) / (self.d_k ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float("-inf"))

        attn = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, t, self.d_model)
        return self.w_out(out)


class ConvolutionModule(nn.Module):
    """Conformer convolution module: pointwise + depthwise + pointwise."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        x = self.layer_norm(x)
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)

        # GLU gate
        x = self.pointwise_conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * torch.sigmoid(x2)

        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = F.silu(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)


class ConformerBlock(nn.Module):
    """Single Conformer block: FF + MHSA + Conv + FF (Macaron style)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.self_attn = RelativeMultiHeadAttention(d_model, num_heads, dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, conv_kernel_size, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        pos_enc: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Macaron FF (half-step)
        x = x + 0.5 * self.ff1(x)

        # Multi-head self-attention with relative position
        residual = x
        x_norm = self.attn_norm(x)
        x = residual + self.attn_dropout(self.self_attn(x_norm, pos_enc, mask))

        # Convolution module
        x = x + self.conv(x)

        # Macaron FF (half-step)
        x = x + 0.5 * self.ff2(x)

        return self.final_norm(x)


class ConformerEncoder(nn.Module):
    """Full Conformer encoder with conv subsampling frontend."""

    def __init__(
        self,
        input_dim: int = 80,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 6,
        ff_expansion: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsampling = ConvSubsampling(input_dim, d_model)
        self.pos_enc = RelativePositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                ConformerBlock(
                    d_model, num_heads, ff_expansion, conv_kernel_size, dropout
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, n_mels) mel features
            lengths: (batch,) time lengths
        Returns:
            out: (batch, time//4, d_model) encoded features
            out_lengths: (batch,) subsampled lengths
        """
        x, lengths = self.subsampling(x, lengths)
        pos = self.pos_enc(x)
        x = self.dropout(x)

        # Create padding mask
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1)  # (batch, 1, time)

        for layer in self.layers:
            x = layer(x, pos, mask)

        return x, lengths
