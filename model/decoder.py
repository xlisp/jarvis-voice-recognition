"""Transformer decoder for autoregressive text generation in ASR."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional import SinusoidalPositionalEncoding


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer: masked self-attn + cross-attn + FFN."""

    def __init__(
        self, d_model: int, num_heads: int, ff_expansion: int = 4, dropout: float = 0.1
    ):
        super().__init__()
        # Masked self-attention
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_dropout = nn.Dropout(dropout)

        # Cross-attention
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_dropout = nn.Dropout(dropout)

        # Feed-forward
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt: (batch, tgt_len, d_model)
            memory: (batch, src_len, d_model) encoder output
            tgt_mask: (tgt_len, tgt_len) causal mask
            tgt_key_padding_mask: (batch, tgt_len) target padding
            memory_key_padding_mask: (batch, src_len) encoder padding
        """
        # Masked self-attention (pre-norm)
        residual = tgt
        tgt_norm = self.self_attn_norm(tgt)
        tgt2, _ = self.self_attn(
            tgt_norm,
            tgt_norm,
            tgt_norm,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        tgt = residual + self.self_attn_dropout(tgt2)

        # Cross-attention
        residual = tgt
        tgt_norm = self.cross_attn_norm(tgt)
        tgt2, _ = self.cross_attn(
            tgt_norm,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
        )
        tgt = residual + self.cross_attn_dropout(tgt2)

        # Feed-forward
        residual = tgt
        tgt = residual + self.ff(self.ff_norm(tgt))

        return tgt


class TransformerDecoder(nn.Module):
    """Full Transformer decoder with token embedding and output projection."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 4,
        num_layers: int = 4,
        ff_expansion: int = 4,
        dropout: float = 0.1,
        max_length: int = 512,
    ):
        super().__init__()
        self.d_model = d_model

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_length, dropout)

        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, num_heads, ff_expansion, dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    @staticmethod
    def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal mask."""
        return torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
        )

    def forward(
        self,
        tgt_tokens: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            tgt_tokens: (batch, tgt_len) token indices
            memory: (batch, src_len, d_model) encoder output
            tgt_key_padding_mask: (batch, tgt_len) True for padded positions
            memory_key_padding_mask: (batch, src_len) True for padded positions
        Returns:
            logits: (batch, tgt_len, vocab_size)
        """
        tgt_len = tgt_tokens.size(1)
        tgt_mask = self._causal_mask(tgt_len, tgt_tokens.device)

        x = self.embedding(tgt_tokens) * (self.d_model ** 0.5)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)

        x = self.final_norm(x)
        return self.output_proj(x)
