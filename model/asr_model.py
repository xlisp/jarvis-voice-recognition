"""Complete ASR model: Conformer Encoder + Transformer Decoder + CTC."""

import torch
import torch.nn as nn

from .feature import MelSpectrogram
from .conformer import ConformerEncoder
from .decoder import TransformerDecoder
from config import ASRConfig


class ASRModel(nn.Module):
    """End-to-end ASR model with CTC + Attention hybrid loss."""

    def __init__(self, config: ASRConfig):
        super().__init__()
        self.config = config
        vocab_size = config.tokenizer.vocab_size

        # Feature extraction
        self.feature_extractor = MelSpectrogram(
            sample_rate=config.audio.sample_rate,
            n_fft=config.audio.n_fft,
            hop_length=config.audio.hop_length,
            win_length=config.audio.win_length,
            n_mels=config.audio.n_mels,
        )

        # Conformer encoder
        self.encoder = ConformerEncoder(
            input_dim=config.encoder.input_dim,
            d_model=config.encoder.d_model,
            num_heads=config.encoder.num_heads,
            num_layers=config.encoder.num_layers,
            ff_expansion=config.encoder.ff_expansion,
            conv_kernel_size=config.encoder.conv_kernel_size,
            dropout=config.encoder.dropout,
        )

        # CTC head
        self.ctc_proj = nn.Linear(config.encoder.d_model, vocab_size)

        # Transformer decoder
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            d_model=config.decoder.d_model,
            num_heads=config.decoder.num_heads,
            num_layers=config.decoder.num_layers,
            ff_expansion=config.decoder.ff_expansion,
            dropout=config.decoder.dropout,
            max_length=config.decoder.max_target_length,
        )

        # Loss functions
        self.ctc_loss = nn.CTCLoss(blank=vocab_size - 1, zero_infinity=True)
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # pad=0

    def encode(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract features and encode.

        Args:
            audio: (batch, samples) raw waveform
            audio_lengths: (batch,) sample counts
        Returns:
            encoder_out: (batch, time, d_model)
            encoder_lengths: (batch,)
        """
        features = self.feature_extractor(audio)
        # Convert sample lengths to frame lengths
        frame_lengths = audio_lengths // self.config.audio.hop_length + 1
        encoder_out, encoder_lengths = self.encoder(features, frame_lengths)
        return encoder_out, encoder_lengths

    def forward(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass with CTC + Attention hybrid loss.

        Args:
            audio: (batch, samples)
            audio_lengths: (batch,)
            targets: (batch, max_target_len) with <sos> prefix
            target_lengths: (batch,) excluding padding
        Returns:
            dict with 'loss', 'ctc_loss', 'att_loss'
        """
        encoder_out, encoder_lengths = self.encode(audio, audio_lengths)

        # --- CTC loss ---
        ctc_logits = self.ctc_proj(encoder_out)  # (batch, time, vocab)
        ctc_log_probs = ctc_logits.log_softmax(dim=-1).transpose(0, 1)  # (time, batch, vocab)

        # CTC targets: remove <sos> token (first column)
        ctc_targets = targets[:, 1:]  # skip <sos>
        ctc_target_lengths = target_lengths - 1  # exclude <sos>

        ctc_loss = self.ctc_loss(
            ctc_log_probs, ctc_targets, encoder_lengths, ctc_target_lengths
        )

        # --- Attention loss ---
        # Decoder input: targets with <sos> prefix (teacher forcing)
        decoder_input = targets[:, :-1]  # exclude last token
        decoder_target = targets[:, 1:]  # shift right

        # Padding masks
        enc_padding_mask = self._make_padding_mask(encoder_lengths, encoder_out.size(1))
        dec_padding_mask = self._make_padding_mask(
            target_lengths - 1, decoder_input.size(1)
        )

        decoder_logits = self.decoder(
            decoder_input,
            encoder_out,
            tgt_key_padding_mask=dec_padding_mask,
            memory_key_padding_mask=enc_padding_mask,
        )

        att_loss = self.ce_loss(
            decoder_logits.reshape(-1, decoder_logits.size(-1)),
            decoder_target.reshape(-1),
        )

        # Combined loss
        ctc_weight = self.config.train.ctc_weight
        loss = ctc_weight * ctc_loss + (1 - ctc_weight) * att_loss

        return {"loss": loss, "ctc_loss": ctc_loss, "att_loss": att_loss}

    @staticmethod
    def _make_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Create boolean padding mask: True for padded positions."""
        return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

    @torch.no_grad()
    def greedy_decode(
        self, audio: torch.Tensor, audio_lengths: torch.Tensor, sos_id: int, eos_id: int, max_len: int = 200
    ) -> list[list[int]]:
        """Simple greedy decoding for inference."""
        self.eval()
        encoder_out, encoder_lengths = self.encode(audio, audio_lengths)
        enc_padding_mask = self._make_padding_mask(encoder_lengths, encoder_out.size(1))

        batch_size = audio.size(0)
        device = audio.device

        # Start with <sos>
        ys = torch.full((batch_size, 1), sos_id, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_len):
            logits = self.decoder(
                ys, encoder_out, memory_key_padding_mask=enc_padding_mask
            )
            next_token = logits[:, -1, :].argmax(dim=-1)  # (batch,)
            next_token = next_token.masked_fill(finished, 0)  # pad finished
            ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos_id)
            if finished.all():
                break

        results = []
        for i in range(batch_size):
            tokens = ys[i, 1:].tolist()  # skip <sos>
            # Trim at <eos>
            if eos_id in tokens:
                tokens = tokens[: tokens.index(eos_id)]
            results.append(tokens)
        return results

    @torch.no_grad()
    def beam_search_decode(
        self,
        audio: torch.Tensor,
        audio_lengths: torch.Tensor,
        sos_id: int,
        eos_id: int,
        beam_size: int = 10,
        max_len: int = 200,
        length_penalty: float = 0.6,
    ) -> list[list[int]]:
        """Beam search decoding (batch_size=1 for simplicity)."""
        self.eval()
        encoder_out, encoder_lengths = self.encode(audio, audio_lengths)
        enc_padding_mask = self._make_padding_mask(encoder_lengths, encoder_out.size(1))

        device = audio.device
        results = []

        for i in range(audio.size(0)):
            enc = encoder_out[i : i + 1]  # (1, T, D)
            enc_mask = enc_padding_mask[i : i + 1]

            # Each beam: (score, token_list)
            beams = [(0.0, [sos_id])]

            for _ in range(max_len):
                candidates = []
                for score, tokens in beams:
                    if tokens[-1] == eos_id:
                        candidates.append((score, tokens))
                        continue

                    ys = torch.tensor([tokens], dtype=torch.long, device=device)
                    logits = self.decoder(ys, enc, memory_key_padding_mask=enc_mask)
                    log_probs = logits[0, -1].log_softmax(dim=-1)

                    topk_probs, topk_ids = log_probs.topk(beam_size)
                    for prob, tok_id in zip(topk_probs.tolist(), topk_ids.tolist()):
                        candidates.append((score + prob, tokens + [tok_id]))

                # Keep top beams
                candidates.sort(key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
                beams = candidates[:beam_size]

                # All beams finished?
                if all(b[1][-1] == eos_id for b in beams):
                    break

            best_tokens = beams[0][1][1:]  # skip <sos>
            if eos_id in best_tokens:
                best_tokens = best_tokens[: best_tokens.index(eos_id)]
            results.append(best_tokens)

        return results
