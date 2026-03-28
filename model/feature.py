"""Audio feature extraction: Mel Spectrogram with optional augmentation."""

import torch
import torch.nn as nn
import torchaudio


class MelSpectrogram(nn.Module):
    """Extract log Mel spectrogram features from raw waveform."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
    ):
        super().__init__()
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            normalized=False,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: (batch, samples)
        Returns:
            features: (batch, time_frames, n_mels) - log mel spectrogram
        """
        # (batch, n_mels, time)
        mel = self.mel_spec(waveform)
        # Log scale with stability
        log_mel = torch.log(mel.clamp(min=1e-9))
        # Normalize per utterance
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True).clamp(min=1e-6)
        log_mel = (log_mel - mean) / std
        # Transpose to (batch, time, n_mels)
        return log_mel.transpose(1, 2)


class ConvSubsampling(nn.Module):
    """Convolutional subsampling frontend.

    Two conv layers with stride 2, reducing time dimension by 4x.
    This is standard in Conformer/Transformer ASR models.
    """

    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # After 2x stride twice: freq_dim = ceil(input_dim / 4)
        conv_out_dim = d_model * ((input_dim + 3) // 4)
        self.linear = nn.Linear(conv_out_dim, d_model)

    def forward(
        self, x: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, time, input_dim) mel features
            lengths: (batch,) original time lengths
        Returns:
            out: (batch, time//4, d_model)
            new_lengths: (batch,) subsampled lengths
        """
        # (batch, 1, time, freq)
        x = x.unsqueeze(1)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
        x = self.linear(x)
        # Update lengths after subsampling
        new_lengths = ((lengths - 1) // 2 + 1)
        new_lengths = ((new_lengths - 1) // 2 + 1)
        return x, new_lengths
