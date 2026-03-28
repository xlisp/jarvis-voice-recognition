"""SpecAugment: data augmentation for speech recognition.

Reference: Park et al., "SpecAugment: A Simple Data Augmentation Method
for Automatic Speech Recognition", 2019.
"""

import torch
import torch.nn as nn


class SpecAugment(nn.Module):
    """Apply SpecAugment (frequency and time masking) to mel features."""

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        num_freq_masks: int = 2,
        num_time_masks: int = 2,
    ):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, freq) mel spectrogram features
        Returns:
            augmented features with same shape
        """
        if not self.training:
            return x

        x = x.clone()
        _, time_steps, freq_bins = x.shape

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = torch.randint(0, min(self.freq_mask_param, freq_bins), (1,)).item()
            f0 = torch.randint(0, max(1, freq_bins - f), (1,)).item()
            x[:, :, f0 : f0 + f] = 0.0

        # Time masking
        for _ in range(self.num_time_masks):
            t = torch.randint(0, min(self.time_mask_param, time_steps), (1,)).item()
            t0 = torch.randint(0, max(1, time_steps - t), (1,)).item()
            x[:, t0 : t0 + t, :] = 0.0

        return x
