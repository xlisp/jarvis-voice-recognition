"""Configuration for the Conformer-Transformer ASR model."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 512
    hop_length: int = 160  # 10ms
    win_length: int = 400  # 25ms
    n_mels: int = 80
    max_audio_seconds: float = 20.0


@dataclass
class TokenizerConfig:
    vocab: str = " abcdefghijklmnopqrstuvwxyz'"
    pad_token: str = "<pad>"
    sos_token: str = "<sos>"
    eos_token: str = "<eos>"
    blank_token: str = "<blank>"

    @property
    def vocab_size(self) -> int:
        return len(self.vocab) + 4  # pad, sos, eos, blank


@dataclass
class ConformerConfig:
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 6
    ff_expansion: int = 4
    conv_kernel_size: int = 31
    dropout: float = 0.1
    input_dim: int = 80  # n_mels


@dataclass
class DecoderConfig:
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 4
    ff_expansion: int = 4
    dropout: float = 0.1
    max_target_length: int = 512


@dataclass
class TrainConfig:
    # Data
    data_root: str = "./data_librispeech"
    train_subset: str = "train-clean-100"
    val_subset: str = "dev-clean"
    num_workers: int = 4

    # Training (tuned for GTX 1080 8GB VRAM)
    epochs: int = 100
    batch_size: int = 8
    grad_accumulation: int = 4  # effective batch = 32
    max_grad_norm: float = 5.0

    # Optimizer
    lr: float = 1e-3
    weight_decay: float = 1e-6
    warmup_steps: int = 5000

    # CTC + Attention
    ctc_weight: float = 0.3  # λ for CTC loss

    # SpecAugment
    freq_mask_param: int = 27
    time_mask_param: int = 100
    num_freq_masks: int = 2
    num_time_masks: int = 2

    # Checkpoint
    checkpoint_dir: str = "./checkpoints"
    log_interval: int = 50
    save_interval: int = 1  # save every N epochs


@dataclass
class InferenceConfig:
    beam_size: int = 10
    ctc_weight: float = 0.3
    max_decode_length: int = 512
    length_penalty: float = 0.6


@dataclass
class ASRConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    encoder: ConformerConfig = field(default_factory=ConformerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
