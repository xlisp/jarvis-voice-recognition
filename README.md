# Jarvis Voice Recognition

A state-of-the-art English speech recognition (ASR) system built from scratch using **Conformer-Transformer** architecture with CTC/Attention hybrid training.

Optimized for training on **NVIDIA GTX 1080** (8GB VRAM) with mixed precision (FP16).

## Architecture

```
Raw Audio
    |
    v
Mel Spectrogram (80-dim, 16kHz)
    |
    v
Conv Subsampling (4x time reduction)
    |
    v
Conformer Encoder (6 layers)
    |  - Macaron Feed Forward (SiLU + half-step residual)
    |  - Relative Multi-Head Self-Attention (4 heads)
    |  - Depthwise Separable Convolution (kernel=31, GLU gate)
    |  - Macaron Feed Forward
    |
    +-------> CTC Head ---------> CTC Loss
    |
    v
Transformer Decoder (4 layers)
    |  - Masked Self-Attention
    |  - Cross-Attention (to encoder)
    |  - Feed Forward (SiLU)
    |
    v
Text Output -----------------> Cross-Entropy Loss

Total Loss = 0.3 * CTC + 0.7 * Attention
```

## Key Features

- **Conformer Encoder** - Combines convolution (local patterns) and self-attention (global context), outperforming pure Transformer on speech tasks
- **Relative Positional Encoding** - Better generalization to variable-length audio
- **CTC + Attention Hybrid** - Joint CTC/Attention training for stable convergence and accurate alignment
- **SpecAugment** - Frequency and time masking for data augmentation
- **Mixed Precision (FP16)** - 1.5-2x faster training, lower VRAM usage
- **Beam Search Decoding** - With length penalty for better results
- **Real-time Recognition** - Microphone input with sliding window inference

## Project Structure

```
jarvis-voice-recognition/
├── config.py              # All hyperparameters
├── model/
│   ├── positional.py      # Sinusoidal + relative positional encoding
│   ├── feature.py         # Mel spectrogram + conv subsampling
│   ├── conformer.py       # Conformer encoder (6 layers)
│   ├── decoder.py         # Transformer decoder (4 layers)
│   └── asr_model.py       # Full ASR model with CTC + greedy/beam decode
├── data/
│   ├── tokenizer.py       # Character-level tokenizer (a-z, space, ')
│   ├── dataset.py         # LibriSpeech dataset loader
│   └── augment.py         # SpecAugment
├── train.py               # Training with AMP, grad accumulation, checkpointing
├── inference.py           # Offline transcription (greedy / beam search)
├── realtime.py            # Real-time microphone recognition
└── requirements.txt
```

## Requirements

- Python 3.10+
- NVIDIA GPU with 8GB+ VRAM (tested on GTX 1080)
- CUDA 11.8+

```bash
pip install torch torchaudio numpy soundfile pyaudio
```

## Quick Start

### 1. Train

LibriSpeech dataset will be downloaded automatically on first run (~6GB for train-clean-100).

```bash
python train.py
```

Training config (GTX 1080 optimized):

| Parameter | Value |
|-----------|-------|
| Batch size | 8 |
| Gradient accumulation | 4 (effective batch = 32) |
| Mixed precision | FP16 |
| Learning rate | Noam schedule (warmup 5000 steps) |
| Optimizer | AdamW (beta 0.9/0.98) |
| Max audio length | 20 seconds |
| Encoder | 6 Conformer layers, d=256, 4 heads |
| Decoder | 4 Transformer layers, d=256, 4 heads |
| Loss | 0.3 * CTC + 0.7 * CrossEntropy |

Expected VRAM usage: ~6-7 GB peak.

Training automatically resumes from the latest checkpoint if interrupted.

### 2. Inference

Transcribe an audio file:

```bash
# Greedy decoding (fast)
python inference.py --audio path/to/audio.wav

# Beam search decoding (better accuracy)
python inference.py --audio path/to/audio.wav --beam-search
```

Supports WAV, FLAC, and other formats via torchaudio.

### 3. Real-time Recognition

Live transcription from microphone:

```bash
python realtime.py
```

Options:
```bash
python realtime.py --chunk-seconds 3.0 --overlap-seconds 0.5
```

## Model Details

### Conformer Block

Each Conformer block follows the Macaron structure:

```
Input
  |
  +-- 0.5 * FeedForward(LayerNorm -> Linear -> SiLU -> Linear)
  |
  +-- RelativeMultiHeadAttention(content_score + position_score)
  |
  +-- ConvModule(Pointwise -> GLU -> DepthwiseConv -> BatchNorm -> SiLU -> Pointwise)
  |
  +-- 0.5 * FeedForward
  |
  +-- LayerNorm
  |
Output
```

### Training Pipeline

1. Raw audio -> 80-dim log Mel spectrogram (25ms window, 10ms hop)
2. Conv subsampling reduces time dimension 4x
3. 6 Conformer layers encode acoustic features
4. CTC branch: linear projection -> CTC loss
5. Decoder branch: 4-layer Transformer with teacher forcing -> cross-entropy loss
6. Combined loss with gradient accumulation over 4 mini-batches
7. Noam LR schedule with 5000-step warmup

### Decoding

- **Greedy**: Fastest, picks argmax at each step
- **Beam Search**: Explores top-k hypotheses with length penalty normalization

## Checkpoints

Saved to `./checkpoints/`:
- `latest.pt` - Most recent checkpoint (for resuming)
- `best.pt` - Best validation loss

Each checkpoint contains model weights, optimizer state, scheduler state, AMP scaler state, and config.

## References

- Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*, 2020
- Park et al., *SpecAugment: A Simple Data Augmentation Method for ASR*, 2019
- Watanabe et al., *Hybrid CTC/Attention Architecture for End-to-End Speech Recognition*, 2017
- Vaswani et al., *Attention Is All You Need*, 2017
