"""LibriSpeech dataset loading and batching for ASR training."""

import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from .tokenizer import CharTokenizer


class LibriSpeechASRDataset(Dataset):
    """Wrapper around torchaudio's LibriSpeech dataset."""

    def __init__(
        self,
        root: str,
        subset: str = "train-clean-100",
        tokenizer: CharTokenizer | None = None,
        sample_rate: int = 16000,
        max_audio_seconds: float = 30.0,
    ):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=root, url=subset, download=True
        )
        self.tokenizer = tokenizer or CharTokenizer()
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_seconds * sample_rate)
        self.resampler_cache = {}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        waveform, sr, transcript, *_ = self.dataset[idx]

        # Resample if needed
        if sr != self.sample_rate:
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self.resampler_cache[sr](waveform)

        # Mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        # Truncate long audio
        if waveform.size(0) > self.max_samples:
            waveform = waveform[: self.max_samples]

        # Tokenize transcript
        tokens = self.tokenizer.encode(transcript)

        return {
            "audio": waveform,
            "audio_length": waveform.size(0),
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "token_length": len(tokens),
            "transcript": transcript,
        }


def collate_fn(batch: list[dict]) -> dict:
    """Collate function with dynamic padding."""
    # Sort by audio length (descending) for efficient packing
    batch.sort(key=lambda x: x["audio_length"], reverse=True)

    audios = [item["audio"] for item in batch]
    tokens = [item["tokens"] for item in batch]
    audio_lengths = torch.tensor([item["audio_length"] for item in batch])
    token_lengths = torch.tensor([item["token_length"] for item in batch])
    transcripts = [item["transcript"] for item in batch]

    # Pad sequences
    audio_padded = pad_sequence(audios, batch_first=True, padding_value=0.0)
    tokens_padded = pad_sequence(tokens, batch_first=True, padding_value=0)

    return {
        "audio": audio_padded,
        "audio_lengths": audio_lengths,
        "tokens": tokens_padded,
        "token_lengths": token_lengths,
        "transcripts": transcripts,
    }
