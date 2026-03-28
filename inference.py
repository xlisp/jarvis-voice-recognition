"""Offline inference script for ASR model."""

import argparse
import torch
import torchaudio

from config import ASRConfig
from model import ASRModel
from data import CharTokenizer


def load_model(checkpoint_path: str, device: torch.device) -> tuple[ASRModel, ASRConfig]:
    """Load trained model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get("config", ASRConfig())
    model = ASRModel(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, config


def transcribe_file(
    model: ASRModel,
    audio_path: str,
    tokenizer: CharTokenizer,
    config: ASRConfig,
    device: torch.device,
    use_beam_search: bool = False,
) -> str:
    """Transcribe a single audio file."""
    # Load audio
    waveform, sr = torchaudio.load(audio_path)

    # Resample if needed
    if sr != config.audio.sample_rate:
        resampler = torchaudio.transforms.Resample(sr, config.audio.sample_rate)
        waveform = resampler(waveform)

    # Mono
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    # Prepare batch
    audio = waveform.unsqueeze(0).to(device)
    audio_lengths = torch.tensor([waveform.size(0)], device=device)

    # Decode
    if use_beam_search:
        predictions = model.beam_search_decode(
            audio,
            audio_lengths,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            beam_size=config.inference.beam_size,
            max_len=config.inference.max_decode_length,
            length_penalty=config.inference.length_penalty,
        )
    else:
        predictions = model.greedy_decode(
            audio,
            audio_lengths,
            sos_id=tokenizer.sos_id,
            eos_id=tokenizer.eos_id,
            max_len=config.inference.max_decode_length,
        )

    return tokenizer.decode(predictions[0])


def main():
    parser = argparse.ArgumentParser(description="ASR Inference")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file (wav/flac)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument("--beam-search", action="store_true", help="Use beam search decoding")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)

    tokenizer = CharTokenizer(config.tokenizer)

    # Transcribe
    decode_method = "beam search" if args.beam_search else "greedy"
    print(f"Transcribing with {decode_method} decoding...")

    text = transcribe_file(model, args.audio, tokenizer, config, device, args.beam_search)
    print(f"\nTranscription: {text}")


if __name__ == "__main__":
    main()
