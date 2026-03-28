"""Real-time microphone speech recognition."""

import argparse
import sys
import threading
import queue

import numpy as np
import torch

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


class RealtimeASR:
    """Real-time speech recognition from microphone input."""

    def __init__(
        self,
        model: ASRModel,
        tokenizer: CharTokenizer,
        config: ASRConfig,
        device: torch.device,
        chunk_seconds: float = 3.0,
        overlap_seconds: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        self.sample_rate = config.audio.sample_rate
        self.chunk_size = int(chunk_seconds * self.sample_rate)
        self.overlap_size = int(overlap_seconds * self.sample_rate)
        self.audio_queue = queue.Queue()
        self.running = False

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio callback to capture microphone input."""
        import pyaudio

        audio_data = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
        self.audio_queue.put(audio_data)
        return (in_data, pyaudio.paContinue)

    def _transcribe_chunk(self, audio_np: np.ndarray) -> str:
        """Transcribe a single audio chunk."""
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0).to(self.device)
        lengths = torch.tensor([waveform.size(1)], device=self.device)

        predictions = self.model.greedy_decode(
            waveform,
            lengths,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
            max_len=200,
        )
        return self.tokenizer.decode(predictions[0])

    def start(self):
        """Start real-time speech recognition."""
        try:
            import pyaudio
        except ImportError:
            print("Error: pyaudio is required for real-time recognition.")
            print("Install with: pip install pyaudio")
            sys.exit(1)

        pa = pyaudio.PyAudio()

        # Find input device
        info = pa.get_default_input_device_info()
        print(f"Using input device: {info['name']}")

        chunk_frames = 1024
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=chunk_frames,
            stream_callback=self._audio_callback,
        )

        print("\n" + "=" * 60)
        print("  REAL-TIME SPEECH RECOGNITION")
        print("  Speak into your microphone... (Ctrl+C to stop)")
        print("=" * 60 + "\n")

        self.running = True
        stream.start_stream()

        audio_buffer = np.array([], dtype=np.float32)

        try:
            while self.running:
                # Collect audio from queue
                while not self.audio_queue.empty():
                    chunk = self.audio_queue.get()
                    audio_buffer = np.concatenate([audio_buffer, chunk])

                # Process when we have enough audio
                if len(audio_buffer) >= self.chunk_size:
                    chunk_to_process = audio_buffer[: self.chunk_size]
                    audio_buffer = audio_buffer[self.chunk_size - self.overlap_size :]

                    # Check if chunk has speech (simple energy-based VAD)
                    energy = np.mean(chunk_to_process ** 2)
                    if energy > 1e-6:  # threshold for speech activity
                        text = self._transcribe_chunk(chunk_to_process)
                        if text.strip():
                            print(f">> {text}")

        except KeyboardInterrupt:
            print("\n\nStopping...")
        finally:
            self.running = False
            stream.stop_stream()
            stream.close()
            pa.terminate()
            print("Real-time recognition stopped.")


def main():
    parser = argparse.ArgumentParser(description="Real-time Speech Recognition")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/best.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=3.0,
        help="Audio chunk length in seconds",
    )
    parser.add_argument(
        "--overlap-seconds",
        type=float,
        default=0.5,
        help="Overlap between chunks in seconds",
    )
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    print(f"Loading model from {args.checkpoint}...")
    model, config = load_model(args.checkpoint, device)
    tokenizer = CharTokenizer(config.tokenizer)

    asr = RealtimeASR(
        model=model,
        tokenizer=tokenizer,
        config=config,
        device=device,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    asr.start()


if __name__ == "__main__":
    main()
