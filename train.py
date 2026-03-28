"""Training script for Conformer-Transformer ASR model.

Optimized for GTX 1080 (8GB VRAM) with mixed precision training.
"""

import os
import time
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from config import ASRConfig
from model import ASRModel
from data import CharTokenizer, LibriSpeechASRDataset, collate_fn, SpecAugment


def get_lr(step: int, d_model: int, warmup_steps: int) -> float:
    """Noam learning rate schedule with warmup."""
    if step == 0:
        step = 1
    return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


def train():
    config = ASRConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        # Enable TF32 for Turing+ GPUs (1080 is Pascal, no TF32, but harmless)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # Tokenizer
    tokenizer = CharTokenizer(config.tokenizer)
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Dataset
    print("Loading training dataset...")
    train_dataset = LibriSpeechASRDataset(
        root=config.train.data_root,
        subset=config.train.train_subset,
        tokenizer=tokenizer,
        sample_rate=config.audio.sample_rate,
        max_audio_seconds=config.audio.max_audio_seconds,
    )
    print("Loading validation dataset...")
    val_dataset = LibriSpeechASRDataset(
        root=config.train.data_root,
        subset=config.train.val_subset,
        tokenizer=tokenizer,
        sample_rate=config.audio.sample_rate,
        max_audio_seconds=config.audio.max_audio_seconds,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=config.train.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model
    model = ASRModel(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(
        f"Training config: batch_size={config.train.batch_size} x "
        f"grad_accum={config.train.grad_accumulation} = "
        f"effective_batch={config.train.batch_size * config.train.grad_accumulation}"
    )

    # SpecAugment
    spec_augment = SpecAugment(
        freq_mask_param=config.train.freq_mask_param,
        time_mask_param=config.train.time_mask_param,
        num_freq_masks=config.train.num_freq_masks,
        num_time_masks=config.train.num_time_masks,
    ).to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr(
            step, config.encoder.d_model, config.train.warmup_steps
        ),
    )

    # Mixed precision scaler for GTX 1080 (FP16)
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("Mixed precision training: ENABLED (FP16)")

    # Checkpoint directory
    os.makedirs(config.train.checkpoint_dir, exist_ok=True)

    # Resume from checkpoint if exists
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    checkpoint_path = os.path.join(config.train.checkpoint_dir, "latest.pt")
    if os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed at epoch {start_epoch}, step {global_step}")

    # Training loop
    for epoch in range(start_epoch, config.train.epochs):
        model.train()
        spec_augment.train()
        epoch_loss = 0.0
        epoch_ctc = 0.0
        epoch_att = 0.0
        num_batches = 0
        t0 = time.time()

        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            audio = batch["audio"].to(device, non_blocking=True)
            audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
            tokens = batch["tokens"].to(device, non_blocking=True)
            token_lengths = batch["token_lengths"].to(device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast(device_type="cuda", enabled=use_amp):
                losses = model(audio, audio_lengths, tokens, token_lengths)
                loss = losses["loss"] / config.train.grad_accumulation

            # Backward with scaler
            scaler.scale(loss).backward()

            if (batch_idx + 1) % config.train.grad_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.train.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            epoch_loss += losses["loss"].item()
            epoch_ctc += losses["ctc_loss"].item()
            epoch_att += losses["att_loss"].item()
            num_batches += 1

            if (batch_idx + 1) % config.train.log_interval == 0:
                avg_loss = epoch_loss / num_batches
                lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0
                gpu_mem_used = (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if device.type == "cuda"
                    else 0
                )
                print(
                    f"Epoch {epoch+1} [{batch_idx+1}/{len(train_loader)}] "
                    f"loss={avg_loss:.4f} ctc={epoch_ctc/num_batches:.4f} "
                    f"att={epoch_att/num_batches:.4f} lr={lr:.2e} "
                    f"mem={gpu_mem_used:.1f}GB time={elapsed:.1f}s"
                )

        avg_train_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - t0
        print(
            f"\nEpoch {epoch+1} train loss: {avg_train_loss:.4f} "
            f"({epoch_time:.0f}s)"
        )

        # Validation
        val_loss = validate(model, val_loader, device, tokenizer, use_amp)
        print(f"Epoch {epoch+1} val loss: {val_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config.train.save_interval == 0:
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            torch.save(ckpt, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = os.path.join(config.train.checkpoint_dir, "best.pt")
                torch.save(ckpt, best_path)
                print(f"Best model saved: {best_path}")

    print("Training complete!")


@torch.no_grad()
def validate(
    model: ASRModel,
    val_loader: DataLoader,
    device: torch.device,
    tokenizer: CharTokenizer,
    use_amp: bool = False,
) -> float:
    """Run validation and print sample predictions."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(val_loader):
        audio = batch["audio"].to(device, non_blocking=True)
        audio_lengths = batch["audio_lengths"].to(device, non_blocking=True)
        tokens = batch["tokens"].to(device, non_blocking=True)
        token_lengths = batch["token_lengths"].to(device, non_blocking=True)

        with autocast(device_type="cuda", enabled=use_amp):
            losses = model(audio, audio_lengths, tokens, token_lengths)
        total_loss += losses["loss"].item()
        num_batches += 1

        # Print sample predictions from first batch
        if batch_idx == 0:
            predictions = model.greedy_decode(
                audio[:3],
                audio_lengths[:3],
                sos_id=tokenizer.sos_id,
                eos_id=tokenizer.eos_id,
            )
            print("\n--- Sample Predictions ---")
            for i, pred_ids in enumerate(predictions):
                pred_text = tokenizer.decode(pred_ids)
                ref_text = batch["transcripts"][i]
                print(f"  REF: {ref_text}")
                print(f"  HYP: {pred_text}")
                print()

    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    train()
