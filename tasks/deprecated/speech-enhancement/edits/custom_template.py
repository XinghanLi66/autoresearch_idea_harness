#!/usr/bin/env python3
"""Self-contained speech enhancement training script.

The EnhancementModel class (EDITABLE region) is the complete enhancement model:
  - STFT analysis/synthesis
  - Mask estimation or direct mapping network
  - Encoder-decoder architecture
  - Any time-domain or frequency-domain approach

FIXED: Noise mixing, training loss (SI-SNR + multi-resolution STFT), data loading, evaluation.

Environment variables:
  DATA_DIR   — path to dataset root
  OUTPUT_DIR — output directory
  SEED       — random seed (default: 42)
  ENV        — dataset label

  # Model config
  N_LAYERS   — number of layers (default: 4)
  D_MODEL    — model dimension (default: 256)
  N_HEAD     — number of attention heads (default: 4)

  # Training config
  MAX_EPOCHS     — max training epochs (default: 50)
  BATCH_SIZE     — batch size (default: 4)
  LEARNING_RATE  — peak learning rate (default: 1e-3)
"""

# ================================================================
# FIXED — imports and utilities (do not modify)
# ================================================================
import json
import math
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ================================================================
# EDITABLE — EnhancementModel: complete speech enhancement model
# Agent should redesign this class to improve enhancement quality.
# Interface contract:
#   __init__(self, n_fft=512, hop_length=256, d_model=256,
#            n_layers=4, n_head=4, dropout=0.1)
#   forward(self, noisy_waveform) -> enhanced_waveform
#     noisy_waveform: (B, T) raw audio at 16kHz
#     enhanced_waveform: (B, T) enhanced audio at 16kHz (same length)
# ================================================================
class EnhancementModel(nn.Module):
    """Speech enhancement / denoising model.

    Default implementation: simple frequency-domain masking with a small
    feed-forward network. This is intentionally basic — a good solution
    should significantly improve upon this.

    Design space includes (but is not limited to):
      - Domain: time-domain (Conv-TasNet), frequency-domain (DCCRN), hybrid
      - Architecture: U-Net, dual-path RNN, Transformer, convolutional
      - Mask type: ideal ratio mask (IRM), complex ideal ratio mask (cIRM),
                   phase-sensitive mask (PSM), direct mapping
      - Encoder/decoder: learnable basis (1D conv), STFT, multi-resolution STFT
      - Sequence modeling: LSTM, TCN, Transformer, S4, Mamba
      - Multi-scale: sub-band processing, multi-resolution
    """

    def __init__(self, n_fft=512, hop_length=256, d_model=256,
                 n_layers=4, n_head=4, dropout=0.1):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.freq_bins = n_fft // 2 + 1

        # Simple frequency-domain masking network
        self.input_proj = nn.Linear(self.freq_bins * 2, d_model)  # real + imag
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 4, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(n_layers)
        ])
        # Output: predict real and imaginary mask
        self.mask_proj = nn.Linear(d_model, self.freq_bins * 2)

    def forward(self, noisy_waveform):
        """
        Args:
            noisy_waveform: (B, T) raw audio at 16kHz
        Returns:
            enhanced_waveform: (B, T) enhanced audio (same length as input)
        """
        T_orig = noisy_waveform.shape[-1]

        # STFT
        window = torch.hann_window(self.n_fft, device=noisy_waveform.device)
        noisy_stft = torch.stft(
            noisy_waveform, self.n_fft, self.hop_length, self.n_fft,
            window=window, return_complex=True,
        )  # (B, F, T_frames)

        # Prepare input: concatenate real and imaginary parts
        real = noisy_stft.real.transpose(1, 2)  # (B, T_frames, F)
        imag = noisy_stft.imag.transpose(1, 2)  # (B, T_frames, F)
        x = torch.cat([real, imag], dim=-1)  # (B, T_frames, 2F)

        # Network forward
        x = self.input_proj(x)  # (B, T_frames, d_model)
        for layer in self.layers:
            x = x + layer(x)  # residual

        # Predict complex mask
        mask = self.mask_proj(x)  # (B, T_frames, 2F)
        mask_real, mask_imag = mask.chunk(2, dim=-1)
        mask_real = mask_real.transpose(1, 2)  # (B, F, T_frames)
        mask_imag = mask_imag.transpose(1, 2)

        # Apply mask in complex domain
        enhanced_real = noisy_stft.real * mask_real - noisy_stft.imag * mask_imag
        enhanced_imag = noisy_stft.real * mask_imag + noisy_stft.imag * mask_real
        enhanced_stft = torch.complex(enhanced_real, enhanced_imag)

        # iSTFT
        enhanced = torch.istft(
            enhanced_stft, self.n_fft, self.hop_length, self.n_fft,
            window=window, length=T_orig,
        )
        return enhanced


# ================================================================
# FIXED — Loss functions, data loading, training loop, evaluation
# (do not modify below this line)
# ================================================================

def si_snr(estimated, target, eps=1e-8):
    """Scale-invariant Signal-to-Noise Ratio (SI-SNR)."""
    # Zero-mean normalization
    estimated = estimated - estimated.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # SI-SNR
    dot = (estimated * target).sum(dim=-1, keepdim=True)
    s_target_energy = (target ** 2).sum(dim=-1, keepdim=True) + eps
    proj = dot * target / s_target_energy

    noise = estimated - proj
    si_snr_val = 10 * torch.log10(
        (proj ** 2).sum(dim=-1) / ((noise ** 2).sum(dim=-1) + eps) + eps
    )
    return si_snr_val.mean()


def multi_resolution_stft_loss(estimated, target, fft_sizes=(512, 1024, 2048)):
    """Multi-resolution STFT loss (spectral convergence + log magnitude)."""
    loss = 0.0
    for n_fft in fft_sizes:
        hop = n_fft // 4
        window = torch.hann_window(n_fft, device=estimated.device)

        est_stft = torch.stft(estimated, n_fft, hop, n_fft, window=window, return_complex=True)
        tgt_stft = torch.stft(target, n_fft, hop, n_fft, window=window, return_complex=True)

        est_mag = est_stft.abs()
        tgt_mag = tgt_stft.abs()

        # Spectral convergence
        sc = torch.norm(tgt_mag - est_mag, p="fro") / (torch.norm(tgt_mag, p="fro") + 1e-8)
        # Log magnitude loss
        lm = F.l1_loss(torch.log(est_mag.clamp(min=1e-9)), torch.log(tgt_mag.clamp(min=1e-9)))

        loss += sc + lm

    return loss / len(fft_sizes)


class SEDataset(Dataset):
    """Speech enhancement dataset with noisy/clean pairs."""

    def __init__(self, manifest_path, max_duration=6.0, sample_rate=16000):
        self.entries = []
        self.sample_rate = sample_rate
        self.max_samples = int(max_duration * sample_rate)

        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        import torchaudio

        # Load noisy
        noisy, sr = torchaudio.load(entry["noisy"])
        if sr != self.sample_rate:
            noisy = torchaudio.functional.resample(noisy, sr, self.sample_rate)
        noisy = noisy.squeeze(0)

        # Load clean
        clean, sr = torchaudio.load(entry["clean"])
        if sr != self.sample_rate:
            clean = torchaudio.functional.resample(clean, sr, self.sample_rate)
        clean = clean.squeeze(0)

        # Truncate/pad to same length
        min_len = min(len(noisy), len(clean), self.max_samples)
        noisy = noisy[:min_len]
        clean = clean[:min_len]

        return noisy, clean


def se_collate_fn(batch):
    """Collate variable-length noisy/clean pairs."""
    noisy_list, clean_list = zip(*batch)
    max_len = max(n.size(0) for n in noisy_list)

    noisy_batch = torch.zeros(len(batch), max_len)
    clean_batch = torch.zeros(len(batch), max_len)

    for i, (n, c) in enumerate(zip(noisy_list, clean_list)):
        noisy_batch[i, :n.size(0)] = n
        clean_batch[i, :c.size(0)] = c

    return noisy_batch, clean_batch


def compute_pesq(enhanced_list, clean_list, sr=16000):
    """Compute PESQ score (if pesq package available)."""
    try:
        from pesq import pesq
        scores = []
        for enh, cln in zip(enhanced_list, clean_list):
            try:
                score = pesq(sr, cln, enh, "wb")
                scores.append(score)
            except Exception:
                continue
        return np.mean(scores) if scores else 0.0
    except ImportError:
        return 0.0


def compute_stoi(enhanced_list, clean_list, sr=16000):
    """Compute STOI score (if pystoi package available)."""
    try:
        from pystoi import stoi
        scores = []
        for enh, cln in zip(enhanced_list, clean_list):
            try:
                score = stoi(cln, enh, sr, extended=False)
                scores.append(score)
            except Exception:
                continue
        return np.mean(scores) if scores else 0.0
    except ImportError:
        return 0.0


def find_manifest(data_dir, split="train"):
    """Find manifest file."""
    candidates = [
        Path(data_dir) / f"{split}.json",
        Path(data_dir) / f"manifest_{split}.json",
        Path(data_dir) / "manifest.json",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"No manifest found in {data_dir} for split={split}")


def main():
    # --- Configuration ---
    data_dir = os.environ.get("DATA_DIR", "/data/speech/se/voicebank-demand")
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/se_output")
    seed = int(os.environ.get("SEED", "42"))
    env_label = os.environ.get("ENV", "voicebank-demand")

    n_layers = int(os.environ.get("N_LAYERS", "4"))
    d_model = int(os.environ.get("D_MODEL", "256"))
    n_head = int(os.environ.get("N_HEAD", "4"))

    max_epochs = int(os.environ.get("MAX_EPOCHS", "50"))
    batch_size = int(os.environ.get("BATCH_SIZE", "4"))
    lr = float(os.environ.get("LEARNING_RATE", "1e-3"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    grad_clip = float(os.environ.get("GRAD_CLIP", "5.0"))
    sisnr_weight = float(os.environ.get("SISNR_WEIGHT", "1.0"))
    stft_weight = float(os.environ.get("STFT_WEIGHT", "1.0"))

    # ================================================================
    # EDITABLE — CONFIG_OVERRIDES: per-method training hyperparameters
    # ================================================================
    # Allowed keys: learning_rate, weight_decay, grad_clip, sisnr_weight,
    #               stft_weight, n_layers, max_epochs.
    # Note: n_layers / max_epochs increase compute; use only when the method
    # paper specifies a deeper or longer-trained recipe.
    CONFIG_OVERRIDES = {}
    # ================================================================
    # FIXED — override application (do not modify)
    # ================================================================

    # Apply per-method hyperparameter overrides (fixed — do not modify).
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': lr = _v
        elif _k == 'weight_decay': weight_decay = _v
        elif _k == 'grad_clip': grad_clip = _v
        elif _k == 'sisnr_weight': sisnr_weight = _v
        elif _k == 'stft_weight': stft_weight = _v
        elif _k == 'n_layers': n_layers = _v
        elif _k == 'max_epochs': max_epochs = _v

    # --- Reproducibility ---
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # --- Data ---
    train_manifest = find_manifest(data_dir, "train")
    train_dataset = SEDataset(train_manifest, max_duration=6.0)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=se_collate_fn, num_workers=4, pin_memory=True,
        drop_last=True,
    )

    test_loader = None
    try:
        test_manifest = find_manifest(data_dir, "test")
        test_dataset = SEDataset(test_manifest, max_duration=10.0)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=se_collate_fn, num_workers=4, pin_memory=True,
        )
    except FileNotFoundError:
        pass

    # --- Model ---
    model = EnhancementModel(
        n_fft=512,
        hop_length=256,
        d_model=d_model,
        n_layers=n_layers,
        n_head=n_head,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    # ── FIXED: Parameter count check ────────────────────────────────
    # Budget based on 1.05x largest baseline (DPT-Net).
    # DPT-Net: encoder(1->256,L=20) + bottleneck(GN+conv) + n_layers dual-path blocks
    #   (each has 2x TransformerEncoderLayer with d=256, ffn=1024) + mask net + decoder
    def _se_param_budget(d, nl):
        """Compute parameter budget from DPT-Net architecture dimensions."""
        N = 256  # DPT-Net encoder filters (fixed at 256)
        # Encoder: Conv1d(1, N, 20) no bias
        enc = 1 * N * 20
        # Bottleneck: GroupNorm(1,N) + Conv1d(N,N,1)
        bn = N*2 + N*N + N
        # Dual-path block: 2x TransformerEncoderLayer(d=N, ffn=4*N)
        # Each TEL: self_attn(4 linear d->d) + 2 linear(d->4d, 4d->d) + 2 LN(d)
        tel = 4*(N*N + N) + N*4*N + 4*N + 4*N*N + N + N*2 + N*2
        dp_block = N*2 + tel + N*2 + tel  # 2x (GroupNorm + TEL)
        # Mask net: PReLU(1) + Conv1d(N,N,1)
        mask = 1 + N*N + N
        # Decoder: ConvTranspose1d(N, 1, 20) no bias
        dec = N * 1 * 20
        total = enc + bn + nl * dp_block + mask + dec
        return int(total * 1.05)

    _budget = _se_param_budget(d_model, n_layers)
    _total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {_total_params:,} (budget: {_budget:,})")

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)

    # --- Training ---
    best_si_snr = -float("inf")

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for noisy, clean in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            enhanced = model(noisy)

            # Combined loss: -SI-SNR + multi-resolution STFT
            loss_sisnr = -si_snr(enhanced, clean)
            loss_stft = multi_resolution_stft_loss(enhanced, clean)
            loss = sisnr_weight * loss_sisnr + stft_weight * loss_stft

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        cur_lr = scheduler.get_last_lr()[0]

        print(
            f"TRAIN_METRICS step={epoch+1} epoch={epoch+1} "
            f"loss={avg_loss:.4f} lr={cur_lr:.6f}",
            flush=True,
        )

        # --- Validation ---
        if test_loader is not None and (epoch + 1) % 5 == 0:
            model.eval()
            total_sisnr = 0.0
            n_eval = 0
            with torch.no_grad():
                for noisy, clean in test_loader:
                    noisy = noisy.to(device)
                    clean = clean.to(device)
                    enhanced = model(noisy)
                    total_sisnr += si_snr(enhanced, clean).item() * noisy.size(0)
                    n_eval += noisy.size(0)

            val_sisnr = total_sisnr / max(n_eval, 1)
            print(
                f"TRAIN_METRICS step={epoch+1} epoch={epoch+1} "
                f"val_si_snr={val_sisnr:.4f}",
                flush=True,
            )

            if val_sisnr > best_si_snr:
                best_si_snr = val_sisnr
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    # --- Final evaluation ---
    best_path = os.path.join(output_dir, "best_model.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, weights_only=True))

    model.eval()
    if test_loader is not None:
        total_sisnr = 0.0
        n_eval = 0
        enhanced_list, clean_list = [], []

        with torch.no_grad():
            for noisy, clean in test_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                enhanced = model(noisy)
                total_sisnr += si_snr(enhanced, clean).item() * noisy.size(0)
                n_eval += noisy.size(0)

                # Collect for PESQ/STOI (first 200 samples)
                if len(enhanced_list) < 200:
                    for j in range(enhanced.size(0)):
                        if len(enhanced_list) < 200:
                            enhanced_list.append(enhanced[j].cpu().numpy())
                            clean_list.append(clean[j].cpu().numpy())

        final_sisnr = total_sisnr / max(n_eval, 1)
        final_pesq = compute_pesq(enhanced_list, clean_list)
        final_stoi = compute_stoi(enhanced_list, clean_list)

        print(
            f"TEST_METRICS si_snr={final_sisnr:.4f} pesq={final_pesq:.4f} stoi={final_stoi:.4f}",
            flush=True,
        )
    else:
        print("TEST_METRICS si_snr=0.0000 pesq=0.0000 stoi=0.0000", flush=True)


if __name__ == "__main__":
    main()
