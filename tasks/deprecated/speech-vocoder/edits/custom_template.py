#!/usr/bin/env python3
"""Self-contained vocoder training script for speech-vocoder task.

The VocoderGenerator class (EDITABLE region) is the complete generator network:
  - Mel-spectrogram → raw waveform conversion
  - Upsampling strategy, residual blocks, activation functions

FIXED: Multi-scale discriminator, GAN training loop, evaluation (PESQ, MRSTFT).

Environment variables:
  DATA_DIR   — path to dataset root
  OUTPUT_DIR — output directory
  SEED       — random seed (default: 42)
  ENV        — dataset label

  # Model config
  UPSAMPLE_RATES — comma-separated upsample rates (default: "8,8,2,2")
  D_MODEL        — initial channel width (default: 512)

  # Training config
  MAX_EPOCHS     — max training epochs (default: 100)
  BATCH_SIZE     — batch size (default: 16)
  LEARNING_RATE  — learning rate (default: 2e-4)
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
# EDITABLE — VocoderGenerator: complete generator network
# Agent should redesign this class to improve synthesis quality.
# Interface contract:
#   __init__(self, n_mels=80, upsample_rates=(8,8,2,2),
#            d_model=512, dropout=0.1)
#   forward(self, mel) -> waveform
#     mel: (B, n_mels, T_mel) mel-spectrogram
#     waveform: (B, 1, T_audio) raw audio
#       where T_audio = T_mel * prod(upsample_rates) * hop_length_factor
# ================================================================
class VocoderGenerator(nn.Module):
    """Neural vocoder generator: mel-spectrogram → raw waveform.

    Default implementation: simple transposed convolution upsampling
    with residual blocks. This is intentionally basic.

    Design space includes (but is not limited to):
      - Upsampling: transposed conv, sub-pixel conv, nearest + conv, polyphase
      - Residual blocks: dilated convolutions, multi-receptive field fusion (MRF)
      - Activation: LeakyReLU, Snake, SiLU, anti-aliased activations
      - Architecture: HiFi-GAN, MelGAN, UnivNet, WaveGlow, Vocos
      - Conditioning: FiLM, adaptive normalization, cross-attention
    """

    def __init__(self, n_mels=80, upsample_rates=(8, 8, 2, 2),
                 d_model=512, dropout=0.1):
        super().__init__()
        self.n_mels = n_mels
        self.upsample_rates = upsample_rates

        # Initial projection
        self.pre_conv = nn.Conv1d(n_mels, d_model, kernel_size=7, padding=3)

        # Upsampling blocks
        ch = d_model
        self.ups = nn.ModuleList()
        self.res_blocks = nn.ModuleList()

        for i, rate in enumerate(upsample_rates):
            next_ch = ch // 2
            self.ups.append(
                nn.ConvTranspose1d(ch, next_ch, kernel_size=rate * 2,
                                   stride=rate, padding=rate // 2)
            )
            self.res_blocks.append(
                nn.Sequential(
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(next_ch, next_ch, kernel_size=3, dilation=1, padding=1),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(next_ch, next_ch, kernel_size=3, dilation=3, padding=3),
                    nn.LeakyReLU(0.1),
                    nn.Conv1d(next_ch, next_ch, kernel_size=3, dilation=5, padding=5),
                )
            )
            ch = next_ch

        # Output projection
        self.post_conv = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Conv1d(ch, 1, kernel_size=7, padding=3),
            nn.Tanh(),
        )

    def forward(self, mel):
        """
        Args:
            mel: (B, n_mels, T_mel) log-mel spectrogram
        Returns:
            waveform: (B, 1, T_audio) synthesized audio
        """
        x = self.pre_conv(mel)

        for up, res in zip(self.ups, self.res_blocks):
            x = F.leaky_relu(x, 0.1)
            x = up(x)
            x = x + res(x)

        x = self.post_conv(x)
        return x


# ================================================================
# FIXED — Discriminator, losses, data loading, training, evaluation
# (do not modify below this line)
# ================================================================

class MultiScaleDiscriminator(nn.Module):
    """Multi-scale discriminator for GAN-based vocoder training."""

    def __init__(self, n_scales=3):
        super().__init__()
        self.discriminators = nn.ModuleList([
            ScaleDiscriminator() for _ in range(n_scales)
        ])
        self.pooling = nn.ModuleList([
            nn.AvgPool1d(kernel_size=4, stride=2, padding=2)
            for _ in range(n_scales - 1)
        ])

    def forward(self, x):
        outputs = []
        for i, disc in enumerate(self.discriminators):
            outputs.append(disc(x))
            if i < len(self.pooling):
                x = self.pooling[i](x)
        return outputs


class ScaleDiscriminator(nn.Module):
    """Single-scale discriminator."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv1d(1, 16, 15, padding=7), nn.LeakyReLU(0.1)),
            nn.Sequential(nn.Conv1d(16, 64, 41, stride=4, padding=20, groups=4), nn.LeakyReLU(0.1)),
            nn.Sequential(nn.Conv1d(64, 256, 41, stride=4, padding=20, groups=16), nn.LeakyReLU(0.1)),
            nn.Sequential(nn.Conv1d(256, 1024, 41, stride=4, padding=20, groups=64), nn.LeakyReLU(0.1)),
            nn.Sequential(nn.Conv1d(1024, 1024, 5, padding=2), nn.LeakyReLU(0.1)),
            nn.Conv1d(1024, 1, 3, padding=1),
        ])

    def forward(self, x):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


def generator_loss(disc_outputs):
    """Generator adversarial loss (hinge)."""
    loss = 0
    for disc_out in disc_outputs:
        loss += torch.mean((disc_out[-1] - 1) ** 2)
    return loss


def discriminator_loss(real_outputs, fake_outputs):
    """Discriminator adversarial loss."""
    loss = 0
    for real_out, fake_out in zip(real_outputs, fake_outputs):
        loss += torch.mean((real_out[-1] - 1) ** 2) + torch.mean(fake_out[-1] ** 2)
    return loss


def feature_matching_loss(real_outputs, fake_outputs):
    """Feature matching loss between real and fake discriminator features."""
    loss = 0
    for real_out, fake_out in zip(real_outputs, fake_outputs):
        for r, f in zip(real_out[:-1], fake_out[:-1]):
            loss += F.l1_loss(f, r.detach())
    return loss


_MEL_BASIS_CACHE: dict = {}


def _get_mel_basis(n_fft: int, n_mels: int, sample_rate: int, device):
    """Create (and cache) a mel filterbank of shape (n_mels, n_fft//2 + 1)."""
    key = (n_fft, n_mels, sample_rate, str(device))
    if key in _MEL_BASIS_CACHE:
        return _MEL_BASIS_CACHE[key]

    fmin, fmax = 0.0, sample_rate / 2.0
    def hz_to_mel(hz):
        return 2595.0 * math.log10(1.0 + hz / 700.0)
    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = (hz_points * n_fft / sample_rate).long()

    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for i in range(n_mels):
        s, c, e = int(bin_points[i]), int(bin_points[i + 1]), int(bin_points[i + 2])
        if s < c:
            fb[i, s:c] = torch.linspace(0, 1, c - s)
        if c < e:
            fb[i, c:e] = torch.linspace(1, 0, e - c)
    fb = fb.to(device)
    _MEL_BASIS_CACHE[key] = fb
    return fb


def mel_spectrogram_loss(estimated, target, n_fft=1024, hop_length=256,
                         n_mels=80, sample_rate=16000):
    """L1 loss on log mel-spectrogram (proper mel filterbank applied)."""
    window = torch.hann_window(n_fft, device=estimated.device)

    est_stft = torch.stft(estimated.squeeze(1), n_fft, hop_length, n_fft,
                          window=window, return_complex=True)
    tgt_stft = torch.stft(target.squeeze(1), n_fft, hop_length, n_fft,
                          window=window, return_complex=True)

    mel_basis = _get_mel_basis(n_fft, n_mels, sample_rate, estimated.device)

    # Apply mel filterbank on power spectrogram.
    est_power = est_stft.abs().pow(2)
    tgt_power = tgt_stft.abs().pow(2)
    est_mel = torch.matmul(mel_basis, est_power)
    tgt_mel = torch.matmul(mel_basis, tgt_power)

    est_logmel = torch.log(est_mel.clamp(min=1e-9))
    tgt_logmel = torch.log(tgt_mel.clamp(min=1e-9))

    return F.l1_loss(est_logmel, tgt_logmel)


class VocoderDataset(Dataset):
    """Vocoder dataset: loads mel-spectrogram + corresponding audio."""

    def __init__(self, manifest_path, n_mels=80, n_fft=1024, hop_length=256,
                 segment_size=8192, sample_rate=16000):
        self.entries = []
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_size = segment_size
        self.sample_rate = sample_rate

        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        import torchaudio

        waveform, sr = torchaudio.load(entry["wav"])
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.squeeze(0)

        # Random segment
        if len(waveform) > self.segment_size:
            start = random.randint(0, len(waveform) - self.segment_size)
            waveform = waveform[start:start + self.segment_size]
        else:
            waveform = F.pad(waveform, (0, self.segment_size - len(waveform)))

        # Compute mel spectrogram
        window = torch.hann_window(self.n_fft)
        stft = torch.stft(waveform, self.n_fft, self.hop_length, self.n_fft,
                          window=window, return_complex=True)
        mag = stft.abs()

        # Proper mel-triangular filterbank via torchaudio
        import torchaudio
        mel_fb = torchaudio.functional.melscale_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=0.0,
            f_max=self.sample_rate / 2.0,
            n_mels=self.n_mels,
            sample_rate=self.sample_rate,
        )  # (n_freqs, n_mels)

        mel = torch.matmul(mel_fb.T.to(mag.device), mag)
        log_mel = torch.log(mel.clamp(min=1e-9))

        return waveform.unsqueeze(0), log_mel  # (1, T), (n_mels, T_mel)


def vocoder_collate_fn(batch):
    """Collate vocoder batch."""
    waveforms, mels = zip(*batch)
    wav_batch = torch.stack(waveforms)
    mel_batch = torch.stack(mels)
    return wav_batch, mel_batch


def find_manifest(data_dir, split="train"):
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
    data_dir = os.environ.get("DATA_DIR", "/data/speech/tts/ljspeech")
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/vocoder_output")
    seed = int(os.environ.get("SEED", "42"))
    env_label = os.environ.get("ENV", "ljspeech")

    upsample_rates_str = os.environ.get("UPSAMPLE_RATES", "8,8,2,2")
    upsample_rates = tuple(int(x) for x in upsample_rates_str.split(","))
    d_model = int(os.environ.get("D_MODEL", "512"))

    max_epochs = int(os.environ.get("MAX_EPOCHS", "100"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    lr = float(os.environ.get("LEARNING_RATE", "2e-4"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.0"))
    fm_weight = float(os.environ.get("FM_WEIGHT", "2.0"))
    mel_weight = float(os.environ.get("MEL_WEIGHT", "45.0"))

    # ================================================================
    # EDITABLE — CONFIG_OVERRIDES: per-method training hyperparameters
    # ================================================================
    # Allowed keys: learning_rate, weight_decay, fm_weight, mel_weight,
    #               max_epochs.
    # Note: max_epochs increases compute; use only when the method paper
    # specifies a longer-trained recipe (e.g. UnivNet trained 500k+ steps).
    CONFIG_OVERRIDES = {}
    # ================================================================
    # FIXED — override application (do not modify)
    # ================================================================

    # Apply per-method hyperparameter overrides (fixed — do not modify).
    for _k, _v in CONFIG_OVERRIDES.items():
        if _k == 'learning_rate': lr = _v
        elif _k == 'weight_decay': weight_decay = _v
        elif _k == 'fm_weight': fm_weight = _v
        elif _k == 'mel_weight': mel_weight = _v
        elif _k == 'max_epochs': max_epochs = _v

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Data
    train_manifest = find_manifest(data_dir, "train")
    train_dataset = VocoderDataset(train_manifest, segment_size=8192)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=vocoder_collate_fn, num_workers=4, pin_memory=True,
        drop_last=True,
    )

    test_loader = None
    try:
        test_manifest = find_manifest(data_dir, "test")
        test_dataset = VocoderDataset(test_manifest, segment_size=16384)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=vocoder_collate_fn, num_workers=4, pin_memory=True,
        )
    except FileNotFoundError:
        pass

    # Model
    generator = VocoderGenerator(
        n_mels=80, upsample_rates=upsample_rates, d_model=d_model, dropout=0.1,
    ).to(device)
    discriminator = MultiScaleDiscriminator().to(device)

    n_params_g = sum(p.numel() for p in generator.parameters())
    n_params_d = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {n_params_g:,}", flush=True)
    print(f"Discriminator parameters: {n_params_d:,}", flush=True)

    # ── FIXED: Parameter count check (generator only) ───────────────
    # Budget based on 1.05x largest baseline (HiFi-GAN V1).
    # HiFi-GAN: pre_conv + 4 upsample stages with MRF (3 ResBlocks each) + post_conv
    def _voc_param_budget(n_m, d, rates):
        """Compute parameter budget from HiFi-GAN V1 architecture."""
        # pre_conv: Conv1d(n_mels, d, 7)
        total = n_m * d * 7 + d
        ch = d
        for rate in rates:
            nch = ch // 2
            # Upsample: ConvTranspose1d(ch, nch, rate*2)
            total += ch * nch * rate * 2 + nch
            # MRF: 3 HiFiResBlocks, each with 3 dilated convs + 3 post convs
            for ks in [3, 7, 11]:
                for _ in range(3):  # 3 dilations
                    total += nch * nch * ks + nch  # dilated conv
                    total += nch * nch * ks + nch  # post conv
            ch = nch
        # post_conv: Conv1d(ch, 1, 7)
        total += ch * 1 * 7 + 1
        return int(total * 1.05)

    _budget = _voc_param_budget(80, d_model, upsample_rates)
    _total_params = sum(p.numel() for p in generator.parameters())
    print(f"Total generator params: {_total_params:,} (budget: {_budget:,})")
    # ────────────────────────────────────────────────────────────────

    # Optimizers
    opt_g = torch.optim.AdamW(generator.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=weight_decay)
    opt_d = torch.optim.AdamW(discriminator.parameters(), lr=lr, betas=(0.8, 0.99), weight_decay=weight_decay)
    sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.999)
    sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=0.999)

    # Training
    best_mel_loss = float("inf")

    for epoch in range(max_epochs):
        generator.train()
        discriminator.train()
        total_g_loss = 0.0
        total_d_loss = 0.0
        n_batches = 0

        for wav_batch, mel_batch in train_loader:
            wav_batch = wav_batch.to(device)
            mel_batch = mel_batch.to(device)

            # Generate
            fake_wav = generator(mel_batch)
            min_len = min(wav_batch.size(-1), fake_wav.size(-1))
            wav_batch = wav_batch[:, :, :min_len]
            fake_wav_trunc = fake_wav[:, :, :min_len]

            # --- Discriminator step ---
            opt_d.zero_grad()
            real_out = discriminator(wav_batch)
            fake_out = discriminator(fake_wav_trunc.detach())
            d_loss = discriminator_loss(real_out, fake_out)
            d_loss.backward()
            opt_d.step()

            # --- Generator step ---
            opt_g.zero_grad()
            fake_out_g = discriminator(fake_wav_trunc)
            real_out_g = discriminator(wav_batch)

            g_adv_loss = generator_loss(fake_out_g)
            # Loss weights per original papers. Defaults match HiFi-GAN / UnivNet
            # (λ_fm=2, λ_mel=45). Baselines with different weighting (e.g. MelGAN
            # uses λ_fm=10 and no mel loss) override these two lines via edit_ops.
            g_fm_loss = feature_matching_loss(real_out_g, fake_out_g) * fm_weight
            g_mel_loss = mel_spectrogram_loss(fake_wav_trunc, wav_batch) * mel_weight

            g_loss = g_adv_loss + g_fm_loss + g_mel_loss
            g_loss.backward()
            opt_g.step()

            total_g_loss += g_loss.item()
            total_d_loss += d_loss.item()
            n_batches += 1

        sch_g.step()
        sch_d.step()

        avg_g = total_g_loss / max(n_batches, 1)
        avg_d = total_d_loss / max(n_batches, 1)
        print(
            f"TRAIN_METRICS step={epoch+1} epoch={epoch+1} "
            f"g_loss={avg_g:.4f} d_loss={avg_d:.4f}",
            flush=True,
        )

        # Save best
        if avg_g < best_mel_loss:
            best_mel_loss = avg_g
            torch.save(generator.state_dict(), os.path.join(output_dir, "best_generator.pt"))

    # --- Final evaluation ---
    best_path = os.path.join(output_dir, "best_generator.pt")
    if os.path.exists(best_path):
        generator.load_state_dict(torch.load(best_path, weights_only=True))

    generator.eval()
    if test_loader is not None:
        total_mel_loss = 0.0
        n_eval = 0
        enhanced_list, clean_list = [], []

        with torch.no_grad():
            for wav_batch, mel_batch in test_loader:
                wav_batch = wav_batch.to(device)
                mel_batch = mel_batch.to(device)

                fake_wav = generator(mel_batch)
                min_len = min(wav_batch.size(-1), fake_wav.size(-1))
                ml = mel_spectrogram_loss(fake_wav[:, :, :min_len], wav_batch[:, :, :min_len])
                total_mel_loss += ml.item() * wav_batch.size(0)
                n_eval += wav_batch.size(0)

                if len(enhanced_list) < 100:
                    for j in range(wav_batch.size(0)):
                        if len(enhanced_list) < 100:
                            enhanced_list.append(fake_wav[j, 0, :min_len].cpu().numpy())
                            clean_list.append(wav_batch[j, 0, :min_len].cpu().numpy())

        avg_mel_loss = total_mel_loss / max(n_eval, 1)

        # PESQ
        pesq_score = 0.0
        try:
            from pesq import pesq
            scores = []
            for enh, cln in zip(enhanced_list, clean_list):
                try:
                    scores.append(pesq(16000, cln, enh, "wb"))
                except Exception:
                    pass
            pesq_score = np.mean(scores) if scores else 0.0
        except ImportError:
            pass

        print(
            f"TEST_METRICS mel_loss={avg_mel_loss:.4f} pesq={pesq_score:.4f}",
            flush=True,
        )
    else:
        print("TEST_METRICS mel_loss=1.0000 pesq=0.0000", flush=True)


if __name__ == "__main__":
    main()
