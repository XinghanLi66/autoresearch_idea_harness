#!/usr/bin/env python3
"""Self-contained ASR training script for speech-asr-encoder task.

The SpeechEncoder class (EDITABLE region) is the complete encoder architecture:
  - Audio feature extraction (waveform → frame-level features)
  - Subsampling / downsampling
  - Encoder blocks (attention, convolution, feed-forward, normalization)
  - Position encoding

FIXED: CTC decoder head, data loading, training loop, evaluation.

Environment variables:
  DATA_DIR   — path to dataset root (e.g. /data/speech/asr/librispeech-100)
  OUTPUT_DIR — output directory for checkpoints
  SEED       — random seed (default: 42)
  ENV        — dataset label (e.g. librispeech-100h, aishell-1, commonvoice-es)

  # Model config (via env or defaults)
  N_LAYERS   — number of encoder layers (default: 4)
  D_MODEL    — model dimension (default: 256)
  N_HEAD     — number of attention heads (default: 4)
  D_FFN      — feed-forward hidden dimension (default: 1024)
  KERNEL_SIZE — convolution kernel size (default: 31)

  # Training config
  MAX_EPOCHS     — max training epochs (default: 30)
  BATCH_SIZE     — batch size in seconds of audio (default: 32)
  LEARNING_RATE  — peak learning rate (default: 1e-3)
  WARMUP_STEPS   — warmup steps (default: 5000)
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
# EDITABLE — SpeechEncoder: complete encoder architecture
# Agent should redesign this class to improve ASR performance.
# Interface contract:
#   __init__(self, n_vocab, d_model=256, n_head=4, n_layers=4,
#            d_ffn=1024, kernel_size=31, dropout=0.1)
#   forward(self, waveform, wav_lens) -> (log_probs, out_lens)
#     waveform: (B, T) raw audio at 16kHz
#     wav_lens: (B,) relative lengths in [0, 1]
#     log_probs: (B, T', n_vocab) log-probabilities for CTC
#     out_lens: (B,) relative output lengths in [0, 1]
# ================================================================
class SpeechEncoder(nn.Module):
    """End-to-end ASR encoder with CTC output.

    Default implementation: simple Transformer encoder with log-mel features.
    This is intentionally basic — a good solution should significantly improve
    upon this baseline by designing a better encoder architecture.

    Design space includes (but is not limited to):
      - Feature extraction: mel-filterbank, learnable filterbanks, SincNet, etc.
      - Subsampling strategy: strided conv, pooling, stacking
      - Encoder block design: Transformer, Conformer, Branchformer, etc.
      - Attention mechanism: vanilla MHA, relative position, RoPE, local, etc.
      - Convolution modules: depthwise separable, multi-kernel, dynamic, etc.
      - Position encoding: sinusoidal, learned, relative, rotary, ALiBi, etc.
      - Normalization: LayerNorm, RMSNorm, pre-norm vs post-norm
      - Activation functions: ReLU, GELU, Swish/SiLU, GLU variants
    """

    def __init__(self, n_vocab, d_model=256, n_head=4, n_layers=4,
                 d_ffn=1024, kernel_size=31, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_vocab = n_vocab

        # --- Feature extraction: 80-dim log-mel filterbank ---
        self.n_mels = 80
        self.n_fft = 400  # 25ms at 16kHz
        self.hop_length = 160  # 10ms at 16kHz
        self.register_buffer(
            "mel_basis",
            self._create_mel_filterbank(16000, self.n_fft, self.n_mels),
        )

        # --- Subsampling: 2-layer CNN with stride 2 → 4x reduction ---
        self.subsample = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Linear projection after reshaping conv output
        self.subsample_proj = nn.Linear(d_model * (self.n_mels // 4), d_model)

        # --- Positional encoding: sinusoidal ---
        self.pos_enc = self._make_sinusoidal_pe(5000, d_model)
        self.dropout = nn.Dropout(dropout)

        # --- Transformer encoder blocks ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ffn,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)

        # --- CTC output projection ---
        self.ctc_proj = nn.Linear(d_model, n_vocab)

    def forward(self, waveform, wav_lens):
        """
        Args:
            waveform: (B, T) raw audio at 16kHz
            wav_lens: (B,) relative lengths in [0, 1]
        Returns:
            log_probs: (B, T', n_vocab) log-probabilities for CTC
            out_lens: (B,) relative output lengths in [0, 1]
        """
        # Feature extraction: waveform → log-mel spectrogram
        feats = self._extract_features(waveform)  # (B, n_mels, T_feat)

        # Subsampling: (B, 1, n_mels, T) → (B, d_model, n_mels//4, T//4)
        x = feats.unsqueeze(1)  # (B, 1, n_mels, T_feat)
        x = self.subsample(x)  # (B, d_model, n_mels//4, T_feat//4)
        B, C, Freq, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, C * Freq)  # (B, T, d_model*Freq)
        x = self.subsample_proj(x)  # (B, T, d_model)

        # Position encoding
        x = x + self.pos_enc[:, :T, :].to(x.device)
        x = self.dropout(x)

        # Compute output lengths (4x reduction from subsampling)
        out_lens = wav_lens  # relative lengths preserved through subsampling

        # Create padding mask
        seq_len = T
        abs_lens = (out_lens * seq_len).long()
        padding_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= abs_lens.unsqueeze(1)

        # Transformer encoder
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        x = self.final_norm(x)

        # CTC output
        log_probs = F.log_softmax(self.ctc_proj(x), dim=-1)
        return log_probs, out_lens

    def _extract_features(self, waveform):
        """Extract log-mel spectrogram features."""
        # STFT
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform, self.n_fft, self.hop_length, self.n_fft,
            window=window, return_complex=True,
        )
        power_spec = stft.abs().pow(2)  # (B, n_fft//2+1, T)

        # Mel filterbank
        mel_spec = torch.matmul(self.mel_basis, power_spec)  # (B, n_mels, T)

        # Log compression
        log_mel = torch.log(mel_spec.clamp(min=1e-9))

        # Normalize per utterance
        mean = log_mel.mean(dim=-1, keepdim=True)
        std = log_mel.std(dim=-1, keepdim=True).clamp(min=1e-6)
        log_mel = (log_mel - mean) / std

        return log_mel

    @staticmethod
    def _create_mel_filterbank(sample_rate, n_fft, n_mels, fmin=0.0, fmax=8000.0):
        """Create mel filterbank matrix."""
        def hz_to_mel(hz):
            return 2595.0 * math.log10(1.0 + hz / 700.0)

        def mel_to_hz(mel):
            return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

        mel_min = hz_to_mel(fmin)
        mel_max = hz_to_mel(fmax)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = mel_to_hz(mel_points)
        bin_points = (hz_points * n_fft / sample_rate).long()

        filterbank = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            start, center, end = bin_points[i], bin_points[i + 1], bin_points[i + 2]
            if start < center:
                filterbank[i, start:center] = torch.linspace(0, 1, center - start)
            if center < end:
                filterbank[i, center:end] = torch.linspace(1, 0, end - center)
        return filterbank

    @staticmethod
    def _make_sinusoidal_pe(max_len, d_model):
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        return pe


# ================================================================
# FIXED — Data loading, training loop, evaluation
# (do not modify below this line)
# ================================================================

class ASRDataset(Dataset):
    """Simple ASR dataset that loads audio + transcript pairs."""

    def __init__(self, manifest_path, vocab, max_duration=20.0):
        self.entries = []
        self.vocab = vocab

        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                if entry.get("duration", 0) <= max_duration:
                    self.entries.append(entry)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        # Load audio
        wav_path = entry["wav"]
        import torchaudio
        try:
            waveform, sr = torchaudio.load(wav_path)
        except (UnicodeDecodeError, RuntimeError):
            # Fallback for opus files with metadata encoding issues
            import subprocess as _sp, io as _io
            _r = _sp.run(["ffmpeg", "-i", wav_path, "-f", "wav", "-ar", "16000", "-ac", "1", "-"],
                         capture_output=True, timeout=30)
            import soundfile as _sf
            _data, sr = _sf.read(_io.BytesIO(_r.stdout))
            waveform = torch.from_numpy(_data).float().unsqueeze(0)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.squeeze(0)  # mono

        # Encode text
        text = entry["text"].lower()
        tokens = self.vocab.encode(text)
        tokens = torch.tensor(tokens, dtype=torch.long)

        return waveform, tokens, len(waveform), len(tokens)


class CharVocab:
    """Character-level vocabulary with CTC blank."""

    def __init__(self, chars=None):
        if chars is None:
            # Default: letters + space + apostrophe + blank
            chars = list("abcdefghijklmnopqrstuvwxyz '")
        self.blank_id = 0
        self.chars = ["<blank>"] + list(chars)
        self.char2idx = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text if c in self.char2idx]

    def decode(self, indices):
        return "".join(self.chars[i] for i in indices if i != self.blank_id)

    def __len__(self):
        return len(self.chars)


class ChineseCharVocab:
    """Character-level vocabulary for Chinese (CER evaluation)."""

    def __init__(self, vocab_file):
        self.blank_id = 0
        self.chars = ["<blank>"]
        with open(vocab_file) as f:
            for line in f:
                char = line.strip().split()[0]
                if char and char not in self.chars:
                    self.chars.append(char)
        self.char2idx = {c: i for i, c in enumerate(self.chars)}

    def encode(self, text):
        return [self.char2idx.get(c, 0) for c in text if c in self.char2idx]

    def decode(self, indices):
        return "".join(self.chars[i] for i in indices if i != self.blank_id)

    def __len__(self):
        return len(self.chars)


def collate_fn(batch):
    """Collate variable-length audio and text."""
    waveforms, tokens_list, wav_lengths, tok_lengths = zip(*batch)

    max_wav_len = max(wav_lengths)
    max_tok_len = max(tok_lengths)

    wav_batch = torch.zeros(len(batch), max_wav_len)
    tok_batch = torch.zeros(len(batch), max_tok_len, dtype=torch.long)
    wav_lens = torch.zeros(len(batch))
    tok_lens = torch.zeros(len(batch), dtype=torch.long)

    for i, (wav, tok, wl, tl) in enumerate(zip(waveforms, tokens_list, wav_lengths, tok_lengths)):
        wav_batch[i, :wl] = wav
        tok_batch[i, :tl] = tok
        wav_lens[i] = wl / max_wav_len  # relative length
        tok_lens[i] = tl

    return wav_batch, tok_batch, wav_lens, tok_lens


def ctc_greedy_decode(log_probs, lengths, vocab):
    """Greedy CTC decoding."""
    predictions = []
    B = log_probs.size(0)
    for i in range(B):
        seq_len = int(lengths[i] * log_probs.size(1))
        pred = log_probs[i, :seq_len].argmax(dim=-1).cpu().tolist()
        # Collapse repeats and remove blanks
        decoded = []
        prev = None
        for p in pred:
            if p != prev and p != vocab.blank_id:
                decoded.append(p)
            prev = p
        predictions.append(vocab.decode(decoded))
    return predictions


def compute_wer(hypotheses, references):
    """Compute Word Error Rate."""
    total_errors = 0
    total_words = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_words = hyp.split()
        ref_words = ref.split()
        # Simple Levenshtein distance
        d = _edit_distance(hyp_words, ref_words)
        total_errors += d
        total_words += len(ref_words)
    return total_errors / max(total_words, 1)


def compute_cer(hypotheses, references):
    """Compute Character Error Rate."""
    total_errors = 0
    total_chars = 0
    for hyp, ref in zip(hypotheses, references):
        hyp_chars = list(hyp.replace(" ", ""))
        ref_chars = list(ref.replace(" ", ""))
        d = _edit_distance(hyp_chars, ref_chars)
        total_errors += d
        total_chars += len(ref_chars)
    return total_errors / max(total_chars, 1)


def _edit_distance(a, b):
    """Compute edit distance between two sequences."""
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[m]


def build_vocab_from_manifest(manifest_path, lang="en"):
    """Build vocabulary from training manifest."""
    if lang == "zh":
        chars = set()
        with open(manifest_path) as f:
            for line in f:
                entry = json.loads(line.strip())
                for c in entry["text"]:
                    chars.add(c)
        vocab = CharVocab(sorted(chars))
    else:
        vocab = CharVocab()
    return vocab


def find_manifest(data_dir, split="train"):
    """Find manifest file in data directory."""
    candidates = [
        Path(data_dir) / f"{split}.json",
        Path(data_dir) / f"manifest_{split}.json",
        Path(data_dir) / f"{split}_manifest.json",
        Path(data_dir) / "manifest.json",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(f"No manifest found in {data_dir} for split={split}")


def main():
    # --- Configuration from environment ---
    data_dir = os.environ.get("DATA_DIR", "/data/speech/asr/librispeech-100")
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/asr_output")
    seed = int(os.environ.get("SEED", "42"))
    env_label = os.environ.get("ENV", "librispeech-100h")

    n_layers = int(os.environ.get("N_LAYERS", "4"))
    d_model = int(os.environ.get("D_MODEL", "256"))
    n_head = int(os.environ.get("N_HEAD", "4"))
    d_ffn = int(os.environ.get("D_FFN", "1024"))
    kernel_size = int(os.environ.get("KERNEL_SIZE", "31"))

    max_epochs = int(os.environ.get("MAX_EPOCHS", "30"))
    batch_size = int(os.environ.get("BATCH_SIZE", "16"))
    lr = float(os.environ.get("LEARNING_RATE", "1e-3"))
    warmup_steps = int(os.environ.get("WARMUP_STEPS", "5000"))
    weight_decay = float(os.environ.get("WEIGHT_DECAY", "0.01"))
    grad_clip = float(os.environ.get("GRAD_CLIP", "5.0"))

    # ================================================================
    # EDITABLE — CONFIG_OVERRIDES: per-method training hyperparameters
    # ================================================================
    # Allowed keys: learning_rate, weight_decay, warmup_steps, grad_clip,
    #               n_layers, max_epochs.
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
        elif _k == 'warmup_steps': warmup_steps = _v
        elif _k == 'grad_clip': grad_clip = _v
        elif _k == 'n_layers': n_layers = _v
        elif _k == 'max_epochs': max_epochs = _v

    # Determine language from env label
    lang = "zh" if "aishell" in env_label else "en"
    use_cer = lang == "zh"

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
    vocab = build_vocab_from_manifest(train_manifest, lang)
    n_vocab = len(vocab)

    train_dataset = ASRDataset(train_manifest, vocab, max_duration=20.0)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True,
        drop_last=True,
    )

    # Try to find test manifest
    test_manifest = None
    for split in ["test", "test-clean", "eval"]:
        try:
            test_manifest = find_manifest(data_dir, split)
            break
        except FileNotFoundError:
            continue

    test_loader = None
    if test_manifest:
        test_dataset = ASRDataset(test_manifest, vocab, max_duration=30.0)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=4, pin_memory=True,
        )

    # --- Model ---
    model = SpeechEncoder(
        n_vocab=n_vocab,
        d_model=d_model,
        n_head=n_head,
        n_layers=n_layers,
        d_ffn=d_ffn,
        kernel_size=kernel_size,
        dropout=0.1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}", flush=True)

    # ── FIXED: Parameter count check ────────────────────────────────
    # Budget based on 1.05x largest baseline (Conformer).
    # Conformer per block: 2x FFN(d_model->d_ffn->d_model) + RelPosAttn(5 linears + biases)
    #   + ConvModule(pointwise d->2d, depthwise d, BN, pointwise d->d) + LayerNorms
    def _param_budget(d, h, L, d_ff, ks, n_v):
        """Compute parameter budget from Conformer architecture dimensions."""
        mels = 80
        # Subsample CNN
        sub = 1*d*9 + d + d*d*9 + d
        sub_proj = d*(mels//4)*d + d
        # Conformer block
        ffn = d*2 + d*d_ff + d_ff + d_ff*d + d  # LN + linear + linear
        attn_ln = d*2
        attn = 4*(d*d + d) + d*d + d*d + d + h*(d//h)*2  # q,k,v,out + w_pos + biases
        conv_ln = d*2
        conv = d*2*d + 2*d + d*ks + d + d*2 + d*d + d  # pw1+dw+bn+pw2
        block_ln = d*2
        block = ffn*2 + attn_ln + attn + conv_ln + conv + block_ln
        # Final
        final = d*2 + d*n_v + n_v  # final_norm + ctc_proj
        total = sub + sub_proj + L*block + final
        return int(total * 1.05)

    _budget = _param_budget(d_model, n_head, n_layers, d_ffn, kernel_size, n_vocab)
    _total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {_total_params:,} (budget: {_budget:,})")

    # --- Optimizer and scheduler ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        decay_steps = max_epochs * len(train_loader) - warmup_steps
        progress = (step - warmup_steps) / max(1, decay_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # --- Training ---
    global_step = 0
    best_metric = float("inf")
    metric_name = "cer" if use_cer else "wer"

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for wav_batch, tok_batch, wav_lens, tok_lens in train_loader:
            wav_batch = wav_batch.to(device)
            tok_batch = tok_batch.to(device)
            wav_lens = wav_lens.to(device)

            log_probs, out_lens = model(wav_batch, wav_lens)

            # CTC loss
            T = log_probs.size(1)
            input_lengths = (out_lens * T).long().clamp(min=1)
            log_probs_ctc = log_probs.permute(1, 0, 2)  # (T, B, V) for CTC

            loss = F.ctc_loss(
                log_probs_ctc, tok_batch, input_lengths, tok_lens,
                blank=vocab.blank_id, zero_infinity=True,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            if global_step % 500 == 0:
                avg_loss = epoch_loss / n_batches
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"TRAIN_METRICS step={global_step} epoch={epoch+1} "
                    f"loss={avg_loss:.4f} lr={cur_lr:.6f}",
                    flush=True,
                )

        # --- Validation at end of epoch ---
        if test_loader is not None:
            model.eval()
            all_hyps, all_refs = [], []
            with torch.no_grad():
                for wav_batch, tok_batch, wav_lens, tok_lens in test_loader:
                    wav_batch = wav_batch.to(device)
                    wav_lens = wav_lens.to(device)

                    log_probs, out_lens = model(wav_batch, wav_lens)
                    hyps = ctc_greedy_decode(log_probs, out_lens, vocab)
                    all_hyps.extend(hyps)

                    # Decode references
                    for j in range(tok_batch.size(0)):
                        ref_tokens = tok_batch[j, :tok_lens[j]].tolist()
                        all_refs.append(vocab.decode(ref_tokens))

            if use_cer:
                metric_val = compute_cer(all_hyps, all_refs)
            else:
                metric_val = compute_wer(all_hyps, all_refs)

            print(
                f"TRAIN_METRICS step={global_step} epoch={epoch+1} "
                f"val_{metric_name}={metric_val:.4f}",
                flush=True,
            )

            if metric_val < best_metric:
                best_metric = metric_val
                torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))

    # --- Final evaluation ---
    if test_loader is not None and os.path.exists(os.path.join(output_dir, "best_model.pt")):
        model.load_state_dict(torch.load(os.path.join(output_dir, "best_model.pt"), weights_only=True))

    model.eval()
    if test_loader is not None:
        all_hyps, all_refs = [], []
        with torch.no_grad():
            for wav_batch, tok_batch, wav_lens, tok_lens in test_loader:
                wav_batch = wav_batch.to(device)
                wav_lens = wav_lens.to(device)

                log_probs, out_lens = model(wav_batch, wav_lens)
                hyps = ctc_greedy_decode(log_probs, out_lens, vocab)
                all_hyps.extend(hyps)

                for j in range(tok_batch.size(0)):
                    ref_tokens = tok_batch[j, :tok_lens[j]].tolist()
                    all_refs.append(vocab.decode(ref_tokens))

        final_wer = compute_wer(all_hyps, all_refs)
        final_cer = compute_cer(all_hyps, all_refs)
        print(f"TEST_METRICS wer={final_wer:.4f} cer={final_cer:.4f}", flush=True)
    else:
        print("TEST_METRICS wer=1.0000 cer=1.0000", flush=True)
        print("WARNING: No test data found, reporting worst-case metrics.", flush=True)


if __name__ == "__main__":
    main()
