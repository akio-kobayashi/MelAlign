#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convert *.pt mel-spectrogram files (shape: [T, 80]) to 22.05-kHz waveforms
using NVIDIA/HiFi-GAN torch-hub model.
"""
from __future__ import annotations

import glob, os, warnings
from argparse import ArgumentParser
from pathlib import Path

import torch
import torchaudio
#from einops import rearrange

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np

def plot_mel_spectrogram_to_file(mel_tensor, filename="mel_output.png", sr=22050, hop_length=256):
    """
    mel_tensor: torch.Tensor or np.ndarray, shape = (n_mel, T)
    filename: 出力画像ファイル名
    sr: サンプリングレート（時間軸表示用）
    hop_length: フレームあたりのサンプル数（時間軸表示用）
    """
    if isinstance(mel_tensor, torch.Tensor):
        mel_tensor = mel_tensor.detach().cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.imshow(mel_tensor, origin='lower', aspect='auto', cmap='magma', interpolation='none')
    plt.title("Generated Mel Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Mel Channels")
    
    num_frames = mel_tensor.shape[1]
    xticks = np.linspace(0, num_frames, 5)
    xticklabels = [f"{(hop_length * x) / sr:.2f}" for x in xticks]
    plt.xticks(xticks, xticklabels)

    plt.colorbar(label="Amplitude (log scale)")
    plt.tight_layout()

    # PNGで保存
    plt.savefig(filename, dpi=300)
    plt.close()
    
# ---------------------------------------------------------------------------

def load_hifigan(device: torch.device):
    """Load pre-trained HiFi-GAN (22.05 kHz, mel-hop 256)."""
    hifigan, train_setup, denoiser = torch.hub.load(
        "NVIDIA/DeepLearningExamples:torchhub", "nvidia_hifigan"
    )
    return hifigan.to(device), train_setup, denoiser.to(device)

# ---------------------------------------------------------------------------

@torch.no_grad()
def mel2wav(
    device,
    mel_path: Path,
    hifigan,
    denoiser,
    max_wav_value: float,
    denoise: float,
    out_dir: Path,
):
    """
    • mel_path : .pt file, tensor shape (T, 80)  or  (80, T)
    • Saves <same-name>.wav to out_dir
    """
    mel: torch.Tensor = torch.load(mel_path).float().to(device)  # (T,80) or (80,T)
    if mel.ndim != 2:
        raise ValueError(f"{mel_path.name}: expected 2-D tensor, got {mel.shape}")
    if mel.size(1) == 80:              # (T,80)
        mel = mel.transpose(0, 1)      # → (80,T)
    if mel.size(0) != 80:
        raise ValueError(f"{mel_path.name}: mel dim != 80 after transpose: {mel.shape}")
    print("plot mel")
    plot_mel_spectrogram_to_file(mel)
    
    mel = mel.unsqueeze(0)             # (1,80,T)

    png_path = out_dir / (mel_path.stem + ".png")
    
    audio = hifigan(mel).squeeze(1)    # (1,T') → (1,T')
    if denoise > 0:
        # HifiGAN の STFT では入力長 >= filter_length が必要なので、足りない分はゼロパディング
        import torch.nn.functional as F
        fl = denoiser.stft.filter_length
        if audio.shape[-1] < fl:
            pad_amt = fl - audio.shape[-1]
            # 末尾にだけゼロパディング
            audio = F.pad(audio, (0, pad_amt), mode='constant', value=0.0)
            # ノイズ除去
        audio = denoiser(audio, denoise)

    audio = torch.clamp(audio, -1.0, 1.0) * max_wav_value   # scale to int16 range
    audio = audio.squeeze(1).short().cpu()
    #print(f"audio.shape = {audio.shape}, dtype = {audio.dtype}, device = {audio.device}")
    wav_path = out_dir / (mel_path.stem + ".wav")
    torchaudio.save(wav_path, audio, 22050)
    print("✓", wav_path)

# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hifigan, setup, denoiser = load_hifigan(device)
    max_wav_value = setup["max_wav_value"]

    in_dir  = Path(args.dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for pt_file in sorted(in_dir.glob("**/*.pt")):
        mel2wav(
            device,
            pt_file,
            hifigan,
            denoiser,
            max_wav_value,
            args.denoising_strength,
            out_dir,
        )

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    parser = ArgumentParser(description="Convert mel *.pt → wav using HiFi-GAN")
    parser.add_argument("--dir", type=str, required=True, help="directory with *.pt mels")
    parser.add_argument("--output_dir", type=str, default="./wav_out", help="save dir")
    parser.add_argument("--denoising_strength", type=float, default=0.005)
    args = parser.parse_args()

    main(args)
