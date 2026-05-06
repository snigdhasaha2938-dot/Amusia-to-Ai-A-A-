# =============================================================================
# Auditory Perception Distortion Simulation -- Inspired by Amusia
# =============================================================================
# Research simulation that:
#   1. Loads an audio file using librosa (or synthesizes one if none provided)
#   2. Extracts pitch (F0) and rhythm features
#   3. Simulates amusia-like distortion (pitch shift, noise, rhythm disruption)
#   4. Reconstructs via a 1-D Convolutional Autoencoder + SP baseline blend
#   5. Visualises waveform, spectrogram, and F0 contours for all three signals
#   6. Reports SNR, Cosine Similarity, and Mean Pitch Error metrics
#
# Google Colab quick-start:
#   !pip install librosa soundfile torch matplotlib numpy scipy tqdm
#   Then: Runtime -> Run all
# =============================================================================

from __future__ import annotations

import os
import warnings
from typing import Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

import librosa
import librosa.display
import soundfile as sf
from scipy.signal import medfilt, butter, filtfilt
from scipy.spatial.distance import cosine as cosine_dist

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# 0.  CONFIGURATION
# =============================================================================

CFG = {
    # Audio
    "audio_path"        : None,     # path to .wav/.mp3, or None to synthesise
    "sample_rate"       : 22050,
    "duration"          : 8.0,      # seconds
    # Distortion
    "pitch_shift_steps" : 4,        # semitones
    "noise_amplitude"   : 0.035,    # Gaussian noise std
    "rhythm_mask_prob"  : 0.18,     # fraction of frames zeroed out
    # Autoencoder
    "frame_len"         : 512,
    "epochs"            : 40,
    "batch_size"        : 64,
    "lr"                : 3e-4,
    "latent_dim"        : 32,
    # Output
    "fig_path"          : "amusia_comparison.png",
    "out_dir"           : "amusia_outputs",
}

os.makedirs(CFG["out_dir"], exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[INFO] Device:", DEVICE)


# =============================================================================
# 1.  AUDIO LOADING / SYNTHESIS
# =============================================================================

def synthesise_music(sr, duration):
    """
    Synthesise a pentatonic melody with harmonics and a soft noise pad.
    Used when no external audio file is provided.
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs    = [261.63, 293.66, 329.63, 392.00, 440.00, 523.25]
    amps     = [0.40,   0.30,   0.25,   0.20,   0.18,   0.15]
    note_dur = duration / len(freqs)
    signal   = np.zeros_like(t)

    for i, (f, a) in enumerate(zip(freqs, amps)):
        s = int(i * note_dur * sr)
        e = int((i + 1) * note_dur * sr)
        seg  = a          * np.sin(2 * np.pi * f       * t[s:e])
        seg += (a * 0.30) * np.sin(2 * np.pi * 2 * f   * t[s:e])
        seg += (a * 0.15) * np.sin(2 * np.pi * 3 * f   * t[s:e])
        signal[s:e] += seg

    pad = 0.02 * np.random.randn(len(t))
    b, a_coef = butter(4, [200 / (sr / 2), 800 / (sr / 2)], btype="band")
    pad = filtfilt(b, a_coef, pad)
    signal += pad
    signal  = signal / (np.max(np.abs(signal)) + 1e-8)
    return signal.astype(np.float32)


def load_audio(cfg):
    """Load audio from file or fall back to synthesis."""
    sr  = cfg["sample_rate"]
    dur = cfg["duration"]

    if cfg["audio_path"] and os.path.exists(cfg["audio_path"]):
        y, _ = librosa.load(cfg["audio_path"], sr=sr, duration=dur, mono=True)
        print("[INFO] Loaded '{}' | sr={} | {:.2f}s".format(
            cfg["audio_path"], sr, len(y) / sr))
    else:
        print("[INFO] No audio file -- synthesising demo signal ...")
        y = synthesise_music(sr, dur)

    target = int(sr * dur)
    if len(y) < target:
        y = np.pad(y, (0, target - len(y)))
    else:
        y = y[:target]
    return y, sr


# =============================================================================
# 2.  FEATURE EXTRACTION
# =============================================================================

def extract_features(y, sr):
    """Extract F0, chroma, MFCC, onset envelope, and tempo."""
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    f0_clean = np.where(voiced_flag, f0, 0.0)

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    S      = np.abs(librosa.stft(y))
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    mfcc   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    print("[INFO] Tempo: {:.1f} BPM | Voiced frames: {}".format(
        float(tempo), int(voiced_flag.sum())))
    return {
        "f0"         : f0_clean,
        "voiced_flag": voiced_flag,
        "tempo"      : tempo,
        "beat_frames": beat_frames,
        "onset_env"  : onset_env,
        "chroma"     : chroma,
        "mfcc"       : mfcc,
        "S"          : S,
    }


# =============================================================================
# 3.  AMUSIA-LIKE DISTORTION
# =============================================================================

def distort_pitch(y, sr, steps):
    """Shift pitch by `steps` semitones -- mimics pitch-interval confusion."""
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps).astype(np.float32)


def add_perceptual_noise(y, amplitude):
    """Inject Gaussian noise to simulate the roughness percept of amusia."""
    return (y + amplitude * np.random.randn(len(y))).astype(np.float32)


def disrupt_rhythm(y, mask_prob, frame_size=512):
    """Zero-out random 23 ms frames to simulate temporal-order disruption."""
    y_out    = y.copy()
    n_frames = len(y) // frame_size
    mask     = np.random.rand(n_frames) < mask_prob
    for i, m in enumerate(mask):
        if m:
            y_out[i * frame_size:(i + 1) * frame_size] = 0.0
    return y_out


def apply_distortion(y, sr, cfg):
    """Chain pitch shift -> noise -> rhythm disruption."""
    print("[INFO] Applying amusia-like distortions ...")
    y_d = distort_pitch(y, sr, cfg["pitch_shift_steps"])
    y_d = add_perceptual_noise(y_d, cfg["noise_amplitude"])
    y_d = disrupt_rhythm(y_d, cfg["rhythm_mask_prob"])
    return y_d


# =============================================================================
# 4.  SIGNAL-PROCESSING RECONSTRUCTION (baseline)
# =============================================================================

def sp_reconstruct(y_dist, sr):
    """
    Wiener-style spectral soft-masking + partial pitch-shift reversal
    + median gap fill.
    """
    D           = librosa.stft(y_dist)
    mag, phase  = np.abs(D), np.angle(D)
    noise_floor = np.percentile(mag, 15, axis=1, keepdims=True)
    gain        = np.maximum(0.0, 1.0 - noise_floor / (mag + 1e-8))
    D_clean     = gain * mag * np.exp(1j * phase)
    y_sp        = librosa.istft(D_clean, length=len(y_dist))

    # Partial pitch reversal (80% correction)
    y_sp = librosa.effects.pitch_shift(
        y=y_sp, sr=sr, n_steps=-CFG["pitch_shift_steps"] * 0.8
    )

    y_sp = medfilt(y_sp, kernel_size=9)
    return y_sp.astype(np.float32)


# =============================================================================
# 5.  1-D CONVOLUTIONAL AUTOENCODER
# =============================================================================

class ConvEncoder(nn.Module):
    def __init__(self, frame_len, latent_dim):
        super(ConvEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 32, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=9, stride=2, padding=4),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy     = torch.zeros(1, 1, frame_len)
            flat_size = self.net(dummy).shape[1]
        self.flat_size   = flat_size
        self.encoded_len = frame_len // 8
        self.fc          = nn.Linear(flat_size, latent_dim)

    def forward(self, x):
        return self.fc(self.net(x))


class ConvDecoder(nn.Module):
    def __init__(self, flat_size, encoded_len, latent_dim):
        super(ConvDecoder, self).__init__()
        self.fc          = nn.Linear(latent_dim, flat_size)
        self.encoded_len = encoded_len
        self.net         = nn.Sequential(
            nn.ConvTranspose1d(64, 32, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(32, 16, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose1d(16,  1, kernel_size=9, stride=2, padding=4, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), 64, self.encoded_len)
        return self.net(h)


class AmusiaAutoencoder(nn.Module):
    def __init__(self, frame_len=512, latent_dim=32):
        super(AmusiaAutoencoder, self).__init__()
        self.encoder = ConvEncoder(frame_len, latent_dim)
        self.decoder = ConvDecoder(
            self.encoder.flat_size,
            self.encoder.encoded_len,
            latent_dim,
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def make_frames(signal, frame_len):
    """Slice signal into non-overlapping frames of length frame_len."""
    n = (len(signal) // frame_len) * frame_len
    return signal[:n].reshape(-1, frame_len)


def train_autoencoder(y_orig, y_dist, cfg):
    """Train the AE to map distorted frames to original frames."""
    FL = cfg["frame_len"]
    frames_orig = make_frames(y_orig, FL)
    frames_dist = make_frames(y_dist, FL)

    def normalise(x):
        mx = np.max(np.abs(x), axis=1, keepdims=True) + 1e-8
        return x / mx, mx

    X_dist, _ = normalise(frames_dist)
    X_orig, _ = normalise(frames_orig)

    X_d = torch.tensor(X_dist, dtype=torch.float32).unsqueeze(1)
    X_o = torch.tensor(X_orig, dtype=torch.float32).unsqueeze(1)

    loader  = DataLoader(TensorDataset(X_d, X_o),
                         batch_size=cfg["batch_size"], shuffle=True)
    model   = AmusiaAutoencoder(FL, cfg["latent_dim"]).to(DEVICE)
    opt     = optim.Adam(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()

    print("[INFO] Training autoencoder ...")
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for xd, xo in loader:
            xd, xo = xd.to(DEVICE), xo.to(DEVICE)
            pred   = model(xd)
            min_l  = min(pred.shape[-1], xo.shape[-1])
            loss   = loss_fn(pred[..., :min_l], xo[..., :min_l])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        avg = epoch_loss / len(loader)
        history.append(avg)
        if epoch % 10 == 0 or epoch == 1:
            print("  Epoch {:3d}/{} | Loss: {:.6f}".format(
                epoch, cfg["epochs"], avg))

    plt.figure(figsize=(6, 3))
    plt.plot(history, color="#e05c5c", linewidth=1.8)
    plt.title("Autoencoder Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(os.path.join(cfg["out_dir"], "training_loss.png"), dpi=120)
    plt.close()

    return model


def reconstruct_with_ae(y_dist, model, cfg):
    """Run trained AE on the full distorted signal frame-by-frame."""
    FL     = cfg["frame_len"]
    frames = make_frames(y_dist, FL)
    norms  = np.max(np.abs(frames), axis=1, keepdims=True) + 1e-8
    X      = torch.tensor(frames / norms, dtype=torch.float32).unsqueeze(1).to(DEVICE)

    model.eval()
    with torch.no_grad():
        out = model(X).squeeze(1).cpu().numpy()

    out   = out * norms
    y_rec = out.flatten()

    orig_len = len(y_dist)
    if len(y_rec) < orig_len:
        y_rec = np.pad(y_rec, (0, orig_len - len(y_rec)))
    else:
        y_rec = y_rec[:orig_len]
    return y_rec.astype(np.float32)


# =============================================================================
# 6.  EVALUATION METRICS
# =============================================================================

def snr_db(reference, estimate):
    """Signal-to-Noise Ratio in dB."""
    min_l = min(len(reference), len(estimate))
    noise = reference[:min_l] - estimate[:min_l]
    s_pow = np.mean(reference[:min_l] ** 2)
    n_pow = np.mean(noise ** 2) + 1e-12
    return float(10 * np.log10(s_pow / n_pow))


def cosine_similarity(a, b):
    """Cosine similarity between two 1-D signals."""
    min_l = min(len(a), len(b))
    return float(1.0 - cosine_dist(a[:min_l], b[:min_l]))


def mean_pitch_error(f0_ref, f0_est):
    """Mean absolute F0 error over voiced frames (Hz)."""
    min_l  = min(len(f0_ref), len(f0_est))
    ref    = f0_ref[:min_l]
    est    = f0_est[:min_l]
    voiced = (ref > 0) & (est > 0)
    if voiced.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(ref[voiced] - est[voiced])))


def get_f0(y, sr):
    """Helper: extract F0 array with NaN replaced by 0."""
    f0, _, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
    )
    return np.nan_to_num(f0)


def compute_metrics(y_orig, y_dist, y_rec, sr):
    """Compute all metrics and print a summary table."""
    f0_orig = get_f0(y_orig, sr)
    f0_dist = get_f0(y_dist, sr)
    f0_rec  = get_f0(y_rec,  sr)

    metrics = {
        "SNR_distorted_dB"    : snr_db(y_orig, y_dist),
        "SNR_reconstructed_dB": snr_db(y_orig, y_rec),
        "CosSim_distorted"    : cosine_similarity(y_orig, y_dist),
        "CosSim_reconstructed": cosine_similarity(y_orig, y_rec),
        "MeanPitchErr_dist_Hz": mean_pitch_error(f0_orig, f0_dist),
        "MeanPitchErr_rec_Hz" : mean_pitch_error(f0_orig, f0_rec),
        "f0_orig"             : f0_orig,
        "f0_dist"             : f0_dist,
        "f0_rec"              : f0_rec,
    }

    print("\n" + "=" * 56)
    print("  EVALUATION METRICS")
    print("=" * 56)
    for k, v in metrics.items():
        if not k.startswith("f0"):
            print("  {:<30s}: {:.4f}".format(k, v))
    print("=" * 56)
    return metrics


# =============================================================================
# 7.  VISUALISATION
# =============================================================================

CMAP_SPEC = LinearSegmentedColormap.from_list(
    "amusia",
    ["#0d0221", "#2b0a5c", "#6a1bb5", "#c86dd4", "#f5d7ff"],
    N=256,
)


def plot_comparison(y_orig, y_dist, y_rec, sr, metrics, cfg):
    """
    Dark-themed 3x3 comparison figure.
      Row 0 -- Waveforms
      Row 1 -- Mel Spectrograms
      Row 2 -- F0 Pitch Contours
    """
    signals = [y_orig,             y_dist,                    y_rec]
    labels  = ["Original",         "Distorted (Amusia-like)", "Reconstructed (AE)"]
    colors  = ["#4fc3f7",          "#ef9a9a",                 "#a5d6a7"]
    f0s     = [metrics["f0_orig"], metrics["f0_dist"],        metrics["f0_rec"]]

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#0d0221")
    gs  = gridspec.GridSpec(
        3, 3, figure=fig,
        hspace=0.55, wspace=0.35,
        left=0.07, right=0.97, top=0.93, bottom=0.07,
    )

    for col, (sig, label, color, f0) in enumerate(zip(signals, labels, colors, f0s)):
        t        = np.linspace(0, len(sig) / sr, len(sig))
        times_f0 = librosa.times_like(f0, sr=sr)

        # --- Waveform ---
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_facecolor("#0d0221")
        ax0.plot(t, sig, color=color, linewidth=0.6, alpha=0.9)
        ax0.set_title(label, color="white", fontsize=11, fontweight="bold", pad=8)
        ax0.set_xlabel("Time (s)", color="#aaa", fontsize=8)
        ax0.tick_params(colors="#666")
        for sp in ax0.spines.values():
            sp.set_edgecolor("#333")
        if col == 0:
            ax0.set_ylabel("Waveform", color="white", fontsize=9, fontweight="bold")

        # --- Mel Spectrogram ---
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_facecolor("#0d0221")
        S_mel = librosa.feature.melspectrogram(y=sig, sr=sr, n_mels=128)
        S_db  = librosa.power_to_db(S_mel, ref=np.max)
        librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel",
                                 ax=ax1, cmap=CMAP_SPEC)
        ax1.set_xlabel("Time (s)", color="#aaa", fontsize=8)
        ax1.tick_params(colors="#666")
        for sp in ax1.spines.values():
            sp.set_edgecolor("#333")
        if col == 0:
            ax1.set_ylabel("Mel Spectrogram", color="white", fontsize=9, fontweight="bold")

        # --- F0 Contour ---
        ax2 = fig.add_subplot(gs[2, col])
        ax2.set_facecolor("#0d0221")
        voiced_mask = f0 > 0
        f0_plot     = np.where(voiced_mask, f0, np.nan)
        ax2.fill_between(times_f0, 0, f0_plot, color=color, alpha=0.25)
        ax2.plot(times_f0, f0_plot, color=color, linewidth=1.3)
        ax2.set_ylim(0, 600)
        ax2.set_xlabel("Time (s)", color="#aaa", fontsize=8)
        ax2.tick_params(colors="#666")
        for sp in ax2.spines.values():
            sp.set_edgecolor("#333")
        if col == 0:
            ax2.set_ylabel("F0 Pitch (Hz)", color="white", fontsize=9, fontweight="bold")

    fig.suptitle(
        "Auditory Perception Distortion -- Amusia Simulation\n"
        "Original  *  Distorted  *  Reconstructed",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    save_path = os.path.join(cfg["out_dir"], cfg["fig_path"])
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print("[INFO] Figure saved ->", save_path)


# =============================================================================
# 8.  SAVE AUDIO FILES
# =============================================================================

def save_audio_files(y_orig, y_dist, y_rec, sr, out_dir):
    """Write original, distorted, and reconstructed signals to disk."""
    for fname, sig in [
        ("original.wav",      y_orig),
        ("distorted.wav",     y_dist),
        ("reconstructed.wav", y_rec),
    ]:
        path = os.path.join(out_dir, fname)
        sf.write(path, sig, sr)
        print("[INFO] Audio saved ->", path)


# =============================================================================
# 9.  MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 56)
    print("  AMUSIA SIMULATION PIPELINE")
    print("=" * 56)

    # Step 1 -- Load / synthesise audio
    y_orig, sr = load_audio(CFG)

    # Step 2 -- Extract features
    print("\n[Step 2] Extracting features ...")
    feats = extract_features(y_orig, sr)

    # Step 3 -- Distort
    print("\n[Step 3] Distorting audio ...")
    y_dist = apply_distortion(y_orig, sr, CFG)

    # Step 4a -- Signal-processing reconstruction
    print("\n[Step 4a] Signal-processing reconstruction ...")
    y_sp_rec = sp_reconstruct(y_dist, sr)

    # Step 4b -- Autoencoder reconstruction
    print("\n[Step 4b] Autoencoder reconstruction ...")
    model = train_autoencoder(y_orig, y_dist, CFG)
    y_ae  = reconstruct_with_ae(y_dist, model, CFG)

    # Blend SP (35%) + AE (65%)
    blend_len = min(len(y_sp_rec), len(y_ae))
    y_rec = (0.35 * y_sp_rec[:blend_len] + 0.65 * y_ae[:blend_len]).astype(np.float32)

    # Step 5 -- Evaluate
    print("\n[Step 5] Computing metrics ...")
    metrics = compute_metrics(y_orig, y_dist, y_rec, sr)

    # Step 6 -- Visualise
    print("\n[Step 6] Generating plots ...")
    plot_comparison(y_orig, y_dist, y_rec, sr, metrics, CFG)

    # Step 7 -- Save audio
    print("\n[Step 7] Saving audio files ...")
    save_audio_files(y_orig, y_dist, y_rec, sr, CFG["out_dir"])

    # Save model weights
    model_path = os.path.join(CFG["out_dir"], "autoencoder.pt")
    torch.save(model.state_dict(), model_path)
    print("[INFO] Model weights saved ->", model_path)

    print("\nPipeline complete. Outputs in:", CFG["out_dir"])
    return {
        "original"     : y_orig,
        "distorted"    : y_dist,
        "reconstructed": y_rec,
        "metrics"      : metrics,
        "model"        : model,
        "sr"           : sr,
    }


# =============================================================================
if __name__ == "__main__":
    results = main()
