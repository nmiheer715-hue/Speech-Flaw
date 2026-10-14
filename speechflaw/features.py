"""Acoustic feature extraction.

Every feature is computed on a shared 10 ms frame grid so baseline and
participant recordings can be compared frame-by-frame after DTW alignment.

Speaker-agnostic normalisation:
  * pitch is expressed in semitones relative to the speaker's own median F0
  * energy is expressed in dB relative to the speaker's own median voiced energy
  * MFCCs get per-utterance mean/variance normalisation (CMVN)
so a deep-voiced and a high-voiced speaker reading the same text with the
same delivery produce near-identical feature curves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import librosa
import numpy as np
import parselmouth
from scipy.ndimage import uniform_filter1d

SR = 16000
HOP_S = 0.01
HOP = int(SR * HOP_S)       # 160 samples
WIN = 400                   # 25 ms analysis window


@dataclass
class Features:
    sr: int
    hop_s: float
    duration: float
    t: np.ndarray            # frame times (s)
    f0_hz: np.ndarray        # raw F0, NaN where unvoiced
    f0_st: np.ndarray        # semitones re speaker median, NaN where unvoiced
    pitch_var: np.ndarray    # rolling std of f0_st over 1.5 s (intonation liveliness)
    energy_db: np.ndarray    # RMS dB re speaker median voiced energy
    centroid: np.ndarray     # spectral centroid (Hz) - brightness / articulation proxy
    hf_ratio: np.ndarray     # share of energy above 2 kHz (dB) - clarity proxy
    silence: np.ndarray      # bool, frame is silent
    syl_rate: np.ndarray     # local syllable-nucleus rate (per second, 1.5 s window)
    mfcc: np.ndarray         # (13, T) CMVN-normalised, used for DTW
    stats: dict = field(default_factory=dict)


def load_audio(path: str) -> np.ndarray:
    y, _ = librosa.load(path, sr=SR, mono=True)
    peak = np.max(np.abs(y)) or 1.0
    return (y / peak * 0.95).astype(np.float32)


def _frame_align(x: np.ndarray, n: int) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    pad_val = x[-1] if len(x) else 0.0
    return np.concatenate([x, np.full(n - len(x), pad_val)])


def _nan_rolling_std(x: np.ndarray, win: int) -> np.ndarray:
    out = np.full_like(x, np.nan, dtype=float)
    half = win // 2
    for i in range(len(x)):
        seg = x[max(0, i - half): i + half + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg) >= max(5, win // 6):
            # robust spread: (P90 - P10) / 2.563 equals sigma for a Gaussian but
            # ignores the octave-jump outliers pitch trackers produce
            p10, p90 = np.percentile(seg, [10, 90])
            out[i] = (p90 - p10) / 2.563
    # fill gaps so the curve is continuous for plotting / comparison
    idx = np.arange(len(out))
    good = ~np.isnan(out)
    if good.any():
        out = np.interp(idx, idx[good], out[good])
    else:
        out[:] = 0.0
    return out


def extract(y: np.ndarray) -> Features:
    n_frames = 1 + len(y) // HOP
    t = np.arange(n_frames) * HOP_S

    # ---- energy -----------------------------------------------------------
    rms = librosa.feature.rms(y=y, frame_length=WIN, hop_length=HOP, center=True)[0]
    rms = _frame_align(rms, n_frames)
    rms_db = 20 * np.log10(rms + 1e-6)
    silence = rms_db < (np.max(rms_db) - 35.0)
    # remove isolated non-silent blips / silent blips shorter than 50 ms
    sm = uniform_filter1d(silence.astype(float), 5)
    silence = sm > 0.5
    voiced_db = rms_db[~silence]
    ref_db = np.median(voiced_db) if len(voiced_db) else np.median(rms_db)
    energy_db = rms_db - ref_db

    # ---- pitch (Praat autocorrelation, robust for speech) ------------------
    snd = parselmouth.Sound(y.astype(np.float64), sampling_frequency=SR)
    pitch = snd.to_pitch(time_step=HOP_S, pitch_floor=70, pitch_ceiling=500)
    p_t = pitch.xs()
    p_f = pitch.selected_array["frequency"]
    p_f[p_f == 0] = np.nan
    f0_hz = np.interp(t, p_t, np.nan_to_num(p_f, nan=0.0), left=0.0, right=0.0)
    voiced_mask = np.interp(t, p_t, (~np.isnan(p_f)).astype(float), left=0, right=0) > 0.5
    f0_hz[~voiced_mask | silence] = np.nan
    med_f0 = np.nanmedian(f0_hz) if np.any(~np.isnan(f0_hz)) else 150.0
    f0_st = 12 * np.log2(f0_hz / med_f0)
    pitch_var = _nan_rolling_std(f0_st, int(1.5 / HOP_S))

    # ---- spectral shape -----------------------------------------------------
    S = np.abs(librosa.stft(y, n_fft=512, hop_length=HOP, win_length=WIN)) ** 2
    freqs = librosa.fft_frequencies(sr=SR, n_fft=512)
    cen = librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR)[0]
    hf = S[freqs >= 2000].sum(0) / (S.sum(0) + 1e-10)
    centroid = _frame_align(cen, n_frames)
    hf_ratio = 10 * np.log10(_frame_align(hf, n_frames) + 1e-6)
    # only meaningful where there is speech: hold last speech value through silence
    for arr in (centroid, hf_ratio):
        last = np.nanmedian(arr[~silence]) if (~silence).any() else 0.0
        for i in range(n_frames):
            if silence[i]:
                arr[i] = last
            else:
                last = arr[i]
    centroid = uniform_filter1d(centroid, 15)
    hf_ratio = uniform_filter1d(hf_ratio, 15)

    # ---- syllable-nucleus rate (energy-peak counting, de Jong & Wempe style)-
    env = uniform_filter1d(np.where(silence, -80, energy_db), 5)
    peaks = librosa.util.peak_pick(env, pre_max=6, post_max=6, pre_avg=10,
                                   post_avg=10, delta=2.0, wait=8)
    peaks = [p for p in peaks if not silence[p] and env[p] > -15]
    impulse = np.zeros(n_frames)
    impulse[peaks] = 1.0
    w = int(1.5 / HOP_S)
    syl_rate = uniform_filter1d(impulse, w) / HOP_S

    # ---- MFCC for alignment ------------------------------------------------
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, n_fft=512,
                                hop_length=HOP, win_length=WIN)
    mfcc = np.stack([_frame_align(r, n_frames) for r in mfcc])
    mfcc = (mfcc - mfcc.mean(1, keepdims=True)) / (mfcc.std(1, keepdims=True) + 1e-8)

    speech_time = (~silence).sum() * HOP_S
    stats = {
        "duration_s": round(len(y) / SR, 2),
        "speech_time_s": round(speech_time, 2),
        "median_f0_hz": round(float(med_f0), 1),
        "pitch_range_st": round(float(np.nanpercentile(f0_st, 95) - np.nanpercentile(f0_st, 5)), 2)
        if np.any(~np.isnan(f0_st)) else 0.0,
        "syllables_per_s": round(len(peaks) / max(speech_time, 1e-3), 2),
        "pause_ratio": round(float(silence.mean()), 3),
    }

    return Features(SR, HOP_S, len(y) / SR, t, f0_hz, f0_st, pitch_var, energy_db,
                    centroid, hf_ratio, silence, syl_rate, mfcc, stats)


def silence_runs(silence: np.ndarray, hop_s: float = HOP_S, min_s: float = 0.15):
    """Return list of (start_s, end_s) silent stretches (excluding leading/trailing)."""
    runs, start = [], None
    for i, s in enumerate(silence):
        if s and start is None:
            start = i
        elif not s and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(silence)))
    out = []
    for a, b in runs:
        if a == 0 or b == len(silence):       # ignore lead-in / tail silence
            continue
        if (b - a) * hop_s >= min_s:
            out.append((a * hop_s, b * hop_s))
    return out
