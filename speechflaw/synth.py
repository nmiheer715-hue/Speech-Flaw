"""Build the contrastive "bad mirror" spectrum from one ideal recording.

For each ideal clip we generate takes from near-perfect to botched by injecting
known delivery flaws into chosen regions. Because we inject them ourselves, the
flaw start/end timestamps in the output file are exact ground truth.

Flaw operators (all signal-level, reproducible with a fixed seed):
  rush      time-compress a region (phase vocoder, pitch preserved)
  drag      time-stretch a region
  monotone  flatten the F0 contour in a region (Praat PSOLA resynthesis)
  quiet     drop gain in a region (trailing off)
  mumble    low-pass at ~900 Hz + gain drop (muffled articulation)
  pause     insert silence at a word boundary
"""
from __future__ import annotations

import json
from pathlib import Path

import librosa
import numpy as np
import parselmouth
import soundfile as sf
from parselmouth.praat import call
from scipy.signal import butter, sosfilt

from .features import SR, extract, silence_runs, load_audio

KIND_NAMES = {
    "rush": "Rushing", "drag": "Dragging", "monotone": "Monotone",
    "quiet": "Too quiet", "mumble": "Mumbled / muffled", "pause": "Unwanted pause",
}

# severity level -> operator strength
STRENGTH = {
    "rush":     {1: 1.3, 2: 1.5, 3: 1.8},      # speed factor
    "drag":     {1: 0.75, 2: 0.65, 3: 0.5},
    "monotone": {1: 0.35, 2: 0.15, 3: 0.0},    # fraction of pitch movement kept
    "quiet":    {1: -9, 2: -14, 3: -20},       # dB
    "mumble":   {1: 1400, 2: 1000, 3: 700},    # low-pass cutoff Hz
    "pause":    {1: 1.0, 2: 1.6, 3: 2.5},      # seconds of inserted silence
}

# The gradient: level 0 is the untouched ideal, level 4 is botched.
SPECTRUM = [
    ("L1_almost_perfect", [("pause", 1)]),
    ("L2_minor",          [("rush", 1), ("quiet", 1)]),
    ("L3_noticeable",     [("monotone", 2), ("pause", 2), ("rush", 2)]),
    ("L4_poor",           [("rush", 3), ("monotone", 3), ("mumble", 2), ("pause", 2)]),
    ("L5_botched",        [("rush", 3), ("monotone", 3), ("quiet", 3), ("mumble", 3),
                           ("pause", 3), ("drag", 3)]),
    # single-flaw isolation takes (clean evaluation of each detector)
    ("S_rush",     [("rush", 2)]),
    ("S_drag",     [("drag", 2)]),
    ("S_monotone", [("monotone", 2)]),
    ("S_quiet",    [("quiet", 2)]),
    ("S_mumble",   [("mumble", 2)]),
    ("S_pause",    [("pause", 2)]),
]


def _flatten_pitch(seg: np.ndarray, keep: float) -> np.ndarray:
    snd = parselmouth.Sound(seg.astype(np.float64), sampling_frequency=SR)
    manip = call(snd, "To Manipulation", 0.01, 75, 500)
    tier = call(manip, "Extract pitch tier")
    n = call(tier, "Get number of points")
    if n < 3:
        return seg
    pts = [(call(tier, "Get time from index", i), call(tier, "Get value at index", i))
           for i in range(1, n + 1)]
    med = float(np.median([v for _, v in pts]))
    call(tier, "Remove points between", 0, snd.duration)
    for t, v in pts:
        st = 12 * np.log2(v / med) * keep
        call(tier, "Add point", t, med * 2 ** (st / 12))
    call([tier, manip], "Replace pitch tier")
    out = call(manip, "Get resynthesis (overlap-add)").values[0]
    return _fit(out, len(seg))


def _fit(x, n):
    return x[:n] if len(x) >= n else np.pad(x, (0, n - len(x)))


def _xfade(a: np.ndarray, b: np.ndarray, ms=15):
    """Concatenate with a short crossfade to avoid clicks."""
    k = int(SR * ms / 1000)
    if len(a) < k or len(b) < k:
        return np.concatenate([a, b])
    r = np.linspace(0, 1, k)
    mid = a[-k:] * (1 - r) + b[:k] * r
    return np.concatenate([a[:-k], mid, b[k:]])


def apply_op(seg: np.ndarray, op: str, level: int) -> np.ndarray:
    s = STRENGTH[op][level]
    if op in ("rush", "drag"):
        return librosa.effects.time_stretch(seg, rate=s)
    if op == "monotone":
        return _flatten_pitch(seg, s)
    if op == "quiet":
        g = 10 ** (s / 20)
        ramp = np.minimum(1, np.linspace(0, 1, len(seg)) * 8)  # 1/8 of region fade-in of the drop
        return seg * (1 - ramp * (1 - g))
    if op == "mumble":
        sos = butter(6, s, btype="low", fs=SR, output="sos")
        return sosfilt(sos, seg) * 10 ** (-4 / 20)
    raise ValueError(op)


def choose_regions(y: np.ndarray, n: int, rng: np.random.Generator):
    """Pick n non-overlapping regions (2-4 s) snapped to pauses, plus pause points."""
    f = extract(y)
    gaps = [(a + b) / 2 for a, b in silence_runs(f.silence, min_s=0.08)]
    dur = len(y) / SR
    bounds = [0.3] + gaps + [dur - 0.3]
    # candidate regions: spans between boundaries of 1.8-4.5 s
    cands = []
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            L = bounds[j] - bounds[i]
            if 1.8 <= L <= 4.5:
                cands.append((bounds[i], bounds[j]))
    if not cands:  # fall back to even slicing
        step = dur / (n + 1)
        cands = [(step * k, step * k + min(3, step * 0.8)) for k in range(1, n + 1)]
    rng.shuffle(cands)
    chosen = []
    for a, b in cands:
        if all(b + 0.4 < c or a > d + 0.4 for c, d in chosen):
            chosen.append((a, b))
        if len(chosen) == n:
            break
    return sorted(chosen), gaps


def make_take(y: np.ndarray, plan, seed=0):
    """Apply a list of (op, level) to y. Returns new audio and exact labels."""
    rng = np.random.default_rng(seed)
    n_region_ops = sum(1 for op, _ in plan if op != "pause")
    n_pause = sum(1 for op, _ in plan if op == "pause")
    regions, gaps = choose_regions(y, n_region_ops, rng)
    events = []  # (orig_start, orig_end, op, level)
    ops = [p for p in plan if p[0] != "pause"]
    for (a, b), (op, lvl) in zip(regions, ops):
        events.append((a, b, op, lvl))
    # pauses go at word gaps not inside other regions
    free = [g for g in gaps if all(not (a - 0.3 <= g <= b + 0.3) for a, b, *_ in events)]
    rng.shuffle(free)
    for g, (op, lvl) in zip(free, [p for p in plan if p[0] == "pause"]):
        events.append((g, g, op, lvl))
    events.sort()

    out, labels, cursor_in, t_out = [], [], 0, 0.0
    pieces = np.zeros(0, dtype=np.float32)
    for a, b, op, lvl in events:
        ia, ib = int(a * SR), int(b * SR)
        pieces = _xfade(pieces, y[cursor_in:ia]) if len(pieces) else y[cursor_in:ia].copy()
        start = len(pieces) / SR
        if op == "pause":
            seg = np.zeros(int(STRENGTH["pause"][lvl] * SR), dtype=np.float32)
            pieces = np.concatenate([pieces, seg])
        else:
            seg = apply_op(y[ia:ib].copy(), op, lvl).astype(np.float32)
            pieces = _xfade(pieces, seg)
        end = len(pieces) / SR
        labels.append({"kind": KIND_NAMES[op], "op": op, "level": lvl,
                       "start": round(start, 3), "end": round(end, 3),
                       "orig_start": round(a, 3), "orig_end": round(b, 3)})
        cursor_in = ib
    pieces = _xfade(pieces, y[cursor_in:]) if len(pieces) else y.copy()
    peak = np.max(np.abs(pieces)) or 1
    return (pieces / peak * 0.95).astype(np.float32), labels


def build(ideal_path: str, out_dir: str, clip_id: str, transcript: str = "", seed: int = 7):
    """Write ideal + spectrum takes and a labels.json manifest for one clip."""
    out = Path(out_dir) / clip_id
    out.mkdir(parents=True, exist_ok=True)
    y = load_audio(ideal_path)
    sf.write(out / "L0_ideal.wav", y, SR)
    if transcript:
        (out / "transcript.txt").write_text(transcript.strip() + "\n")
    manifest = {"clip_id": clip_id, "source": Path(ideal_path).name, "sr": SR,
                "baseline": "L0_ideal.wav", "takes": []}
    for k, (name, plan) in enumerate(SPECTRUM):
        audio, labels = make_take(y, plan, seed=seed + k)
        fn = f"{name}.wav"
        sf.write(out / fn, audio, SR)
        manifest["takes"].append({"file": fn, "level": name, "flaws": labels})
        print(f"  {clip_id}/{fn}: {', '.join(l['kind'] for l in labels)}")
    (out / "labels.json").write_text(json.dumps(manifest, indent=2))
    return manifest
