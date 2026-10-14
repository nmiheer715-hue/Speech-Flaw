"""Contrastive comparison: align a participant take to the ideal baseline with
DTW, compute per-frame deltas, ground flaw regions in time, explain them, and
score the take against a fixed rubric.

Everything is deterministic (no randomness, no learned weights), so the same
pair of files always produces the same timestamps and scores.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d

from .features import Features, HOP_S, silence_runs

# ---------------------------------------------------------------------------
# Rubric thresholds (documented in docs/METHOD.md). Tuned once on the
# synthetic stress-test set, then frozen.
# ---------------------------------------------------------------------------
TH = {
    "rush": 1.25,          # local speech-rate ratio above this = rushing
    "rush_strong": 1.7,
    "drag": 0.78,          # below this = dragging
    "drag_strong": 0.55,
    "mono": 0.50,          # pitch-variability ratio below this = monotone
    "mono_strong": 0.2,
    "quiet": -7.0,         # dB below baseline = too quiet / trailing off
    "quiet_strong": -18.0,
    "loud": 7.0,
    "loud_strong": 15.0,
    "muffle_hf": -5.0,     # high-frequency energy share drop (dB) = muffled
    "muffle_hf_strong": -14.0,
    "pause_excess": 0.6,   # seconds of extra silence = unwanted pause
    "pause_strong": 2.0,
    "min_region": 0.45,    # s
    "merge_gap": 0.35,     # s
}

CATEGORY = {
    "Rushing": "Pace", "Dragging": "Pace",
    "Monotone": "Intonation",
    "Too quiet": "Volume", "Too loud": "Volume",
    "Mumbled / muffled": "Clarity",
    "Unwanted pause": "Pausing", "Skipped pause": "Pausing",
}
WEIGHTS = {"Pace": 0.25, "Intonation": 0.25, "Volume": 0.15, "Clarity": 0.15, "Pausing": 0.20}


@dataclass
class Flaw:
    kind: str
    category: str
    start: float
    end: float
    severity: float          # 0..1
    metric: str
    value: float
    baseline_value: float
    explanation: str
    words: str = ""

    def to_dict(self):
        d = asdict(self)
        for k in ("start", "end", "severity", "value", "baseline_value"):
            d[k] = round(float(d[k]), 3)
        return d


@dataclass
class Comparison:
    base: Features
    test: Features
    map_t2b: np.ndarray      # for each test frame, the aligned baseline frame
    curves: dict             # aligned curves for plotting (test time axis)
    flaws: list
    scores: dict
    dtw_cost: float


# ---------------------------------------------------------------------------
def align(base: Features, test: Features, stride: int = 2) -> np.ndarray:
    """DTW on CMVN MFCCs (decimated for speed). Returns test-frame -> base-frame map."""
    X = base.mfcc[:, ::stride]
    Y = test.mfcc[:, ::stride]
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean")
    wp = wp[::-1]
    nb, nt = base.mfcc.shape[1], test.mfcc.shape[1]
    m = np.full(Y.shape[1], np.nan)
    for i, j in wp:
        m[j] = i if np.isnan(m[j]) else (m[j] + i) / 2
    m = np.interp(np.arange(nt) / stride, np.arange(len(m)), m) * stride
    m = np.maximum.accumulate(np.clip(m, 0, nb - 1))
    cost = float(D[-1, -1] / len(wp))
    return np.round(m).astype(int), cost


def _regions(mask: np.ndarray, min_s: float, gap_s: float):
    """Boolean frame mask -> list of (a, b) frame index regions."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    regs, a, prev = [], idx[0], idx[0]
    gap = int(gap_s / HOP_S)
    for i in idx[1:]:
        if i - prev > gap:
            regs.append((a, prev + 1))
            a = i
        prev = i
    regs.append((a, prev + 1))
    return [(a, b) for a, b in regs if (b - a) * HOP_S >= min_s]


def _sev(mag, th, strong):
    """Map how far a metric overshoots its threshold onto 0.3..1."""
    x = (abs(mag) - abs(th)) / (abs(strong) - abs(th) + 1e-9)
    return float(np.clip(0.3 + 0.7 * x, 0.3, 1.0))


def _words_in(words, a, b):
    if not words:
        return ""
    return " ".join(w["word"] for w in words if w["end"] > a and w["start"] < b).strip()


def compare(base: Features, test: Features, base_words=None) -> Comparison:
    m, cost = align(base, test)
    nt = len(test.t)
    speech = ~test.silence
    b_speech = ~base.silence

    # ---- pace: speech-only warp slope (immune to inserted pauses) -----------
    ct = np.cumsum(speech).astype(float)
    cb = np.cumsum(b_speech).astype(float)[m]
    W = int(0.6 / HOP_S)
    lo, hi = np.clip(np.arange(nt) - W, 0, nt - 1), np.clip(np.arange(nt) + W, 0, nt - 1)
    dt = ct[hi] - ct[lo]
    db = cb[hi] - cb[lo]
    rate_ratio = np.where(dt > 10, db / np.maximum(dt, 1), 1.0)
    rate_ratio = uniform_filter1d(rate_ratio, 21)

    # ---- intonation ----------------------------------------------------------
    pv_b = base.pitch_var[m]
    pv_ratio = test.pitch_var / np.maximum(pv_b, 0.3)

    # ---- volume (only compare speech frames to speech frames) ---------------
    e_b = base.energy_db[m]
    both = speech & b_speech[m]
    e_delta = np.where(both, test.energy_db - e_b, 0.0)
    e_delta = uniform_filter1d(e_delta, 41)

    # ---- clarity ----------------------------------------------------------
    hf_delta = uniform_filter1d(np.where(both, test.hf_ratio - base.hf_ratio[m], 0.0), 41)
    cen_ratio = test.centroid / np.maximum(base.centroid[m], 1.0)

    curves = {
        "t": test.t,
        "base_t": base.t[m],
        "f0_test": test.f0_st, "f0_base": base.f0_st[m],
        "pv_test": test.pitch_var, "pv_base": pv_b,
        "e_test": test.energy_db, "e_base": e_b,
        "rate_ratio": rate_ratio,
        "hf_test": test.hf_ratio, "hf_base": base.hf_ratio[m],
        "cen_ratio": cen_ratio,
    }

    flaws: list[Flaw] = []

    def add(kind, mask, metric, series, base_series, th, strong, unit, explain):
        for a, b in _regions(mask & speech_fill, TH["min_region"], TH["merge_gap"]):
            seg = series[a:b]
            val = float(np.median(seg))
            bval = float(np.median(base_series[a:b])) if base_series is not None else 0.0
            ta, tb = a * HOP_S, b * HOP_S
            sev = _sev(val if base_series is None else val, th, strong)
            words = ""
            if base_words:
                words = _words_in(base_words, base.t[m[a]], base.t[m[min(b, nt - 1)]])
            flaws.append(Flaw(kind, CATEGORY[kind], ta, tb, sev, metric, val, bval,
                              explain(val, bval, ta, tb), words))

    # fill short silences inside speech so a region is not split by every gap
    speech_fill = uniform_filter1d(speech.astype(float), int(0.4 / HOP_S)) > 0.2

    add("Rushing", rate_ratio > TH["rush"], "speech-rate ratio", rate_ratio, None,
        TH["rush"], TH["rush_strong"], "x",
        lambda v, b, a, e: (f"Between {a:.2f}s and {e:.2f}s the speaker covers {v:.2f} s of the ideal "
                            f"delivery per second of their own speech (DTW warp slope = {v:.2f}, "
                            f"threshold {TH['rush']}). Speech here is ~{(v - 1) * 100:.0f}% faster than "
                            f"baseline, compressing syllables and stress contrasts."))
    add("Dragging", rate_ratio < TH["drag"], "speech-rate ratio", rate_ratio, None,
        TH["drag"], TH["drag_strong"], "x",
        lambda v, b, a, e: (f"Between {a:.2f}s and {e:.2f}s speech runs at {v:.2f}x the baseline rate "
                            f"(DTW warp slope, threshold {TH['drag']}): ~{(1 - v) * 100:.0f}% slower, "
                            f"which reads as hesitant or laboured."))
    add("Monotone", pv_ratio < TH["mono"], "pitch-variability ratio", pv_ratio, None,
        TH["mono"], TH["mono_strong"], "x",
        lambda v, b, a, e: (f"Between {a:.2f}s and {e:.2f}s pitch movement collapses: the rolling "
                            f"standard deviation of F0 is {v:.2f}x the baseline's at the aligned "
                            f"position ({np.median(test.pitch_var[int(a / HOP_S):int(e / HOP_S)]):.2f} st vs "
                            f"{np.median(pv_b[int(a / HOP_S):int(e / HOP_S)]):.2f} st). The ideal speaker "
                            f"uses intonation to mark emphasis here; this take is flat."))
    add("Too quiet", e_delta < TH["quiet"], "energy delta (dB)", e_delta, None,
        TH["quiet"], TH["quiet_strong"], "dB",
        lambda v, b, a, e: (f"Between {a:.2f}s and {e:.2f}s RMS energy is {v:.1f} dB relative to the "
                            f"baseline at the same words (speaker-normalised; threshold {TH['quiet']} dB). "
                            f"Voice drops away / trails off, so these words lose projection."))
    add("Too loud", e_delta > TH["loud"], "energy delta (dB)", e_delta, None,
        TH["loud"], TH["loud_strong"], "dB",
        lambda v, b, a, e: (f"Between {a:.2f}s and {e:.2f}s energy is +{v:.1f} dB over baseline "
                            f"(threshold +{TH['loud']} dB): over-projected / shouting."))
    add("Mumbled / muffled", hf_delta < TH["muffle_hf"], "HF-energy delta (dB)", hf_delta, None,
        TH["muffle_hf"], TH["muffle_hf_strong"], "dB",
        lambda v, b, a, e: (f"Between {a:.2f}s and {e:.2f}s the share of spectral energy above 2 kHz "
                            f"falls {abs(v):.1f} dB below baseline (spectral centroid "
                            f"{np.median(cen_ratio[int(a / HOP_S):int(e / HOP_S)]):.2f}x baseline). "
                            f"Consonant bursts and fricatives live above 2 kHz, so articulation is "
                            f"mumbled or muffled here."))

    # ---- pauses ---------------------------------------------------------------
    b_runs = silence_runs(base.silence)
    for ta, tb in silence_runs(test.silence, min_s=0.3):
        a, b = int(ta / HOP_S), min(int(tb / HOP_S), nt - 1)
        ba, bb = base.t[m[a]], base.t[m[b]]
        base_sil = sum(max(0, min(e, bb + 0.15) - max(s, ba - 0.15)) for s, e in b_runs)
        excess = (tb - ta) - base_sil
        if excess > TH["pause_excess"]:
            words = _words_in(base_words, ba - 0.6, bb + 0.6) if base_words else ""
            flaws.append(Flaw(
                "Unwanted pause", "Pausing", ta, tb, _sev(excess, TH["pause_excess"], TH["pause_strong"]),
                "extra silence (s)", excess, base_sil,
                f"Silence of {tb - ta:.2f}s at {ta:.2f}s where the ideal delivery pauses for only "
                f"{base_sil:.2f}s at the aligned point (+{excess:.2f}s, threshold {TH['pause_excess']}s). "
                f"The pause breaks the phrase instead of marking it.", words))
    for s, e in b_runs:
        if e - s < 0.35:
            continue
        ia, ib = np.searchsorted(m, int(s / HOP_S)), np.searchsorted(m, int(e / HOP_S))
        if ib <= ia:
            ib = ia + 1
        test_sil = test.silence[ia:ib].sum() * HOP_S
        if test_sil < 0.25 * (e - s) and (ib - ia) * HOP_S < 0.6 * (e - s):
            ta, tb = ia * HOP_S, max(ib * HOP_S, ia * HOP_S + TH["min_region"])
            flaws.append(Flaw(
                "Skipped pause", "Pausing", ta, tb, 0.35, "pause kept (s)", test_sil, e - s,
                f"The ideal delivery holds a {e - s:.2f}s dramatic/phrase pause here; this take keeps "
                f"only {test_sil:.2f}s and runs straight through it.",
                _words_in(base_words, s - 0.6, e + 0.6) if base_words else ""))

    flaws.sort(key=lambda f: f.start)
    scores = score(flaws, test)
    return Comparison(base, test, m, curves, flaws, scores, cost)


def score(flaws, test: Features) -> dict:
    """Rubric score 0-10 per category + weighted overall. Deterministic."""
    speech_t = max(test.stats["speech_time_s"], 1.0)
    out = {}
    for cat in WEIGHTS:
        pen = sum((f.end - f.start) * f.severity for f in flaws if f.category == cat)
        if cat == "Pausing":   # pauses are short but disruptive: weight count too
            pen += sum(0.8 * f.severity for f in flaws if f.category == cat)
        out[cat] = round(10 * float(np.exp(-6 * pen / speech_t)), 1)
    out["Overall"] = round(sum(out[c] * w for c, w in WEIGHTS.items()), 1)
    return out
