# SpeechFlaw — Technical Documentation

## 1. Problem

Judges of spoken-performance events (declamation, interpretive reading, oratory) score delivery
subjectively and give vague feedback ("work on your pacing"). SpeechFlaw makes delivery feedback
objective and *located*: given an ideal reading and a participant reading of the same text, it reports
**where** (start/end timestamps) the delivery deviates, **what** kind of deviation it is, and **why**,
in measurable acoustic terms.

The core idea is *contrastive*. We never judge a speech in isolation. Every frame of the participant
take is compared with the aligned frame of the ideal take. That makes the system text-aware without
needing a language model, and it lets the explanation cite a concrete baseline number.

## 2. Contrastive dataset construction

### 2.1 The "good" baseline
Each clip is a 30–60 s expressive reading of a well-known public-domain speech. A clip folder holds
`L0_ideal.wav` (16 kHz mono, peak-normalised) and its transcript.

### 2.2 The "bad" spectrum
From each ideal clip, `speechflaw/synth.py` produces 11 takes of the same transcript:

| Take | Injected flaws |
|---|---|
| L1_almost_perfect | one 1.0 s extra pause |
| L2_minor | mild rushing (1.3×) + mild trailing off (−9 dB) |
| L3_noticeable | monotone (15% pitch movement kept) + 1.6 s pause + rushing (1.5×) |
| L4_poor | strong rushing (1.8×) + flat pitch + muffling (1 kHz low-pass) + pause |
| L5_botched | all of the above at maximum strength + quiet + drag |
| S_* | one flaw each (rush, drag, monotone, quiet, mumble, pause) for isolated testing |

Flaw operators:

- **Rush / drag:** phase-vocoder time-stretch of a region (pitch preserved).
- **Monotone:** Praat PSOLA manipulation. Each pitch-tier point is pulled toward the region median, so
  only a fraction *k* of the original semitone excursion is kept, and the region is resynthesised.
- **Too quiet:** a gain ramp down to −9/−14/−20 dB, which models a voice trailing off.
- **Mumble:** 6th-order Butterworth low-pass at 1400/1000/700 Hz plus −4 dB. This removes the
  >2 kHz band that carries consonant bursts and fricatives.
- **Pause:** silence inserted at a natural word gap.

Regions are 1.8–4.5 s spans whose boundaries are snapped to existing pauses, so edits never cut a
word. Short crossfades (15 ms) prevent clicks.

### 2.3 Labels
Because the flaws are injected programmatically, `labels.json` stores each flaw's type, severity
level, and **exact start/end time in the output take**. Time-changing operators shift every later
timestamp, and the builder tracks that offset. This gives sample-accurate ground truth, which hand
labelling cannot. Quality matters more than quantity here: each take differs from its baseline
*only* in the labelled regions.

## 3. Feature extraction (`features.py`)

Everything sits on a common 10 ms hop (25 ms window, 16 kHz).

| Feature | Method | Captures |
|---|---|---|
| F0 | Praat autocorrelation pitch, 70–500 Hz | intonation contour |
| f0_st | 12·log2(F0 / speaker-median F0) | speaker-normalised pitch |
| pitch variability | rolling robust σ = (P90−P10)/2.563 of f0_st over 1.5 s | liveliness vs monotone |
| energy | 20·log10 RMS, minus the speaker's median voiced level | projection / volume |
| silence | energy < max−35 dB, 50 ms smoothing | pauses |
| spectral centroid | from STFT (512-pt) | brightness |
| HF share | 10·log10(E>2 kHz / E total) | articulation clarity |
| syllable rate | energy-peak picking (de Jong & Wempe style) | pace (descriptive) |
| MFCC-13 | per-utterance CMVN | alignment |

**Speaker-agnostic design.** Pitch is in semitones relative to the speaker's own median, energy is
relative to the speaker's own median voiced level, and MFCCs are mean/variance normalised. A deep and
a high voice delivering identically therefore produce near-identical curves. The comparison measures
*delivery*, not *voice*.

The robust σ matters. Pitch trackers make octave jumps at voicing boundaries, and plain std inflated
by those jumps hid real monotone regions in early tests (recall 0 → 1.0 after the change).

## 4. Alignment and temporal grounding (`compare.py`)

### 4.1 DTW
MFCC sequences of baseline and participant are aligned with dynamic time warping (Euclidean cost,
decimated ×2 for speed, then interpolated back). This gives a monotone map *m(j)*: test frame →
baseline frame. Every baseline feature, and every baseline word when alignment is enabled, can then be
read off at the aligned position of any participant frame.

### 4.2 Per-frame deltas
- **Pace (speech-rate ratio).** Let C_t(j) be cumulative *speech* frames in the test and C_b(i) in the
  baseline. Over a ±0.6 s window, ratio = ΔC_b(m(j)) / ΔC_t(j). Counting only speech frames makes the
  measure immune to inserted pauses: a pause does not register as "slow speech". A ratio of 1.5 means
  the participant covers 1.5 s of ideal speech per second of their own.
- **Intonation.** pitch-variability(test) ÷ pitch-variability(baseline at m(j)).
- **Volume.** energy(test) − energy(baseline at m(j)), only where both frames are speech, smoothed 0.4 s.
- **Clarity.** HF-share(test) − HF-share(baseline at m(j)).
- **Pauses.** For each test silence ≥ 0.3 s we measure the baseline silence inside the aligned span.
  The excess is unwanted pause time. Conversely, a baseline pause ≥ 0.35 s that the participant
  mostly skips becomes a *skipped pause* (lost dramatic effect).

### 4.3 Region extraction
Each delta is thresholded into a boolean mask. Masks are restricted to speech, then converted to
regions: gaps < 0.35 s are merged and regions < 0.45 s dropped. Each region keeps its start/end in
participant time, the median metric value, and a severity in [0.3, 1] that grows linearly from the
threshold to a "strong" level.

### 4.4 Causal explanation
Each region becomes a sentence built from its numbers. Example:

> *Between 12.30 s and 15.10 s the speaker covers 1.48 s of the ideal delivery per second of their own
> speech (DTW warp slope = 1.48, threshold 1.25). Speech here is ~48% faster than baseline,
> compressing syllables and stress contrasts.*

The explanation always states the measured value, the baseline value or threshold, and the
perceptual consequence. With word alignment on, it also lists the words affected.

## 5. Rubric and scoring

Thresholds are fixed once and frozen (shown in the dashboard sidebar):

| Category | Flaw | Threshold |
|---|---|---|
| Pace | rushing / dragging | rate ratio > 1.25 / < 0.78 |
| Intonation | monotone | variability ratio < 0.50 |
| Volume | too quiet / too loud | < −7 dB / > +7 dB vs baseline |
| Clarity | mumbled / muffled | HF share < −5 dB vs baseline |
| Pausing | unwanted / skipped pause | +0.6 s excess / < 25% of baseline pause kept |

Per category: penalty P = Σ(duration × severity) / speech time (pauses also add 0.8 × severity each,
since a short pause is disruptive out of proportion to its length). Score = 10·exp(−6P).
Overall = 0.25·Pace + 0.25·Intonation + 0.15·Volume + 0.15·Clarity + 0.20·Pausing.

The scoring is fully deterministic: identical inputs always give identical scores.

## 6. Evaluation

`python cli.py eval` runs every take against its baseline. A predicted flaw is matched to a gold flaw
of the same type if IoU ≥ 0.2, with ±0.25 s edge tolerance for point-like pauses.

| Flaw | Recall | Mean IoU | Mean onset error |
|---|---|---|---|
| Dragging | 1.00 | 0.97 | 0.09 s |
| Monotone | 1.00 | 0.84 | 0.36 s |
| Mumbled / muffled | 1.00 | 0.96 | 0.09 s |
| Rushing | 0.80 | 0.84 | 0.22 s |
| Too quiet | 1.00 | 0.65 | 0.36 s |
| Unwanted pause | 1.00 | 0.88 | 0.11 s |
| **All** | **0.95** | | precision **0.91** |

Sanity checks:
- The ideal compared with itself gives 0 flaws and a score of 10.0.
- The overall score falls monotonically along the spectrum: 10.0 (L0) → 9.6 → 9.9 → 7.3 → 5.8 → 4.4 (L5).
  L2 scores slightly above L1 because its rush region was missed.

## 7. Limitations (honest)

- **Synthetic flaws.** Injected flaws are cleaner than human mistakes. Real speakers rush *and* drop
  pitch at once, with gradual onsets. Evaluation on self-recorded human flawed takes is the next step.
- **Small dataset.** A one-night build covers only a few clips. Thresholds tuned on it may need
  re-tuning on more voices and recording conditions.
- **Coupled features.** Time-stretching also slows pitch movement, so a drag region can also trip the
  monotone detector (the one false positive in `S_drag`).
- **Same-text assumption.** The participant must read the same transcript. Paraphrasing or skipped
  sentences degrade the DTW alignment.
- **Recording conditions.** Very different microphones or rooms shift the HF-share and energy features.
  Per-utterance normalisation reduces but does not remove this.
- **Delivery only.** Content quality (argument, word choice) is out of scope.
- **Edge precision.** Onset errors of 0.1–0.4 s come from the 0.6–1.5 s analysis windows needed for
  stable rate and variability estimates. That is enough to point at a phrase, not a single syllable.

## 8. Future work
Learned flaw classifiers trained on this contrastive data, human-recorded flawed takes, emphasis and
stress placement per word, and filler-word detection ("um", "uh") from the alignment.
