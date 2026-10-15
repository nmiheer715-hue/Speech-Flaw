"""Word-level forced alignment of the baseline transcript (optional).

Uses faster-whisper's word timestamps (CTC/attention alignment). Only the
*baseline* needs aligning: DTW then carries every word onto the participant
take, so each flaw region is reported together with the words it covers.

If faster-whisper is not installed the rest of the system still runs; flaws
are just reported without words.
"""
from __future__ import annotations

import json
from pathlib import Path


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:
        return False


def align_words(wav_path: str, transcript: str = "", model_size: str = "base.en"):
    from faster_whisper import WhisperModel
    import librosa
    # decode with librosa (16 kHz mono float32) instead of faster-whisper's PyAV
    # decoder, which breaks on some PyAV versions
    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio, word_timestamps=True,
                                   initial_prompt=transcript or None, beam_size=5,
                                   temperature=0.0)       # deterministic decoding
    words = []
    for seg in segments:
        for w in seg.words or []:
            words.append({"word": w.word.strip(), "start": round(w.start, 3), "end": round(w.end, 3)})
    return words


def cached_words(wav_path: str, transcript: str = ""):
    """Align once and cache next to the wav as <name>.words.json."""
    p = Path(wav_path)
    cache = p.with_suffix(".words.json")
    if cache.exists():
        return json.loads(cache.read_text())
    if not available():
        return None
    try:
        words = align_words(str(p), transcript)
    except Exception:          # e.g. model download blocked/offline: run without word labels
        return None
    cache.write_text(json.dumps(words, indent=1))
    return words
