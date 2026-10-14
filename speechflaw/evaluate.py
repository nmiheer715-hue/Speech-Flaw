"""Score the detector against the injected ground-truth labels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .compare import compare
from .features import extract, load_audio


def iou(a, b):
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def match(pred, gold, tol=0.25):
    """Greedy match of predicted flaws to gold flaws of the same kind.
    A pause is a point event in the source, so we allow +-tol s slack on edges."""
    used, hits = set(), []
    for g in gold:
        best, bi = 0.0, None
        for i, p in enumerate(pred):
            if i in used or p["kind"] != g["kind"]:
                continue
            o = iou((p["start"] - tol, p["end"] + tol), (g["start"] - tol, g["end"] + tol))
            if o > best:
                best, bi = o, i
        if bi is not None and best >= 0.2:
            used.add(bi)
            p = pred[bi]
            hits.append({"kind": g["kind"], "iou": iou((p["start"], p["end"]), (g["start"], g["end"])),
                         "onset_err": abs(p["start"] - g["start"]), "offset_err": abs(p["end"] - g["end"])})
    return hits, len(pred) - len(used)


def run(data_dir: str, out_json: str | None = None):
    rows, per_kind = [], {}
    for lab in sorted(Path(data_dir).glob("*/labels.json")):
        man = json.loads(lab.read_text())
        clip = lab.parent
        base = extract(load_audio(str(clip / man["baseline"])))
        # ideal vs itself: should yield no flaws (false-positive sanity check)
        self_cmp = compare(base, base)
        rows.append({"clip": man["clip_id"], "take": "L0_ideal (self)", "gold": 0,
                     "pred": len(self_cmp.flaws), "tp": 0, "fp": len(self_cmp.flaws),
                     "overall": self_cmp.scores["Overall"]})
        for take in man["takes"]:
            test = extract(load_audio(str(clip / take["file"])))
            c = compare(base, test)
            pred = [f.to_dict() for f in c.flaws]
            gold = take["flaws"]
            hits, fp = match(pred, gold)
            for g in gold:
                per_kind.setdefault(g["kind"], {"gold": 0, "tp": 0, "iou": [], "onset": []})["gold"] += 1
            for h in hits:
                k = per_kind[h["kind"]]
                k["tp"] += 1
                k["iou"].append(h["iou"])
                k["onset"].append(h["onset_err"])
            rows.append({"clip": man["clip_id"], "take": take["level"], "gold": len(gold),
                         "pred": len(pred), "tp": len(hits), "fp": fp,
                         "overall": c.scores["Overall"]})
    tp = sum(r["tp"] for r in rows)
    gold = sum(r["gold"] for r in rows)
    pred = sum(r["pred"] for r in rows)
    summary = {
        "recall": round(tp / gold, 3) if gold else None,
        "precision": round(tp / pred, 3) if pred else None,
        "per_kind": {k: {"recall": round(v["tp"] / v["gold"], 3),
                         "mean_iou": round(float(np.mean(v["iou"])), 3) if v["iou"] else None,
                         "mean_onset_err_s": round(float(np.mean(v["onset"])), 3) if v["onset"] else None,
                         "n": v["gold"]}
                     for k, v in sorted(per_kind.items())},
        "rows": rows,
    }
    if out_json:
        Path(out_json).write_text(json.dumps(summary, indent=2))
    return summary
