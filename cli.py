"""Command-line entry point.

  python cli.py build   --ideal clips/jobs.wav --id jobs [--transcript clips/jobs.txt]
  python cli.py compare --base data/jobs/L0_ideal.wav --test data/jobs/L3_noticeable.wav
  python cli.py eval
"""
import argparse
import json
from pathlib import Path

from speechflaw.compare import compare
from speechflaw.features import extract, load_audio


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build contrastive spectrum from an ideal clip")
    b.add_argument("--ideal", required=True)
    b.add_argument("--id", required=True)
    b.add_argument("--transcript", default="")
    b.add_argument("--out", default="data")
    c = sub.add_parser("compare", help="compare a take against a baseline")
    c.add_argument("--base", required=True)
    c.add_argument("--test", required=True)
    e = sub.add_parser("eval", help="evaluate detector on the dataset")
    e.add_argument("--data", default="data")
    e.add_argument("--out", default="results/eval.json")
    a = ap.parse_args()

    if a.cmd == "build":
        from speechflaw.synth import build
        text = Path(a.transcript).read_text() if a.transcript else ""
        build(a.ideal, a.out, a.id, text)
    elif a.cmd == "compare":
        r = compare(extract(load_audio(a.base)), extract(load_audio(a.test)))
        print(json.dumps({"scores": r.scores, "flaws": [f.to_dict() for f in r.flaws]}, indent=2))
    elif a.cmd == "eval":
        from speechflaw.evaluate import run
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        s = run(a.data, a.out)
        print(f"precision={s['precision']}  recall={s['recall']}")
        for k, v in s["per_kind"].items():
            print(f"  {k:20s} recall={v['recall']}  IoU={v['mean_iou']}  onset_err={v['mean_onset_err_s']}s  n={v['n']}")
        for r in s["rows"]:
            print(f"  {r['clip']:10s} {r['take']:20s} gold={r['gold']} pred={r['pred']} tp={r['tp']} fp={r['fp']} overall={r['overall']}")


if __name__ == "__main__":
    main()
