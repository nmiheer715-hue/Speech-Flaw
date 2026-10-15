"""SpeechFlaw dashboard.   Run:  streamlit run app.py"""
import json
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from speechflaw import align
from speechflaw.compare import compare, TH
from speechflaw.features import extract, load_audio

DATA = Path("data")
COLORS = {"Pace": "#E4572E", "Intonation": "#7B5EA7", "Volume": "#29A3A3",
          "Clarity": "#E8A33D", "Pausing": "#3D7DD8"}

st.set_page_config(page_title="SpeechFlaw", layout="wide")
st.title("SpeechFlaw — contrastive speech delivery analysis")
st.caption("Align a participant's delivery to an ideal reading of the same text, find exactly "
           "where it deviates, and explain why in acoustic terms.")


@st.cache_data(show_spinner=False)
def analyse(base_path: str, test_path: str, transcript: str):
    base = extract(load_audio(base_path))
    test = extract(load_audio(test_path))
    words = align.cached_words(base_path, transcript) if align.available() else None
    return compare(base, test, base_words=words), words is not None


# ---------------------------------------------------------------- input --------
with st.sidebar:
    st.header("Input")
    clips = sorted(p.parent.name for p in DATA.glob("*/labels.json"))
    mode = st.radio("Source", ["Dataset pair", "Upload your own"] if clips else ["Upload your own"])
    gold, transcript = [], ""
    if mode == "Dataset pair":
        clip = st.selectbox("Clip", clips)
        man = json.loads((DATA / clip / "labels.json").read_text())
        takes = {t["level"]: t for t in man["takes"]}
        names = list(takes)
        want = st.query_params.get("take", names[0])
        level = st.selectbox("Participant take", names, index=names.index(want) if want in names else 0)
        base_path = str(DATA / clip / man["baseline"])
        test_path = str(DATA / clip / takes[level]["file"])
        gold = takes[level]["flaws"]
        tp = DATA / clip / "transcript.txt"
        transcript = tp.read_text() if tp.exists() else ""
        show_gold = st.checkbox("Overlay ground-truth labels", value=True)
    else:
        bf = st.file_uploader("Ideal / baseline recording", type=["wav", "mp3", "m4a", "flac", "ogg"])
        tf = st.file_uploader("Participant recording (same text)", type=["wav", "mp3", "m4a", "flac", "ogg"])
        transcript = st.text_area("Transcript (optional, improves word labels)")
        show_gold = False
        if not (bf and tf):
            st.info("Upload both recordings to start.")
            st.stop()
        tmp = Path(tempfile.mkdtemp())
        base_path, test_path = str(tmp / ("base" + Path(bf.name).suffix)), str(tmp / ("test" + Path(tf.name).suffix))
        Path(base_path).write_bytes(bf.getvalue())
        Path(test_path).write_bytes(tf.getvalue())
    st.divider()
    st.caption("Rubric thresholds (frozen)")
    st.json({k: v for k, v in TH.items() if not k.endswith("strong")}, expanded=False)

with st.spinner("Extracting features, aligning with DTW, grounding flaws…"):
    c, has_words = analyse(base_path, test_path, transcript)

# ---------------------------------------------------------------- scores -------
s = c.scores
cols = st.columns(6)
cols[0].metric("Overall", f"{s['Overall']}/10")
for col, k in zip(cols[1:], ["Pace", "Intonation", "Volume", "Clarity", "Pausing"]):
    col.metric(k, f"{s[k]}/10")

a1, a2 = st.columns(2)
a1.markdown("**Ideal baseline**")
a1.audio(base_path)
a1.caption(" · ".join(f"{k}: {v}" for k, v in c.base.stats.items()))
a2.markdown("**Participant take**")
a2.audio(test_path)
a2.caption(" · ".join(f"{k}: {v}" for k, v in c.test.stats.items()))

# ---------------------------------------------------------------- plot ---------
cv = c.curves
t = cv["t"]
fig = make_subplots(rows=5, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    subplot_titles=["Pitch contour (semitones re speaker median)",
                                    "Pitch variability (robust σ, 1.5 s window)",
                                    "Energy (dB re speaker median)",
                                    "Local speech rate (participant ÷ baseline, from DTW slope)",
                                    "High-frequency energy share >2 kHz (dB) — clarity"])
BASE, TEST = "#9AA0A6", ("#58A6FF" if st.get_option("theme.base") == "dark" else "#1F6FEB")


def pair(row, yt, yb, name):
    fig.add_trace(go.Scatter(x=t, y=yb, name="baseline (aligned)", line=dict(color=BASE, width=1.5),
                             showlegend=row == 1, legendgroup="b"), row=row, col=1)
    fig.add_trace(go.Scatter(x=t, y=yt, name="participant", line=dict(color=TEST, width=1.5),
                             showlegend=row == 1, legendgroup="t"), row=row, col=1)


def f0_view(x):
    """Display-only: drop octave-jump outliers and median-smooth voiced runs."""
    from scipy.ndimage import median_filter
    x = np.where(np.abs(x) > 8, np.nan, x)
    good = ~np.isnan(x)
    y = x.copy()
    y[good] = median_filter(x[good], 9)
    return y


def e_view(x):
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(np.clip(x, -40, 15), 15)


pair(1, f0_view(cv["f0_test"]), f0_view(cv["f0_base"]), "f0")
pair(2, cv["pv_test"], cv["pv_base"], "pv")
pair(3, e_view(cv["e_test"]), e_view(cv["e_base"]), "e")
fig.add_trace(go.Scatter(x=t, y=cv["rate_ratio"], name="rate ratio", line=dict(color=TEST),
                         showlegend=False), row=4, col=1)
for y0 in (TH["rush"], 1.0, TH["drag"]):
    fig.add_hline(y=y0, line=dict(dash="dot" if y0 != 1 else "solid", color="#999", width=1), row=4, col=1)
pair(5, cv["hf_test"], cv["hf_base"], "hf")

for f in c.flaws:
    fig.add_vrect(x0=f.start, x1=f.end, fillcolor=COLORS[f.category], opacity=0.18, line_width=0,
                  row="all", col=1)
    fig.add_annotation(x=f.start, y=1.0, xref="x", yref="y domain", text=f.kind, showarrow=False,
                       xanchor="left", yanchor="top", font=dict(size=10, color=COLORS[f.category]))
if show_gold:
    for g in gold:
        fig.add_vrect(x0=g["start"], x1=max(g["end"], g["start"] + 0.05), line=dict(color="black", dash="dash",
                      width=1), fillcolor="rgba(0,0,0,0)", row=1, col=1)
DARK = st.get_option("theme.base") == "dark"
fig.update_layout(height=980, margin=dict(l=40, r=20, t=40, b=30),
                  legend=dict(orientation="h", y=1.04, x=0), hovermode="x unified",
                  template="plotly_dark" if DARK else "plotly_white",
                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
fig.update_xaxes(title_text="participant time (s)", row=5, col=1)
st.plotly_chart(fig, use_container_width=True)
if show_gold:
    st.caption("Shaded bands = detected flaws (colour = rubric category). Dashed boxes on the top "
               "row = injected ground-truth flaw regions.")

# ---------------------------------------------------------------- flaws --------
st.subheader(f"Grounded flaws ({len(c.flaws)})")
if not c.flaws:
    st.success("No delivery deviations beyond the rubric thresholds.")
for i, f in enumerate(c.flaws):
    with st.expander(f"{f.start:6.2f}s – {f.end:6.2f}s   ·   {f.kind}   ·   severity {f.severity:.2f}",
                     expanded=i < 3):
        st.markdown(f"**{f.category}** — {f.explanation}")
        if f.words:
            st.markdown(f"Words affected: *“{f.words}”*")
        st.caption(f"metric: {f.metric} = {f.value:.3f}")
        st.audio(test_path, start_time=max(0, int(f.start - 0.5)))

st.download_button("Download JSON report",
                   json.dumps({"scores": s, "flaws": [f.to_dict() for f in c.flaws],
                               "baseline_stats": c.base.stats, "participant_stats": c.test.stats,
                               "dtw_cost": c.dtw_cost}, indent=2),
                   file_name="speechflaw_report.json")
if not has_words:
    st.caption("Tip: `pip install faster-whisper` to label each flaw with the words it covers.")
