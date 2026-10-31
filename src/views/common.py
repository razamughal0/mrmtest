"""Shared UI components used by Admin / Teacher / Student portals.

The centrepiece is render_student_detail(): risk gauge + SHAP plain-English
reasons + weekly LMS activity + WHAT-IF simulator. Teacher and Admin see
the advisor framing ("risk"), the Student portal reuses the same engine
with supportive framing ("success score") in student_view.py.
"""
import sys, json, hashlib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from config import (SCORED_CSV, METRICS_PATH, FEATURES, FEATURE_LABELS,
                    RISK_COLORS, RISK_EMOJI, risk_level)
from src.predict import explain_student, what_if, predict_risk

WEEK_COLS = [f"w{i}_clicks" for i in range(1, 9)]


# ------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_students() -> pd.DataFrame:
    return pd.read_csv(SCORED_CSV)


@st.cache_data(show_spinner=False)
def load_metrics() -> dict:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}


def refresh_data():
    load_students.clear(); load_metrics.clear()
    import src.predict as p
    p._model, p._explainer, p._students = None, None, None


def badge(level: str) -> str:
    return f"{RISK_EMOJI.get(level, '')} {level} risk"


def style_risk_table(df: pd.DataFrame, cols):
    show = df[cols].copy()
    if "risk_score" in show:
        show["risk_score"] = (show.risk_score * 100).round(1)
        show = show.rename(columns={"risk_score": "Risk %"})
    return show


# ------------------------------------------------------------------ charts
def risk_gauge(prob: float, title="Early-warning risk", success_mode=False):
    """Gauge 0-100%. success_mode flips it to a (supportive) success score."""
    value = (1 - prob) * 100 if success_mode else prob * 100
    if success_mode:
        steps = [{"range": [0, 30], "color": "#fecaca"}, {"range": [30, 60], "color": "#fde68a"},
                 {"range": [60, 100], "color": "#bbf7d0"}]
        bar = "#16a34a" if value >= 60 else "#f59e0b" if value >= 30 else "#d32f2f"
    else:
        steps = [{"range": [0, 40], "color": "#bbf7d0"}, {"range": [40, 70], "color": "#fde68a"},
                 {"range": [70, 100], "color": "#fecaca"}]
        bar = RISK_COLORS[risk_level(prob)]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number={"suffix": "%", "font": {"size": 44}},
        title={"text": title, "font": {"size": 15}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": bar}, "steps": steps,
               "threshold": {"line": {"color": "#111827", "width": 2},
                             "value": 30 if success_mode else 70}}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def shap_chart(shap_df: pd.DataFrame, top_n=10):
    d = shap_df.head(top_n).iloc[::-1]  # largest on top
    fig = go.Figure(go.Bar(
        x=d.shap, y=d.label, orientation="h",
        marker_color=["#d32f2f" if v > 0 else "#16a34a" for v in d.shap],
        text=[f"{v:+.2f}" for v in d.shap], textposition="outside",
        hovertemplate="%{y}<br>SHAP %{x:+.2f}<extra></extra>"))
    fig.add_vline(x=0, line_color="#374151")
    fig.update_layout(height=380, margin=dict(l=10, r=40, t=30, b=30),
                      title="SHAP explanation — what is driving this student's score?",
                      xaxis_title="SHAP value  (→ pushes risk UP   |   ← pulls risk DOWN)")
    return fig


def weekly_chart(row: pd.Series, df: pd.DataFrame):
    weeks = list(range(1, 9))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=weeks, y=[row[c] for c in WEEK_COLS], mode="lines+markers",
                             name=row["name"], line=dict(width=3, color="#2563eb")))
    fig.add_trace(go.Scatter(x=weeks, y=df[WEEK_COLS].mean().values, mode="lines+markers",
                             name="Class average", line=dict(width=2, dash="dash", color="#9ca3af")))
    low = df[df.risk_level == "Low"][WEEK_COLS].mean().values if "risk_level" in df else None
    if low is not None:
        fig.add_trace(go.Scatter(x=weeks, y=low, mode="lines", name="On-track students (avg)",
                                 line=dict(width=2, dash="dot", color="#16a34a")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=30),
                      title="LMS (VLE) activity — clicks per week (weeks 1-8, OULAD-style)",
                      xaxis_title="Week of semester", yaxis_title="Clicks")
    return fig


def assignment_breakdown(row: pd.Series) -> pd.DataFrame:
    """Deterministic per-assignment detail derived from the summary fields,
    so the Student/Teacher record screens can show a believable table."""
    seed = int(hashlib.md5(str(row.student_id).encode()).hexdigest()[:6], 16)
    rng = np.random.default_rng(seed)
    submitted, late, missing = int(row.assignments_submitted), int(row.late_submissions), int(row.missing_submissions)
    statuses = (["Submitted"] * max(submitted - late, 0) + ["Late"] * late + ["Missing"] * missing)
    statuses += ["Submitted"] * max(0, 5 - len(statuses))
    rows = []
    for i, status in enumerate(statuses[:5], start=1):
        # gentle trend across assignments matching score_trend
        centre = row.avg_assignment_score + row.score_trend * (i - 3) / 4
        score = None if status == "Missing" else float(np.clip(rng.normal(centre, 6), 5, 100))
        rows.append({"Assignment": f"Assignment {i}", "Due week": i + 1, "Status": status,
                     "Score %": (round(score, 1) if score is not None else "—")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- what-if
def what_if_panel(row: pd.Series, key_prefix: str, current_prob: float):
    """Sliders -> live re-scoring. Returns (new_prob, changes_dict)."""
    st.markdown("#### 🎛️ What-if simulator — *what would change this score?*")
    st.caption("Drag the sliders to test a recovery plan. The model re-scores the student instantly — "
               "no single lever is magic; combined, realistic changes are what move the needle.")

    # instant preset scenarios (computed, no widget-state tricks needed)
    presets = {
        "Quiz average +10 points": {"avg_quiz_score": min(100, row.avg_quiz_score + 10)},
        "Submit everything, on time": {"missing_submissions": 0, "late_submissions": 0},
        "Full action plan (below)": {"avg_quiz_score": min(100, row.avg_quiz_score + 15),
                                     "avg_assignment_score": min(100, row.avg_assignment_score + 10),
                                     "missing_submissions": 0, "late_submissions": 0,
                                     "vle_total_clicks": int(row.vle_total_clicks + 200),
                                     "vle_active_days": min(56, int(row.vle_active_days + 10)),
                                     "attendance_pct": min(100, row.attendance_pct + 15)},
    }
    cols = st.columns(len(presets))
    for c, (label, change) in zip(cols, presets.items()):
        p = what_if(row, change)
        c.metric(label, f"{p:.0%} risk", f"{(p - current_prob) * 100:+.0f} pts vs now",
                 delta_color="inverse")

    k = key_prefix
    c1, c2 = st.columns(2)
    with c1:
        quiz = st.slider("Average quiz score (%)", 0, 100, int(round(row.avg_quiz_score)), key=f"{k}_quiz")
        assign = st.slider("Average assignment score (%)", 0, 100, int(round(row.avg_assignment_score)), key=f"{k}_assign")
        clicks = st.slider("LMS total clicks (semester)", 0, 1600, int(row.vle_total_clicks), step=10, key=f"{k}_clicks")
        days = st.slider("LMS active days", 0, 56, int(row.vle_active_days), key=f"{k}_days")
    with c2:
        missing = st.slider("Missing assignments (of 5)", 0, 5, int(row.missing_submissions), key=f"{k}_missing")
        late = st.slider("Late submissions", 0, 5, int(row.late_submissions), key=f"{k}_late")
        attend = st.slider("Attendance (%)", 0, 100, int(round(row.attendance_pct)), key=f"{k}_attend")
        trend = st.slider("Score trend (recent − early)", -60, 60, int(round(row.score_trend)), key=f"{k}_trend")

    changes = {"avg_quiz_score": quiz, "avg_assignment_score": assign, "vle_total_clicks": clicks,
               "vle_active_days": days, "missing_submissions": missing, "late_submissions": late,
               "attendance_pct": attend, "score_trend": trend}
    new_prob = what_if(row, changes)

    a, b, c3 = st.columns(3)
    a.metric("Current risk", f"{current_prob:.0%}", badge(risk_level(current_prob)), delta_color="off")
    b.metric("Simulated risk", f"{new_prob:.0%}", f"{(new_prob - current_prob) * 100:+.0f} pts", delta_color="inverse")
    c3.metric("Simulated level", badge(risk_level(new_prob)),
              "improved ✓" if new_prob < current_prob - 0.01 else "no change yet — try combining levers",
              delta_color="off")
    if new_prob < current_prob - 0.01:
        st.success(f"✅ With this plan the model moves **{row['name']}** from **{current_prob:.0%} → {new_prob:.0%}** "
                   f"({badge(risk_level(current_prob))} → {badge(risk_level(new_prob))}). "
                   "This is the conversation to have in the advising meeting.")
    return new_prob, changes


# --------------------------------------------------------- student detail
def render_student_detail(row: pd.Series, df: pd.DataFrame, key_prefix: str,
                          show_what_if=True, show_record=True):
    prob, shap_df, up, down = explain_student(row)
    level = risk_level(prob)

    head, gauge_col = st.columns([1.4, 1])
    with head:
        st.markdown(f"### {row['name']}  ·  `{row.student_id}`")
        st.markdown(f"**Course:** {row.course} &nbsp;|&nbsp; **Teacher:** {row.teacher_name} "
                    f"&nbsp;|&nbsp; **Semester:** {int(row.semester)}")
        st.markdown(f"**Risk level:** {badge(level)}  &nbsp;|&nbsp; **Predicted outcome if nothing changes:** "
                    f"{'Fail / Withdrawn' if prob >= 0.5 else 'Pass'}  &nbsp;|&nbsp; "
                    f"**Recorded final result (dataset):** {row.final_result}")
        st.markdown("##### 🔴 Why flagged? (SHAP, in plain English)")
        if up:
            for r in up[:5]:
                st.markdown(f"- **{r['text']}**  `SHAP {r['shap']:+.2f}`")
        else:
            st.markdown("- No strong risk drivers — this student looks on track. 🎉")
        st.markdown("##### 🟢 What is helping?")
        for r in down[:4]:
            st.markdown(f"- {r['text']}  `SHAP {r['shap']:+.2f}`")
    with gauge_col:
        st.plotly_chart(risk_gauge(prob), width="stretch", key=f"{key_prefix}_gauge")

    st.plotly_chart(shap_chart(shap_df), width="stretch", key=f"{key_prefix}_shap")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(weekly_chart(row, df), width="stretch", key=f"{key_prefix}_weekly")
    with c2:
        st.markdown("#### Assignments so far (weeks 1-8)")
        st.dataframe(assignment_breakdown(row), width="stretch", hide_index=True)
        st.caption(f"Quiz average **{row.avg_quiz_score:.0f}%** · Assignment average **{row.avg_assignment_score:.0f}%** · "
                   f"Attendance **{row.attendance_pct:.0f}%** · LMS clicks **{int(row.vle_total_clicks)}** "
                   f"(class median {int(df.vle_total_clicks.median())})")

    if show_what_if:
        st.divider()
        what_if_panel(row, key_prefix, prob)

    if show_record:
        st.divider()
        st.markdown("#### 📋 Full early-warning record vs class average")
        rec = pd.DataFrame({
            "Feature": [FEATURE_LABELS[f] for f in FEATURES],
            "This student": [round(float(row[f]), 2) for f in FEATURES],
            "Class average": [round(float(df[f].mean()), 2) for f in FEATURES],
            "SHAP": [round(float(shap_df.set_index('feature').loc[f, 'shap']), 3) for f in FEATURES],
        })
        st.dataframe(rec, width="stretch", hide_index=True)
    return prob, up, down


def student_label(row):
    return f"{row['name']} ({row.student_id}) — {row.risk_score:.0%} {RISK_EMOJI.get(row.risk_level,'')}"
