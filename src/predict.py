"""
Prediction + SHAP explanation engine (shared by all 3 portals).

Key function: explain_student(row) ->
  (risk_probability, shap_values, reasons)
where `reasons` is a list of plain-English sentences like:
  🔴 "3 late submissions — pushing risk UP (+0.42)"
  🟢 "Average quiz score is 82% — pulling risk DOWN (-0.31)"
That translation layer is what makes this more than a black-box score:
an advisor (or student) sees *why*, and therefore *what to do next*.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
import shap
from config import MODEL_PATH, SCORED_CSV, RAW_CSV, FEATURES, FEATURE_LABELS, risk_level

_model, _explainer, _students = None, None, None


def load_model():
    global _model, _explainer
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        _explainer = shap.TreeExplainer(_model)
    return _model, _explainer


def load_students() -> pd.DataFrame:
    global _students
    path = SCORED_CSV if SCORED_CSV.exists() else RAW_CSV
    _students = pd.read_csv(path)
    return _students


def predict_risk(feature_row: dict | pd.Series | pd.DataFrame) -> float:
    """Return P(at-risk) 0..1 for one student's feature dict/Series."""
    model, _ = load_model()
    if isinstance(feature_row, pd.DataFrame):
        X = feature_row[FEATURES].astype(float)
    else:
        X = pd.DataFrame([{f: float(feature_row[f]) for f in FEATURES}])
    return float(model.predict_proba(X)[:, 1][0])


def _shap_for(X: pd.DataFrame) -> np.ndarray:
    _, explainer = load_model()
    sv = explainer.shap_values(X)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    return sv


# --- plain-English templates: (feature, value, increases_risk?) -> sentence
def _sentence(feat: str, v: float, increases: bool) -> str:
    t = {
        "late_submissions":      f"{int(v)} late submission(s) so far" if increases else f"Submits on time (only {int(v)} late)",
        "missing_submissions":   f"{int(v)} missing assignment(s)" if increases else "No missing assignments",
        "assignments_submitted": f"Only {int(v)} of 5 assignments submitted" if increases else f"All/most assignments submitted ({int(v)} of 5)",
        "avg_quiz_score":        f"Average quiz score is low ({v:.0f}%)" if increases else f"Strong quiz scores ({v:.0f}% average)",
        "avg_assignment_score":  f"Average assignment score is low ({v:.0f}%)" if increases else f"Good assignment scores ({v:.0f}% average)",
        "score_trend":           f"Scores are FALLING over time ({v:+.0f} points)" if increases else f"Scores are improving ({v:+.0f} points)",
        "vle_total_clicks":      f"Very low LMS activity ({int(v)} clicks this semester)" if increases else f"Active on the LMS ({int(v)} clicks)",
        "vle_active_days":       f"Logged in on only {int(v)} days" if increases else f"Regularly active online ({int(v)} days)",
        "attendance_pct":        f"Attendance is only {v:.0f}%" if increases else f"Good attendance ({v:.0f}%)",
        "prev_gpa":              f"Previous GPA is low ({v:.2f})" if increases else f"Solid previous GPA ({v:.2f})",
        "prev_sem_grade":        f"Previous semester grade was {v:.0f}%" if increases else f"Previous semester grade was {v:.0f}%",
        "admission_grade":       f"Admission grade was relatively low ({v:.0f}%)" if increases else f"Came in with a solid admission grade ({v:.0f}%)",
        "credits_enrolled":      f"Heavy course load this semester ({int(v)} credits)" if increases else f"Manageable course load ({int(v)} credits)",
        "debtor":                "Has outstanding dues (debtor) — possible financial stress" if increases else "No outstanding dues",
        "tuition_up_to_date":    "Tuition fees are NOT up to date" if increases else "Tuition fees up to date",
        "scholarship_holder":    "No scholarship support" if increases else "Scholarship holder (financial support)",
        "first_generation":      "First-generation university student (less guidance at home)" if increases else "Not first-generation",
        "credits_approved_prev_sem": f"Passed only {int(v)} credits last semester" if increases else f"Passed {int(v)} credits last semester",
        "quiz_attempts":         f"Few quiz attempts ({int(v)}) — low practice" if increases else f"Practices regularly ({int(v)} quiz attempts)",
        "forum_posts":           f"Barely participates in forums ({int(v)} posts)" if increases else f"Participates in forums ({int(v)} posts)",
        "resources_viewed":      f"Few learning resources opened ({int(v)})" if increases else f"Uses learning resources well ({int(v)} viewed)",
    }
    return t.get(feat, f"{FEATURE_LABELS.get(feat, feat)} = {v:g}")


# Concrete, actionable advice tied to each risk-driving feature
RECOMMENDATIONS = {
    "late_submissions": "Use a planner / set reminders 2 days before every deadline.",
    "missing_submissions": "Talk to your teacher TODAY about missed work — partial credit may still be possible.",
    "assignments_submitted": "Submit every assignment, even if incomplete — submitted beats missing.",
    "avg_quiz_score": "Re-attempt practice quizzes weekly; aim to raise quiz average step by step.",
    "avg_assignment_score": "Visit office hours / tutoring for feedback before the next assignment.",
    "score_trend": "Your scores are slipping — review the last topic again before moving on.",
    "vle_total_clicks": "Log in to the LMS daily, even for 15 minutes — small regular sessions add up.",
    "vle_active_days": "Build a fixed daily study slot on the LMS.",
    "attendance_pct": "Prioritise attending every class for the next 3 weeks.",
    "prev_gpa": "Ask your advisor about a study-skills / foundation support plan.",
    "debtor": "Visit the finance office — installment plans and aid options exist.",
    "tuition_up_to_date": "Contact student finance about payment plans before it affects enrolment.",
    "first_generation": "Join the first-generation student mentoring circle.",
    "quiz_attempts": "Attempt every practice quiz — attempts alone improve scores.",
    "resources_viewed": "Open the slides/recordings for every lecture you attend.",
    "forum_posts": "Ask one question a week on the course forum — it counts as engagement.",
}


def explain_student(row: pd.Series | dict, top_n: int = 6):
    """
    Returns (risk_prob, shap_df, risk_reasons, protective_reasons)
    shap_df columns: feature, label, value, shap ( >0 pushes risk UP )
    """
    X = pd.DataFrame([{f: float(row[f]) for f in FEATURES}])
    prob = predict_risk(X)
    sv = _shap_for(X)[0]
    shap_df = pd.DataFrame({
        "feature": FEATURES,
        "label": [FEATURE_LABELS[f] for f in FEATURES],
        "value": [float(row[f]) for f in FEATURES],
        "shap": sv,
    }).sort_values("shap", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    risk_reasons, protective = [], []
    for _, r in shap_df.iterrows():
        item = {"feature": r.feature, "shap": float(r.shap), "value": float(r.value),
                "text": _sentence(r.feature, float(r.value), r.shap > 0),
                "advice": RECOMMENDATIONS.get(r.feature, "")}
        (risk_reasons if r.shap > 0 else protective).append(item)
    return prob, shap_df, risk_reasons[:top_n], protective[:top_n]


def what_if(row: pd.Series | dict, changes: dict) -> float:
    """Re-score a student after applying hypothetical changes (sliders)."""
    new = {f: float(row[f]) for f in FEATURES}
    new.update({k: float(v) for k, v in changes.items() if k in FEATURES})
    # keep submission arithmetic consistent for the simulator
    if "missing_submissions" in changes or "assignments_submitted" in changes:
        new["assignments_submitted"] = float(np.clip(5 - new["missing_submissions"], 0, 5))
    new["late_submissions"] = float(np.clip(new["late_submissions"], 0, new["assignments_submitted"]))
    return predict_risk(new)


if __name__ == "__main__":
    df = load_students()
    # demo with the student nearest 72% risk (the classic what-if story),
    # rather than the single most extreme case
    demo = df.iloc[(df.risk_score - 0.72).abs().argsort().iloc[0]] if "risk_score" in df else df.iloc[0]
    prob, shap_df, up, down = explain_student(demo)
    print(f"{demo['name']} ({demo['student_id']})  risk={prob:.0%}  level={risk_level(prob)}")
    print("Pushing risk UP:")
    for r in up: print(f"  🔴 {r['text']}  (SHAP {r['shap']:+.2f})")
    print("Pulling risk DOWN:")
    for r in down: print(f"  🟢 {r['text']}  (SHAP {r['shap']:+.2f})")
    new = what_if(demo, {"avg_quiz_score": min(100, demo.avg_quiz_score * 1.10)})
    print(f"What-if quiz +10% only: risk {prob:.0%} -> {new:.0%}  (one lever alone is honest but small)")
    plan = what_if(demo, {"avg_quiz_score": min(100, demo.avg_quiz_score + 15),
                          "missing_submissions": 0, "late_submissions": 0,
                          "vle_total_clicks": demo.vle_total_clicks + 200,
                          "attendance_pct": min(100, demo.attendance_pct + 15)})
    print(f"What-if combined action plan: risk {prob:.0%} -> {plan:.0%}")
