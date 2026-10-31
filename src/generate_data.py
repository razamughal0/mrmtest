"""
Dataset builder — OULAD (#99) + UCI Student Dropout (#100), combined schema.

WHY a generator?
  * OULAD (32,593 students, 7 modules, VLE clickstream by week, assessments)
    and UCI Dropout (4,424 students, demographic/academic/financial) are
    both free, but they cannot be row-merged (different universities /
    students). The standard approach for this combined project is to use
    ONE schema that takes the best features of both:
        UCI  -> background + prior record + financial features
        OULAD -> week-by-week VLE activity + submission behaviour
  * This script generates a realistic, statistically-consistent dataset in
    that combined schema (n=4,424, mirroring UCI) so the project runs
    anywhere with zero downloads. `download_uci()` below shows how to pull
    the REAL UCI file when internet is available, and the README explains
    how to swap in OULAD's studentInfo/studentVLE/studentAssessment CSVs.

Latent-factor design (so correlations are believable, not random noise):
  ability        -> admission grade, prev GPA, quiz/assignment scores
  engagement     -> VLE clicks, active days, attendance, late/missing
  financial_stress -> debtor, tuition_up_to_date, attendance, engagement
  at_risk (target) is then sampled from those realised features, so the ML
  model has to *re-discover* the pattern — exactly like real data.

Run:  python -m src.generate_data
Out:  data/students_raw.csv
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from config import RAW_CSV, DATA_DIR, TEACHERS

FIRST_M = ["Ahmed","Ali","Muhammad","Hassan","Usman","Bilal","Omar","Hamza","Talha","Shahid",
           "Arslan","Danish","Fahad","Imran","Kamran","Noman","Rizwan","Saad","Waqas","Yasir",
           "James","John","Michael","David","Daniel","Liam","Noah","Oliver","Ethan","Lucas"]
FIRST_F = ["Fatima","Ayesha","Zainab","Maryam","Hira","Sana","Iqra","Mahnoor","Amna","Nadia",
           "Areeba","Dua","Eman","Fizza","Hafsa","Kiran","Laiba","Mehwish","Nimra","Rabia",
           "Emma","Sophia","Olivia","Ava","Mia","Isabella","Charlotte","Amelia","Harper","Ella"]
LAST = ["Khan","Ahmed","Ali","Sheikh","Malik","Hussain","Raza","Iqbal","Tariq","Shah","Butt",
        "Chaudhry","Farooq","Siddiqui","Akhtar","Mehmood","Aslam","Javed","Naz","Anwar",
        "Smith","Johnson","Brown","Taylor","Wilson","Davies","Evans","Thomas","Roberts","Walker"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def generate(n: int = 4424, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---------- latent factors ----------
    ability = rng.normal(0, 1, n)
    engagement = 0.45 * ability + rng.normal(0, 0.9, n)          # correlated, not identical
    fin_stress = rng.random(n) < 0.28                              # ~28% under financial pressure
    engagement = engagement - fin_stress * rng.uniform(0, 0.8, n)  # stress hurts engagement

    # ---------- identity / enrolment ----------
    teacher_idx = rng.integers(0, len(TEACHERS), n)
    courses = [TEACHERS[i]["course"] for i in teacher_idx]
    teacher_names = [TEACHERS[i]["name"] for i in teacher_idx]
    teacher_ids = [TEACHERS[i]["teacher_id"] for i in teacher_idx]
    gender_male = (rng.random(n) < 0.52).astype(int)
    names = []
    for m in gender_male:
        first = rng.choice(FIRST_M if m else FIRST_F)
        names.append(f"{first} {rng.choice(LAST)}")

    student_ids = [f"STU-{10001 + i}" for i in range(n)]
    emails = [f"{sid.lower()}@student.university.edu" for sid in student_ids]

    # ---------- UCI-style background / prior record ----------
    age = np.clip(rng.normal(20.5, 2.2, n), 17, 38).round().astype(int)
    first_generation = (rng.random(n) < 0.32).astype(int)
    scholarship_holder = ((rng.random(n) < 0.22) | ((ability > 1.1) & (rng.random(n) < 0.5))).astype(int)
    debtor = ((fin_stress) & (rng.random(n) < 0.55)).astype(int)
    tuition_up_to_date = np.where(fin_stress, (rng.random(n) < 0.45), (rng.random(n) < 0.96)).astype(int)
    tuition_up_to_date = np.where(debtor == 1, 0, tuition_up_to_date)

    admission_grade = np.clip(66 + 11 * ability + rng.normal(0, 7, n), 35, 100)
    prev_gpa = np.clip(2.75 + 0.45 * ability + 0.15 * engagement + rng.normal(0, 0.35, n), 0.5, 4.0)
    credits_enrolled = rng.integers(12, 21, n)
    credits_approved_prev_sem = np.clip(
        np.round(credits_enrolled * sigmoid(0.9 + 0.7 * ability + 0.5 * engagement + rng.normal(0, .5, n))), 0, 21).astype(int)
    prev_sem_grade = np.clip(62 + 10 * ability + 5 * engagement + rng.normal(0, 8, n), 20, 100)

    # ---------- OULAD-style VLE activity (weeks 1-8) ----------
    attendance_pct = np.clip(78 + 11 * engagement - 6 * fin_stress + rng.normal(0, 8, n), 15, 100)
    # weekly clicks: engaged students stay active; disengaged fade week by week
    base_clicks = np.exp(3.4 + 0.55 * engagement + rng.normal(0, 0.35, n))   # ~ clicks / week
    weekly = {}
    fade = np.clip(-engagement, -1, 2)                                       # >0 => fading
    for w in range(1, 9):
        level = base_clicks * np.exp(-fade * (w - 1) * 0.13)
        weekly[w] = np.maximum(0, rng.poisson(np.maximum(level, 0.1))).astype(int)
    vle_total_clicks = sum(weekly.values())
    vle_active_days = np.clip(np.round(vle_total_clicks / rng.uniform(6, 14, n)), 0, 56).astype(int)
    forum_posts = np.maximum(0, rng.poisson(np.exp(0.6 + 0.6 * engagement))).astype(int)
    resources_viewed = np.clip(np.round(vle_total_clicks * rng.uniform(0.08, 0.22, n)), 0, 300).astype(int)
    quiz_attempts = np.clip(np.round(rng.normal(7 + 2.2 * engagement, 2.2, n)), 0, 14).astype(int)

    # ---------- assessments (5 due by week 8) ----------
    avg_quiz_score = np.clip(63 + 11 * ability + 5 * engagement + rng.normal(0, 9, n), 5, 100)
    avg_assignment_score = np.clip(64 + 10 * ability + 6 * engagement + rng.normal(0, 9, n), 5, 100)
    # first vs latest score -> trend (falling scores are a classic OULAD signal)
    first_score = np.clip(avg_assignment_score + rng.normal(4, 6, n), 0, 100)
    last_score = np.clip(avg_assignment_score + 6 * engagement + rng.normal(-2, 7, n), 0, 100)
    score_trend = np.clip(last_score - first_score, -80, 80)

    # submission behaviour driven mainly by engagement
    p_submit = sigmoid(1.6 + 1.1 * engagement - 0.6 * fin_stress)
    assignments_submitted = np.array([rng.binomial(5, p) for p in p_submit])
    missing_submissions = 5 - assignments_submitted
    p_late = sigmoid(-1.4 - 0.9 * engagement + 0.5 * fin_stress)
    late_submissions = np.array([rng.binomial(s, p) if s else 0
                                 for s, p in zip(assignments_submitted, p_late)])

    df = pd.DataFrame({
        "student_id": student_ids, "name": names, "email": emails,
        "course": courses, "teacher_id": teacher_ids, "teacher_name": teacher_names,
        "semester": rng.integers(1, 7, n),
        "age": age, "gender_male": gender_male, "first_generation": first_generation,
        "scholarship_holder": scholarship_holder, "debtor": debtor,
        "tuition_up_to_date": tuition_up_to_date,
        "admission_grade": admission_grade.round(1), "prev_gpa": prev_gpa.round(2),
        "credits_enrolled": credits_enrolled, "credits_approved_prev_sem": credits_approved_prev_sem,
        "prev_sem_grade": prev_sem_grade.round(1), "attendance_pct": attendance_pct.round(1),
        "vle_total_clicks": vle_total_clicks.astype(int), "vle_active_days": vle_active_days,
        "forum_posts": forum_posts, "resources_viewed": resources_viewed,
        "quiz_attempts": quiz_attempts, "avg_quiz_score": avg_quiz_score.round(1),
        "assignments_submitted": assignments_submitted.astype(int),
        "late_submissions": late_submissions.astype(int),
        "missing_submissions": missing_submissions.astype(int),
        "avg_assignment_score": avg_assignment_score.round(1),
        "score_trend": score_trend.round(1),
    })
    for w in range(1, 9):
        df[f"w{w}_clicks"] = weekly[w]

    # ---------- target: at_risk (Fail / Withdrawn by end of semester) ----------
    # Sampled from the REALISED features above (with noise) - the model only
    # ever sees the features, never the latent factors.
    # Coefficients deliberately MODERATE (+ generous noise): real student
    # outcomes are messy. This targets ~35% at-risk (UCI dropout rate is
    # ~32%, OULAD Fail+Withdrawn ~38%) and keeps predicted probabilities
    # spread out - a model that only ever says 0% or 100% is useless for
    # early-warning triage and for what-if simulations.
    z = (
        -1.85
        # non-linear / interaction effects (why tree models beat a plain
        # logistic regression on real OULAD data: risk spikes at thresholds
        # and when several warning signs appear together)
        + 0.90 * ((df.vle_total_clicks < 120) & (df.score_trend < -10))
        + 0.75 * (df.missing_submissions >= 2)
        + 0.65 * (df.avg_quiz_score < 50)
        + 0.55 * (df.attendance_pct < 60)
        + 0.65 * ((df.debtor == 1) & (df.tuition_up_to_date == 0))
        + 0.70 * ((df.avg_quiz_score < 55) & (df.score_trend < -15))
        - 0.018 * (df.avg_quiz_score - 63)
        - 0.015 * (df.avg_assignment_score - 64)
        - 0.012 * (df.attendance_pct - 78)
        + 0.25 * df.late_submissions
        + 0.35 * df.missing_submissions
        - 0.0012 * (df.vle_total_clicks - df.vle_total_clicks.median())
        - 0.012 * df.score_trend
        - 0.25 * (df.prev_gpa - 2.75)
        + 0.25 * df.debtor
        - 0.20 * df.tuition_up_to_date
        - 0.15 * df.scholarship_holder
        + 0.10 * df.first_generation
        + rng.normal(0, 0.90, n)
    )
    p_risk = sigmoid(z)
    df["at_risk"] = (rng.random(n) < p_risk).astype(int)

    # OULAD-style final result label (for display / reporting only)
    def final_result(row):
        if row.at_risk == 0:
            return "Distinction" if row.avg_assignment_score >= 78 and row.avg_quiz_score >= 78 else "Pass"
        return "Withdrawn" if (row.missing_submissions >= 3 or row.attendance_pct < 45) else "Fail"
    df["final_result"] = df.apply(final_result, axis=1)
    return df


def download_uci() -> bool:
    """Optional: fetch the REAL UCI Dropout dataset via ucimlrepo (pip install ucimlrepo).
    Kept as a documented hook - the app works fully on the generated combined
    dataset without this. See README 'Using the real datasets'."""
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
        ds = fetch_ucirepo(name="Predict Students Dropout and Academic Success")
        X, y = ds.data.features, ds.data.targets
        out = DATA_DIR / "uci_real_raw.csv"
        pd.concat([X, y], axis=1).to_csv(out, index=False)
        print(f"Saved real UCI data -> {out}  shape={X.shape}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[optional] Real UCI download not available ({e}). "
              "Continuing with the generated combined dataset.")
        return False


if __name__ == "__main__":
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_uci()
    df = generate()
    df.to_csv(RAW_CSV, index=False)
    print(f"Saved {RAW_CSV}  rows={len(df)}  at_risk rate={df.at_risk.mean():.1%}")
    print(df.final_result.value_counts().to_string())
