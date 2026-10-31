"""
Central configuration for the Student Early-Warning System.
Project: #99 OULAD + #100 UCI Student Dropout (combined).

Idea in one line:
  Flag at-risk students EARLY in the semester (by week 8) using LMS (VLE)
  activity + assessments + academic/financial record, EXPLAIN every flag
  with SHAP in plain English, and let advisors/students run WHAT-IF
  simulations ("if quiz scores rise 10%, risk drops 72% -> 40%").
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_CSV = DATA_DIR / "students_raw.csv"          # generated / downloaded data
SCORED_CSV = DATA_DIR / "students_scored.csv"     # raw + risk_score/risk_level
DB_PATH = DATA_DIR / "app.db"

MODEL_PATH = MODELS_DIR / "model.pkl"
METRICS_PATH = MODELS_DIR / "metrics.json"
TEST_PRED_PATH = MODELS_DIR / "test_predictions.csv"

# ---------------------------------------------------------------- features
# Model features (all knowable by ~week 8 of the semester -> truly "early").
# Sources:
#   UCI Dropout (#100): admission_grade, prev qualifications, credits
#                       approved/enrolled, debtor, tuition_up_to_date,
#                       scholarship_holder, age, gender ...
#   OULAD (#99):        VLE clicks / active days, assignment submission
#                       behaviour and scores, score trend ...
# NOTE (fairness by design): `age` and `gender` exist in the raw datasets
# and in our CSVs, but they are deliberately NOT model features. A student
# must never be flagged "at risk" because of who they are - only because of
# changeable behaviour/record (activity, submissions, scores, finances),
# which an advisor can actually act on. See docs/PROJECT_REPORT.md §Ethics.
FEATURES = [
    # background / support context (UCI)
    "first_generation",
    "scholarship_holder",
    "debtor",
    "tuition_up_to_date",
    # prior academic record (UCI)
    "admission_grade",            # 0-100 (%)
    "prev_gpa",                   # 0-4
    "credits_enrolled",
    "credits_approved_prev_sem",
    "prev_sem_grade",             # 0-100
    # engagement / attendance
    "attendance_pct",             # 0-100
    # VLE / LMS activity (OULAD)
    "vle_total_clicks",
    "vle_active_days",
    "forum_posts",
    "resources_viewed",
    "quiz_attempts",
    # assessments so far (OULAD) - 5 assignments due by week 8
    "avg_quiz_score",             # 0-100
    "assignments_submitted",      # 0-5
    "late_submissions",           # 0-5
    "missing_submissions",        # 0-5
    "avg_assignment_score",       # 0-100
    "score_trend",                # last score - first score (-100..100)
]

TARGET = "at_risk"  # 1 = Fail / Withdrawn at end of semester

FEATURE_LABELS = {
    "age": "Age",
    "gender_male": "Gender (Male)",
    "first_generation": "First-generation student",
    "scholarship_holder": "Scholarship holder",
    "debtor": "Has outstanding dues (debtor)",
    "tuition_up_to_date": "Tuition fees up to date",
    "admission_grade": "Admission grade",
    "prev_gpa": "Previous GPA",
    "credits_enrolled": "Credits enrolled",
    "credits_approved_prev_sem": "Credits passed (prev. semester)",
    "prev_sem_grade": "Previous semester grade",
    "attendance_pct": "Attendance %",
    "vle_total_clicks": "LMS (VLE) total clicks",
    "vle_active_days": "LMS active days",
    "forum_posts": "Forum posts",
    "resources_viewed": "Resources viewed",
    "quiz_attempts": "Quiz attempts",
    "avg_quiz_score": "Average quiz score",
    "assignments_submitted": "Assignments submitted (of 5)",
    "late_submissions": "Late submissions",
    "missing_submissions": "Missing submissions",
    "avg_assignment_score": "Average assignment score",
    "score_trend": "Score trend (recent - early)",
}

# Risk bands (single source of truth, used by every screen)
def risk_level(score: float) -> str:
    """score is a probability 0..1"""
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"

RISK_COLORS = {"High": "#d32f2f", "Medium": "#f59e0b", "Low": "#16a34a"}
RISK_EMOJI = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}

# ---------------------------------------------------------------- people
# Teachers are shared by the data generator and the DB seeder so the
# Teacher portal always matches the Students table.
TEACHERS = [
    {"teacher_id": "T01", "name": "Dr. Sarah Ahmed",   "username": "sarah.ahmed",   "course": "CS101 - Intro to Programming"},
    {"teacher_id": "T02", "name": "Prof. Imran Khan",   "username": "imran.khan",   "course": "CS201 - Data Structures"},
    {"teacher_id": "T03", "name": "Dr. Ayesha Malik",   "username": "ayesha.malik",   "course": "MTH101 - calculus"},
    {"teacher_id": "T04", "name": "Mr. David Wilson",    "username": "david.wilson",     "course": "ENG101 - English Composition"},
    {"teacher_id": "T05", "name": "Dr. Fatima Noor",    "username": "fatima.noor",    "course": "BUS201 - Principles of Management"},
    {"teacher_id": "T06", "name": "Prof. Hassan Raza",  "username": "hassan.raza",  "course": "PHY101 - Physics I"},
    {"teacher_id": "T07", "name": "Ms. Zainab Tariq",   "username": "zainab.tariq",   "course": "CS301 - Database Systems"},
    {"teacher_id": "T08", "name": "Dr. Omar Farooq",    "username": "omar.farooq",    "course": "STA201 - Statistics"},
]

COURSES = [t["course"] for t in TEACHERS]

# Demo accounts (also shown on the login screen for viva/demo)
DEMO_ACCOUNTS = {
    "admin":   {"username": "admin",        "password": "admin123",   "role": "admin"},
    "teacher": {"username": "sarah.ahmed",  "password": "teacher123", "role": "teacher"},
    "student": {"username": "demo.student", "password": "student123", "role": "student"},
}
