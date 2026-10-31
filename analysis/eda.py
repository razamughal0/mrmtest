"""
Exploratory analysis (run: python analysis/eda.py)
Prints the headline findings quoted in docs/PROJECT_REPORT.md §4.
This is the script version of the EDA notebook work: target balance,
group contrasts (future-Fail vs Pass), and feature-target correlations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import RAW_CSV, FEATURES

df = pd.read_csv(RAW_CSV)
pd.set_option("display.width", 140)

print(f"Rows (students): {len(df):,}   Courses: {df.course.nunique()}   "
      f"At-risk rate: {df.at_risk.mean():.1%}")
print("\nFinal-result mix (OULAD-style labels):")
print(df.final_result.value_counts(normalize=True).mul(100).round(1).to_string())

print("\n=== Students who end up AT-RISK vs ON-TRACK (mean, week-8 features) ===")
key = ["avg_quiz_score", "avg_assignment_score", "vle_total_clicks", "vle_active_days",
       "attendance_pct", "missing_submissions", "late_submissions", "score_trend", "prev_gpa"]
print(df.groupby("at_risk")[key].mean().round(1).T.rename(columns={0: "On-track", 1: "At-risk"}).to_string())
print("\nReading: the gap is visible by week 8 in EVERY behavioural signal - "
      "that is what makes early warning possible at all.")

print("\n=== Weekly LMS clicks: the disengagement curve ===")
wk = [f"w{i}_clicks" for i in range(1, 9)]
print(df.groupby("at_risk")[wk].mean().round(1).rename(index={0: "On-track", 1: "At-risk"}).to_string())

print("\n=== Correlation of each feature with at_risk (sorted) ===")
corr = df[FEATURES + ["at_risk"]].corr()["at_risk"].drop("at_risk").sort_values(key=abs, ascending=False)
print(corr.round(3).to_string())

print("\n=== At-risk rate by course ===")
print(df.groupby("course").at_risk.agg(["mean", "count"]).round(3).to_string())

print("\n=== Financial-stress signal (UCI features) ===")
print(df.groupby(["debtor", "tuition_up_to_date"]).at_risk.agg(["mean", "count"]).round(3).to_string())
