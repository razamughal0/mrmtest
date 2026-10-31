"""
Generate the figures used in README.md and docs/PROJECT_REPORT.md.
Run AFTER train_model:   python -m src.make_figures   (needs matplotlib)
Outputs: docs/figures/*.png
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import SCORED_CSV, METRICS_PATH, TEST_PRED_PATH, FEATURE_LABELS, BASE_DIR

OUT = BASE_DIR / "docs" / "figures"
COLORS = {"High": "#d32f2f", "Medium": "#f59e0b", "Low": "#16a34a"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SCORED_CSV)
    metrics = json.loads(METRICS_PATH.read_text())

    # 1) weekly clicks by risk level (the core OULAD insight)
    wk = [f"w{i}_clicks" for i in range(1, 9)]
    plt.figure(figsize=(7, 4))
    for lvl in ["Low", "Medium", "High"]:
        plt.plot(range(1, 9), df[df.risk_level == lvl][wk].mean().values,
                 marker="o", linewidth=2.5, color=COLORS[lvl], label=f"{lvl} risk")
    plt.title("LMS engagement fades early for at-risk students (OULAD-style)")
    plt.xlabel("Week of semester"); plt.ylabel("Average clicks per week"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "weekly_clicks_by_risk.png", dpi=160); plt.close()

    # 2) risk score distribution
    plt.figure(figsize=(7, 4))
    for lvl in ["Low", "Medium", "High"]:
        plt.hist(df[df.risk_level == lvl].risk_score, bins=30, alpha=.75,
                 color=COLORS[lvl], label=lvl)
    plt.axvline(.4, color="#374151", ls="--"); plt.axvline(.7, color="#374151", ls="--")
    plt.title("Distribution of predicted risk scores (n=4,424)")
    plt.xlabel("Predicted P(at-risk)"); plt.ylabel("Students"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT / "risk_distribution.png", dpi=160); plt.close()

    # 3) global SHAP importance
    imp = metrics["shap_global_importance"]
    top = dict(list(imp.items())[:12])
    plt.figure(figsize=(7, 4.6))
    plt.barh([FEATURE_LABELS.get(k, k) for k in reversed(top)], list(reversed(top.values())), color="#dc2626")
    plt.title("Global SHAP importance (mean |SHAP|, test sample)")
    plt.xlabel("Mean |SHAP value|"); plt.tight_layout()
    plt.savefig(OUT / "shap_global.png", dpi=160); plt.close()

    # 4) ROC curve
    roc = pd.read_csv(TEST_PRED_PATH)
    plt.figure(figsize=(5.5, 4.5))
    plt.plot(roc.fpr, roc.tpr, linewidth=2.5, color="#2563eb",
             label=f"XGBoost (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Random")
    plt.title("ROC curve (held-out test set)"); plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate"); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "roc_curve.png", dpi=160); plt.close()

    # 5) confusion matrix
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15,
                    color="white" if cm[i, j] > cm.max() / 2 else "#111827")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred. on-track", "Pred. at-risk"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actually on-track", "Actually at-risk"])
    ax.set_title(f"Confusion matrix (test set, n={metrics['n_test']})")
    plt.tight_layout(); plt.savefig(OUT / "confusion_matrix.png", dpi=160); plt.close()

    # 6) a single student's SHAP explanation (~75% risk, recoverable case)
    from src.predict import explain_student
    cand = df[(df.risk_level == "High") & (df.assignments_submitted >= 2)]
    row = cand.iloc[(cand.risk_score - 0.75).abs().argsort().iloc[0]]
    prob, sdf, _, _ = explain_student(row)
    top10 = sdf.head(10).iloc[::-1]
    plt.figure(figsize=(7.5, 4.6))
    plt.barh(top10.label, top10.shap,
             color=["#d32f2f" if v > 0 else "#16a34a" for v in top10.shap])
    plt.axvline(0, color="#374151")
    plt.title(f"SHAP explanation for one student\n{row['name']} ({row.student_id}) — risk {prob:.0%}", fontsize=11)
    plt.xlabel("SHAP value (→ risk up | ← risk down)"); plt.tight_layout()
    plt.savefig(OUT / "shap_one_student.png", dpi=160); plt.close()

    print("Figures written to", OUT)
    for p in sorted(OUT.glob("*.png")):
        print(" -", p.name, f"{p.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
