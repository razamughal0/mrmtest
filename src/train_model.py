"""
Train & compare models, score every student, compute global SHAP importance.

Models compared: XGBoost (selected), Random Forest, Logistic Regression.
Why XGBoost? Best AUC on tabular data like this + native SHAP TreeExplainer
support for exact, fast per-student explanations.

Early-warning care points (mention these in viva/report):
  * ONLY features knowable by week 8 are used -> no target leakage from
    end-of-semester results.
  * Recall matters more than raw accuracy here: missing a truly at-risk
    student (FN) is worse than a false alarm (FP) that only costs a
    friendly check-in email. Threshold bands (Low<40%<=Medium<70%<=High)
    reflect that.
  * Stratified split keeps the at-risk rate identical in train/test.

Run:  python -m src.train_model
Out:  models/model.pkl, models/metrics.json, models/test_predictions.csv,
      data/students_scored.csv  (every student + risk_score + risk_level)
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, classification_report)
from xgboost import XGBClassifier

from config import (RAW_CSV, SCORED_CSV, MODEL_PATH, METRICS_PATH,
                    TEST_PRED_PATH, MODELS_DIR, DATA_DIR, FEATURES, TARGET,
                    risk_level)


def evaluate(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "model": name,
        "accuracy": round(float(accuracy_score(y_test, pred)), 4),
        "precision": round(float(precision_score(y_test, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
    }, proba, pred


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not RAW_CSV.exists():
        from src.generate_data import generate
        generate().to_csv(RAW_CSV, index=False)
        print(f"Raw data was missing - generated {RAW_CSV}")

    df = pd.read_csv(RAW_CSV)
    X, y = df[FEATURES].astype(float), df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, df.index, test_size=0.20, random_state=42, stratify=y)

    pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            scale_pos_weight=pos_w, eval_metric="logloss",
            random_state=42, n_jobs=-1),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
    }

    results, best, best_proba, best_pred = [], None, None, None
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        metrics, proba, pred = evaluate(name, m, X_te, y_te)
        results.append(metrics)
        print(metrics)
        if name == "XGBoost":
            best, best_proba, best_pred = m, proba, pred

    print("\nClassification report (XGBoost, test set):")
    print(classification_report(y_te, best_pred, target_names=["On track", "At risk"]))

    cm = confusion_matrix(y_te, best_pred).tolist()
    fpr, tpr, _ = roc_curve(y_te, best_proba)

    # ----- global SHAP importance (mean |SHAP| on a test sample) -----
    explainer = shap.TreeExplainer(best)
    sample = X_te.sample(min(800, len(X_te)), random_state=42)
    sv = explainer.shap_values(sample)
    if isinstance(sv, list):        # older SHAP returns list per class
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                # newer SHAP: (n, features, classes)
        sv = sv[:, :, 1]
    global_imp = {f: round(float(v), 4) for f, v in
                  sorted(zip(FEATURES, np.abs(sv).mean(axis=0)),
                         key=lambda kv: -kv[1])}

    xgb_metrics = next(r for r in results if r["model"] == "XGBoost")
    metrics_out = {
        **xgb_metrics,
        "selected_model": "XGBoost",
        "n_students": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "at_risk_rate": round(float(y.mean()), 4),
        "confusion_matrix": cm,   # [[TN, FP], [FN, TP]]
        "comparison": results,
        "shap_global_importance": global_imp,
        "features": FEATURES,
    }
    METRICS_PATH.write_text(json.dumps(metrics_out, indent=2))
    joblib.dump(best, MODEL_PATH)
    pd.DataFrame({"y_true": y_te.values, "proba": best_proba,
                  "fpr": np.interp(np.linspace(0, 1, len(y_te)), fpr, fpr)}).to_csv(TEST_PRED_PATH, index=False)
    # store the ROC curve properly for plotting
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(TEST_PRED_PATH, index=False)

    # ----- score EVERY student for the dashboards -----
    scored = df.copy()
    scored["risk_score"] = best.predict_proba(X)[:, 1].round(4)
    scored["risk_level"] = scored.risk_score.apply(risk_level)
    scored["success_score"] = (100 * (1 - scored.risk_score)).round(1)
    scored = scored.sort_values("risk_score", ascending=False).reset_index(drop=True)
    scored.to_csv(SCORED_CSV, index=False)

    print(f"\nSaved {MODEL_PATH}\nSaved {METRICS_PATH}\nSaved {SCORED_CSV} ({len(scored)} students)")
    print(scored.risk_level.value_counts().to_string())
    print("Top SHAP features:", list(global_imp)[:6])
    return metrics_out


if __name__ == "__main__":
    main()
