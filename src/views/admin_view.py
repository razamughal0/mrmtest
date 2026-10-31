"""ADMIN portal — whole-institution view.

Admin can: monitor every cohort, manage students (add / batch-score /
export), manage user accounts & teachers, inspect & retrain the model
(metrics, confusion matrix, ROC, global SHAP), and oversee all
interventions logged by teachers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (FEATURES, FEATURE_LABELS, RISK_COLORS, TEACHERS, COURSES,
                    SCORED_CSV, RAW_CSV, TEST_PRED_PATH, risk_level)
from src import database as db
from src.predict import predict_risk
from src.views.common import load_students, load_metrics, refresh_data, render_student_detail, student_label

TABLE_COLS = ["student_id", "name", "course", "teacher_name", "semester", "risk_score", "risk_level",
              "avg_quiz_score", "avg_assignment_score", "missing_submissions", "late_submissions",
              "vle_total_clicks", "attendance_pct", "final_result"]


def _show(df, cols=TABLE_COLS, **kw):
    show = df[cols].copy()
    if "risk_score" in show:
        show["risk_score"] = (show.risk_score * 100).round(1)
        show = show.rename(columns={"risk_score": "Risk %"})
    st.dataframe(show, width="stretch", hide_index=True, **kw)
    return show


def render(user):
    df = load_students(); metrics = load_metrics()
    page = st.sidebar.radio("Admin menu", ["🏠 Overview", "🎓 Students", "➕ Add / Batch-Score Students",
                                           "👥 Users & Teachers", "🤖 Model Center", "🗂️ All Interventions"])

    # ---------------- Overview ----------------
    if page.startswith("🏠"):
        st.title("🏠 Admin Overview — whole institution")
        high, med = df[df.risk_level == "High"], df[df.risk_level == "Medium"]
        inter = db.list_interventions()
        a, b, c, d, e = st.columns(5)
        a.metric("Total students", f"{len(df):,}")
        b.metric("🔴 High risk", f"{len(high):,}", f"{len(high)/len(df):.0%} of all")
        c.metric("🟠 Medium risk", f"{len(med):,}", f"{len(med)/len(df):.0%}")
        d.metric("Model ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", metrics.get("selected_model", ""))
        e.metric("Interventions logged", f"{len(inter):,}",
                 f"{int((inter.status!='Resolved').sum()) if len(inter) else 0} still open")

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.pie(df, names="risk_level", hole=.45, title="Students by risk level",
                                   color="risk_level", color_discrete_map=RISK_COLORS,
                                   category_orders={"risk_level": ["High", "Medium", "Low"]}),
                            width="stretch", key="a_pie")
        with c2:
            by_course = df.groupby("course").risk_score.mean().mul(100).round(1).reset_index()
            st.plotly_chart(px.bar(by_course, x="course", y="risk_score", color="risk_score",
                                   color_continuous_scale=["#16a34a", "#f59e0b", "#d32f2f"],
                                   title="Average risk % by course", text="risk_score"),
                            width="stretch", key="a_course")

        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(px.histogram(df, x="risk_score", nbins=30, color="risk_level",
                                         color_discrete_map=RISK_COLORS, title="Risk-score distribution (all students)"),
                            width="stretch", key="a_hist")
        with c4:
            wt = [f"w{i}_clicks" for i in range(1, 9)]
            trend = df.groupby("risk_level")[wt].mean().T
            trend.index = range(1, 9)
            fig = go.Figure()
            for lvl in ["Low", "Medium", "High"]:
                if lvl in trend:
                    fig.add_trace(go.Scatter(x=trend.index, y=trend[lvl], mode="lines+markers",
                                             name=f"{lvl} risk", line=dict(color=RISK_COLORS[lvl], width=3)))
            fig.update_layout(title="Average LMS clicks per week — at-risk students disengage early",
                              xaxis_title="Week", yaxis_title="Avg clicks", height=360,
                              margin=dict(l=10, r=10, t=50, b=30))
            st.plotly_chart(fig, width="stretch", key="a_trend")
            st.caption("☝️ This is the core OULAD insight the whole system is built on: disengagement is "
                       "visible **weeks** before failure.")

        st.markdown("### 🚨 Highest-risk students institution-wide (top 20)")
        _show(df.sort_values("risk_score", ascending=False).head(20))

    # ---------------- Students ----------------
    elif page.startswith("🎓"):
        st.title("🎓 Students — search, inspect, export")
        f1, f2, f3, f4 = st.columns(4)
        levels = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"], key="a_lv")
        course = f2.selectbox("Course", ["All courses"] + sorted(df.course.unique()), key="a_course_f")
        teacher = f3.selectbox("Teacher", ["All teachers"] + sorted(df.teacher_name.unique()), key="a_teacher_f")
        search = f4.text_input("Search name / ID", key="a_search")
        view = df[df.risk_level.isin(levels)] if levels else df.iloc[0:0]
        if course != "All courses": view = view[view.course == course]
        if teacher != "All teachers": view = view[view.teacher_name == teacher]
        if search:
            s = search.lower()
            view = view[view.name.str.lower().str.contains(s) | view.student_id.str.lower().str.contains(s)]
        view = view.sort_values("risk_score", ascending=False)
        st.caption(f"{len(view):,} students match.")
        _show(view.head(200))
        st.download_button("⬇️ Download filtered students (CSV)", view.to_csv(index=False).encode(),
                           "students_filtered.csv", "text/csv")
        st.download_button("⬇️ Download FULL scored dataset (CSV)", df.to_csv(index=False).encode(),
                           "students_scored.csv", "text/csv")

        if len(view):
            st.divider(); st.markdown("### 🔎 Inspect one student (full advisor view)")
            labels = [student_label(r) for _, r in view.head(300).iterrows()]
            pick = st.selectbox("Student", labels, key="a_pick")
            row = view.iloc[labels.index(pick)]
            render_student_detail(row, df, key_prefix=f"a_{row.student_id}")

    # ---------------- Add / batch ----------------
    elif page.startswith("➕"):
        st.title("➕ Add a student / batch-score a CSV")
        st.markdown("#### Add one student (scored instantly by the model)")
        with st.form("add_student"):
            n1, n2, n3 = st.columns(3)
            name = n1.text_input("Full name", "New Student")
            course = n2.selectbox("Course", COURSES)
            semester = n3.number_input("Semester", 1, 8, 1)
            st.caption("Early-warning features (defaults = class averages — adjust to the real student):")
            vals, cols = {}, st.columns(3)
            defaults = {"age": 20, "gender_male": 0}
            for i, f in enumerate(FEATURES):
                avg = float(df[f].mean())
                if f in ("first_generation", "scholarship_holder", "debtor", "tuition_up_to_date"):
                    vals[f] = cols[i % 3].selectbox(FEATURE_LABELS[f], [0, 1], index=int(round(avg)), key=f"add_{f}")
                elif f in ("assignments_submitted", "late_submissions", "missing_submissions"):
                    vals[f] = cols[i % 3].number_input(FEATURE_LABELS[f], 0, 5, int(round(avg)), key=f"add_{f}")
                elif f == "prev_gpa":
                    vals[f] = cols[i % 3].number_input(FEATURE_LABELS[f], 0.0, 4.0, round(avg, 2), key=f"add_{f}")
                else:
                    vals[f] = cols[i % 3].number_input(FEATURE_LABELS[f], value=round(avg, 1), key=f"add_{f}")
            create_login = st.checkbox("Also create a Student-portal login (username = student ID, password = student123)", value=True)
            if st.form_submit_button("🎓 Add student & score", type="primary"):
                teacher = next(t for t in TEACHERS if t["course"] == course)
                nums = [int(s.replace("STU-", "")) for s in df.student_id if str(s).startswith("STU-")]
                sid = f"STU-{max(nums) + 1}"
                prob = predict_risk(vals)
                weekly = np.maximum(0, np.random.default_rng(abs(hash(sid)) % 9999).poisson(
                    max(vals["vle_total_clicks"] / 8, 1), 8))
                new = {**vals, **defaults, "student_id": sid, "name": name,
                       "email": f"{sid.lower()}@student.university.edu", "course": course,
                       "teacher_id": teacher["teacher_id"], "teacher_name": teacher["name"],
                       "semester": int(semester), "at_risk": int(prob >= 0.5),
                       "final_result": "In progress", "risk_score": round(prob, 4),
                       "risk_level": risk_level(prob), "success_score": round(100 * (1 - prob), 1)}
                for i, w in enumerate(weekly, 1):
                    new[f"w{i}_clicks"] = int(w)
                new_df = pd.DataFrame([new])
                for path in (SCORED_CSV, RAW_CSV):
                    old = pd.read_csv(path)
                    pd.concat([old, new_df.reindex(columns=old.columns)], ignore_index=True).to_csv(path, index=False)
                if create_login:
                    db.create_user(sid.lower(), "student123", "student", name, new["email"], linked_student_id=sid)
                refresh_data()
                st.success(f"Added **{name}** ({sid}) → risk **{prob:.0%}** ({risk_level(prob)}). "
                           f"Find them under 🎓 Students." + (" Login created: " + sid.lower() + " / student123" if create_login else ""))
                st.rerun()

        st.divider()
        st.markdown("#### Batch-score an uploaded CSV")
        st.caption(f"CSV must contain these columns: `student_id` (optional) + the {len(FEATURES)} model features below.")
        with st.expander("📄 Required columns & downloadable template"):
            template = df[["student_id", "name"] + FEATURES].head(5)
            st.dataframe(template, width="stretch", hide_index=True)
            st.download_button("⬇️ Download template CSV", template.to_csv(index=False).encode(),
                               "batch_template.csv", "text/csv")
        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            batch = pd.read_csv(up)
            missing = [f for f in FEATURES if f not in batch.columns]
            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                from src.predict import load_model
                model, _ = load_model()
                batch["risk_score"] = model.predict_proba(batch[FEATURES].astype(float))[:, 1].round(4)
                batch["risk_level"] = batch.risk_score.apply(risk_level)
                batch = batch.sort_values("risk_score", ascending=False)
                st.success(f"Scored {len(batch)} students: "
                           + ", ".join(f"{k}={v}" for k, v in batch.risk_level.value_counts().items()))
                st.dataframe(batch, width="stretch", hide_index=True)
                st.download_button("⬇️ Download scored batch", batch.to_csv(index=False).encode(),
                                   "batch_scored.csv", "text/csv")

    # ---------------- Users & teachers ----------------
    elif page.startswith("👥"):
        st.title("👥 Users & Teachers")
        users = db.list_users()
        a, b, c = st.columns(3)
        a.metric("Admins", int((users.role == "admin").sum()))
        b.metric("Teachers", int((users.role == "teacher").sum()))
        c.metric("Student logins", int((users.role == "student").sum()))

        st.markdown("#### Teachers & their cohorts")
        rows = []
        inter = db.list_interventions()
        for t in TEACHERS:
            cohort = df[df.teacher_id == t["teacher_id"]]
            rows.append({"Teacher": t["name"], "Username": t["username"], "Course": t["course"],
                         "Students": len(cohort), "High risk": int((cohort.risk_level == "High").sum()),
                         "Avg risk %": round(cohort.risk_score.mean() * 100, 1),
                         "Interventions logged": int((inter.teacher_name == t["name"]).sum()) if len(inter) else 0})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.markdown("#### All user accounts")
        role_f = st.multiselect("Role", ["admin", "teacher", "student"], default=["admin", "teacher", "student"], key="u_role")
        uview = users[users.role.isin(role_f)]
        st.dataframe(uview.head(400), width="stretch", hide_index=True)
        st.caption(f"{len(uview):,} accounts (student logins are auto-seeded for a demo sample; create more below).")

        st.markdown("#### Create a user")
        with st.form("create_user"):
            u1, u2, u3 = st.columns(3)
            uname = u1.text_input("Username")
            pw = u2.text_input("Password", value="student123")
            role = u3.selectbox("Role", ["student", "teacher", "admin"])
            fname = st.text_input("Full name")
            link_s = st.text_input("Linked student ID (student role, e.g. STU-10001)", value="")
            link_t = st.selectbox("Linked teacher (teacher role)", [""] + [t["teacher_id"] + " - " + t["name"] for t in TEACHERS])
            if st.form_submit_button("➕ Create user", type="primary"):
                ok = db.create_user(uname, pw, role, fname or uname,
                                    linked_student_id=(link_s or None),
                                    linked_teacher_id=(link_t.split(" - ")[0] if link_t else None))
                st.success("User created ✓") if ok else st.error("Username already exists.")
                if ok: st.rerun()

        st.markdown("#### Deactivate / delete a user")
        others = users[users.username != "admin"]
        if len(others):
            target = st.selectbox("User", [f"{r.id} · {r.username} ({r.role})" for r in others.itertuples()], key="del_user")
            uid = int(target.split(" · ")[0])
            d1, d2 = st.columns(2)
            if d1.button("⏸️ Deactivate / reactivate"):
                cur = int(users.set_index("id").loc[uid, "is_active"])
                db.set_user_active(uid, not cur); st.success("Updated ✓"); st.rerun()
            if d2.button("🗑️ Delete user"):
                db.delete_user(uid); st.success("Deleted ✓"); st.rerun()

    # ---------------- Model Center ----------------
    elif page.startswith("🤖"):
        st.title("🤖 Model Center")
        if not metrics:
            st.warning("No metrics found — train the model first (button below).")
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Selected model", metrics["selected_model"])
            m2.metric("Accuracy", f"{metrics['accuracy']:.1%}")
            m3.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
            m4.metric("Recall (at-risk caught)", f"{metrics['recall']:.1%}")
            m5.metric("F1 score", f"{metrics['f1']:.3f}")
            st.caption(f"Trained on {metrics['n_train']:,} students · tested on {metrics['n_test']:,} unseen students · "
                       f"overall at-risk rate {metrics['at_risk_rate']:.0%} · features use **week 1-8 data only** (no leakage). "
                       "Recall is emphasised: missing a truly struggling student is worse than a false alarm.")

            st.markdown("#### Model comparison (same test set)")
            comp = pd.DataFrame(metrics["comparison"])
            st.dataframe(comp, width="stretch", hide_index=True)
            st.caption("All three models land within ~0.01 AUC of each other on this dataset. **XGBoost is deployed** because "
                       "it gives *exact* SHAP TreeExplainer explanations per student and captures non-linear "
                       "threshold effects (e.g. '2+ missing assignments') that dominate real OULAD data.")

            c1, c2 = st.columns(2)
            with c1:
                cm = np.array(metrics["confusion_matrix"])
                fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                x=["Predicted on-track", "Predicted at-risk"],
                                y=["Actually on-track", "Actually at-risk"],
                                title=f"Confusion matrix (test set, n={metrics['n_test']:,})")
                st.plotly_chart(fig, width="stretch", key="a_cm")
                tn, fp, fn, tp = cm.ravel()
                st.caption(f"✅ Correctly flagged **{tp}** at-risk students · ⚠️ missed **{fn}** (false negatives — the costly error) · "
                           f"false alarms **{fp}** (cost: one friendly check-in).")
            with c2:
                if TEST_PRED_PATH.exists():
                    roc = pd.read_csv(TEST_PRED_PATH)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=roc.fpr, y=roc.tpr, mode="lines", name=f"XGBoost (AUC {metrics['roc_auc']:.3f})",
                                             line=dict(width=3, color="#2563eb")))
                    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guessing",
                                             line=dict(dash="dash", color="#9ca3af")))
                    fig.update_layout(title="ROC curve", xaxis_title="False-positive rate",
                                      yaxis_title="True-positive rate", height=420, margin=dict(l=10, r=10, t=50, b=30))
                    st.plotly_chart(fig, width="stretch", key="a_roc")

            imp = metrics["shap_global_importance"]
            idf = pd.DataFrame({"feature": [FEATURE_LABELS.get(k, k) for k in imp], "mean |SHAP|": list(imp.values())})
            st.plotly_chart(px.bar(idf.head(15).iloc[::-1], x="mean |SHAP|", y="feature", orientation="h",
                                   title="Global SHAP importance — what drives risk across ALL students",
                                   color="mean |SHAP|", color_continuous_scale="Reds"),
                            width="stretch", key="a_shap_global")
            st.caption("Reading: LMS activity and submission behaviour (things a student can change *this week*) "
                       "outweigh background factors — exactly what an early-warning system needs.")

        st.divider()
        st.markdown("#### ♻️ Retrain model")
        st.caption("Regenerates scores for every student from `data/students_raw.csv` "
                   "(run `python -m src.generate_data` first if you replaced the raw data with real OULAD/UCI extracts).")
        if st.button("🚀 Retrain now", type="primary"):
            with st.spinner("Training XGBoost + comparing models + computing SHAP… (~10 seconds)"):
                from src.train_model import main as train_main
                out = train_main()
            refresh_data()
            st.success(f"Retrained ✓  Accuracy {out['accuracy']:.1%} · ROC-AUC {out['roc_auc']:.3f} · Recall {out['recall']:.1%}")
            st.rerun()

    # ---------------- Interventions ----------------
    else:
        st.title("🗂️ All Interventions (institution oversight)")
        inter = db.list_interventions()
        if not len(inter):
            st.info("No interventions logged yet."); return
        a, b, c, d = st.columns(4)
        a.metric("Total", len(inter))
        b.metric("Open", int((inter.status == "Open").sum()))
        c.metric("In progress", int((inter.status == "In Progress").sum()))
        d.metric("Resolved", int((inter.status == "Resolved").sum()))
        st.plotly_chart(px.histogram(inter, x="teacher_name", color="status", barmode="group",
                                     title="Interventions by teacher & status"),
                        width="stretch", key="a_inter_hist")
        tf = st.selectbox("Filter by teacher", ["All teachers"] + sorted(inter.teacher_name.dropna().unique()), key="a_inter_tf")
        view = inter if tf == "All teachers" else inter[inter.teacher_name == tf]
        st.dataframe(view[["id", "created_at", "student_id", "student_name", "teacher_name", "kind", "note", "status"]],
                     width="stretch", hide_index=True)
        iid = st.selectbox("Intervention ID to update", view.id.tolist(), key="a_inter_id")
        new = st.selectbox("New status", ["Open", "In Progress", "Resolved"], key="a_inter_new")
        if st.button("Update status", type="primary"):
            db.update_intervention_status(int(iid), new); st.success("Updated ✓"); st.rerun()
        st.download_button("⬇️ Download interventions (CSV)", inter.to_csv(index=False).encode(),
                           "interventions.csv", "text/csv")
