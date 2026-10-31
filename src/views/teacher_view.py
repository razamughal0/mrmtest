"""TEACHER / ADVISOR portal.

Workflow it supports (the demo story):
  1. Dashboard: "how is my cohort doing, who needs me today?"
  2. Student Explorer: pick a student -> risk + SHAP reasons + what-if
     -> log an intervention (which the student then sees in their portal).
  3. Class Analytics + Interventions history.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RISK_COLORS, RISK_EMOJI
from src import database as db
from src.views.common import load_students, render_student_detail, student_label, badge

QUEUE_COLS = ["student_id", "name", "course", "risk_score", "risk_level",
              "avg_quiz_score", "missing_submissions", "late_submissions",
              "vle_total_clicks", "attendance_pct"]


def _cohort(user, df):
    tid = user.get("linked_teacher_id")
    mine = df[df.teacher_id == tid] if tid else df
    show_all = st.sidebar.checkbox("Show ALL students (not only my cohort)", value=False)
    return (df if show_all else mine), show_all


def render(user):
    df_all = load_students()
    df, show_all = _cohort(user, df_all)
    st.sidebar.caption(f"Cohort: **{'All students' if show_all else user.get('full_name','My cohort')}** · {len(df)} students")

    page = st.sidebar.radio("Teacher menu", ["🏠 Dashboard", "🔎 Student Explorer",
                                             "📊 Class Analytics", "🗂️ My Interventions"])

    # ---------------- Dashboard ----------------
    if page.startswith("🏠"):
        st.title("🏠 Teacher Dashboard")
        st.caption("Early warnings are generated from data available by **week 8** — while there is still time to help.")
        high = df[df.risk_level == "High"]; med = df[df.risk_level == "Medium"]
        my_open = db.list_interventions(teacher_name=user.get("full_name"))
        n_open = len(my_open[my_open.status != "Resolved"]) if len(my_open) else 0
        a, b, c, d = st.columns(4)
        a.metric("Students in view", f"{len(df):,}")
        b.metric("🔴 High risk — act this week", f"{len(high):,}", f"{len(high)/max(len(df),1):.0%} of cohort")
        c.metric("🟠 Medium risk — monitor", f"{len(med):,}")
        d.metric("Open interventions (mine)", n_open)

        c1, c2 = st.columns(2)
        with c1:
            fig = px.pie(df, names="risk_level", title="Risk mix of my cohort", hole=.45,
                         color="risk_level", color_discrete_map=RISK_COLORS,
                         category_orders={"risk_level": ["High", "Medium", "Low"]})
            st.plotly_chart(fig, width="stretch", key="t_dash_pie")
        with c2:
            fig = px.histogram(df, x="risk_score", nbins=25, color="risk_level",
                               color_discrete_map=RISK_COLORS, title="Distribution of risk scores",
                               category_orders={"risk_level": ["High", "Medium", "Low"]})
            st.plotly_chart(fig, width="stretch", key="t_dash_hist")

        st.markdown("### 🚨 Today's priority queue (highest risk first)")
        st.caption("Open **Student Explorer** from the sidebar to see *why* each student is flagged (SHAP) and to log help.")
        queue = df.sort_values("risk_score", ascending=False).head(25)
        show = queue[QUEUE_COLS].copy()
        show["risk_score"] = (show.risk_score * 100).round(1)
        st.dataframe(show.rename(columns={"risk_score": "Risk %"}),
                     width="stretch", hide_index=True)

        st.markdown("### Common warning signs in my high-risk group (average)")
        if len(high):
            signs = pd.DataFrame({
                "Signal": ["Quiz average %", "Assignment average %", "Missing (of 5)", "Late",
                           "LMS clicks", "Attendance %", "Score trend"],
                "High-risk students": [high.avg_quiz_score.mean(), high.avg_assignment_score.mean(),
                                       high.missing_submissions.mean(), high.late_submissions.mean(),
                                       high.vle_total_clicks.mean(), high.attendance_pct.mean(), high.score_trend.mean()],
                "Low-risk students": [df[df.risk_level == "Low"][c].mean() for c in
                                      ["avg_quiz_score", "avg_assignment_score", "missing_submissions",
                                       "late_submissions", "vle_total_clicks", "attendance_pct", "score_trend"]],
            }).round(1)
            st.dataframe(signs, width="stretch", hide_index=True)

    # ---------------- Student Explorer ----------------
    elif page.startswith("🔎"):
        st.title("🔎 Student Explorer — risk, reasons & what-if")
        f1, f2, f3 = st.columns([1.2, 1, 2.2])
        levels = f1.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
        courses = sorted(df.course.unique())
        course = f2.selectbox("Course", ["All courses"] + courses)
        search = f3.text_input("Search by name or ID", placeholder="e.g. Fatima or STU-10234")
        view = df[df.risk_level.isin(levels)] if levels else df.iloc[0:0]
        if course != "All courses":
            view = view[view.course == course]
        if search:
            s = search.lower()
            view = view[view.name.str.lower().str.contains(s) | view.student_id.str.lower().str.contains(s)]
        view = view.sort_values("risk_score", ascending=False)
        st.caption(f"{len(view):,} students match — sorted by risk (highest first).")

        if not len(view):
            st.warning("No students match these filters."); return
        labels = [student_label(r) for _, r in view.iterrows()]
        pick = st.selectbox("👤 Select a student", labels, key="t_pick")
        row = view.iloc[labels.index(pick)]

        show = view.head(50)[QUEUE_COLS].copy()
        show["risk_score"] = (show.risk_score * 100).round(1)
        with st.expander(f"📋 Matching students ({min(len(view),50)} shown)"):
            st.dataframe(show.rename(columns={"risk_score": "Risk %"}), width="stretch", hide_index=True)

        st.divider()
        prob, up, down = render_student_detail(row, df_all, key_prefix=f"t_{row.student_id}")

        # ---- intervention logger (appears instantly in Student portal) ----
        st.divider()
        st.markdown(f"#### 🤝 Log an intervention for {row['name']}")
        st.caption("Saved to the database. The student sees this under **Messages from Teacher** in their portal.")
        with st.form("intervention_form"):
            kind = st.selectbox("Type of help", ["Meeting", "Email", "Call", "Tutoring referral",
                                                 "Finance referral", "Study-plan review", "Wellbeing referral"])
            default_note = ""
            if up:
                default_note = (f"We discussed: {up[0]['text'].lower()}. "
                                + (up[0]["advice"] if up[0]["advice"] else "Agreed next steps together."))
            note = st.text_area("Note / message to student", value=default_note, height=90)
            status = st.selectbox("Status", ["Open", "In Progress", "Resolved"])
            if st.form_submit_button("💾 Save intervention", type="primary"):
                db.add_intervention(row.student_id, row["name"], user.get("full_name", "Teacher"),
                                    kind, note, risk_at_time=float(prob), status=status)
                st.success("Intervention saved ✓ — visible now in the student's portal and under My Interventions.")

        hist = db.list_interventions(student_id=row.student_id)
        st.markdown(f"##### History for this student ({len(hist)})")
        if len(hist):
            st.dataframe(hist[["created_at", "teacher_name", "kind", "note", "status"]],
                         width="stretch", hide_index=True)
        else:
            st.info("No interventions logged yet for this student.")

    # ---------------- Class Analytics ----------------
    elif page.startswith("📊"):
        st.title("📊 Class Analytics")
        by_course = df.groupby("course").agg(
            students=("student_id", "count"), avg_risk=("risk_score", "mean"),
            high=("risk_level", lambda s: (s == "High").sum())).reset_index()
        by_course["avg_risk"] = (by_course.avg_risk * 100).round(1)
        st.plotly_chart(px.bar(by_course, x="course", y="avg_risk", color="avg_risk",
                               color_continuous_scale=["#16a34a", "#f59e0b", "#d32f2f"],
                               title="Average risk % by course", text="avg_risk"),
                        width="stretch", key="t_ana_course")
        st.dataframe(by_course, width="stretch", hide_index=True)

        st.plotly_chart(px.scatter(df, x="vle_total_clicks", y="avg_quiz_score", color="risk_level",
                                   color_discrete_map=RISK_COLORS, opacity=.55,
                                   hover_name="name", hover_data=["student_id", "course"],
                                   title="Engagement vs performance — the bottom-left corner is the danger zone",
                                   labels={"vle_total_clicks": "LMS clicks (engagement)", "avg_quiz_score": "Quiz average %"}),
                        width="stretch", key="t_ana_scatter")
        st.plotly_chart(px.box(df, x="risk_level", y="attendance_pct", color="risk_level",
                               color_discrete_map=RISK_COLORS, title="Attendance by risk level",
                               category_orders={"risk_level": ["Low", "Medium", "High"]}),
                        width="stretch", key="t_ana_box")

    # ---------------- Interventions ----------------
    else:
        st.title("🗂️ My Interventions")
        mine = db.list_interventions(teacher_name=user.get("full_name"))
        if not len(mine):
            st.info("No interventions yet. Log one from **Student Explorer** after reviewing a flagged student."); return
        a, b, c = st.columns(3)
        a.metric("Total logged", len(mine))
        b.metric("Open / In progress", int((mine.status != "Resolved").sum()))
        c.metric("Resolved", int((mine.status == "Resolved").sum()))
        st.dataframe(mine[["id", "created_at", "student_id", "student_name", "kind", "note", "status", "risk_at_time"]],
                     width="stretch", hide_index=True)
        st.markdown("#### Update a status")
        iid = st.selectbox("Intervention ID", mine.id.tolist())
        new = st.selectbox("New status", ["Open", "In Progress", "Resolved"])
        if st.button("Update status", type="primary"):
            db.update_intervention_status(int(iid), new); st.success("Updated ✓"); st.rerun()
