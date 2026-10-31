"""STUDENT portal — "check my own record".

Framing matters: students see a *Success Score* (100 - risk%) and friendly,
actionable guidance — never a scary "you will fail" label. Same model and
same SHAP engine underneath; different, supportive presentation. This is a
deliberate ethical/UX choice worth stating in the report/viva.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from config import RISK_COLORS
from src import database as db
from src.predict import explain_student
from src.views.common import (load_students, risk_gauge, shap_chart, weekly_chart,
                              assignment_breakdown, what_if_panel)


def _my_row(user, df):
    sid = user.get("linked_student_id")
    match = df[df.student_id == sid]
    if len(match):
        return match.iloc[0]
    st.warning("Your login isn't linked to a student record yet — showing a sample record. Ask Admin to link your account.")
    return df.iloc[0]


def _status_message(prob):
    if prob >= 0.70:
        return ("🔴", "Needs attention now", "Some warning signs have built up — but every one of them is fixable, "
                "and your teachers can help. Start with the Action Plan page and book a meeting this week.")
    if prob >= 0.40:
        return ("🟠", "Keep an eye on it", "You're in the middle zone. A few small, steady changes now will "
                "move you firmly into the green — see your personalised plan below.")
    return ("🟢", "On track — great work!", "Your activity, submissions and scores look healthy. "
            "Keep the routine going and check in here each week.")


def render(user):
    df = load_students()
    row = _my_row(user, df)
    prob, shap_df, up, down = explain_student(row)
    emoji, headline, msg = _status_message(prob)

    page = st.sidebar.radio("Student menu", ["🏠 My Dashboard", "📋 My Record & Activity",
                                             "🎯 Why My Score? + Action Plan", "💬 Messages from Teacher"])

    if page.startswith("🏠"):
        st.title(f"👋 Welcome, {row['name'].split()[0]}!")
        st.caption(f"{row.student_id} · {row.course} · Semester {int(row.semester)} · Teacher: {row.teacher_name}")
        g, info = st.columns([1, 1.3])
        with g:
            st.plotly_chart(risk_gauge(prob, title="My Success Score", success_mode=True),
                            width="stretch", key="s_gauge")
        with info:
            st.markdown(f"## {emoji} {headline}")
            st.write(msg)
            st.caption("Your Success Score is an early-warning estimate from your LMS activity, "
                       "submissions and scores up to week 8. It is **not** a grade and **not** a final judgement — "
                       "it exists so you (and your teacher) can act early.")
            if up:
                st.markdown("**Biggest things to work on:**")
                for r in up[:3]:
                    st.markdown(f"- {r['text']}" + (f"  →  *{r['advice']}*" if r["advice"] else ""))

        a, b, c, d, e = st.columns(5)
        a.metric("Quiz average", f"{row.avg_quiz_score:.0f}%", f"class {df.avg_quiz_score.mean():.0f}%")
        b.metric("Assignment avg", f"{row.avg_assignment_score:.0f}%", f"class {df.avg_assignment_score.mean():.0f}%")
        c.metric("Submitted", f"{int(row.assignments_submitted)} / 5", f"{int(row.missing_submissions)} missing",
                 delta_color="inverse" if row.missing_submissions else "normal")
        d.metric("Attendance", f"{row.attendance_pct:.0f}%", f"class {df.attendance_pct.mean():.0f}%")
        e.metric("LMS clicks", f"{int(row.vle_total_clicks)}", f"class median {int(df.vle_total_clicks.median())}")
        st.plotly_chart(weekly_chart(row, df), width="stretch", key="s_weekly")
        msgs = db.list_interventions(student_id=row.student_id)
        if len(msgs):
            st.info(f"💬 You have **{len(msgs)} message(s)/note(s)** from your teachers — open **Messages from Teacher**.")

    elif page.startswith("📋"):
        st.title("📋 My Record & Activity")
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("#### My profile")
            profile = pd.DataFrame({"Field": ["Student ID", "Name", "Email", "Course", "Teacher", "Semester",
                                              "Age", "Admission grade", "Previous GPA", "Previous semester grade",
                                              "Credits enrolled", "Credits passed (prev sem)", "Scholarship",
                                              "Tuition up to date", "Final result (recorded)"],
                                    "Value": [row.student_id, row["name"], row.email, row.course, row.teacher_name,
                                              int(row.semester), int(row.age), f"{row.admission_grade:.0f}%",
                                              f"{row.prev_gpa:.2f}", f"{row.prev_sem_grade:.0f}%",
                                              int(row.credits_enrolled), int(row.credits_approved_prev_sem),
                                              "Yes" if row.scholarship_holder else "No",
                                              "Yes" if row.tuition_up_to_date else "No — please visit Student Finance",
                                              row.final_result]})
            st.dataframe(profile, width="stretch", hide_index=True)
        with p2:
            st.markdown("#### My assignments (weeks 1-8)")
            st.dataframe(assignment_breakdown(row), width="stretch", hide_index=True)
            st.markdown("#### My engagement")
            eng = pd.DataFrame({"Activity": ["LMS total clicks", "LMS active days", "Resources viewed",
                                             "Quiz attempts", "Forum posts", "Late submissions", "Missing submissions"],
                                "Me": [int(row.vle_total_clicks), int(row.vle_active_days), int(row.resources_viewed),
                                       int(row.quiz_attempts), int(row.forum_posts), int(row.late_submissions),
                                       int(row.missing_submissions)],
                                "Class average": [round(df[c].mean(), 1) for c in
                                                  ["vle_total_clicks", "vle_active_days", "resources_viewed",
                                                   "quiz_attempts", "forum_posts", "late_submissions", "missing_submissions"]]})
            st.dataframe(eng, width="stretch", hide_index=True)
        st.plotly_chart(weekly_chart(row, df), width="stretch", key="s_rec_weekly")
        wk = pd.DataFrame({"Week": range(1, 9), "My clicks": [row[f"w{i}_clicks"] for i in range(1, 9)],
                           "Class average": df[[f"w{i}_clicks" for i in range(1, 9)]].mean().round(1).values})
        st.dataframe(wk.T.rename(columns={i: f"Week {i+1}" for i in range(8)}), width="stretch")
        st.caption("Something wrong in your record? Contact your teacher or the Admin office — "
                   "early-warning data should always be correctable by the student it describes.")

    elif page.startswith("🎯"):
        st.title("🎯 Why My Score? + My Action Plan")
        st.markdown(f"Your early-warning risk is currently **{prob:.0%}** (Success Score **{100*(1-prob):.0f}**). "
                    "Here is exactly what is influencing it — nothing hidden.")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 🔴 Pulling my score down")
            if up:
                for r in up[:5]:
                    st.markdown(f"- **{r['text']}**")
                    if r["advice"]:
                        st.caption(f"   💡 {r['advice']}")
            else:
                st.markdown("- Nothing major — keep it up! 🎉")
        with c2:
            st.markdown("##### 🟢 Working in my favour")
            for r in down[:5]:
                st.markdown(f"- {r['text']}")
        with st.expander("🔬 Show the technical SHAP chart (what your teacher sees)"):
            st.plotly_chart(shap_chart(shap_df), width="stretch", key="s_shap")
            st.caption("SHAP values explain the model's reasoning for *you specifically*: "
                       "red bars push risk up, green bars pull it down.")

        st.divider()
        st.markdown("### 🗓️ Build my recovery / improvement plan")
        new_prob, changes = what_if_panel(row, f"s_{row.student_id}", prob)
        st.markdown("#### ✅ My commitments this month")
        for i, r in enumerate(up[:4]):
            if r["advice"]:
                st.checkbox(r["advice"], key=f"s_commit_{i}")
        st.checkbox("Book a 15-minute meeting with my teacher", key="s_commit_meet")
        st.caption("Ticking these is for your own tracking in this demo. In a real deployment, "
                   "your plan would be shared with your advisor so they can support you.")

    else:
        st.title("💬 Messages from Teacher")
        msgs = db.list_interventions(student_id=row.student_id)
        if not len(msgs):
            st.info("No messages yet. When your teacher logs a meeting, email or referral for you, it will appear here.")
            return
        for _, m in msgs.iterrows():
            icon = {"Meeting": "🤝", "Email": "✉️", "Call": "📞", "Tutoring referral": "📚",
                    "Finance referral": "💷"}.get(m.kind, "📝")
            with st.container(border=True):
                st.markdown(f"{icon} **{m.kind}** · {m.teacher_name} · `{str(m.created_at)[:10]}` · Status: **{m.status}**")
                st.write(m.note)
        st.caption("These notes are written by your teachers in the Teacher portal (Interventions). "
                   "Reply through your normal university email / office hours.")
