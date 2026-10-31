"""
🎓 Student Early-Warning System  —  OULAD (#99) + UCI Student Dropout (#100)

One app, three portals (login decides which one you see):
  👑 ADMIN    — institution overview, student/user management, Model Center
  👩‍🏫 TEACHER  — advisor dashboard: at-risk queue, SHAP reasons, what-if, interventions
  🎓 STUDENT  — "my record": success score, my activity, why-my-score, action plan, messages

Run everything from scratch:
    pip install -r requirements.txt
    python -m src.generate_data     # build combined OULAD+UCI-schema dataset
    python -m src.train_model       # train XGBoost, score all students, SHAP
    python -m src.database          # create SQLite DB + demo logins
    streamlit run app.py

Demo logins (also one-click buttons on the login screen):
    admin / admin123  ·  sarah.ahmed / teacher123  ·  demo.student / student123
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st

st.set_page_config(page_title="Student Early-Warning System", page_icon="🎓",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
  div[data-testid="stMetric"] {background:#f8fafc; border:1px solid #e5e7eb;
      border-radius:12px; padding:10px 12px;}
  div[data-testid="stSidebar"] {background:#0f172a;}
  div[data-testid="stSidebar"] * {color:#e2e8f0 !important;}
  div[data-testid="stSidebar"] input, div[data-testid="stSidebar"] select {color:#111827 !important;}
  h1 {letter-spacing:-.02em;}
</style>""", unsafe_allow_html=True)

from config import SCORED_CSV, MODEL_PATH, RAW_CSV, DEMO_ACCOUNTS  # noqa: E402
from src import database as db  # noqa: E402


# ------------------------------------------------------------- first run
def bootstrap():
    """If data/model don't exist yet, build them (one click, ~15 seconds)."""
    if SCORED_CSV.exists() and MODEL_PATH.exists():
        db.init_db()
        return True
    st.title("🎓 Student Early-Warning System")
    st.warning("First run: dataset/model not found yet.")
    st.markdown("Click below to (1) build the combined **OULAD + UCI-schema** dataset (4,424 students), "
                "(2) train **XGBoost** and compute **SHAP** explanations, and (3) create the "
                "database with demo **Admin / Teacher / Student** logins.")
    if st.button("🚀 Build dataset, train model & create demo accounts", type="primary"):
        with st.spinner("Building… this takes about 15 seconds."):
            if not RAW_CSV.exists():
                from src.generate_data import generate
                from config import DATA_DIR
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                generate().to_csv(RAW_CSV, index=False)
            from src.train_model import main as train_main
            train_main()
            db.init_db()
        st.success("Ready! Please log in.")
        st.rerun()
    st.code("pip install -r requirements.txt\npython -m src.generate_data\n"
            "python -m src.train_model\npython -m src.database\nstreamlit run app.py", language="bash")
    return False


# ------------------------------------------------------------------ login
def do_login(username, password):
    user = db.authenticate(username, password)
    if user:
        st.session_state.user = user
        st.rerun()
    else:
        st.error("❌ Invalid username or password (or account deactivated). Try a one-click demo login below.")


def login_screen():
    left, right = st.columns([1.25, 1])
    with left:
        st.title("🎓 Student Early-Warning System")
        st.markdown("#### Flags at-risk students **early** — while there's still time to help.")
        st.markdown("""
Universities usually notice a struggling student only *after* they fail. This system
combines **OULAD** week-by-week LMS activity with **UCI Dropout** academic/financial
records, predicts risk by **week 8** with **XGBoost**, and then goes further than
"at-risk: yes/no":

- 🔎 **SHAP explanations in plain English** — *"flagged because of 3 late submissions
  and falling quiz scores"*
- 🎛️ **What-if simulator** — *"submit everything on time + quiz +15 → risk 72% → 28%"*
- 🤝 **Intervention workflow** — teacher logs help; the student sees it instantly

**Three portals, one login:** 👑 Admin · 👩‍🏫 Teacher/Advisor · 🎓 Student (my record)
""")
        st.info("👉 **Fastest demo path:** one-click **Teacher** login → *Student Explorer* → pick the top "
                "student → read the SHAP reasons → drag the **what-if sliders** → log an intervention → "
                "log out → one-click **Student** login → *Messages from Teacher*.")
    with right:
        st.markdown("### 🔐 Log in")
        with st.form("login"):
            u = st.text_input("Username", placeholder="e.g. sarah.ahmed")
            p = st.text_input("Password", type="password")
            if st.form_submit_button("Log in", type="primary", width="stretch"):
                do_login(u, p)
        st.markdown("##### ⚡ One-click demo logins")
        c1, c2, c3 = st.columns(3)
        if c1.button("👑\nAdmin", width="stretch"):
            do_login(DEMO_ACCOUNTS["admin"]["username"], DEMO_ACCOUNTS["admin"]["password"])
        if c2.button("👩‍🏫\nTeacher", width="stretch"):
            do_login(DEMO_ACCOUNTS["teacher"]["username"], DEMO_ACCOUNTS["teacher"]["password"])
        if c3.button("🎓\nStudent", width="stretch"):
            do_login(DEMO_ACCOUNTS["student"]["username"], DEMO_ACCOUNTS["student"]["password"])
        with st.expander("📋 All demo accounts"):
            st.markdown("""
| Role | Username | Password | What you see |
|---|---|---|---|
| 👑 Admin | `admin` | `admin123` | Whole institution, users, Model Center, retrain |
| 👩‍🏫 Teacher | `sarah.ahmed` | `teacher123` | My cohort, at-risk queue, SHAP + what-if, interventions |
| 👩‍🏫 Other teachers | `imran.khan`, `ayesha.malik`, … | `teacher123` | Their own course cohort |
| 🎓 Student (demo) | `demo.student` | `student123` | A high-risk student's own record + action plan |
| 🎓 Any student | their ID, e.g. `stu-10001` | `student123` | That student's own record (seeded sample) |
""")
        st.caption("Demo passwords are intentionally simple for presentation. "
                   "Passwords are stored hashed (SHA-256 + salt) in SQLite — see `src/database.py`.")


# ------------------------------------------------------------------- main
def main():
    if not bootstrap():
        return
    user = st.session_state.get("user")
    if not user:
        login_screen()
        return

    role_icons = {"admin": "👑 Admin", "teacher": "👩‍🏫 Teacher / Advisor", "student": "🎓 Student"}
    with st.sidebar:
        st.markdown("## 🎓 Early-Warning System")
        st.markdown(f"**{user.get('full_name') or user['username']}**")
        st.caption(f"@{user['username']} · {role_icons.get(user['role'], user['role'])}")
        if user["role"] == "student" and user.get("linked_student_id"):
            st.caption(f"Student ID: `{user['linked_student_id']}`")
        st.divider()

    if user["role"] == "admin":
        from src.views.admin_view import render
        render(user)
    elif user["role"] == "teacher":
        from src.views.teacher_view import render
        render(user)
    else:
        from src.views.student_view import render
        render(user)

    with st.sidebar:
        st.divider()
        st.caption("Model: XGBoost · Explanations: SHAP · Data: OULAD (#99) + UCI Dropout (#100) schema · "
                   "Predictions use week 1–8 data only.")
        if st.button("🚪 Log out", width="stretch"):
            st.session_state.pop("user", None)
            st.rerun()


if __name__ == "__main__":
    main()
