"""
SQLite database: users (admin / teacher / student) + interventions log.

Students' academic records live in data/students_scored.csv (the ML side);
this DB is the *application* side: who can log in, and what help was given.
Separating them mirrors a real deployment (SIS/LMS extracts -> ML scoring
-> advisor app DB).

Demo logins (also on the login screen):
  admin         / admin123
  sarah.ahmed   / teacher123     (all teachers: first.last / teacher123)
  demo.student  / student123     (linked to a HIGH-risk student so the
                                  Student portal shows the interesting case)
  + any student can log in with their ID, e.g. stu-10001 / student123
    (accounts are auto-seeded for the top-200 highest-risk students and
    a random sample, covering demo needs without 4,424 rows of clutter;
    Admin can create more from the Admin panel.)
"""
import sys, hashlib, sqlite3
from pathlib import Path
from datetime import date, timedelta
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import DB_PATH, DATA_DIR, SCORED_CSV, TEACHERS, DEMO_ACCOUNTS


def _hash(pw: str) -> str:
    return hashlib.sha256(("ews-salt::" + pw).encode()).hexdigest()


def conn():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def init_db():
    c = conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('admin','teacher','student')),
        full_name TEXT, email TEXT,
        linked_student_id TEXT,   -- student role: which record is "mine"
        linked_teacher_id TEXT,   -- teacher role: which cohort is "mine"
        is_active INTEGER DEFAULT 1,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS interventions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        student_id TEXT NOT NULL,
        student_name TEXT,
        teacher_name TEXT,
        kind TEXT,               -- Meeting / Email / Call / Tutoring / Finance referral ...
        note TEXT,
        status TEXT DEFAULT 'Open',   -- Open / In Progress / Resolved
        risk_at_time REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );
    """)
    if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        _seed(c)
    c.commit(); c.close()


def _seed(c):
    add = ("INSERT OR IGNORE INTO users (username,password_hash,role,full_name,email,"
           "linked_student_id,linked_teacher_id) VALUES (?,?,?,?,?,?,?)")
    c.execute(add, ("admin", _hash("admin123"), "admin", "System Administrator",
                    "admin@university.edu", None, None))
    for t in TEACHERS:
        c.execute(add, (t["username"], _hash("teacher123"), "teacher", t["name"],
                        f"{t['username']}@university.edu", None, t["teacher_id"]))

    demo_student_id = None
    if SCORED_CSV.exists():
        df = pd.read_csv(SCORED_CSV)
        # demo.student := high-risk but RECOVERABLE (risk nearest 75% and at
        # least 2 assignments submitted). The extreme ~100% cases make a poor
        # demo: no realistic what-if plan moves them, and a "Success Score 0"
        # is the wrong first impression for the Student portal.
        cand = df[(df.risk_level == "High") & (df.assignments_submitted >= 2)]
        if len(cand):
            demo = cand.iloc[(cand.risk_score - 0.75).abs().argsort().iloc[0]]
            demo_student_id = demo.student_id
            c.execute(add, ("demo.student", _hash("student123"), "student", demo["name"],
                            demo.email, demo.student_id, None))
        # student logins: top-150 by risk + every 25th student (varied demos)
        sample = pd.concat([df.head(150), df.iloc[::25]]).drop_duplicates("student_id")
        for _, s in sample.iterrows():
            c.execute(add, (s.student_id.lower(), _hash("student123"), "student",
                            s["name"], s.email, s.student_id, None))
        _seed_interventions(c, df, demo_student_id)
    print("DB seeded.")


def _seed_interventions(c, df, demo_student_id):
    """A few believable past interventions so no screen starts empty."""
    import random
    rng = random.Random(7)
    kinds_notes = [
        ("Email", "Sent check-in email about low LMS activity this week."),
        ("Meeting", "1-to-1 advising meeting; agreed a 2-week catch-up plan."),
        ("Tutoring referral", "Referred to peer tutoring for quiz preparation."),
        ("Call", "Phone call with student about two missing assignments."),
        ("Finance referral", "Referred to student finance for installment plan."),
        ("Email", "Shared deadline planner and office-hours timings."),
    ]
    pool = df[df.risk_level.isin(["High", "Medium"])].head(40)
    rows = []
    for _, s in pool.iterrows():
        for _ in range(rng.choice([0, 1, 1, 2])):
            kind, note = rng.choice(kinds_notes)
            d = date.today() - timedelta(days=rng.randint(2, 40))
            rows.append((s.student_id, s["name"], s.teacher_name, kind, note,
                         rng.choice(["Open", "In Progress", "Resolved"]),
                         float(s.risk_score), d.isoformat()))
    if demo_student_id:  # guarantee the demo student sees teacher messages
        d = df[df.student_id == demo_student_id].iloc[0]
        rows.append((demo_student_id, d["name"], d.teacher_name, "Meeting",
                     "We noticed your quiz scores and LMS activity have dipped. "
                     "Please book a 15-minute meeting this week — we will make a catch-up plan together.",
                     "Open", float(d.risk_score), (date.today() - timedelta(days=3)).isoformat()))
        rows.append((demo_student_id, d["name"], d.teacher_name, "Email",
                     "Reminder: Assignment 4 deadline is Friday. Office hours Tue/Thu 2-4pm if you need help.",
                     "Open", float(d.risk_score), (date.today() - timedelta(days=1)).isoformat()))
    c.executemany(
        "INSERT INTO interventions (student_id,student_name,teacher_name,kind,note,status,risk_at_time,created_at)"
        " VALUES (?,?,?,?,?,?,?,?)", rows)


# ------------------------------------------------------------------ auth
def authenticate(username: str, password: str):
    init_db()
    c = conn()
    row = c.execute("SELECT * FROM users WHERE username=? AND is_active=1",
                    (username.strip().lower(),)).fetchone()
    c.close()
    if row and row["password_hash"] == _hash(password):
        return dict(row)
    return None


def list_users():
    init_db(); c = conn()
    rows = [dict(r) for r in c.execute("SELECT id,username,role,full_name,email,linked_student_id,linked_teacher_id,is_active,created_at FROM users ORDER BY role, username")]
    c.close(); return pd.DataFrame(rows)


def create_user(username, password, role, full_name, email="", linked_student_id=None, linked_teacher_id=None):
    c = conn()
    try:
        c.execute("INSERT INTO users (username,password_hash,role,full_name,email,linked_student_id,linked_teacher_id) VALUES (?,?,?,?,?,?,?)",
                  (username.strip().lower(), _hash(password), role, full_name, email, linked_student_id, linked_teacher_id))
        c.commit(); ok = True
    except sqlite3.IntegrityError:
        ok = False
    c.close(); return ok


def set_user_active(user_id, active: bool):
    c = conn(); c.execute("UPDATE users SET is_active=? WHERE id=?", (int(active), user_id)); c.commit(); c.close()


def delete_user(user_id):
    c = conn(); c.execute("DELETE FROM users WHERE id=?", (user_id,)); c.commit(); c.close()


# ---------------------------------------------------------- interventions
def add_intervention(student_id, student_name, teacher_name, kind, note, risk_at_time=None, status="Open"):
    c = conn()
    c.execute("INSERT INTO interventions (student_id,student_name,teacher_name,kind,note,status,risk_at_time) VALUES (?,?,?,?,?,?,?)",
              (student_id, student_name, teacher_name, kind, note, status, risk_at_time))
    c.commit(); c.close()


def list_interventions(student_id=None, teacher_name=None):
    init_db(); c = conn(); q, args = "SELECT * FROM interventions", []
    where = []
    if student_id: where.append("student_id=?"); args.append(student_id)
    if teacher_name: where.append("teacher_name=?"); args.append(teacher_name)
    if where: q += " WHERE " + " AND ".join(where)
    rows = [dict(r) for r in c.execute(q + " ORDER BY created_at DESC, id DESC", args)]
    c.close(); return pd.DataFrame(rows)


def update_intervention_status(intervention_id, status):
    c = conn(); c.execute("UPDATE interventions SET status=? WHERE id=?", (status, intervention_id)); c.commit(); c.close()


if __name__ == "__main__":
    init_db()
    print(list_users().groupby("role").size().to_string())
    print("auth test:", bool(authenticate("admin", "admin123")),
          bool(authenticate("sarah.ahmed", "teacher123")),
          bool(authenticate("demo.student", "student123")))
