"""Smoke/unit tests — run:  python -m unittest discover -s tests -v
(no pytest needed; standard-library unittest only)
Requires the pipeline to have been run first (generate_data + train_model + database).
"""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from config import FEATURES, SCORED_CSV, risk_level
from src.predict import load_students, predict_risk, explain_student, what_if
from src import database as db


class TestRiskBands(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(risk_level(0.10), "Low")
        self.assertEqual(risk_level(0.39), "Low")
        self.assertEqual(risk_level(0.40), "Medium")
        self.assertEqual(risk_level(0.69), "Medium")
        self.assertEqual(risk_level(0.70), "High")
        self.assertEqual(risk_level(0.99), "High")


class TestDataAndModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_students()

    def test_dataset_shape_and_features(self):
        self.assertGreater(len(self.df), 4000)
        for f in FEATURES:
            self.assertIn(f, self.df.columns)
            self.assertFalse(self.df[f].isna().any(), f"NaNs in {f}")

    def test_no_demographic_features_in_model(self):
        # fairness-by-design: age/gender are in the CSV but must NOT be model inputs
        self.assertNotIn("age", FEATURES)
        self.assertNotIn("gender_male", FEATURES)

    def test_scores_are_probabilities(self):
        self.assertTrue(self.df.risk_score.between(0, 1).all())
        self.assertTrue(set(self.df.risk_level.unique()) <= {"High", "Medium", "Low"})

    def test_explain_returns_reasons(self):
        row = self.df.iloc[10]
        prob, sdf, up, down = explain_student(row)
        self.assertTrue(0 <= prob <= 1)
        self.assertEqual(len(sdf), len(FEATURES))
        self.assertTrue(all(r["shap"] > 0 for r in up))
        self.assertTrue(all(r["shap"] <= 0 for r in down))

    def test_what_if_improving_behaviour_lowers_risk(self):
        """Core product promise: fixing submissions/engagement must move risk DOWN
        for a typical high-risk-but-recoverable student."""
        cand = self.df[(self.df.risk_level == "High") & (self.df.assignments_submitted >= 2)]
        row = cand.iloc[(cand.risk_score - 0.75).abs().argsort().iloc[0]]
        before = predict_risk(row)
        after = what_if(row, {"missing_submissions": 0, "late_submissions": 0,
                              "avg_quiz_score": min(100, row.avg_quiz_score + 20),
                              "vle_total_clicks": int(row.vle_total_clicks + 300)})
        self.assertLess(after, before)

    def test_what_if_keeps_submission_arithmetic_consistent(self):
        row = self.df.iloc[0]
        p = what_if(row, {"missing_submissions": 0})  # implies submitted=5, late<=5
        self.assertTrue(0 <= p <= 1)


class TestAuthRoles(unittest.TestCase):
    def test_three_roles_login(self):
        self.assertEqual(db.authenticate("admin", "admin123")["role"], "admin")
        self.assertEqual(db.authenticate("sarah.ahmed", "teacher123")["role"], "teacher")
        stu = db.authenticate("demo.student", "student123")
        self.assertEqual(stu["role"], "student")
        self.assertTrue(str(stu["linked_student_id"]).startswith("STU-"))

    def test_wrong_password_rejected(self):
        self.assertIsNone(db.authenticate("admin", "wrong"))
        self.assertIsNone(db.authenticate("nobody", "admin123"))

    def test_demo_student_has_teacher_messages(self):
        stu = db.authenticate("demo.student", "student123")
        msgs = db.list_interventions(student_id=stu["linked_student_id"])
        self.assertGreaterEqual(len(msgs), 1)


if __name__ == "__main__":
    unittest.main()
