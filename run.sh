#!/usr/bin/env bash
# Student Early-Warning System - one-command setup + launch
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then python3 -m venv .venv; fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q -r requirements.txt

[ -f data/students_raw.csv ]  || python -m src.generate_data
[ -f models/model.pkl ]       || python -m src.train_model
python -m src.database

echo ""
echo "🎓 Early-Warning System starting at http://localhost:8501"
echo "   Demo logins:  admin/admin123 · sarah.ahmed/teacher123 · demo.student/student123"
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
