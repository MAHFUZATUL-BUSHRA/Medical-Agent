import sqlite3

import pandas as pd

from medagent.schemas import CANCER, HEART
from scripts.build_databases import build_database


def _fake_csv(spec, path, n=5):
    df = pd.DataFrame({name: [i % 2 for i in range(n)] for name in spec.column_names})
    df.to_csv(path, index=False)


def test_heart_db_built_with_correct_types(tmp_path):
    csv = tmp_path / "heart.csv"
    _fake_csv(HEART, csv)
    db = build_database(HEART, csv, tmp_path)
    assert db.name == "heart_disease.db"
    with sqlite3.connect(db) as conn:
        info = {r[1]: r[2] for r in conn.execute(f'PRAGMA table_info("{HEART.table}")')}
        assert info["age"] == "INTEGER" and info["oldpeak"] == "REAL"
        assert conn.execute(f'SELECT COUNT(*) FROM "{HEART.table}"').fetchone()[0] == 5


def test_cancer_db_handles_case_insensitive_headers(tmp_path):
    csv = tmp_path / "The_Cancer_data.csv"
    df = pd.DataFrame({n.lower(): [1, 0, 1] for n in CANCER.column_names})
    df.to_csv(csv, index=False)
    db = build_database(CANCER, csv, tmp_path)
    assert db.name == "cancer.db"
    with sqlite3.connect(db) as conn:
        assert conn.execute(f'SELECT SUM(Diagnosis) FROM "{CANCER.table}"').fetchone()[0] == 2
