"""Convert the two Kaggle CSVs into typed SQLite databases.

Usage:
    python -m scripts.build_databases                    # auto-detects CSVs in data/raw/
    python -m scripts.build_databases --heart-csv path/heart.csv --cancer-csv path/cancer.csv
    python -m scripts.build_databases --download         # fetch via kagglehub (needs Kaggle creds)
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from medagent.config import DB_DIR, RAW_DIR
from medagent.schemas import CANCER, HEART, DatasetSpec


def find_csv(spec: DatasetSpec, explicit: Path | None) -> Path:
    if explicit:
        if not explicit.exists():
            raise FileNotFoundError(f"CSV not found: {explicit}")
        return explicit
    for pattern in spec.csv_globs:
        matches = sorted(RAW_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No CSV for '{spec.key}' found in {RAW_DIR}. Download it from "
        f"https://www.kaggle.com/datasets/{spec.kaggle_slug} or pass --{spec.key}-csv."
    )


def download_from_kaggle(spec: DatasetSpec) -> None:
    try:
        import kagglehub  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise SystemExit("Install kagglehub first: pip install kagglehub") from e
    folder = Path(kagglehub.dataset_download(spec.kaggle_slug))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for csv in folder.glob("*.csv"):
        shutil.copy(csv, RAW_DIR / csv.name)
        print(f"  downloaded {csv.name}")


def load_dataframe(spec: DatasetSpec, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Case-insensitive match of file headers to the expected schema
    lookup = {c.strip().lower(): c for c in df.columns}
    missing = [c for c in spec.column_names if c.lower() not in lookup]
    if missing:
        raise ValueError(f"{csv_path.name} is missing expected columns: {missing}")
    df = df[[lookup[c.lower()] for c in spec.column_names]]
    df.columns = spec.column_names
    for name, sql_type, _ in spec.columns:
        df[name] = pd.to_numeric(df[name], errors="coerce")
    return df


def build_database(spec: DatasetSpec, csv_path: Path, db_dir: Path = DB_DIR) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / spec.db_file
    df = load_dataframe(spec, csv_path)

    types = [t for _, t, _ in spec.columns]
    rows = [
        tuple(
            None if pd.isna(v) else (int(v) if t == "INTEGER" else float(v))
            for v, t in zip(row, types)
        )
        for row in df.itertuples(index=False, name=None)
    ]

    ddl_cols = ",\n    ".join(f'"{n}" {t}' for n, t, _ in spec.columns)
    placeholders = ", ".join("?" for _ in spec.columns)

    if db_path.exists():
        db_path.unlink()
    with sqlite3.connect(db_path) as conn:
        conn.execute(f'CREATE TABLE "{spec.table}" (\n    {ddl_cols}\n)')
        conn.executemany(f'INSERT INTO "{spec.table}" VALUES ({placeholders})', rows)
        conn.commit()
    print(f"  {db_path.name}: table '{spec.table}' <- {len(rows)} rows from {csv_path.name}")
    return db_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--heart-csv", type=Path)
    p.add_argument("--cancer-csv", type=Path)
    p.add_argument("--download", action="store_true", help="Download CSVs with kagglehub first")
    args = p.parse_args(argv)

    if args.download:
        print("Downloading datasets from Kaggle...")
        for spec in (HEART, CANCER):
            download_from_kaggle(spec)

    print("Building SQLite databases...")
    try:
        build_database(HEART, find_csv(HEART, args.heart_csv))
        build_database(CANCER, find_csv(CANCER, args.cancer_csv))
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
