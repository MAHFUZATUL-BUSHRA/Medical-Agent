"""Single source of truth for the two datasets.

Used by:
  * scripts/build_databases.py  -> to create typed SQLite tables
  * medagent/db_tools.py        -> to give the SQL agents a data dictionary
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    db_file: str
    table: str
    csv_globs: Tuple[str, ...]
    kaggle_slug: str
    description: str
    # (column_name, sqlite_type, human description)
    columns: Tuple[Tuple[str, str, str], ...]

    @property
    def column_names(self) -> list[str]:
        return [c[0] for c in self.columns]

    @property
    def data_dictionary(self) -> str:
        return "\n".join(f"- {n} ({t}): {d}" for n, t, d in self.columns)


HEART = DatasetSpec(
    key="heart",
    db_file="heart_disease.db",
    table="heart_disease_patients",
    csv_globs=("heart*.csv",),
    kaggle_slug="johnsmith88/heart-disease-dataset",
    description="Clinical measurements for patients, with a heart disease label.",
    columns=(
        ("age", "INTEGER", "Age in years"),
        ("sex", "INTEGER", "1 = male, 0 = female"),
        ("cp", "INTEGER", "Chest pain type: 0 typical angina, 1 atypical angina, 2 non-anginal pain, 3 asymptomatic"),
        ("trestbps", "INTEGER", "Resting blood pressure (mm Hg)"),
        ("chol", "INTEGER", "Serum cholesterol (mg/dl)"),
        ("fbs", "INTEGER", "Fasting blood sugar > 120 mg/dl: 1 = true, 0 = false"),
        ("restecg", "INTEGER", "Resting ECG result (0 normal, 1 ST-T abnormality, 2 left ventricular hypertrophy)"),
        ("thalach", "INTEGER", "Maximum heart rate achieved"),
        ("exang", "INTEGER", "Exercise-induced angina: 1 = yes, 0 = no"),
        ("oldpeak", "REAL", "ST depression induced by exercise relative to rest"),
        ("slope", "INTEGER", "Slope of the peak exercise ST segment (0-2)"),
        ("ca", "INTEGER", "Number of major vessels colored by fluoroscopy (0-4)"),
        ("thal", "INTEGER", "Thalassemia test result code (0-3)"),
        ("target", "INTEGER", "Heart disease label: 1 = disease present, 0 = no disease"),
    ),
)

CANCER = DatasetSpec(
    key="cancer",
    db_file="cancer.db",
    table="cancer_patients",
    csv_globs=("*ancer*.csv",),
    kaggle_slug="rabieelkharoua/cancer-prediction-dataset",
    description="Lifestyle / genetic risk factors for patients, with a cancer diagnosis label.",
    columns=(
        ("Age", "INTEGER", "Age in years (20-80)"),
        ("Gender", "INTEGER", "0 = male, 1 = female"),
        ("BMI", "REAL", "Body Mass Index (15-40)"),
        ("Smoking", "INTEGER", "0 = non-smoker, 1 = smoker"),
        ("GeneticRisk", "INTEGER", "Genetic risk level: 0 low, 1 medium, 2 high"),
        ("PhysicalActivity", "REAL", "Hours of physical activity per week (0-10)"),
        ("AlcoholIntake", "REAL", "Alcohol units per week (0-5)"),
        ("CancerHistory", "INTEGER", "Personal history of cancer: 0 = no, 1 = yes"),
        ("Diagnosis", "INTEGER", "Cancer diagnosis: 1 = cancer, 0 = no cancer"),
    ),
)

SPECS = {HEART.key: HEART, CANCER.key: CANCER}
