"""Database tools: a LangChain SQL agent per SQLite DB.

Each DatabaseTool wraps a LangChain SQL agent (AgentExecutor) that
  1. inspects the schema, 2. writes SQL, 3. runs it (read-only), 4. answers in plain English.

The same objects are exposed to the OpenAI Agents SDK and to the LangChain
AgentExecutor main agent (see agents_sdk.py / lc_agent.py).
"""
from __future__ import annotations

from pathlib import Path

from langchain_community.agent_toolkits import create_sql_agent
from langchain_community.utilities import SQLDatabase

from . import config
from .llm import make_chat_model
from .schemas import CANCER, HEART, DatasetSpec

# NOTE: {dialect} and {top_k} are filled in by create_sql_agent.
_PREFIX = """You are an expert data analyst answering questions about the {title} dataset
stored in a SQLite database (dialect: {{dialect}}).

Dataset: {description}
Main table: {table}

Data dictionary:
{dictionary}

Rules:
- Always inspect the schema first, then write a correct SQL query.
- Only run read-only SELECT queries. Never modify data. Limit raw row output to {{top_k}} rows
  unless the user asks for more.
- Use aggregate functions (COUNT, AVG, MIN, MAX, GROUP BY) for statistics.
- Columns coded as 0/1/2... are categorical: use the data dictionary to translate codes
  into words in your final answer (e.g. sex=1 means male).
- If a query errors, rewrite it and try again.
- Final answer: a short natural-language answer that includes the actual numbers
  (and percentages when helpful). Mention the number of rows that supports the answer.
- This is a research dataset, so do not present results as medical advice."""


def _read_only_db(path: Path) -> SQLDatabase:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m scripts.build_databases` first."
        )
    uri = f"sqlite:///file:{path.as_posix()}?mode=ro&uri=true"
    return SQLDatabase.from_uri(uri, sample_rows_in_table_info=3)


class DatabaseTool:
    """Natural-language -> SQL -> natural-language tool for one SQLite DB."""

    def __init__(self, name: str, title: str, spec: DatasetSpec, db_path: Path | None = None):
        self.name = name
        self.spec = spec
        self.title = title
        self.db_path = db_path or (config.DB_DIR / spec.db_file)
        self._executor = None  # built lazily so importing never needs an API key / DB

    def _build(self):
        llm = make_chat_model()
        prefix = _PREFIX.format(
            title=self.title,
            description=self.spec.description,
            table=self.spec.table,
            dictionary=self.spec.data_dictionary,
        )
        return create_sql_agent(
            llm=llm,
            db=_read_only_db(self.db_path),
            agent_type="openai-tools",
            prefix=prefix,
            top_k=10,
            max_iterations=8,
            verbose=config.VERBOSE,
        )

    def run(self, question: str) -> str:
        """Answer a data question and return a natural-language result."""
        try:
            if self._executor is None:
                self._executor = self._build()
            return self._executor.invoke({"input": question})["output"]
        except Exception as exc:  # surface errors to the router agent instead of crashing
            return f"{self.name} failed: {exc}"


def make_heart_tool() -> DatabaseTool:
    return DatabaseTool("HeartDiseaseDBTool", "Heart Disease", HEART)


def make_cancer_tool() -> DatabaseTool:
    return DatabaseTool("CancerDBTool", "Cancer Prediction", CANCER)


HEART_TOOL_DESCRIPTION = (
    "Query the Heart Disease patient dataset (SQLite). Use for statistics, counts, averages, "
    "distributions, comparisons or filtering on columns like age, sex, chest pain (cp), "
    "cholesterol, blood pressure, max heart rate, and heart disease target. "
    "Input: the user's data question in plain English."
)

CANCER_TOOL_DESCRIPTION = (
    "Query the Cancer Prediction patient dataset (SQLite). Use for statistics, counts, averages, "
    "distributions, comparisons or filtering on Age, Gender, BMI, Smoking, GeneticRisk, "
    "PhysicalActivity, AlcoholIntake, CancerHistory and cancer Diagnosis. "
    "Input: the user's data question in plain English."
)
