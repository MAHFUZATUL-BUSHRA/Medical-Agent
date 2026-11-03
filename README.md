# Multi-Tool Medical AI Agent

An AI agent that answers **data questions** from two medical datasets (Heart Disease, Cancer)
via SQL, and **general medical questions** via live web search.

| Tool | Purpose | Implementation |
|---|---|---|
| `HeartDiseaseDBTool` | Stats on `heart_disease.db` | LangChain SQL agent (read-only SQLite) |
| `CancerDBTool` | Stats on `cancer.db` | LangChain SQL agent (read-only SQLite) |
| `MedicalWebSearchTool` | Definitions, symptoms, treatments, cures | Tavily (default) or SerpAPI |

The **main router agent** decides which tool(s) to call. Two interchangeable backends are provided:

- `agents` (default): **OpenAI Agents SDK** `Agent` + `Runner`
- `langchain`: **LangChain `AgentExecutor`** (tool-calling agent)

The DB tools themselves are LangChain SQL agents in both cases, so the project uses
both the OpenAI Agents SDK and LangChain, as required.

## Architecture

```
                         User question
                               │
                    ┌──────────▼──────────┐
                    │   Main Router Agent │   (OpenAI Agents SDK  |  LangChain AgentExecutor)
                    └───┬──────┬───────┬──┘
        statistics/data │      │       │ definitions / symptoms / cures
          ┌─────────────▼┐  ┌──▼────────────┐  ┌────▼──────────────────┐
          │HeartDiseaseDB│  │  CancerDBTool │  │ MedicalWebSearchTool  │
          │    Tool      │  │               │  │ (Tavily / SerpAPI)    │
          └──────┬───────┘  └──────┬────────┘  └───────────────────────┘
        LangChain SQL agent  LangChain SQL agent
                 │                 │
          heart_disease.db      cancer.db      (SQLite, opened read-only)
```

## Project layout

```
main.py                    CLI (interactive or one-shot)
medagent/
  config.py                env/config loading
  schemas.py               table schemas + data dictionaries (single source of truth)
  db_tools.py              HeartDiseaseDBTool / CancerDBTool (LangChain SQL agents)
  web_tool.py              MedicalWebSearchTool
  prompts.py               routing prompt
  agents_sdk.py            main agent: OpenAI Agents SDK
  lc_agent.py              main agent: LangChain AgentExecutor
scripts/build_databases.py CSV -> SQLite converter
tests/                     pytest tests for the DB builder
data/raw/                  put the Kaggle CSVs here
databases/                 generated heart_disease.db, cancer.db
```

## Setup

**1. Install** (Python 3.10+)

```bash
git clone <your-repo-url> && cd medical-multitool-agent
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure keys**

```bash
cp .env.example .env
# edit .env: set LLM_PROVIDER (openai | groq | gemini) and the matching key,
# plus TAVILY_API_KEY (or SERPAPI_API_KEY + SEARCH_PROVIDER=serpapi)
```

No OpenAI credit? Use a free provider: Gemini key at https://aistudio.google.com/apikey
(`LLM_PROVIDER=gemini`) or Groq key at https://console.groq.com/keys (`LLM_PROVIDER=groq`).
Free tiers have per-minute rate limits, so wait a few seconds if you see a 429.

Get a free Tavily key at https://tavily.com.

**3. Get the datasets**

Download both CSVs from Kaggle and place them in `data/raw/`:

- Heart Disease: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset (`heart.csv`)
- Cancer Prediction: https://www.kaggle.com/datasets/rabieelkharoua/cancer-prediction-dataset (`The_Cancer_data_1500_V2.csv`)

(Or `pip install kagglehub` and run the build step with `--download`; this needs Kaggle credentials.)

**4. Build the SQLite databases**

```bash
python -m scripts.build_databases
# explicit paths:  python -m scripts.build_databases --heart-csv path/a.csv --cancer-csv path/b.csv
```

This creates `databases/heart_disease.db` (table `heart_disease_patients`) and
`databases/cancer.db` (table `cancer_patients`) with explicit column types
(`INTEGER` / `REAL`). Column meanings are documented in `medagent/schemas.py`.

**5. Run the agent**

```bash
python main.py                          # interactive chat, OpenAI Agents SDK router
python main.py --backend langchain      # LangChain AgentExecutor router
python main.py -q "What is the average cholesterol of patients with heart disease?"
python main.py -v                       # verbose: show SQL agent reasoning
```

Each answer is prefixed with the tools that were used, e.g. `[tools used: HeartDiseaseDBTool]`.

## Example questions

| Question | Routed to |
|---|---|
| How many patients in the heart dataset are over 60? | `HeartDiseaseDBTool` |
| What's the average cholesterol for males vs females with heart disease? | `HeartDiseaseDBTool` |
| What percentage of smokers have a cancer diagnosis? | `CancerDBTool` |
| Average BMI of patients with high genetic risk? | `CancerDBTool` |
| What are the symptoms of a heart attack? | `MedicalWebSearchTool` |
| What is chemotherapy? | `MedicalWebSearchTool` |
| What is hypertension, and what is the average resting BP in the heart data? | web + `HeartDiseaseDBTool` |

## Tests

```bash
pytest
```

The tests cover the CSV -> SQLite step (types, row counts, case-insensitive headers) and
do not require API keys.

## Design notes

- **Read-only DB access**: SQLite is opened with `mode=ro`, so generated SQL cannot modify data.
- **Lazy initialisation**: tools build their LLM/SQL agent on first use, so importing never needs keys.
- **Data dictionaries in prompts**: coded columns (e.g. `sex`, `cp`, `GeneticRisk`) are explained to the SQL agent so answers use words, not codes.
- **Graceful errors**: tool failures are returned as text to the router agent rather than crashing the chat.
- **Conversation memory**: both backends keep chat history within a session.

## Known limitations

- The Kaggle heart dataset contains duplicate rows (1,025 rows, many repeats). They are loaded as-is; ask the agent to count `DISTINCT` rows if needed.
- LLM-generated SQL can be wrong; use `-v` to inspect queries.
- Outputs are informational and **not medical advice**.

## Troubleshooting

- `<PROVIDER>_API_KEY is not set` -> create `.env` from `.env.example` and match `LLM_PROVIDER` to the key you filled in.
- `heart_disease.db not found` -> run `python -m scripts.build_databases`.
- `MedicalWebSearchTool failed: TAVILY_API_KEY is not set` -> add the key to `.env`.
- `No CSV found` -> check the files are in `data/raw/` or pass `--heart-csv` / `--cancer-csv`.
