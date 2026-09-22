
Author: Goutami Chenumalla
# Zepto Data & AI Platform

One repository, three connected modules built as part of the Zepto analytics-guild capstone:

| Module | Folder | Marks | What it does |
|---|---|---|---|
| Data Pipeline | `/data_pipeline` | 25 | Scrapes books.toscrape.com → cleans → converts currency → loads into normalized SQLite → queries with SQL + pandas |
| Analytics Pipeline | `/analytics` | 50 | Profiles, cleans, visualizes and models the Titanic dataset end to end (classification + regression) |
| Support Assistant | `/support_assistant` | 25 | RAG-based FastAPI service answering Zepto policy questions, grounded in 8 local docs via ChromaDB + LangGraph |

Total: 100 marks.

## 1. Setup

### 1.1 Clone the repo
```bash
git clone <your-repo-url>
cd zepto-project
```

### 1.2 Create and activate a virtual environment
```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 1.3 Install dependencies
One consolidated `requirements.txt` at the repo root covers all three modules (chosen over per-module files to keep setup to a single command):
```bash
pip install -r requirements.txt
```

### 1.4 Register the venv as a Jupyter/VS Code kernel
All module code is written as `.py` files using VS Code's `# %%` cell markers (Jupyter-style interactive cells), not `.ipynb` notebooks. To run cells interactively in VS Code:
1. Install the **Python** and **Jupyter** extensions in VS Code.
2. Open any `.py` file in this repo (e.g. `data_pipeline/scrape_and_load.py`).
3. In the bottom-right of VS Code, click the interpreter selector and choose `.venv`.
4. Click **Run Cell** above any `# %%` block (or `Shift+Enter`) — VS Code opens an Interactive Window and runs that cell, keeping variables alive between cells, exactly like a notebook.
5. To run the whole file top-to-bottom non-interactively: `python data_pipeline/scrape_and_load.py`.

## 2. How to run each module

### 2.1 Data Pipeline (`/data_pipeline`)
```bash
cd data_pipeline
python scrape_and_load.py      # scrapes, cleans, builds books.db
python queries.py              # runs the 5+ SQL queries + pandas equivalents
```
Currency conversion uses a fixed baseline rate: **1 GBP = 105.50 INR** (project-defined constant, no live lookup).

### 2.2 Analytics (`/analytics`)
```bash
cd analytics
python 01_eda.py                # loads titanic once, profiles, cleans, saves titanic.csv, EDA charts
python 02_modeling.py           # reads titanic.csv, trains/evaluates 3 classifiers + regression, saves pipeline.joblib
```

### 2.3 Support Assistant (`/support_assistant`)
```bash
cd support_assistant
python ingest.py                       # embeds docs/*.txt into ChromaDB
uvicorn main:app --reload --port 7860  # start the API (MOCK_LLM=1 by default, no key needed)
```
Test it:
```bash
curl -X POST http://localhost:7860/ask -H "Content-Type: application/json" -d '{"query": "What is your return policy?"}'
```
Docker:
```bash
docker build -t zepto-assistant .
docker run -p 7860:7860 zepto-assistant
```

## 3. Design decisions

_(Filled in as each module is built — see each module's section below / module-level README.)_

### 3.1 Data Pipeline
- TBD

### 3.2 Analytics
- TBD

### 3.3 Support Assistant
- TBD

## 4. Git workflow

Development happened on feature branches (e.g. `feature/data-pipeline`, `feature/analytics`, `feature/support-assistant`), each committed to at least twice, then merged into `main`. Visible via:
```bash
git log --graph --all --oneline
```
