# 💠 TalentPulse — HR Attrition Intelligence Platform

**Built by [Pawan Dubey](https://github.com/pawandubey23)** — Full-Stack & AIML Developer

An end-to-end data analytics system that predicts **who** is likely to leave,
**when**, and **why** — combining Python (ETL + ML), MySQL (relational backend),
Power BI (executive dashboard), and Streamlit (live interactive simulator).

Unlike typical single-notebook attrition projects, TalentPulse uses
**simulated longitudinal data** (24 months of history per employee), a
**four-model ML layer** (classification + SHAP explainability, survival
analysis, clustering, NLP sentiment), and **two front ends** (Power BI for
executives, Streamlit for a live what-if simulator) — all reading from the
same MySQL source of truth.

**Repo:** `TalentPulse-HR-Attrition-Intelligence`
**Live demo:** https://talentpulse-hr-attrition-intelligence-pawan-da.streamlit.app/

---

## Architecture

```
                         ┌─────────────────────┐
                         │   generate_data.py   │  synthetic 24-month
                         │  (Faker + NumPy)     │  longitudinal HR data
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   MySQL (3NF)        │  employees, salary_history,
                         │   schema.sql         │  performance_reviews,
                         └──────────┬───────────┘  engagement_surveys, etc.
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   train_models.py     │  XGBoost + SHAP
                         │                       │  KMeans clustering
                         │                       │  Cox survival model
                         │                       │  VADER sentiment
                         └──────────┬───────────┘
                                    │  writes back
                                    ▼
                    attrition_predictions / sentiment_scores
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                              ▼
          ┌─────────────────────┐      ┌─────────────────────────┐
          │   Power BI (Service) │      │   Streamlit (Cloud)      │
          │   Executive dashboard│      │   Live what-if simulator │
          │   DirectQuery→MySQL  │      │   + animated visuals     │
          └─────────────────────┘      └─────────────────────────┘
```

## Why this is different from a typical attrition project

| Typical project | TalentPulse |
|---|---|
| Static Kaggle CSV | Simulated 24-month longitudinal history (salary, reviews, surveys, attendance) |
| One classifier, accuracy score | Classifier + SHAP explainability + Cox survival model + clustering + NLP sentiment |
| CSV import into Power BI | Live DirectQuery connection to MySQL |
| Dashboard only | Dashboard **and** a live Streamlit "what-if" risk simulator with real-time SHAP explanations |
| Manual, one-off | Scheduled automated retraining via GitHub Actions |
| Generic default styling | Custom branded UI — animated burnt-orange (`#C05800`) theme, hover effects, animated gauges |

---

## Project structure

```
talentpulse/
├── sql/schema.sql              # MySQL DDL (3NF) + stored procedures
├── data_generator/
│   ├── generate_data.py        # synthetic longitudinal data generator
│   └── load_to_mysql.py        # loads CSVs into a real MySQL instance
├── ml/train_models.py          # classification, SHAP, clustering, survival, sentiment
├── app/streamlit_app.py        # interactive dashboard + what-if simulator
├── powerbi/POWERBI_GUIDE.md    # DAX measures + page-by-page build guide
├── .github/workflows/retrain.yml  # monthly automated retraining
├── requirements.txt
├── data/talentpulse.db         # trained SQLite DB, committed so the
│                                #   Streamlit Cloud deploy works with zero setup
└── models/*.pkl                # trained XGBoost model + feature frame, committed
```

---

## Local setup (5 minutes)

```bash
pip install -r requirements.txt

# 1. Generate synthetic longitudinal data (writes SQLite + CSVs)
python data_generator/generate_data.py --n_employees 1500 --months 24

# 2. Train all four ML models and write predictions back to the DB
python ml/train_models.py

# 3. Launch the interactive dashboard
python -m streamlit run app/streamlit_app.py
```

This runs fully on SQLite locally (zero setup) — the schema and queries are
written to be MySQL-compatible, so moving to production is a config change,
not a rewrite. The repo ships with `data/talentpulse.db` and `models/*.pkl`
already generated, so you can also just run step 3 directly.

> Tip: if `streamlit` isn't recognized as a command on Windows, use
> `python -m streamlit run app/streamlit_app.py` instead — it works
> regardless of your PATH setup.

---

## Deployment (all free-tier)

### 1. Streamlit — Streamlit Community Cloud
1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), **New app** → select
   your repo → branch `main` → main file path `app/streamlit_app.py` → **Deploy**.
3. No secrets needed for the default deploy — it reads the committed
   `data/talentpulse.db` directly.

**Read-only filesystem note:** Streamlit Community Cloud mounts your repo
as read-only, but SQLite needs to write a small lock/journal file next to
the `.db` even for plain reads. `streamlit_app.py` handles this automatically
by copying the DB into a writable temp directory on first load
(`get_writable_db_path()`) — no action needed, but worth knowing if you ever
see an `sqlite3.OperationalError` on a from-scratch deploy elsewhere.

### 2. MySQL — Railway or Aiven (optional, for the "real" production setup)
1. Create a free MySQL instance on [Railway](https://railway.app) or [Aiven](https://aiven.io).
2. Run `sql/schema.sql` against it (via their web console or `mysql` CLI).
3. Set env vars locally: `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, `MYSQL_DB`.
4. Load data: `python data_generator/load_to_mysql.py`.
5. To point the deployed Streamlit app at MySQL instead of the bundled
   SQLite file, add those credentials under **App settings → Secrets** on
   Streamlit Cloud and swap `sqlite3.connect(...)` for a SQLAlchemy MySQL
   engine (same pattern as `load_to_mysql.py`).

### 3. Power BI — Power BI Service
Follow `powerbi/POWERBI_GUIDE.md` — connect Power BI Desktop to your MySQL
instance with **DirectQuery**, build the 5 pages, then
**File → Publish → Power BI Service** for a shareable link.

### 4. Automation — GitHub Actions
`.github/workflows/retrain.yml` regenerates data, retrains models, and
pushes updated artifacts monthly. Add your MySQL credentials as **GitHub
repo secrets** (`Settings → Secrets and variables → Actions`) if you're
running the MySQL-backed version.

---

## ML layer summary

- **Classification (XGBoost)**: predicts attrition probability per employee; evaluated on a held-out 25% test split (see console output of `train_models.py` for AUC/precision/recall — report these honestly in interviews, and note that with real-world data the model's separation would be less clean than on this synthetic set).
- **Explainability (SHAP)**: top-3 drivers written per employee, and a live per-employee SHAP chart in the Streamlit what-if simulator.
- **Survival analysis (Cox Proportional Hazards, `lifelines`)**: predicts expected remaining tenure in months — most student projects skip this entirely.
- **Clustering (KMeans)**: segments employees into flight-risk personas (e.g. "Burnt-out High Performer", "Disengaged Veteran").
- **NLP sentiment (VADER)**: scores engagement-survey comments and exit-interview text.

## Suggested LinkedIn post sequence
1. Problem framing — "Most attrition dashboards tell you who left. This one tells you who's about to."
2. Architecture reveal — the diagram above, explain the MySQL → ML → BI/Streamlit flow.
3. Deep-dive post — SHAP explainability + Cox survival curves, with a screenshot.
4. Demo video — walk through the Streamlit what-if simulator live.

## Honest limitations to mention (shows maturity in interviews)
- Data is synthetically generated, not from a real company — the model's near-perfect test AUC reflects synthetic signal strength, not real-world performance.
- Survival model assumes proportional hazards, a simplifying assumption worth naming if asked.
- Sentiment analysis uses VADER (rule-based) rather than a fine-tuned transformer, a deliberate scope/cost tradeoff worth being able to explain.

## Suggested GitHub topics
`python` `machine-learning` `xgboost` `shap` `streamlit` `power-bi` `mysql`
`hr-analytics` `data-analytics` `survival-analysis` `nlp` `data-science`