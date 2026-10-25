# Lead IQ Scoring System

Lead IQ is a machine-learning pipeline for prioritizing inbound boat-shopping
leads. It uses an XGBoost binary classifier to estimate conversion intent,
maps the model output to a calibrated score from 1 to 100, and assigns each
lead to a `Cold`, `Warm`, or `Hot` tier.

The repository contains the exploratory analysis, Snowflake feature and target
construction, model training and evaluation, online feature retrieval, and a
small FastAPI service for single-lead scoring.

> **Internal project:** This repository contains company-specific database
> names, feature definitions, and business logic. Confirm that you are
> authorized to publish it and use a private repository unless the project
> owner has explicitly approved public release. Never commit `.env` files,
> credentials, customer data, model artifacts, or notebook outputs containing
> sensitive records.

## How it works

1. Lead, listing, and historical behavior data are transformed into model
   features in Snowflake.
2. Feature views are registered in the Snowflake Feature Store for online
   retrieval.
3. An XGBoost model is trained and evaluated on time-aware lead outcomes.
4. The fitted model and a frozen reference-score distribution are stored as
   versioned artifacts.
5. At inference time, features are fetched for one `LEAD_ID`, scored, and
   converted to a 1-100 percentile-based score and tier.
6. The FastAPI wrapper exposes health and prediction endpoints for local or
   service-based integration.

The inference feature vector currently contains 17 fields covering prior lead
history, source channel, listing context, contact details, and message-intent
signals. Tier boundaries are defined in `model_inference.py` and must remain in
sync with the evaluation code.

## Repository layout

```text
.
├── app.py                         # FastAPI service
├── model_inference.py             # Online feature lookup and scoring
├── model_eval.py                  # Lift, AUC, calibration, and plots
├── snowflake_session.py           # Local Snowflake session factory
├── feature_table_and_target.sql   # Feature/target table construction
├── model_train_deploy.ipynb       # Training and model artifact workflow
├── training.ipynb                 # Earlier training workflow
├── fs_code_draft.ipynb            # Feature Store development notebook
├── feature ablation test.ipynb    # Feature ablation analysis
├── feature heatmap.ipynb          # Feature relationship analysis
├── eda/                           # Lead, listing, outcome, and market EDA
├── tests/                         # SQL checks and test scaffolding
├── .github/workflows/             # CI and deployment workflow drafts
├── .env.example                   # Environment variable template
└── requirements.txt               # Python dependencies
```

## Prerequisites

- Python 3.11 or 3.12
- Access to the required Snowflake account, database, warehouse, and schema
- A Snowflake role with permission to read the registered feature views and
  model stage
- A valid programmatic access token (PAT) supplied through environment
  variables

The code currently references the following internal Snowflake resources:

- Database: `AWS_DP_PROD`
- Feature Store: `BOATS_FEATURE_STORE`
- Feature Group: `LEADIQ_FG`, version `V1`
- Model Stage: `AWS_DP_PROD.ML_OPS.LEADIQ_MODEL_STAGE`

These names may need to be parameterized before the project is used in another
environment.

## Local setup

Clone the repository and create an isolated environment:

```bash
git clone <repository-url>
cd bg-ml-leadiq
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current application also imports `fastapi`, `uvicorn`, `python-dotenv`,
`joblib`, and `matplotlib`. Until those packages are added to the locked
dependency manifest, install them explicitly:

```bash
python -m pip install fastapi uvicorn python-dotenv joblib matplotlib
```

Copy the environment template and fill in your own values:

```bash
cp .env.example .env
```

```dotenv
SNOWFLAKE_ACCOUNT=your_account_identifier
SNOWFLAKE_USER=your_username
SNOWFLAKE_PAT=your_programmatic_access_token
SNOWFLAKE_ROLE=DATA_SCIENTIST
SNOWFLAKE_WAREHOUSE=your_warehouse
SNOWFLAKE_DATABASE=AWS_DP_PROD
SNOWFLAKE_SCHEMA=ML_OPS
```

Do not paste a real token into source code, a notebook, a terminal transcript,
an issue, or a pull request. The `.gitignore` excludes `.env`, but credentials
should still be rotated immediately if they are exposed.

## Verify the Snowflake connection

After configuring `.env`, run:

```bash
python test_local_session.py
```

The script opens a Snowflake session, prints the active user, role, warehouse,
database, and schema, and then closes the session. It does not train or score a
model.

## Training and evaluation

The primary workflow is `model_train_deploy.ipynb`. It trains the classifier,
evaluates ranking quality, creates the frozen reference-score distribution,
and prepares versioned artifacts for the Snowflake stage and model registry.

Evaluation helpers in `model_eval.py` include:

- lift at 1%, 5%, 10%, and 20%;
- ROC-AUC and PR-AUC;
- cumulative-gains, ROC, precision-recall, and score-distribution plots;
- decile lift and segment-level lift tables;
- Hot-tier precision and recall; and
- conversion of raw probabilities to the 1-100 score and tier.

Run notebooks only in an authorized environment with access to the referenced
Snowflake objects. Review every SQL cell before execution, particularly cells
that create or replace tables, upload artifacts, alter users, or deploy models.

## Run the scoring API

The API expects valid Snowflake credentials and access to online Feature Store
data. Start it from the repository root:

```bash
uvicorn app:app --host 127.0.0.1 --port 8000
```

Check service health:

```bash
curl http://127.0.0.1:8000/health
```

Score a lead:

```bash
curl http://127.0.0.1:8000/predict/123456
```

Replace `123456` with an authorized positive integer `LEAD_ID`. The response is
expected to include the raw model score, calibrated 1-100 score, tier, model
version, and timing metadata.

## Testing and quality checks

The repository currently includes SQL validation queries and Python test
scaffolding, but the automated test suite is not yet complete.

```bash
ruff check .
pytest tests/unit -v
```

The existing GitHub Actions CI file still references a `src/` directory that
is not present, so it should be corrected before CI is treated as authoritative.
The feature-refresh and model-deploy workflows are placeholders and currently
only print the intended actions.

## Current limitations

- The code depends on internal Snowflake resources and has not been packaged
  as a standalone application.
- Online scoring requires a deployed model artifact and populated online
  Feature Store.
- The FastAPI and MLflow/Snowflake execution paths should be integration-tested
  in the target environment before production use.
- The dependency file is not yet a fully reproducible lockfile.
- Unit and integration test coverage is incomplete.
- Database and schema names are currently hard-coded in several files.
- Latency targets should be verified with per-stage production telemetry.

## Safe GitHub publishing checklist

Before pushing this project to GitHub:

- obtain authorization from the code and data owner;
- create the repository as private by default;
- remove `__pycache__/`, `*.pyc`, generated plots, local artifacts, and outputs;
- clear notebook outputs and inspect notebook metadata;
- scan the full working tree and Git history for secrets;
- confirm `.snowflake/settings.json` contains no sensitive environment details;
- review SQL for personal identifiers, privileged commands, and internal-only
  object names;
- add an approved license or mark the repository as proprietary; and
- enable branch protection and secret scanning where available.

## Status

This repository is a working project snapshot. Core feature, training,
evaluation, and inference logic is present, while CI, automated tests, external
configuration, and production deployment still require hardening.

## License

No open-source license is currently provided. Unless the project owner states
otherwise, treat all contents as proprietary and confidential.
