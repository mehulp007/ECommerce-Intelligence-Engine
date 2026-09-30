# E-Commerce Intelligence Engine

A local customer analytics application built with FastAPI and Streamlit. It uses historical retail transactions to show customer personas, a 30-day inactivity probability, model contributions, and product recommendations.

## Features

- Four customer personas based on recency, frequency, and monetary value.
- Calibrated XGBoost predictions with SHAP contributions.
- BM25 recommendations that exclude previously purchased products.
- Bearer-key authentication and a responsive dashboard.
- Local MLflow tracking, a SQLite model registry, and versioned inference bundles.

The application uses the [UCI Online Retail dataset](https://archive.ics.uci.edu/dataset/352/online+retail), provided by Daqing Chen under CC BY 4.0. It serves a fixed historical snapshot dated **November 1, 2011**. No cloud service is deployed.

## Run on your PC

These instructions use **Windows PowerShell**. Run each command from the project folder. Docker is not required.

### 1. Install prerequisites and download the project

Install **Python 3.12** and **Git for Windows**. Internet access is needed to download dependencies and the dataset. This repository is public; you can clone or download it without a GitHub account or signing in.

```powershell
git clone https://github.com/mehulp007/ecommerce-intelligence-engine.git
cd ecommerce-intelligence-engine
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt -r requirements-train.txt -r requirements-recommend.txt -r requirements-api.txt -r requirements-ui.txt
```

Alternatively, open the [repository](https://github.com/mehulp007/ecommerce-intelligence-engine), select **Code > Download ZIP**, extract it, and open PowerShell in the extracted folder. Start with the `py -3.12` command. The commands use the virtual environment directly, so you do not need to activate it or change PowerShell execution policies.

### 2. Download the data, train the models, and create a bundle

A fresh checkout does **not** contain raw data, trained models, customer records, or serving bundles. Run these commands in order, waiting for each one to finish:

```powershell
.\.venv\Scripts\python.exe -m ecommerce_intelligence.data.pipeline
.\.venv\Scripts\python.exe -m ecommerce_intelligence.models.train
.\.venv\Scripts\python.exe -m ecommerce_intelligence.recommendations.train
.\.venv\Scripts\python.exe -m ecommerce_intelligence.serving.promote
```

The data pipeline downloads the approximately 24 MB UCI workbook automatically. Training time depends on your PC. Promotion prints the generated bundle version. Keep the generated files on your PC; the API needs them to run. Repeat this step only when rebuilding the data or models.

### 3. Start the API

In the same PowerShell window, run:

```powershell
$bundle = Get-ChildItem artifacts/bundles -Directory | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$env:EIE_BUNDLE_DIR = $bundle.FullName
$env:EIE_API_KEY = Read-Host "Choose a local API key"
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Enter a key of your choice and remember it for the dashboard. Wait for **Application startup complete**. Keep this window running.

### 4. Start the dashboard

Open a **second PowerShell window**, navigate to the same project folder, and run:

```powershell
cd "C:\path\to\ecommerce-intelligence-engine"
$env:EIE_API_URL = "http://127.0.0.1:8000"
$env:EIE_API_KEY = Read-Host "Enter the same API key"
.\.venv\Scripts\python.exe -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501
```

Replace `C:\path\to\ecommerce-intelligence-engine` with your actual folder path. Enter exactly the same key as in the API window.

Open **http://127.0.0.1:8501/** in your browser. Choose a verified sample ID and click **View customer intelligence**. To use a typed ID, leave the sample selector on **Choose a sample**.

Keep both PowerShell windows running while using the application. Press **Ctrl+C** in each window to stop it. On later visits, repeat steps 3 and 4 only.

### Troubleshooting

| Problem | What to do |
|---|---|
| `py` is not recognized | Install Python 3.12 with its Windows launcher, then reopen PowerShell. |
| No bundle found | Complete step 2 before starting the API. |
| API unavailable | Start the API, wait for startup to complete, and check that both windows use port 8000. |
| API key rejected | Restart both services with matching keys. If using `.streamlit/secrets.toml`, its settings take precedence over dashboard environment variables. |
| Customer not found | Choose a verified sample ID. The API accepts only the saved November 2011 cohort. |
| Port already in use | Stop the previous server with Ctrl+C before starting another instance. |
| `/` on port 8000 shows `Not Found` | This is expected: use port 8501 for the dashboard, `/health` for readiness, or `/docs` for API documentation. |

## API routes

| Method | Route | Authentication | Purpose |
|---|---|---|---|
| GET | `/health` | None | Readiness and bundle version |
| GET | `/demo/customers` | Bearer key | Verified sample customer IDs |
| POST | `/predict/churn` | Bearer key | Inactivity probability, persona, and contributions |
| POST | `/recommend` | Bearer key | Up to five product recommendations |

Prediction requests use `{"customer_id":"12347"}`. Recommendation requests use `{"customer_id":"12347","limit":5}`. Unknown IDs return 404; malformed inputs return 422. Swagger documentation is available at **http://127.0.0.1:8000/docs** when the API is running.

Keys are read from environment variables. The dashboard also supports `API_URL` and `API_KEY` in an untracked `.streamlit/secrets.toml` file; see `.streamlit/secrets.toml.example`. API requests are made by the Streamlit Python server, so the bearer key is not placed in browser code or URLs.

## Architecture

```text
UCI transactions -> cleaning -> dated features -> model training -> frozen bundle
                                                                    |
Browser <-> Streamlit dashboard <-> authenticated FastAPI <-----------+
```

| Folder | Contents |
|---|---|
| `src/ecommerce_intelligence/` | Data, features, models, evaluation, recommendations, and serving |
| `api/` | FastAPI routes and request/response schemas |
| `ui/` | Streamlit application, API client, and CSS |
| `tests/` | Data, temporal leakage, model, API, and client checks |
| `reports/` | Aggregate evaluation results and charts |
| `data/` | Generated local transactions, snapshots, and predictions |
| `artifacts/` | Generated local models and inference bundles |
| `docker/` | API Dockerfile; optional Compose configuration is at the project root |

## Measured results

The inactivity model trains on June-August 2011, calibrates on September, and evaluates on October-November. April and May are excluded from supervised training because their 180-day histories are incomplete. The recommender trains on purchases before November 1 and evaluates on November purchases.

| Metric | Result |
|---|---|
| Held-out inactivity ROC AUC | 0.710 |
| Held-out average precision | 0.795 |
| Held-out Brier score | 0.194 |
| Persona silhouette score | 0.366 |
| BM25 Recall@5 | 3.12% |
| Popularity baseline Recall@5 | 1.14% |
| BM25 catalog coverage@5 | 19.46% |

These are recorded local training results, not guarantees for other data. Detailed metrics are in `reports/milestone2_metrics.json` and `reports/recommender_metrics.json`.

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests api ui
.\.venv\Scripts\python.exe -m black --check src tests api ui
```

Checks that need locally generated models or predictions are skipped until training and promotion are complete.

## Limitations

- The target is a **30-day inactivity proxy**, not confirmed account cancellation.
- SHAP contributions explain raw XGBoost log-odds, not additive calibrated probabilities or causal effects.
- Product prices are historical medians, not live prices or inventory.
- Recommendations exclude products already bought before the cutoff. Cohort members without merchandise history receive a popularity fallback.
- Raw data, customer-level outputs, MLflow files, trained models, bundles, and real secrets stay outside Git.
- Docker/Compose configuration is included, but its image has not been built or tested on this PC. It requires a generated local bundle.
