# E-Commerce Intelligence Engine

A local retail intelligence studio built on **31.8 million real fashion purchases** from the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset: 1.37 million customers, 105,542 products, and 105,100 product photos.

The finished project will combine a two-stage product recommender, visual and natural-language product
search, churn and lifetime-value models, a campaign simulator, a Gemini-powered analyst copilot, and a
Next.js frontend. It runs entirely on a local PC; no cloud service is required.

> **Status: v2 is being built phase by phase.** The original UCI Online Retail version (FastAPI +
> Streamlit) is preserved at the
> [`v1-uci`](https://github.com/mehulp007/ECommerce-Intelligence-Engine/tree/v1-uci) tag.

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository preparation | Done |
| 1 | Data foundation: H&M ingestion, DuckDB warehouse, integrity checks, thumbnails | Done |
| 2 | Two-stage recommender (candidate generation + LightGBM ranker, MAP@12) | **Done** |
| 3 | FashionCLIP image and text embeddings, similar-item and text search | Planned |
| 4 | Churn, lifetime value, segments, campaign simulator | Planned |
| 5 | FastAPI v2 | Planned |
| 6 | Gemini analyst copilot | Planned |
| 7 | Next.js frontend | Planned |
| 8 | CI, Docker Compose, final documentation | Planned |

## Data foundation (Phase 1)

```mermaid
flowchart LR
    Kaggle["Kaggle archive<br/>28.7 GB ZIP"] --> CSV["3 CSV tables<br/>3.7 GB"]
    Kaggle --> Thumbs["105,100 WebP thumbnails<br/>256x384, 0.59 GB"]
    CSV --> Parquet["Typed Parquet<br/>zstd, 0.52 GB"]
    Parquet --> Warehouse["DuckDB warehouse<br/>tables + sales view"]
    Parquet --> Quality["Integrity report<br/>reports/data_quality.json"]
```

- **Typed Parquet.** Article IDs stay ten-digit text so leading zeros survive; dates are real `DATE`
  values; customer flags become booleans; transactions are sorted by date for fast time filtering.
- **DuckDB warehouse.** `data/warehouse.duckdb` holds the `transactions`, `customers`, and `articles`
  tables plus a denormalized `sales` view (product, colour, department, channel, customer age) for
  analytics and the future copilot.
- **Integrity checks.** The pipeline fails if any transaction has nulls, a non-positive price, an unknown
  customer or article, or a date outside the published range, or if customer or article IDs repeat.
- **Thumbnails.** Photos are decoded straight from the ZIP in parallel worker processes, so the
  29 GB of images never needs to be unpacked.

### Measured results

| Item | Result |
|---|---|
| Transactions | 31,788,324 rows, 2018-09-20 to 2020-09-22 |
| Customers | 1,371,980 (1,362,281 with a purchase) |
| Articles | 105,542 (104,547 sold at least once; 105,100 with a photo) |
| Integrity checks | All passed |
| Online share of purchase rows | 70.4% |
| Parquet size | 0.52 GB (from 3.7 GB of CSV) |
| Runtime on an i7-14650HX, 16 GB RAM | ingest 28 s, warehouse 8 s, quality 10 s, thumbnails 225 s |

Weekly purchase volume peaks in late June 2019 (563k rows) and falls to its lowest full week in mid-March
2020 (184k rows) as COVID closed stores. Later phases treat this as a real distribution shift.

## Two-stage recommender (Phase 2)

The recommender predicts which articles each customer will buy next week. Like production
recommenders at large retailers, it works in two stages: cheap **candidate generators** propose about
100 articles per customer, then a **learned ranker** orders them.

```mermaid
flowchart LR
    TX["31.8M purchases<br/>(before the as-of date)"] --> R["Buy again<br/>recent purchases"]
    TX --> P["Trending<br/>last week's best sellers"]
    TX --> A["Popular in age group"]
    TX --> C["Bought together<br/>same-basket pairs"]
    TX --> M["Similar shoppers<br/>ALS matrix factorization"]
    R & P & A & C & M --> U["~108 candidates per customer"]
    U --> F["45 features<br/>article momentum, customer profile,<br/>customer-article affinity"]
    F --> L["LightGBM LambdaRank"]
    L --> T["Top 12 + reason"]
```

**Evaluation protocol.** Training labels come from the weeks starting 26 Aug, 2 Sep, and 9 Sep 2020;
early stopping uses the 9 Sep week; the final week of data (16–22 Sep 2020, 68,984 buyers) is held out
and only used for the scores below. Every feature is computed from purchases strictly before the
as-of date, and a test inserts future purchases to prove they change nothing. The metric is MAP@12,
the metric of the original Kaggle competition.

### Held-out results (16–22 Sep 2020)

| Model | MAP@12 | Recall@12 |
|---|---|---|
| **Two-stage ranker** | **0.0344** | **5.67%** |
| Buy again, then trending | 0.0257 | 3.89% |
| Bought together, then trending | 0.0116 | 2.70% |
| ALS (similar shoppers) only | 0.0112 | 2.01% |
| Trending (global popularity) | 0.0088 | 2.29% |

The ranker is **3.9x the popularity baseline** and **34% better than the strongest single
heuristic**. For context, the competition's winning private-leaderboard score was about 0.037 on a
different, hidden week, so the comparison is indicative only.

### What matters (ablations on the same held-out week)

| Removed | MAP@12 | Change |
|---|---|---|
| Buy-again candidates | 0.0279 | -18.7% |
| Customer-article affinity features | 0.0311 | -9.5% |
| Article features (momentum, price, category) | 0.0318 | -7.5% |
| Bought-together candidates | 0.0337 | -2.0% |
| Customer features | 0.0340 | -1.0% |
| Trending candidates | 0.0341 | -0.7% |
| ALS candidates | 0.0343 | -0.2% |
| Age-group candidates | 0.0345 | +0.2% (noise) |

Repeat purchases dominate fashion shopping, and personal affinity (how recently the customer bought
the article, how often they buy that product family and garment type) is the most informative feature
group. The two popularity sources overlap heavily, and ALS adds little beyond the other sources;
Phase 3 adds image-embedding candidates as a stronger content signal.

### Production scoring

The production ranker is retrained including the held-out week, then scores every customer as of
23 Sep 2020: **16,463,760 recommendations for all 1,371,980 customers** in 9.5 minutes, each with a
reason (`buy_again`, `bought_together`, `similar_shoppers`, `popular_in_age_group`, `trending`).
Recommendations cover 21.6% of the catalog. The full run, including evaluation and eight ablation
models, takes 20 minutes on an i7-14650HX with 16 GB RAM.

## Run it on your PC

These steps use **Windows PowerShell** and **Python 3.12**. Run one command at a time.

### 1. Create the environment

From the project folder (the one containing `pyproject.toml`):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[data,train,recsys,dev]"
```

### 2. Get access to the data

The dataset is not redistributed in this repository. Each user downloads it from Kaggle:

1. Sign in to Kaggle, open the
   [competition page](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations),
   choose **Join Competition**, and accept the rules.
2. In **Kaggle → Settings → API**, create a new token and copy it.
3. Save it to your user profile (outside the project) and check that the file is not empty:

```powershell
New-Item -ItemType Directory -Force "$env:USERPROFILE\.kaggle"
notepad "$env:USERPROFILE\.kaggle\access_token"
(Get-Item "$env:USERPROFILE\.kaggle\access_token").Length
```

Paste the token into Notepad, save, and close it. The last command should print a number above zero.
Never commit or share the token.

### 3. Build the data foundation

```powershell
.\.venv\Scripts\eie.exe data all
```

This runs five steps, which can also be run individually:

| Command | What it does |
|---|---|
| `eie data download` | Downloads the 28.7 GB archive into `data/raw/hm/` (about 30 minutes at 17 MB/s) |
| `eie data ingest` | Extracts the CSV tables and writes typed Parquet to `data/parquet/` |
| `eie data warehouse` | Builds `data/warehouse.duckdb` |
| `eie data quality` | Writes `reports/data_quality.json` and fails if a check does not pass |
| `eie data thumbnails` | Writes WebP thumbnails to `data/thumbnails/` |

Optional settings: `EIE_DUCKDB_MEMORY` (default `6GB`) and `EIE_DUCKDB_THREADS` limit DuckDB's resource
use.

### 4. Train the recommender

```powershell
.\.venv\Scripts\eie.exe recsys train
```

This evaluates on the held-out week, writes `reports/recommender_metrics.json`, saves the ranker to
`artifacts/recsys/`, and writes every customer's top 12 to `data/serving/recommendations.parquet`.
Add `--no-ablations` to skip the eight comparison models (about 5 minutes faster) or `--no-score` to
skip scoring all customers.

### 5. Explore the warehouse

```powershell
.\.venv\Scripts\python.exe -c "import duckdb; print(duckdb.connect('data/warehouse.duckdb', read_only=True).sql('SELECT index_group_name, count(*) AS purchases FROM sales GROUP BY ALL ORDER BY purchases DESC'))"
```

## Project layout

| Path | Contents |
|---|---|
| `src/ecommerce_intelligence/cli.py` | The `eie` command-line tool |
| `src/ecommerce_intelligence/data/` | Download, ingestion, warehouse, quality checks, thumbnails |
| `src/ecommerce_intelligence/recsys/` | Candidate generators, ALS, features, LightGBM ranker, metrics, pipeline |
| `src/ecommerce_intelligence/evaluation/` | Shared metric helpers |
| `tests/` | Tests that run on a miniature synthetic H&M archive |
| `reports/` | Data-quality, thumbnail, and recommender reports |
| `data/`, `artifacts/` | Local data and generated outputs (Git-ignored) |

## Checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests
.\.venv\Scripts\python.exe -m black --check src tests
```

The tests build a small fake archive with the real file layout, so they run in seconds without the
real dataset.

## Data notes and limitations

- **Terms of use.** The H&M data is provided through Kaggle for non-commercial and academic use. This is
  a non-commercial portfolio project; raw data, images, and customer-level outputs are never committed.
- **Prices are normalized** by H&M and are not a real currency.
- **Repeated rows are kept.** 2,974,905 rows repeat an earlier row exactly; they represent several units
  bought together.
- **Sales channels.** `sales_channel_id = 2` is interpreted as online and `1` as store; the dataset does
  not document these codes.
- The data is historical (2018–2020), not live.
- Recommendations can include articles a customer already bought; these are labelled `buy_again`
  because repeat purchases are common in fashion.
