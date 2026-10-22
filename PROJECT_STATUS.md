# Project Status

Last updated: 2026-10-23

## Current phase

**All build phases complete.** A design upgrade is under way in five stages: Graphite Mint restyle (done), Product Galaxy (done), 3D scroll story (done), UX extras, and a README GIF.

## Phase log

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository preparation, `v1-uci` tag, ignore rules, environment template | Complete |
| 1 | H&M download, typed Parquet, DuckDB warehouse, integrity report, WebP thumbnails, `eie` CLI | Complete |
| 2 | Candidate generation + LightGBM ranker, MAP@12 on the final week, baselines and ablations | Complete |
| 3 | FashionCLIP image/text embeddings, FAISS similar-item and text search | Complete |
| 4 | Churn (inactivity proxy), lifetime value, segments, campaign simulator | Complete |
| 5 | FastAPI v2 | Complete |
| 6 | Gemini analyst copilot | Complete |
| 7 | Next.js frontend | Complete |
| 8 | CI, Docker Compose, documentation | Complete |

## Phase 1 measured results

| Item | Value |
|---|---|
| Transactions | 31,788,324 rows, 2018-09-20 to 2020-09-22 |
| Customers | 1,371,980 (1,362,281 with at least one purchase) |
| Articles | 105,542 (104,547 sold at least once) |
| Integrity checks | Passed: no nulls, orphans, non-positive prices, or duplicate keys |
| Parquet size | 423 MB transactions, 92 MB customers, 3.5 MB articles |
| Thumbnails | 105,100 WebP images (256x384), 0 failures, 0.59 GB |
| Articles without a photo | 442 |

## Phase 2 measured results

| Item | Value |
|---|---|
| Held-out week | 2020-09-16 to 2020-09-22, 68,984 buyers, 213,728 purchased pairs |
| Ranker MAP@12 / Recall@12 | 0.0344 / 5.67% (popularity 0.0088, buy-again + trending 0.0257) |
| Candidates | ~108 per customer, covering 13.9% of held-out purchases |
| Ranker | LightGBM LambdaRank, 45 features, 120 rounds chosen by early stopping on the 9 Sep week |
| Production output | 16,463,760 rows: top 12 for all 1,371,980 customers as of 2020-09-23 |

Phase 2's sparse categorical codes were re-encoded to consecutive integers in Phase 3; LightGBM now
reports no warnings.

## Phase 3 measured results

| Item | Value |
|---|---|
| Embeddings | FashionCLIP (MIT), 512-d; 105,542 descriptions and 105,100 photos; 207 MB (float16) |
| Cross-modal sanity check | A photo's nearest description is its own for 60.2% of 2,022 sampled products |
| Text search precision@10 | 0.95 on 12 labelled queries after mean-centring each modality (0.22 before); weights chosen on the same queries |
| Recommender MAP@12 | 0.0346 (Phase 2: 0.0344), 48 features, ~133 candidates per customer, candidate recall 14.4% |
| Style signal | `style_similarity` is 11th of 48 features by gain; the style candidate source alone does not help (removing it: 0.0348) |
| Production output | 16,463,760 rows for 1,371,980 customers |

Possible follow-up: drop the similar-style candidate source (keep the feature) and confirm on a new
validation week rather than the held-out week.

## Phase 4 measured results

| Item | Value |
|---|---|
| Churn split | Train Oct 2019–Feb 2020 (40% sample, 1.96M rows), calibrate Mar–Apr 2020, test May–Jul 2020 (2.96M customer-months) |
| Churn test | ROC AUC 0.795, AP 0.857, Brier 0.178 (recency-only logistic 0.701; all-feature logistic 0.789) |
| Calibration | Calibrating on COVID months worsened Brier (0.178 vs 0.175 uncalibrated); production calibrates on Jun–Jul 2020 |
| Lifetime value holdout | Spend MAE 0.139, correlation 0.668, total +7.3% vs actual (previous-period baseline MAE 0.145) |
| Segments | 5 clusters (silhouette 0.193) on behaviour + visual style; 378,488 lapsed customers |
| Scores | 1,371,980 customers; 993,492 active; mean churn probability 0.627; total 180-day value 212,005 units |

## Phase 5 measured results

| Item | Value |
|---|---|
| Serving build | Purchases sorted by customer (225 MB), product statistics, dashboard aggregates |
| Routes | 15 OpenAPI paths with typed schemas; bearer-key auth; public health and thumbnails |
| Start-up | FashionCLIP warms up in a background thread from the local cache, so the first search does not wait for the model |
| Tests | 10 API tests on synthetic shoppers run through the real pipelines; 34 tests in total |

## Phase 6 measured results

| Item | Value |
|---|---|
| Model | `gemini-3.8-flash` (GA 2026-09-02), fallback `gemini-3.6-flash`; google-genai 2.30 |
| Tools | run_sql, create_chart, get_customer, search_products, similar_products, plan_campaign |
| SQL safety | One SELECT; allow-listed tables via syntax tree; read-only attach with external access off and configuration locked; 20 s / 500-row limits; 15 attack queries tested |
| Live check | Growth question answered in 32 s (about 15.8k tokens); every figure matched an independent query |
| Fixes from live use | `active` renamed `account_active`; prompt defines active customers, exact drivers, and model types; customer figures labelled by period |
| Resilience | 503 retry with backoff; fallback before the first answer; free-tier daily quota (20 requests per model) detected, remembered until reset, and restarted once on the other model |
| Tests | 38 copilot tests (all-models-exhausted case added in Phase 7) |

## Phase 7 measured results

| Item | Value |
|---|---|
| Stack | Next.js 16.4 (App Router, Turbopack), React 19.3, TypeScript, Tailwind CSS 4, TanStack Query 5, Recharts 3 |
| Pages | Overview, Customer 360 (list and profile), Products (browse, describe, detail), Segments, Campaigns, Copilot, Model Lab |
| API access | Next.js route handlers forward `/api/*` and `/images/*` and add the bearer key on the server; 0 key matches in the client bundle, server build, or served HTML |
| Types | Generated from the OpenAPI schema (`eie serve openapi`, `npm run gen:api`); a pytest check catches drift |
| Responsive check | No horizontal overflow at 375 px on any page (light and dark themes at the time; dark only since the redesign) |
| Copilot UI | All event types rendered from a real-tool replay; live Gemini call showed the quota message (free tier exhausted) |
| Checks | `npm run lint`, `npm run typecheck`, `npm run build` clean; 73 Python tests |

## Phase 8 results

| Item | Value |
|---|---|
| CI | GitHub Actions: Python lint and tests (without PyTorch), web lint, type check and build, and both container images |
| Setup | `scripts/setup.ps1` (environment, PyTorch CUDA or CPU, project, web app, `.env` with a generated key), `eie build` (every pipeline in order), `scripts/start.ps1` (API and web app together) |
| Docker | `compose.yaml` with an API image (CPU PyTorch) and a standalone Next.js image; data and reports mounted read-only; ports bound to 127.0.0.1 |
| Tests | 74 Python tests, including the build order; they also pass without PyTorch, as CI runs them |
| README | Rewritten: features, held-out results, architecture, stack, three-step setup |

## Design stage 1 — Graphite Mint restyle

| Item | Value |
|---|---|
| Theme | One dark theme: graphite surfaces (#0A0A0A, #141414, #1C1C1C), mint accent #3DDC97, violet #9B8CFF; the light/dark toggle was removed |
| Typography | Geist Sans for text; Geist Mono for headline numbers and chart axes |
| Contrast | All text and accent pairs pass WCAG AA (lowest 5.07:1, muted text on raised surfaces) |
| Components | Cards, stats, buttons, inputs, chips, badges, charts, product cards, and a grouped sidebar with a mint active indicator; pages fade in unless reduced motion is set |
| Responsive check | No horizontal overflow at 375 px on any of the 9 routes checked |
| Security | Next.js upgraded 16.4.0 → 16.4.2 for a critical advisory (GHSA-2464-fhr4-w9xw); `npm audit --omit=dev` reports 0 vulnerabilities |

## Design stage 2 — Product Galaxy

| Item | Value |
|---|---|
| Layout | Joint photo and description vectors (shared with text search via `joint_vectors`), PCA to 50, then UMAP to 3 (30 neighbours, min_dist 0.15, cosine, seed 42), scaled so 99% of points fall within radius 1 |
| Build | `eie search galaxy`: 105,542 products in 96.8 s on CPU; part of `eie build`; UMAP lives in a separate `galaxy` extra so the API image stays lean |
| Neighbourhood quality | 92.2% of each product's 10 nearest map neighbours share its product group (23.3% by chance); 60.3% share its colour group (8.2% by chance) |
| API | `GET /api/galaxy` (names and field layout) and `GET /api/galaxy/points` (one binary buffer, 21 bytes per product, 2.2 MB); 503 until the map is built |
| Page | `/galaxy`: one GPU point cloud with a custom shader, four colour modes, a legend that filters, search fly-to, hover preview with photo, click to open; "Show in the Galaxy" on every product page rings the product and lights its 39 nearest neighbours |
| Performance | three.js (941 KB uncompressed) loads only with the Galaxy scene, not in any page's initial JavaScript; phones load every third product (35,181) |
| Tests | 7 new (layout order and codes, determinism, scaling, truncated files, API buffer against the layout and sales, 503); 82 in total |
| CI fix | The lifetime-value fits occasionally stopped with L-BFGS-B "ABNORMAL" on the tiny synthetic test data (rounding noise from parallel sums); they now polish with Powell before failing. Full-data results are unchanged because L-BFGS-B already converges there |

## Design stage 3 — 3D scroll story

| Item | Value |
|---|---|
| Page | `/story`, full screen outside the app shell (app pages moved into an `(app)` route group; URLs unchanged); linked from the sidebar |
| Scenes | 1 Galaxy hero; 2 weekly revenue as stacked online/store columns drawn in left to right, with the store-closure weeks marked; 3 six candidate sources converging on a ranker and 12 slots; 4 a typed query diving to its 50 matches in their real garment colours; 5 a 30,000-customer risk cloud with a live budget slider; 6 a recorded copilot run; 7 held-out results |
| Engine | One React Three Fiber canvas with four actors that crossfade on scroll position; a camera rig eases between shots; native scrolling (no scroll-jacking library); all text is HTML |
| Data | New `serving/story.py` writes an anonymous 30,000-customer sample (no IDs; 22 bytes each) during `eie serve prepare`, served by `GET /api/story/customers` and `/points`; all other figures come from existing endpoints |
| Copilot replay | A real run (gemini-3.8-flash, 79 s, 14.7k tokens) on monthly online share in 2020; every figure matched an independent warehouse query |
| Finding surfaced | At a 300-unit budget, only 5.2% of the 60,000 customers the planner contacts are high risk; their median predicted value is 5× the active average |
| Accessibility | Reduced motion gives a static, stacked page with a 2D chart and no WebGL; phones load lighter samples and a compact copilot card; `?capture=1` hides product photos for recordings |
| Performance | About 95 fps while scrolling (integrated Intel GPU, Chrome, dev build); heap stable at 39–41 MB across six full passes after GC |
| Tests | 3 new (sample contents and anonymity, axis direction, API); 85 in total |

## Known data facts

- Prices are normalized by H&M (median 0.0254, max 0.5915); they are not a real currency.
- 2,974,905 transaction rows repeat an earlier row exactly; these represent multiple units bought together and are kept.
- 70.4% of transaction rows have `sales_channel_id = 2`, interpreted as online (the dataset does not document the codes).
- Weekly volume dips sharply from mid-March 2020 (COVID store closures); later models must account for this shift.
- Channel code 1 has zero rows in the five weeks from 2020-03-23 to 2020-04-26 (and 5 rows the week after), which supports reading code 2 as online.
