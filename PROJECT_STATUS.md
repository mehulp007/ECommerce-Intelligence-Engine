# Project Status

Last updated: 2026-10-23

## Current phase

**All build phases complete.** A design upgrade is under way in five stages: Graphite Mint restyle (done), Product Galaxy, 3D scroll story, UX extras, and a README GIF.

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

## Known data facts

- Prices are normalized by H&M (median 0.0254, max 0.5915); they are not a real currency.
- 2,974,905 transaction rows repeat an earlier row exactly; these represent multiple units bought together and are kept.
- 70.4% of transaction rows have `sales_channel_id = 2`, interpreted as online (the dataset does not document the codes).
- Weekly volume dips sharply from mid-March 2020 (COVID store closures); later models must account for this shift.
- Channel code 1 has zero rows in the five weeks from 2020-03-23 to 2020-04-26 (and 5 rows the week after), which supports reading code 2 as online.
