# Project Status

Last updated: 2026-10-10

## Current phase

**Phase 5 — API: complete.** Phase 6 (Gemini analyst copilot) is next.

## Phase log

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository preparation, `v1-uci` tag, ignore rules, environment template | Complete |
| 1 | H&M download, typed Parquet, DuckDB warehouse, integrity report, WebP thumbnails, `eie` CLI | Complete |
| 2 | Candidate generation + LightGBM ranker, MAP@12 on the final week, baselines and ablations | Complete |
| 3 | FashionCLIP image/text embeddings, FAISS similar-item and text search | Complete |
| 4 | Churn (inactivity proxy), lifetime value, segments, campaign simulator | Complete |
| 5 | FastAPI v2 | Complete |
| 6 | Gemini analyst copilot | Planned |
| 7 | Next.js frontend | Planned |
| 8 | CI, Docker Compose, documentation | Planned |

## Phase 1 measured results

| Item | Value |
|---|---|
| Transactions | 31,788,324 rows, 2018-09-20 to 2020-09-22 |
| Customers | 1,371,980 (1,362,281 with at least one purchase) |
| Articles | 105,542 (104,547 sold at least once) |
| Integrity checks | Passed: no nulls, orphans, non-positive prices, or duplicate keys |
| Parquet size | 423 MB transactions, 92 MB customers, 3.5 MB articles |
| Ingest / warehouse / quality runtime | 28.3 s / 7.7 s / 9.8 s on an i7-14650HX |
| Thumbnails | 105,100 WebP images (256x384), 0 failures, 0.59 GB, 225 s |
| Articles without a photo | 442 |

## Phase 2 measured results

| Item | Value |
|---|---|
| Held-out week | 2020-09-16 to 2020-09-22, 68,984 buyers, 213,728 purchased pairs |
| Ranker MAP@12 / Recall@12 | 0.0344 / 5.67% (popularity 0.0088, buy-again + trending 0.0257) |
| Candidates | ~108 per customer, covering 13.9% of held-out purchases |
| Ranker | LightGBM LambdaRank, 45 features, 120 rounds chosen by early stopping on the 9 Sep week |
| Production output | 16,463,760 rows: top 12 for all 1,371,980 customers as of 2020-09-23 |
| Runtime | 20 minutes end to end, including eight ablation models |

Phase 2's sparse categorical codes were re-encoded to consecutive integers in Phase 3; LightGBM now
reports no warnings.

## Phase 3 measured results

| Item | Value |
|---|---|
| Embeddings | FashionCLIP (MIT), 512-d; 105,542 descriptions in 48 s and 105,100 photos in 8.6 min on an RTX 4060 Laptop GPU; 207 MB |
| Cross-modal sanity check | A photo's nearest description is its own for 60.2% of 2,022 sampled products |
| Text search precision@10 | 0.95 on 12 labelled queries after mean-centring each modality (0.22 before); weights chosen on the same queries |
| Recommender MAP@12 | 0.0346 (Phase 2: 0.0344), 48 features, ~133 candidates per customer, candidate recall 14.4% |
| Style signal | `style_similarity` is 11th of 48 features by gain; the style candidate source alone does not help (removing it: 0.0348) |
| Production output | 16,463,760 rows for 1,371,980 customers; full run 33 minutes |

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
| Runtime | 13 minutes; GPU TreeSHAP cut churn scoring from 12.8 to 4.0 minutes |

## Phase 5 measured results

| Item | Value |
|---|---|
| Serving build | 26 s: purchases sorted by customer (225 MB), product statistics, dashboard aggregates |
| Routes | 15 OpenAPI paths with typed schemas; bearer-key auth; public health and thumbnails |
| Latency (local) | Customer 360 0.21 s; recommendations and history 0.06–0.07 s; text search 0.08–0.14 s; campaign plan 0.26 s |
| Start-up | About 1 s; FashionCLIP warms up in the background from the local cache (first search was 53 s before) |
| Tests | 10 API tests on synthetic shoppers run through the real pipelines; 34 tests in total |

## Known data facts

- Prices are normalized by H&M (median 0.0254, max 0.5915); they are not a real currency.
- 2,974,905 transaction rows repeat an earlier row exactly; these represent multiple units bought together and are kept.
- 70.4% of transaction rows have `sales_channel_id = 2`, interpreted as online (the dataset does not document the codes).
- Weekly volume dips sharply from mid-March 2020 (COVID store closures); later models must account for this shift.
