# Project Status

Last updated: 2026-10-10

## Current phase

**Phase 2 — Two-stage recommender: complete.** Phase 3 (image and text embeddings) is next.

## Phase log

| Phase | Scope | Status |
|---|---|---|
| 0 | Repository preparation, `v1-uci` tag, ignore rules, environment template | Complete |
| 1 | H&M download, typed Parquet, DuckDB warehouse, integrity report, WebP thumbnails, `eie` CLI | Complete |
| 2 | Candidate generation + LightGBM ranker, MAP@12 on the final week, baselines and ablations | Complete |
| 3 | FashionCLIP image/text embeddings, FAISS similar-item and text search | Planned |
| 4 | Churn (inactivity proxy), lifetime value, segments, campaign simulator | Planned |
| 5 | FastAPI v2 | Planned |
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

Follow-up for Phase 3: re-encode sparse categorical codes (some are -1 or very large) to consecutive
integers to remove LightGBM warnings, then re-measure with image-embedding candidates.

## Known data facts

- Prices are normalized by H&M (median 0.0254, max 0.5915); they are not a real currency.
- 2,974,905 transaction rows repeat an earlier row exactly; these represent multiple units bought together and are kept.
- 70.4% of transaction rows have `sales_channel_id = 2`, interpreted as online (the dataset does not document the codes).
- Weekly volume dips sharply from mid-March 2020 (COVID store closures); later models must account for this shift.
