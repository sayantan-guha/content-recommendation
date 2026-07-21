# content_recommendation

## Overview

A content recommendation system for Hoichoi: tags titles by storyline/tone, groups them into interpretable Programming Categories, segments users into taste clusters by watch behavior, and ranks titles per cluster by lift — validated end-to-end against held-out real watch history.

## Project layout

```
content_recommendation/
├── README.md
├── docs/         reference docs (methodology, schema, taxonomy, results)
├── src/          pipeline + recommendation code, incl. recommender.py (shared model logic)
├── backend/      FastAPI service that fits the model and serves it as JSON
├── ui/           Streamlit sample UI, styled with the hoichoi brand design system
├── .streamlit/   Streamlit theme config (light theme, brand colors)
├── data/         tagged datasets, watch samples, model outputs (gitignored — see note below)
└── _archive/     superseded scripts/notebooks (gitignored)
```

`data/*.csv` is intentionally excluded from version control (raw/derived data containing user watch history) — see the root `.gitignore` comment. Nothing in `src/` persists a fitted model artifact; every script refits from scratch on each run.

## Running the sample UI

The sample UI is a two-service setup: a FastAPI backend that fits the model once and serves it as JSON, and a Streamlit frontend that renders it.

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the backend (fits the model at startup — takes a few seconds)
cd content_recommendation/backend
uvicorn app:app --port 8000

# 3. In a second terminal, start the UI
cd content_recommendation
streamlit run ui/app.py
```

- Backend: http://localhost:8000 (`/health`, `/users`, `/users/{uid}/history`, `/users/{uid}/recommendations`)
- UI: http://localhost:8501 — pick a viewer from the dropdown in the top nav to see their top recommendations

The UI points at `http://localhost:8000` by default; override with the `HC_RECS_API` environment variable if the backend runs elsewhere.

## Docs

| Doc | What's in it |
|---|---|
| [docs/DATA_AND_METHODOLOGY.md](docs/DATA_AND_METHODOLOGY.md) | **Start here** — what data was fetched, what fields were engineered, and every technique/methodology used, in plain + technical terms |
| [docs/APPROACH.md](docs/APPROACH.md) | The overall step-by-step design (content → category, audience → title) |
| [docs/SCHEMA.md](docs/SCHEMA.md) | Database tables, relationships, and known data-quality quirks |
| [docs/TAG_TAXONOMY.md](docs/TAG_TAXONOMY.md) | The closed vocabulary of storyline/tone tags |
| [docs/CATEGORIES.md](docs/CATEGORIES.md) | The 8 Programming Categories, v1 vs. v2 weighting, validation results |
| [docs/AUDIENCE_CLUSTERS.md](docs/AUDIENCE_CLUSTERS.md) | The 4 audience clusters, title affinity scoring, creator-boost weighting, at-scale validation |

## Code

| File | What it does |
|---|---|
| [src/pipeline_series_structured.py](src/pipeline_series_structured.py) | Original model on the 500-title tagged sample: content tagging → category clustering → audience clustering → series/season structured linkage → discovery-ranked recommendations, validated against held-out watch history |
| [src/pipeline_full_catalog.py](src/pipeline_full_catalog.py) | Same model, scaled to the full published catalog (775 titles: 472 movies + 303 series) — structurally simpler since the full pull is already one row per show/movie, no episode-rollup step needed |
| [src/season_continuation.py](src/season_continuation.py) | Next-season recommender + single-list merge helper, kept separate from (but combinable with) the discovery ranking output |
| [src/recommender.py](src/recommender.py) | Framework-agnostic extraction of the full-catalog model (content + audience clustering, scoring) — imported by `backend/app.py` |
| [backend/app.py](backend/app.py) | FastAPI service: fits the model once at startup, serves `/users`, `/users/{uid}/history`, and `/users/{uid}/recommendations` as JSON |
| [ui/app.py](ui/app.py) | Streamlit sample UI (hero banner + recommendation rail), styled with the hoichoi brand design system, backed entirely by the FastAPI service above |
| [src/experiments/k_tuning_sweep.py](src/experiments/k_tuning_sweep.py) | Grid search over K_CONTENT/K_USER cluster counts against the held-out methodology; not wired into production, kept as a reference experiment |
| [src/pipeline_item_cf.py](src/pipeline_item_cf.py) | Item-item collaborative filtering ("users who watched X also watched Y", cosine similarity over the binary watch matrix) — outperforms the cluster-based model by a wide margin; see Status below |
| [src/eval_recommender.py](src/eval_recommender.py) | Evaluation harness for `recommender.py`: the standard warm-item LOO test (mean rank, Hit@10/20, NDCG@10/20, MRR) plus a cold-start exposure check (does the fallback actually surface low-data titles?) |

## Status

Core pipeline built and validated at scale (2,218 users), now at the series/season level with structured `content_series_id`/`content_season_id` linkage (permalink-slug matching retired): the adopted model (8 categories, 6 audience clusters, `popularity^0.7 × cluster_rate^0.3 × creator_boost` scoring) hits 60.8% Hit@10 / 74.5% Hit@20 in held-out testing on the 500-title sample, matching/ahead of a naive popularity baseline. A separate season-continuation mechanism (`src/season_continuation.py`) recommends the next season once a user has watched any of the current one (no completion threshold), merged into the same single ranked list as regular discovery recommendations — no separate rail.

**Content catalog scaled to all 775 published titles** (`src/pipeline_full_catalog.py`, `data/content_features_full_tagged.csv`): all 626 net-new titles were genuinely re-tagged (each synopsis actually read against the closed taxonomy, not keyword-matched) — `tag_low_confidence` dropped from 538/779 to 108/775, and the remaining low-confidence rows are now real thin-content cases (reality TV, one-line documentary blurbs), not tagging-heuristic artifacts. Held-out performance on the same 2,218-user sample is statistically unchanged (60.4% Hit@10 / 73.7% Hit@20) — expected, since none of the ~575 newly-added titles have any watch history in this particular user sample yet, so they can't be recommended or validated until a larger/fresher user pull is done. Next: pull a larger, more recent user sample so the newly-tagged titles actually get evaluated.

**Sample UI added** (`backend/app.py` + `ui/app.py`): a FastAPI service fits the full-catalog model once and serves recommendations as JSON; a Streamlit frontend, styled with the hoichoi brand design system, lets you pick a real viewer and see their top-10 picks. See "Running the sample UI" above.

**Movie/series type-affinity quota** (`src/recommender.py`'s `apply_type_quota`): the recommendation slate (now top-20) is reordered so its movie/series split matches the user's own watch-history proportions, instead of a flat top-N cut that can skew toward whichever type happens to score higher. The UI also gained a **Watched History rail** and a **Watch Mix donut chart** (`ui/app.py`) showing this same movie/series split.

**Watch sample scaled to the full 5,000 users** (`data/user_title_watch_sample_5000.csv`): `src/pipeline_full_catalog.py` and `backend/app.py` now read from this sample instead of the older 2,218-user one, giving the full 775-title catalog real evaluation coverage for the first time (752 distinct titles now have watch history, vs. 183 before). Held-out results on this sample: **17.6% Hit@10 / 27.3% Hit@20** for the model vs. **16.8% Hit@10 / 25.8% Hit@20** for a naive popularity baseline. The absolute numbers are much lower than the old 60.4%/73.7% -- that's expected, not a regression: the eligible candidate pool grew ~4x (183 → 752 titles) competing for the same fixed top-10/20 slots, which mechanically lowers any method's raw hit rate. What matters is the model's **lift over the popularity baseline**, which actually improved slightly (vs. roughly tied before) now that there's real signal to distinguish it from raw popularity.

**K-tuning grid search at this scale** (`src/experiments/k_tuning_sweep.py`): swept 16 combinations of K_CONTENT (content-category clusters: 6/8/10/12) x K_USER (audience clusters: 4/6/8/10) against the same held-out methodology. Result: **no meaningful improvement over the current 8/6 setting** — all 16 combinations landed within 0.174-0.179 Hit@10, a band narrower than the ~0.6-point standard error on a proportion this size (n_eval=4,200). The "best" combo (K_CONTENT=12, K_USER=10, Hit@10=0.179) is statistically indistinguishable from production (8/6, Hit@10=0.177); cluster count isn't the current bottleneck.

**Item-item collaborative filtering added** (`src/pipeline_item_cf.py`) — and it decisively outperforms the cluster-based model:

| Model | Hit@10 | Hit@20 | Mean rank |
|---|---|---|---|
| Popularity baseline | 16.8% | 25.8% | 110.1 |
| Cluster model (8 categories, 6 audience clusters, popularity^0.3 scoring) | 18.4% | 27.7% | 102.7 |
| **Item-item CF** (cosine similarity over the binary watch matrix) | **29.6%** | **39.8%** | **72.7** |

CF's lift over the popularity baseline (+12.8pp) is ~8x the cluster model's (+1.6pp). Root cause traced through the whole investigation: the 6 audience clusters concentrate 69% of users into just 2 "generic drama viewer" blobs whose item-popularity correlates 0.95+ with *global* popularity — so the cluster model's "personalized" score is barely distinct from popularity for most users. CF sidesteps this entirely by scoring every user against their own specific co-viewers rather than a cluster average. K-tuning, tag-density fixes, and audience-profile enrichment (recency/completion weighting, tested at 500-user scale) were all tried first and made no difference — the cluster *architecture* was the ceiling, not any of its tunable parameters.

Refinements to CF were tested and rejected — weighting co-occurrence by seconds_watched (no change), shrinkage regularization (hurt), and blending in creator_boost or the cluster model's score (both hurt, diluting CF's sharper signal). Plain binary cosine similarity is the best variant found.

**Known limitation:** CF only scores titles with enough co-occurrence data (same >=5-viewer floor used for evaluation) and users with >=1 watched eligible title — it needs the content-model and popularity baseline as fallback tiers for cold-start titles/users, not a full replacement. Not yet wired into `backend/app.py`/`ui/app.py`.

**`item-item-cf` and `scale-user-sample-4800` merged into `main`.** Production (`backend/app.py`/`ui/app.py`) now fits on the 5,000-user watch sample (4,210 eval users, up from 1,485) via the same cluster-based model documented above — CF itself is still not wired into serving, only the larger sample and its K-tuning findings.

**Cold-start fallback blending added** (`src/recommender.py`'s `cold_start_candidates`): titles with fewer than 5 distinct viewers are excluded from the cluster/CF-scored candidate pool entirely (no reliable signal to rank them), so brand-new or low-viewership titles could never surface. The fallback scores these titles by cosine similarity between the user's taste profile and the title's content-category mixture (the same tag-derived vector used for clustering, available the moment a title is tagged — no watch history needed), and reserves ~10% of each recommendation slate (2 of 20 slots) for the best-matching cold titles. Verified with `src/eval_recommender.py`: cold-title exposure goes from **0% of users → 100% of users** getting at least one low-data title in their top-20, with the warm-item ranking metrics below computed on the unchanged warm-only path (so this doesn't cannibalize the tested ranking quality).

**NDCG@k and MRR added alongside Hit@10/20** (`src/eval_recommender.py`), since hit-rate alone treats a held-out title landing at rank 1 the same as rank 20 — no signal on ranking quality within the top-N. On the 5,000-user sample, 4,210 eval users: **mean_rank=102.8, Hit@10=17.6%, Hit@20=27.2%, NDCG@10=0.098, NDCG@20=0.122, MRR=0.091**. The low absolute NDCG/MRR (vs. Hit@10/20) confirms recovered titles tend to land in the back half of the top-20 rather than at the top — a ranking-sharpness gap the cluster/CF blend doesn't currently address.

**Segmented evaluation by task** (`src/eval_recommender.py`): rather than one blended metric, evaluation is now split by what's actually being tested — **warm-item recovery** (can a title the user already watched, with >=5 viewers overall, be recovered from their remaining history? — the standard LOO test) vs. **cold-start exposure** (does the fallback actually get low-data titles in front of users at all? — LOO can't test "recovery" of a title only 1-4 people watched, so this measures exposure rate instead, with/without the fallback).
