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

## Status

Core pipeline built and validated at scale (2,218 users), now at the series/season level with structured `content_series_id`/`content_season_id` linkage (permalink-slug matching retired): the adopted model (8 categories, 6 audience clusters, `popularity^0.7 × cluster_rate^0.3 × creator_boost` scoring) hits 60.8% Hit@10 / 74.5% Hit@20 in held-out testing on the 500-title sample, matching/ahead of a naive popularity baseline. A separate season-continuation mechanism (`src/season_continuation.py`) recommends the next season once a user has watched any of the current one (no completion threshold), merged into the same single ranked list as regular discovery recommendations — no separate rail.

**Content catalog scaled to all 775 published titles** (`src/pipeline_full_catalog.py`, `data/content_features_full_tagged.csv`): all 626 net-new titles were genuinely re-tagged (each synopsis actually read against the closed taxonomy, not keyword-matched) — `tag_low_confidence` dropped from 538/779 to 108/775, and the remaining low-confidence rows are now real thin-content cases (reality TV, one-line documentary blurbs), not tagging-heuristic artifacts. Held-out performance on the same 2,218-user sample is statistically unchanged (60.4% Hit@10 / 73.7% Hit@20) — expected, since none of the ~575 newly-added titles have any watch history in this particular user sample yet, so they can't be recommended or validated until a larger/fresher user pull is done. Next: pull a larger, more recent user sample so the newly-tagged titles actually get evaluated.

**Sample UI added** (`backend/app.py` + `ui/app.py`): a FastAPI service fits the full-catalog model once and serves recommendations as JSON; a Streamlit frontend, styled with the hoichoi brand design system, lets you pick a real viewer and see their top-10 picks. See "Running the sample UI" above.

**Tag-density audit and fix for the 625-title scale-up batch**: an audit (reading synopsis text against `docs/TAG_TAXONOMY.md` for a random sample) found the scale-up batch averaged only 2.27 storyline tags/title vs. 2.96 for the original 150-title batch it shares the catalog with -- a systematic under-tagging gap, with ~1 in 5 sampled titles below the taxonomy's 2-tag floor and not always caught by the `tag_low_confidence` flag. Re-ran the same 5-parallel-agent tagging process on all 625 titles with explicit instructions to capture secondary storyline threads, not just the single most obvious tag. Result: storyline tags/title rose to 2.41 (partial close of the gap), and `tag_low_confidence` correctly rose to 19.5% (agents flagged thin content honestly rather than force-fitting tags). Held-out Hit@10/Hit@20 and clustering silhouette score were both unchanged (within noise) -- confirming this data-scale's bottleneck is watch-history coverage, not tagging granularity, but the tags themselves are now more accurate for anything that surfaces them directly (browse/filter UI, editorial tooling).
