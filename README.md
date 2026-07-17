# content_recommendation

## Overview

A content recommendation system for Hoichoi: tags titles by storyline/tone, groups them into interpretable Programming Categories, segments users into taste clusters by watch behavior, and ranks titles per cluster by lift — validated end-to-end against held-out real watch history.

## Project layout

```
content_recommendation/
├── README.md
├── docs/       reference docs (methodology, schema, taxonomy, results)
├── src/        pipeline + recommendation code
├── data/       tagged datasets, watch samples, model outputs (gitignored — see note below)
└── _archive/   superseded scripts/notebooks (gitignored)
```

`data/*.csv` is intentionally excluded from version control (raw/derived data containing user watch history) — see the root `.gitignore` comment. Nothing in `src/` persists a fitted model artifact; every script refits from scratch on each run.

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

## Status

Core pipeline built and validated at scale (2,218 users), now at the series/season level with structured `content_series_id`/`content_season_id` linkage (permalink-slug matching retired): the adopted model (8 categories, 6 audience clusters, `popularity^0.7 × cluster_rate^0.3 × creator_boost` scoring) hits 60.8% Hit@10 / 74.5% Hit@20 in held-out testing on the 500-title sample, matching/ahead of a naive popularity baseline. A separate season-continuation mechanism (`src/season_continuation.py`) recommends the next season once a user has watched any of the current one (no completion threshold), merged into the same single ranked list as regular discovery recommendations — no separate rail.

**Content catalog scaled to all 775 published titles** (`src/pipeline_full_catalog.py`, `data/content_features_full_tagged.csv`): all 626 net-new titles were genuinely re-tagged (each synopsis actually read against the closed taxonomy, not keyword-matched) — `tag_low_confidence` dropped from 538/779 to 108/775, and the remaining low-confidence rows are now real thin-content cases (reality TV, one-line documentary blurbs), not tagging-heuristic artifacts. Held-out performance on the same 2,218-user sample is statistically unchanged (60.4% Hit@10 / 73.7% Hit@20) — expected, since none of the ~575 newly-added titles have any watch history in this particular user sample yet, so they can't be recommended or validated until a larger/fresher user pull is done. Next: pull a larger, more recent user sample so the newly-tagged titles actually get evaluated.
