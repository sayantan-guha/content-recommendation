# content_recommendation

## Overview

A content recommendation system for Hoichoi: tags titles by storyline/tone, groups them into interpretable Programming Categories, segments users into taste clusters by watch behavior, and ranks titles per cluster by lift — validated end-to-end against held-out real watch history.

## Docs

| Doc | What's in it |
|---|---|
| [DATA_AND_METHODOLOGY.md](DATA_AND_METHODOLOGY.md) | **Start here** — what data was fetched, what fields were engineered, and every technique/methodology used, in plain + technical terms |
| [APPROACH.md](APPROACH.md) | The overall step-by-step design (content → category, audience → title) |
| [SCHEMA.md](SCHEMA.md) | Database tables, relationships, and known data-quality quirks |
| [TAG_TAXONOMY.md](TAG_TAXONOMY.md) | The closed vocabulary of storyline/tone tags |
| [CATEGORIES.md](CATEGORIES.md) | The 8 Programming Categories, v1 vs. v2 weighting, validation results |
| [AUDIENCE_CLUSTERS.md](AUDIENCE_CLUSTERS.md) | The 4 audience clusters, title affinity scoring, creator-boost weighting, at-scale validation |

## Status

Core pipeline built and validated at scale (2,218 users), now at the series/season level with structured `content_series_id`/`content_season_id` linkage (permalink-slug matching retired): the adopted model (8 categories, 6 audience clusters, `popularity^0.7 × cluster_rate^0.3 × creator_boost` scoring) hits 60.8% Hit@10 / 74.5% Hit@20 in held-out testing, matching/ahead of a naive popularity baseline. A separate season-continuation mechanism (`season_continuation.py`) guarantees a next-season nudge once a user has genuinely finished the current one. Next: extend structured linkage past the current 500-title tagged sample to the full catalog.
