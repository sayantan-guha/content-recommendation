# content_recommendation

## Overview

A content recommendation system for Hoichoi. Titles are tagged by storyline/tone and grouped into interpretable Programming Categories (used for content-similarity scoring and cold-start fallback); users are recommended titles primarily via item-item collaborative filtering ("users who watched X also watched Y"), with content-similarity and popularity as fallback tiers for cold-start titles/users — validated end-to-end against held-out real watch history. See "Status" below for how the model evolved from the original cluster-based approach to this.

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
| [docs/DATA_AND_METHODOLOGY.md](docs/DATA_AND_METHODOLOGY.md) | **Start here** — has an up-to-date current-production summary at the top, then the full history of data pulled/engineered/tested (including the retired cluster model) below |
| [docs/APPROACH.md](docs/APPROACH.md) | The original step-by-step design rationale (content → category, audience → title) — audience-clustering steps are retired, tagging/category steps are still live for cold-start fallback; see the status note at the top |
| [docs/SCHEMA.md](docs/SCHEMA.md) | Database tables, relationships, known data-quality quirks, and how they're actually queried now (PostHog HogQL) |
| [docs/TAG_TAXONOMY.md](docs/TAG_TAXONOMY.md) | The closed vocabulary of storyline/tone tags |
| [docs/CATEGORIES.md](docs/CATEGORIES.md) | The 8 Programming Categories — still computed and used today, just for cold-start-fallback content similarity instead of audience-cluster scoring |
| [docs/AUDIENCE_CLUSTERS.md](docs/AUDIENCE_CLUSTERS.md) | **Retired from production** — the audience-cluster mechanism and its validation history, kept as historical record; see the status note at the top |

## Code

| File | What it does |
|---|---|
| [src/pipeline_series_structured.py](src/pipeline_series_structured.py) | **Historical/reference, not production** — original cluster-based model on the 500-title tagged sample: content tagging → category clustering → audience clustering → series/season structured linkage → discovery-ranked recommendations |
| [src/pipeline_full_catalog.py](src/pipeline_full_catalog.py) | **Historical/reference, not production** — same cluster-based model, scaled to the full 775-title catalog. Superseded by `src/recommender.py`'s CF-based scorer |
| [src/season_continuation.py](src/season_continuation.py) | Next-season recommender + single-list merge helper, kept separate from (but combinable with) the discovery ranking output |
| [src/recommender.py](src/recommender.py) | Production model: three-tier scoring (item-item CF → content-similarity cold-start fallback → popularity fallback) plus the movie/series type-affinity quota — imported by `backend/app.py` |
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

**`item-item-cf` and `scale-user-sample-4800` merged into `main`, and CF is now wired into serving** (`src/recommender.py`), replacing the cluster-based scorer entirely. Production (`backend/app.py`/`ui/app.py`) fits on the 5,000-user watch sample (4,210 eval users, up from 1,485) and scores with a three-tier scheme instead of one blended formula (each tier only activates when the one above it has nothing to work with — blending them was tested and rejected, see above):

1. **Item-item CF (primary)** — for any user with >=1 watched eligible title, score candidates by summed cosine-similarity to everything they've watched.
2. **Content-similarity cold-start fallback** (unchanged from the cluster-model era) — titles with <5 viewers have no CF co-occurrence signal for *any* user, so they're scored by cosine similarity to the user's content-tag profile instead. Still reserves ~10% of every slate.
3. **Popularity fallback** — for a user with 0 watched eligible titles, CF has nothing to sum similarities over, so candidates are ranked by raw popularity. (Verified directly: a synthetic zero-history user's top-20 matches the raw popularity ranking exactly.)

The movie/series type-affinity quota still applies on top, unchanged — it's agnostic to which tier produced the ranked list. The cluster-based model (8 categories, 6 audience clusters) is fully retired from serving; `src/pipeline_full_catalog.py` and `src/pipeline_series_structured.py` remain as reference/history, not production code paths.

**Cold-start fallback blending added** (`src/recommender.py`'s `cold_start_candidates`): titles with fewer than 5 distinct viewers are excluded from CF scoring entirely (no reliable co-occurrence signal to rank them), so brand-new or low-viewership titles could never surface. The fallback scores these titles by cosine similarity between the user's taste profile and the title's content-category mixture (the same tag-derived vector used for content clustering, available the moment a title is tagged — no watch history needed), and reserves ~10% of each recommendation slate (2 of 20 slots) for the best-matching cold titles. Verified with `src/eval_recommender.py`: cold-title exposure goes from **0% of users → 100% of users** getting at least one low-data title in their top-20, with the warm-item ranking metrics below computed on the unchanged warm-only path (so this doesn't cannibalize the tested ranking quality).

**NDCG@k and MRR added alongside Hit@10/20** (`src/eval_recommender.py`), since hit-rate alone treats a held-out title landing at rank 1 the same as rank 20 — no signal on ranking quality within the top-N. On the now-live CF scorer, 5,000-user sample, 4,210 eval users: **mean_rank=72.4, Hit@10=29.6%, Hit@20=40.0%, NDCG@10=0.181, NDCG@20=0.207, MRR=0.162** — matching `pipeline_item_cf.py`'s standalone benchmark, confirming the wiring is correct, and both NDCG@10 and MRR are roughly double the retired cluster model's (0.098 and 0.091 respectively), meaning CF doesn't just recover more titles but ranks them meaningfully closer to the top.

**Segmented evaluation by task** (`src/eval_recommender.py`): rather than one blended metric, evaluation is now split by what's actually being tested — **warm-item recovery** (can a title the user already watched, with >=5 viewers overall, be recovered from their remaining history? — the standard LOO test) vs. **cold-start exposure** (does the fallback actually get low-data titles in front of users at all? — LOO can't test "recovery" of a title only 1-4 people watched, so this measures exposure rate instead, with/without the fallback).

**Watch-history recency, tested three separate ways, rejected all three:** (1) profile-weighting by a decayed timestamp per watch (both an unverified `_id`-decode proxy and, later, real `created_at` pulled via PostHog) showed no reliable Hit@10/20 movement on the cluster model; (2) the same real-timestamp weighting applied to CF scoring showed no benefit either, worse if anything (Hit@10 43.9%→42.5% on a 500-user real-timestamp sample); (3) restricting CF scoring to only a user's K most-recent watches (instead of full history) **monotonically hurt** — the smaller the recency window, the worse the result (Hit@10 dropped from 43.9% at full history to 30.9% using only the 3 most recent watches). Conclusion: don't pursue watch-history recency in any form for this model; CF benefits from as much history as it can get, not less.

**Completion-rate filtering added** (`src/recommender.py`'s `load_audience_model`): pulled real `content_run_length_secs` + `created_at` via PostHog (`cms_v_watch_history_with_content`) for a sample of users, and computed `completion_pct = seconds_watched / content_run_length_secs`. Validated on a 500-user real-timestamp sample (same LOO methodology): dropping watches confirmed below **60% completion** from the CF matrix improved Hit@10 43.9%→52.7% (+8.8pp), Hit@20 56.6%→67.3% (+10.7pp), NDCG@10 0.289→0.379. Initially wired in as a partial overlay on top of the 5,000-user sample (only ~22% coverage), which diluted the effect to near-noise at production scale — see below for why that was superseded.

**Production data source switched from the 5,000-user sample to the 1,100-user completion sample, for consistency.** Rather than overlaying completion data onto a larger sample it only partially covers, `load_audience_model` now reads `data/user_watch_completion_sample_1100.csv` as the sole watch source, with the ≥60%-completion filter applied to every row (100% coverage, not 22%). **The 5,000-user sample is stashed, not deleted** — `data/user_title_watch_sample_5000.csv` remains in the repo for a possible future re-expansion once completion data is pulled for more users, but nothing in the current pipeline reads it. Trade-off: fewer eval users (765, down from 3,928) and a smaller eligible-item pool (384 titles vs. 634, both at the same ≥5-viewer floor), but Hit@10/20 rise to **45.6%/60.0%** (from 29.6%/40.0% pre-completion-filter) — much closer to the validated 500-user result, as expected now that every user in the pool actually has the completion signal applied.

**Storyline/tone tag vocabulary rebuilt from scratch (v3)** (`docs/TAG_TAXONOMY.md`, `data/content_features_full_tagged.csv`): the original 42-storyline/16-tone vocabulary was designed from a 30-title pilot sample; a v2 patch added 8 tags for genres the pilot missed (Sci-Fi, Fantasy, Musical, Sports, Mythological/Devotional). This rebuild replaced both entirely — a new 65-storyline/16-tone vocabulary designed from a ~130-title stratified sample spanning the full 775-title catalog, then all 775 titles re-tagged against it from scratch (12 parallel tagging passes). 100% vocabulary compliance verified, every tag used at least once, `tag_low_confidence` at 87/775 (genuine thin-content cases, comparable to prior versions). CF metrics are unaffected (CF never touches these tags). The real, verified effect is on the content-similarity fallback specifically: e.g. "Golondaaj" (a sports underdog film) previously top-matched with unrelated Documentary/Social Drama titles at cosine similarity=1.0 under the old vocabulary; under the rebuild it correctly surfaces "East Bengaler Chhele" (another sports underdog film) instead. Documented, not fixed by this rebuild: the K=8 mixture compression still caps how much any vocabulary can sharpen *mainstream*-genre similarity (see taxonomy doc for detail).

**Cluster-based model re-tested with the new tags, weight sweep, and K sweep — confirmed the architecture itself is the ceiling, not the tags.** With CF deliberately set aside, the retired cluster scorer was re-run on the current 1,100-user completion-filtered data across 6 feature-weight combinations and 7 K_CONTENT/K_USER combinations. Every configuration landed in the same narrow 20.7–22.3% Hit@10 band — notably, a genre-only weighting (storyline/tone effectively zeroed out) performed as well as any tag-heavy configuration, and a direct v2-vs-v3 tag comparison at identical settings was statistically indistinguishable. Confirms the root cause already diagnosed earlier (audience-cluster concentration correlating with global popularity) rather than tag quality or cluster count. For scale: CF on this identical data scores Hit@10=45.6%/Hit@20=60.0%, roughly double the cluster model's best result (22.3%/33.3%) regardless of tuning.

**Creator (director/actor) overlap boost: rejected for CF, adopted for the cold-start fallback.** Re-tested creator boost on top of CF on the current data (previously rejected on the older 5,000-user sample) — same result holds: monotonically worse the stronger it's applied (Hit@10 45.6%→33.3% at strength 0.5). But tested in isolation on the content-similarity cold-start fallback specifically (never tried there before), it roughly **doubles** the fallback's own Hit@10/20 (3.5%→7.7%, 7.1%→13.0% at strength 0.3) — the fallback's pure content-tag similarity is comparatively weak on its own (many mainstream titles saturate near cosine-similarity 1.0, per the tag-rebuild finding above), so a concrete signal like shared cast/crew gives it something to discriminate on that content tags alone don't reliably capture. Wired into `cold_start_candidates` at strength 0.3/0.3 (`COLD_START_DIR_W`/`COLD_START_ACTOR_W`); the primary CF scorer is untouched.

**Edge cases in `recommend_for_user`, found by manually tracing real users whose watch history skewed toward eras/genres with little co-viewing data** (some gaps only show up by hand-testing individual real profiles, not aggregate metrics):

1. **"Low watch history" was tried, tested, and REMOVED — it isn't actually a real edge case.** Initially added a `MIN_WATCHED_FOR_CF` count threshold on the assumption that few watched titles means CF's co-occurrence sum is too thin to trust. Then explicitly tested this breakeven point (`src/experiments/cf_vs_content_breakeven_v2.py`, same LOO methodology, bucketed by number of remaining watched titles, run *without* the standard eval's `>=4 total watches` filter so 1-2-title users were actually visible): **CF beats content-similarity at every single bucket, including users with just 1 remaining watched title** (Hit@10 35.4% vs. 10.4%). There is no watch-count breakeven point — the threshold was an unvalidated assumption and would have actively hurt users with few-but-informative watches, so it was removed.
2. **No similar users found** (`CF_SIGNAL_EPSILON`) — the real edge case isn't count, it's whether CF found *any* co-viewer signal at all. A user can have several watched titles and still get zero real co-viewers for all of them (verified directly on a real user: 3 of 4 watched titles had *zero* other viewers in our sample). CF's ranking in that case is tie-broken noise, not a real signal — silently returning modern titles regardless of the user's actual taste. Now detected by checking `scores.max() <= CF_SIGNAL_EPSILON` and routed to content-similarity ranking instead. Verified this is correctly narrow: a case with even thin-but-real signal (2 co-viewers) does *not* trigger it, matching the breakeven test showing thin CF signal still beats content-similarity.
3. **Candidate pool exhaustion** (found by code review, not by a specific user) — if a power user has watched nearly every eligible title, or the cold-start pool is empty, the slate could silently come up short of `top_n`. Now backfilled from the full unwatched catalog by content-similarity (or popularity, if there's no profile at all) so a full `top_n` is always returned when the catalog has enough unwatched titles.

All three route through the same new `content_based_ranking` helper (refactored out of `cold_start_candidates`, which now just calls it) — the eligibility floor and warm/cold split are unaffected; only *which scorer* produces the ranked list changes. Verified no regression: `eval_recommender.py` numbers are unchanged (real production users overwhelmingly have both enough history and real CF signal, so these paths are true edge cases, not the common path).
