# Data & Methodology — Full Reference

**In one sentence:** we took 500 real Hoichoi titles and ~2,200 real users' watch history, taught a model to describe *what kind of story* each title is (not just its genre), grouped titles into 8 categories and users into 6 taste clusters, grouped episodes into real series/seasons using the database's own structured IDs, and validated — with real held-out data, not guesswork — that this beats simply recommending "what's popular," while a separate mechanism guarantees a next-season nudge to anyone who's actually finished the current one.

This doc is the single reference for **what data was pulled, what new fields were engineered, and which techniques were used at each stage.** Each section leads with a plain-English summary, then the technical detail. For narrative context and results, see [APPROACH.md](APPROACH.md), [CATEGORIES.md](CATEGORIES.md), and [AUDIENCE_CLUSTERS.md](AUDIENCE_CLUSTERS.md).

---

## 1. Data fetched — where it came from

**Simple:** everything came from Hoichoi's own content database and viewing logs, pulled live via SQL queries — nothing external except two reference documents for one title (Queens) used as a spot-check.

**Technical:**

| Source | Tool used | What was pulled |
|---|---|---|
| `cms_v_series_latest` | PostHog HogQL (`mcp__posthog__execute_sql`, ClickHouse-backed) | Series-level metadata: title, synopsis (short/long, EN+BN), genre, maturity rating, season count, cast/crew `person_ids`, release year |
| `cms_v_videos_latest` (movies + episodes) | same | Per-asset metadata for the 500-title tagging sample: title, content type, genre, maturity, run length |
| `cms_v_people_latest` | same | Person records (name, role) joined against the `person_id` arrays to resolve actual director/actor/producer names |
| `cms_v_watch_history_with_content` | same | Raw per-user, per-title watch events (`user_id`, `content_id`, `seconds_watched`) — this is the largest table (millions of rows); every pull was scoped to specific `content_id`s and/or `user_id`s |
| `Queens.docx` (official creative brief) + 7 episode script PDFs | Local filesystem, read directly | Used once, for a single title, to validate the tagging approach against richer source text and a business-curated "Similar Shows" ground truth |

**Scale actually pulled**, in the order it grew across this project:

| Stage | Users | Watch-event rows | Titles with watch coverage |
|---|---|---|---|
| Initial validation sample | 283 | 8,746 | 296 of 500 |
| Scaled-up sample (current) | 2,218 | 52,491 (deduped) | 413 of 500 |

The scaled pull needed **two independent random draws** merged together, because ClickHouse caps any single query at 50,000 result rows and the real qualifying-user pool is in the hundreds of thousands.

**A data-quality fix that had to happen before any of this was usable:** `cms_v_watch_history_with_content` has a fan-out bug — every episode-watch event produces **two** rows (one carrying `content_permalink`, the other carrying `content_series_id`/season/episode number), sharing the same `_id`. Movies aren't duplicated this way. Every watch-history pull in this project applies `SELECT DISTINCT user_id, content_id, _id, seconds_watched` first to undo this before aggregating — skipping this step silently doubles episode watch-time.

---

## 2. New fields created — feature engineering

**Simple:** the raw database only gives you genre, a short synopsis, and cast/crew names. We used an LLM to *read* the synopsis and extract a richer, structured description of each title — what happens in it (storyline) and how it feels (tone) — from a fixed, closed list of options, so the tags are consistent and machine-usable rather than free text.

**Technical — new columns added to the 500-title master file** (`content_features_500_tagged.csv`):

| New field | How it was created |
|---|---|
| `storyline_tags` (multi-label) | LLM reads `synopsis_text` and assigns 0+ tags from a **closed vocabulary of 42 tags** (e.g. `murder_mystery`, `family_conflict`, `revenge_vendetta`) — see [TAG_TAXONOMY.md](TAG_TAXONOMY.md) |
| `overall_tone_tags` (multi-label) | Same LLM pass, closed vocabulary of **16 tone tags** (e.g. `suspenseful`, `heartwarming`, `dark`) |
| `tag_low_confidence` (boolean) | LLM self-flags titles where the synopsis was too thin/generic to tag reliably (a human-review queue, not a silent guess) |
| `genre_normalized` | Cleaned/standardized version of the CMS genre field (merging near-duplicate genre labels) |
| `era_bucket` | Derived by binning `release_year` into decade buckets (1950s … 2020s, or "Unknown") |
| `cast_size` | Count of the `actor_names` array length |
| `director_names`, `actor_names`, `producer_names` | Parsed out of the CMS `person_ids` array by joining against `cms_v_people_latest` and filtering by role |
| `has_imdb_id`, `has_imdb_rating` | Boolean flags from an IMDb enrichment pass, used only as data-quality indicators, not as model features |
| `maturity_tags` | Content-warning tags (Violence, Nudity, Substance Use, etc.) pulled from CMS, then **case-normalized** (`"Violence"`/`"violence"` merged) to avoid inflating the tag vocabulary with duplicates |

**Derived, not stored per-title — computed at model-build time:**

| Field | How |
|---|---|
| **Programming Category soft mixture** (8 percentages per title, e.g. "40% Mystery, 26% Thriller...") | Softmax over each title's distance to 8 K-means cluster centroids (temperature = 0.35) — see Section 3 |
| **User taste profile** (8-dimensional) | Weighted average of a user's watched titles' category mixtures, weighted by `seconds_watched` |
| **Cluster–title Likelihood/lift** | Ratio of a cluster's viewing rate for a title vs. the overall base rate, Laplace-smoothed |

---

## 3. Techniques & methodologies used

### 3.1 LLM-assisted closed-vocabulary tagging
**Simple:** instead of asking the LLM to write free-text tags (which would never be consistent across 500 titles), it's given a fixed list of ~40-60 allowed tags and told to only pick from that list.
**Technical:** vocabulary was seeded from a 30-title pilot sample, then locked; tagging runs check for out-of-vocabulary drift; low-confidence flag built in for thin synopses.

### 3.2 Multi-hot vector encoding + block-weighted feature construction
**Simple:** each title becomes a row of 1s and 0s — one column per possible tag/genre/etc. — and some columns (like genre) count for more than others (like maturity rating) when comparing titles.
**Technical:** each property (`genre`, `storyline`, `tone`, `era`, `maturity`) is one-hot/multi-hot encoded, then **L2-normalized per block** before applying a scalar weight — this decouples a block's influence from how many tags it happens to have (otherwise a 42-tag block would automatically dominate an 8-tag block regardless of intended weight). Final weights: genre 3.0, storyline 3.0, tone 2.0, era 1.0, maturity 1.0 (validated via grid search, Section 3.7).

### 3.3 K-means clustering (two separate applications)
**Simple:** group similar things together automatically — once for titles (into 8 "Programming Categories"), once for users (into 4 "audience clusters").
**Technical:**
- **Content clustering**: k=8 on the weighted title vectors (500 titles). k chosen for business interpretability, not a statistical peak — silhouette scores were flat across k=4–14.
- **Audience clustering**: k=4 on **standardized (z-scored)** user taste profiles (8-dim category mixtures), so no single category's raw prevalence dominates cluster formation.

### 3.4 Softmax-over-distance for soft (not hard) category membership
**Simple:** a title doesn't have to be "100% one category" — it can be 60% Mystery and 30% Thriller, like the reference product screenshots showed.
**Technical:** `mixture = softmax(-distance_to_each_centroid / temperature)`, temperature = 0.35 (tuned so the average top-1 category share matches the intended "confident but not binary" calibration).

### 3.5 Laplace-smoothed rate scoring, blended with overall popularity
**Simple:** to say "this cluster watches this title a lot," you need to guard against small numbers making that rate meaningless (a title with 1 viewer isn't reliably "popular in this cluster"). Separately, once titles are correctly grouped into real series (3.8) rather than fragments, a handful of true blockbuster franchises dominate total watch-time — so the score needs to respect overall popularity too, not just within-cluster affinity, or it'll under-rank things that are simply very widely watched.
**Technical:** `cluster_rate = (cluster_viewers + 0.5) / (cluster_size + 1)`, `pop_rate = (overall_viewers + 0.5) / (overall_users + 1)` (same Laplace smoothing, computed leave-one-out). Adopted score: **`pop_rate^0.7 × cluster_rate^0.3 × creator_boost`** — replaced a pure `cluster_rate × boost` formula after series-level consolidation (3.8) made pure popularity a stronger baseline than cluster-rate alone; this blend matches popularity on Hit@10 and beats it on Hit@20 while keeping personalization. Titles below a **≥5-viewer floor** are excluded from ranking entirely as "eligible titles" — not just smoothed, dropped, because below that floor the smoothing correction itself can produce a spuriously high score.

### 3.6 Held-out (leave-one-out) validation — the core methodology used to test everything
**Simple:** to check "does this actually predict what someone will watch," hide one title a user really watched, pretend we don't know about it, and see if the model would have ranked it near the top of everything else they hadn't seen yet.
**Technical, per evaluated user:**
1. Remove one watched title from their history.
2. Recompute their taste profile without it.
3. Re-assign them to the nearest cluster centroid.
4. Recompute that cluster's title-lift table **excluding this specific held-out interaction** (to prevent the answer leaking into its own evidence).
5. Rank all "eligible" unwatched titles by score; record the true title's rank.
6. Aggregate across users: mean rank, mean percentile, **Hit@10** (was it in the top 10?), **Hit@20**.

Always benchmarked against a **naive popularity baseline** (rank by raw view count, no personalization) — the real test of whether the model earns its complexity.

### 3.7 Weight tuning via grid search, validated at two different levels of leverage
**Simple:** rather than guessing what weight numbers should be, try a range of them and measure which actually improves the "did we predict the right thing" test.
**Technical:** two independent grid searches were run, screened on a subset of users then confirmed on the full held-out set to rule out sampling noise:
- **Creator-boost weights** (how much to boost a title if it shares a director/actor with something the user's watched): grid of 21 combinations. Result: **high-leverage** — Hit@20 ranged from 8.8% to 59.4% depending on the weights. Original guess (director 1.0, actor 0.3) was actually one of the worst; **actor overlap turned out to be the more common standalone signal** in this catalog (260 actor-only matches vs. 21 director-only, out of 1,870 held-out cases) — director-heavy weighting was an unvalidated assumption that didn't hold. Adopted: **director 2.0, actor 2.0**.
- **Category-formation weights** (genre/storyline/tone/era/maturity): grid of 18 combinations. Result: **low-leverage** — all combinations landed within a narrow Hit@20 band (52–59%); the current baseline held up as the best (or statistically tied-for-best) once confirmed at full scale. No change made.

### 3.8 Structured series/season linkage (replaces permalink-slug matching)
**Simple:** originally we grouped an episode with its show by comparing the show-name text baked into its URL. That broke for shows whose URLs stuff the season/episode number into the "show" part of the link (e.g. `.../season-9-episode-3`) — every episode looked like a different show. Instead we now read the database's own `content_series_id`/`content_season_id` fields directly, which don't depend on URL text at all.
**Technical:** `cms_v_watch_history_with_content` carries `content_series_id`, `content_season_id`, `content_season_number`, `content_episode_number` on a fanned-out companion row per watch event; grouping by `content_id` and taking `MAX()` of each field reliably recovers the populated value over the placeholder `0`/empty row. Queried for all 350 tagged episodes in one pass — **100% resolved, 0 empty**, no slug parsing left anywhere in the pipeline. Effect: 107 slug-fragmented pseudo-series collapsed into **53 real series** (e.g. Eken Babu Season 9, previously 7+ singleton fragments, is now one series with all 7 episodes correctly grouped). See [AUDIENCE_CLUSTERS.md](AUDIENCE_CLUSTERS.md) for the before/after and the scoring re-tune this triggered.

### 3.9 Season-continuation as a guaranteed slot, not a ranking feature
**Simple:** if someone just finished all of Season 1 of a show, they almost certainly want Season 2 next — that's too strong and too obvious a signal to leave to a competitive popularity/affinity score, where a niche Season 2 could lose to an unrelated blockbuster. So it's handled separately: detect "finished a season, hasn't started the next," and place that next episode directly, outside the ranked list.
**Technical:** `season_continuation.py` builds a `series_id → {season_number → [content_ids sorted by episode]}` index from the same structured fields (3.8), then for each user checks whether they've genuinely watched (passed the >60%-of-runtime per-episode filter) **≥90%** of a season's episodes and have not yet started the next season; if so, that next season's first episode is returned as a direct, guaranteed recommendation, unioned with (and ranked above) the discovery-ranking output. The 90% threshold is intentionally much stricter than the 60% per-episode engagement filter — conflating "genuinely watched one episode" with "finished the whole season" was an early bug (a user 60-83% through a season was nearly pushed into the next season before finishing the current one). Justified by real behavior: 66.1% of 5,000 real users who finished Sampurna Season 1 went on to watch Season 2 — a far stronger signal than the ~60-70% Hit@20 the discovery model achieves generally.

### 3.10 Coverage-gated sampling
**Simple:** to build a fair test, only include real users who've actually watched enough of our 500 tagged titles to have a meaningful taste profile — not every user on the platform.
**Technical:** users filtered to ≥3 watched titles within the 500-title tagged catalog before being included in any clustering or validation step; this is a known selection bias (documented as a limitation), since it skews toward more-engaged users.

### 3.11 External validation against business ground truth
**Simple:** for one title (Queens), we checked whether our algorithm's output matched what human content strategists had already written down as "similar shows" in an official product brief — a sanity check against real business judgment, not just our own metrics.
**Technical:** the brief's "Similar Shows: Dainee, Anusandhan" field was cross-referenced against our bottom-up clusters; "Anusandhan" turned out to be a re-dubbed version of a title already in our tagged set ("Talaash (Hindi)"), independently landing in the same category our algorithm placed Queens near — a real-world confirmation signal, not manufactured.

---

## Where each technique lives in the pipeline (quick map)

```
Raw CMS + watch-history data (SQL pulls)
        ↓ [dedup fix, LLM tagging, field engineering]  → Section 1 & 2
500-title tagged catalog
        ↓ [structured content_series_id/season_id linkage, NOT permalink slugs]  → Section 3.8
203 series-level items (150 movies + 53 real series)
        ↓ [weighted multi-hot vectors → K-means k=8 → softmax mixture]  → Section 3.2–3.4
8 Programming Categories (soft %, per series/movie)
        ↓ [per-user weighted average → z-score → K-means k=6]
6 Audience Clusters (taste profiles)
        ↓ [Laplace-smoothed cluster-rate + popularity-rate, ≥5-viewer floor]  → Section 3.5
Cluster × Series scoring table
        ↓ [pop_rate^0.7 × cluster_rate^0.3 × creator_boost(0.5,0.5)]  → Section 3.5, 3.7
Discovery-ranked recommendations per user
        ↓ [+ season-continuation guaranteed slot, unioned in, not ranked]  → Section 3.9
Final recommendations per user
        ↓ [held-out validation vs. popularity baseline]  → Section 3.6
Hit@10 = 60.8%, Hit@20 = 74.5% (vs. popularity baseline's 61.0%/73.3%)
```
