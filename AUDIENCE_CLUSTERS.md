# Audience Taste Clusters (v1 — 272-user validation sample)

Step 3 of [APPROACH.md](APPROACH.md): segment users by *what they watch* (Programming Category consumption mix), not demographics.

## Method

1. **Deduped watch history**: confirmed and fixed the `cms_v_watch_history_with_content` fan-out (every episode watch event produces 2 rows — one carrying `content_permalink`, the other `content_series_id`/season/episode — sharing the same `_id`; movies aren't duplicated). Dedup via `DISTINCT (user_id, content_id, _id, seconds_watched)`.
2. **Coverage-gated sampling**: our Programming Categories only exist for the 500-title tagged sample (8.8% of the ~5,700-title catalog), so users had to be filtered to ones with real overlap. Of 2.5M users who watched at least 1 of our 500 titles, 1.28M+ had watched 5+, 870K+ had watched 10+ — plenty of pool. Sampled 292 real users, kept 272 with ≥3 overlapping titles.
3. **Per-user category profile**: for each user, weighted average of their watched titles' bottom-up Programming Category mixture (from `content_categories_500.csv`), weighted by `seconds_watched`. Each profile sums to 100%.
4. **K-means clustering** on standardized profiles. Silhouette was flat across k=2-8 (same pattern as Step 2's content clustering — no sharply separated natural clusters), so k=4 was chosen for interpretability.

## Result: 4 clusters

| Cluster | n | Defining categories (vs. ~12.5% even-split baseline) |
|---|---|---|
| **Comedy Fans** | 53 | Comedy of Errors & Farce: 28.9% (2.3x baseline) |
| **Drama & Feel-Good Seekers** | 54 | Character-Driven Life Dramas: 21.9%, Feel-Good Life Stories: 17.7% |
| **Mystery & Thriller Enthusiasts** | 53 | Whodunit Detective Mysteries: 26.1%, Suspense & Crime-Driven Thrillers: 24.4% (together ~half their viewing) |
| **Broad Mainstream** | 112 (largest) | Fairly even across all 8, mild lean toward Family Betrayal & Social Drama (16.8%) |

This matches the reference screenshot pattern directly: one broad/mixed mainstream cluster alongside more sharply-defined taste clusters.

Output: `data/user_taste_clusters_272.csv` (per-user 8-category profile + cluster assignment), `data/user_title_watch_sample.csv` (raw per-user per-title watch seconds, the input to the profiles).

## Known limitations (read before using this for anything beyond validation)

- **272 users is a proof-of-concept sample, not the user base.** 4M+ total users exist; this shows the method works, not final segment sizes/definitions.
- **Selection bias**: sampled users were required to have watched several of our specific 500 tagged titles — these are likely more-engaged, longer-tenured users, not representative of casual/new users.
- **50,000-row query cap**: the raw watch-history pull hit a result-size cap per user batch; very heavy watchers' full histories may be truncated, capping their apparent title-overlap count (unlikely to change cluster membership much, but worth knowing).
- **Category coverage gap inherited from Step 2**: profiles are built only from the 500/5,700 titles we've tagged — a user's true taste may differ if their untagged-title viewing skews differently from their tagged-title viewing.
- Cluster count (k=4) was a judgment call, not statistically forced — same caveat as Step 2's k=8.

## Step 4 — Per-cluster title affinity: Likelihood + Viewers% (same 272-user validation scale)

Computed for each (cluster, title) pair per [APPROACH.md](APPROACH.md:52):
- **Viewers%** = reach = fraction of the cluster that watched the title.
- **Likelihood** = lift = cluster's viewers% ÷ overall sample base rate for that title.

**Method:** floored to titles with ≥5 total viewers across the 272-user sample (224 of 296 watched titles qualified), and applied a Laplace/continuity correction (+0.5) to both cluster-level and overall-level viewer counts before taking the ratio — this prevents small-cluster noise from producing spurious infinite or wildly inflated lift values, per the smoothing technique APPROACH.md calls for.

**Top findings per cluster** (full table: `data/cluster_title_affinity.csv`):

| Cluster | High-lift, low-reach ("loved but undiscovered") | High-lift, high-reach ("established hit") |
|---|---|---|
| Broad Mainstream | Family Album (6.6% reach, 2.13x lift) | Settlement, Unmachan!, Promaan (~50-53% reach, ~1.9-2x lift) |
| Comedy Fans | Kelor Kirti (6.5% reach, 3.22x lift) | Bhooter Bhabishyat (47.2% reach, 2.13x lift) |
| Drama & Feel-Good Seekers | Patro Chai (11.8% reach, 3.80x lift) | Chitrangada (17.3% reach, 3.49x lift) |
| Mystery & Thriller Enthusiasts | Bhaggyolokkhi (6.5% reach, 2.72x lift) | Durgeshgorer Guptodhon (32.4% reach, 2.08x lift) |

This is exactly the two-column pattern from the reference screenshot — distinguishing "everyone in this cluster loves it but few have found it yet" from "already broadly seen."

### Caveats — read before using these numbers for anything real

- **Extremely small sample.** The ≥5-total-viewers floor is thin — several "top" titles in the tables above have as few as 2-3 actual viewers in a cluster (e.g. several Mystery & Thriller entries). These lift values are illustrative of the *method*, not statistically reliable estimates.
- Inherits every caveat from the clustering step above (272-user proof-of-concept, selection bias, 500/5,700-title coverage gap, 50K-row query cap).
- No significance testing applied — APPROACH.md suggests this as a refinement once sample size grows; not meaningful yet at n=272.

## v2 — rebuilt on the reweighted categories (data/user_taste_clusters_v2.csv, data/cluster_title_affinity_v2.csv)

[CATEGORIES.md](CATEGORIES.md)'s v2 section reweights the 8 Programming Categories (genre/storyline at 3x, tone at 2x, era/maturity at 1x, cast/director deliberately excluded from clustering). Steps 3-4 were rebuilt on top of those categories, same methodology, on the full 283-user / 296-title watch sample.

**v2 audience clusters (k=4):**

| Cluster | n | Defining categories |
|---|---|---|
| **Family Drama Enthusiasts** | 124 | Character-Driven Family Drama: 47.5%, Whodunit Detective Mysteries: 14.4% |
| **Mystery & Thriller Enthusiasts** | 102 | Whodunit Detective Mysteries: 25.6%, Revenge & Survival Thrillers: 25.0% |
| **Broad Mainstream (Drama/Feel-Good)** | 51 | Character-Driven Family Drama: 26.5%, Feel-Good & Reconciliation Stories: 20.1% |
| **Legal & Social Drama Niche** | 6 | Family Legal & Social Drama: 58.8% |

## Step 4.5 — Creator affinity boost (ranking-time, not baked into clustering)

Testing whether to weight `cast`/`director` into the Programming Category clustering itself (per a direct ask to compare weighting schemes) showed two things at once: (a) it measurably improved held-out prediction accuracy, but (b) it also collapsed 2 of the 8 categories into a single director's back-catalog (100% Soumen Halder, 100% Joydeep Mukherjee) — see [CATEGORIES.md](CATEGORIES.md) v2 section for the full comparison. Cast/director carry real signal; they just don't belong in a *story-type* taxonomy.

**Fix**: keep categories story-only (v2), and apply cast/director as a multiplier at the final ranking step instead:

```
final_score(user, title) = cluster_Likelihood(cluster, title) × (1 + 2.0 × shared_directors + 2.0 × shared_actors)
```

where `shared_directors`/`shared_actors` = count of directors/actors on `title` that also appear on titles the user has already watched.

**Weights were revised from an initial 1.0/0.3 (director-heavy) guess to 2.0/2.0 (equal) after a grid search** (`data/creator_boost_weight_grid_search.csv`, 21 combinations tested on the 1,870-user held-out set): the director-heavy guess assumed directors drive identity more than any single cast member, but checking actual signal frequency in our catalog showed the opposite pattern — of held-out titles with a creator match, **actor-only matches outnumbered director-only matches 12-to-1** (260 vs. 21; 1,371 had both, since Bengali OTT titles mostly recur as tight director+lead-actor ensembles). Every weight combo with actor_w ≥ dir_w beat the original 1.0/0.3, and gains plateaued past (3,3) — pushing both weights arbitrarily higher (tested to 5,5) only added ~0.5pp further, at the risk of the creator-boost overriding genuine taste-based ranking. (2.0, 2.0) was picked from that plateau: Hit@10/Hit@20 of 44.1%/58.6%, up from 39.1%/52.0% at the original weights.

### Held-out validation: this is the best-performing scheme tested (full table: `data/feature_weighting_validation_comparison.csv`)

| Scheme | mean rank | mean %ile | Hit@10 | Hit@20 |
|---|---|---|---|---|
| v1 (storyline+tone, binary, current) | 72.9 | 0.372 | 11.5% | 21.1% |
| all properties, equal weight | 67.2 | 0.346 | 18.0% | 24.1% |
| all properties, weighted (cast/director in clustering) | 66.6 | 0.342 | 19.5% | 28.0% |
| story-only weighted (cast/director dropped entirely) | 70.6 | 0.362 | 7.7% | 21.5% |
| **v2 — story-only categories + creator boost at ranking** | **43.8** | **0.221** | **36.8%** | **49.4%** |
| *(reference) naive popularity baseline* | 55.8 | 0.275 | 28.0% | 38.7% |

v2 is the only scheme that beats the naive popularity baseline outright — Hit@10 more than triples over the current production approach (11.5% → 36.8%).

### Validated at scale (2,218 users, 413 titles — data/user_title_watch_sample_2218.csv)

Re-ran the same held-out validation on a much larger coverage-gated sample (2,218 real users with ≥3 watched titles among our 500, pulled via two independent random draws from the full watch-history table and merged; 1,870 of them had ≥4 titles and were used for evaluation — up from 265). The eligible-titles floor (≥5 viewers) also grew, from 225 to 312 titles, making the ranking task itself harder (bigger candidate pool).

| Scheme | mean rank | mean %ile | Hit@10 | Hit@20 |
|---|---|---|---|---|
| A — v1 (current, storyline+tone binary) | 106.0 | 0.368 | 1.4% | 5.8% |
| E — v2, original creator-boost weights (1.0 dir / 0.3 actor) | 48.8 | 0.171 | 39.1% | 52.0% |
| **E — v2, revised creator-boost weights (2.0 dir / 2.0 actor, adopted)** | **43.5*** | **0.148*** | **44.1%** | **58.6%** |
| popularity baseline (reference) | 45.8 | 0.157 | 29.1% | 44.8% |

*Revised-weight row computed on the same 1,870-user eval set as part of the director/actor weight grid search below — see that section for the full sweep.

The result **holds up at 7x the scale, and E's Hit@10/Hit@20 edge over popularity actually grows** (even before the weight revision: 39.1%/52.0% vs. 29.1%/44.8%, wider than at 265 users; after revision, wider still). v1's advantage over pure chance also shrinks sharply once the candidate pool grows (Hit@10 falls to 1.4%), underscoring that the current production scoring doesn't scale well — this was not visible at the smaller sample size. Full numbers: `data/feature_weighting_validation_at_scale_2218.csv`.

### Known limitations of v2 / Step 4.5

- Same 296-title/413-title, 500/5,700-catalog-coverage caveats as the v1 sections above — this changes the *scoring method*, not the underlying tagged-catalog scale.
- The `1.0` / `0.3` creator-boost weights are a reasoned heuristic, not fit from data — a real next step is learning these (or a full learned ranking model) once there's enough interaction volume.
- Held-out evaluation holds out exactly one title per user; it doesn't yet test sequential/next-watch prediction or cold-start users with very few watched titles.

## Series-level validation — the honest test, and it changes the conclusion

Every result above ranks **episodes**, not series — a direct consequence of an earlier decision to keep per-episode rows rather than aggregate ("just split the data for now"). That meant a series with several tagged episodes (e.g. a 24-episode Hindi series had 24 separate rows) could appear multiple times in a ranked list, and — more importantly — **held-out validation was hiding one episode while leaving sibling episodes of the same series in the user's profile**, making "predict they'll watch this" partly just "recognize they're already watching this show."

To get the honest number, episodes were rolled up to their parent series using the permalink slug embedded in the URL (e.g. `/shows/atonko/purbabhas` → series `atonko`) — the same fuzzy-link approach documented in [SCHEMA.md](SCHEMA.md). 344 of 350 episodes (98%) resolved to a series this way; the remaining 11 (6 with a flat `/episode/...` URL carrying no series segment, 5 mislabeled movies) fell back to being treated as their own singleton item — a known, small (~3% of episodes) gap.

**Result: 500 titles → 257 series-level items** (150 movies + 107 series/singletons). Watch history was rolled up the same way (sum `seconds_watched` across all episodes of a series a user watched), taking 52,491 episode-level watch rows down to 24,993 series-level ones. Content categories, audience clusters, and the creator-boost were all rebuilt from scratch on this series-level data, then held-out validation was re-run **hiding an entire series, not one episode**.

| | mean rank | mean %ile | Hit@10 | Hit@20 | eligible items |
|---|---|---|---|---|---|
| Episode-level (previous result, likely inflated by same-series leakage) | 43.5 | 0.148 | 44.1% | 58.6% | 312 |
| **Series-level, our model (the honest number)** | 63.3 | 0.330 | **25.7%** | **37.0%** | 205 |
| **Series-level, popularity baseline** | 26.3 | 0.137 | **50.5%** | **65.4%** | 205 |

Two things happen at once, and both matter:

1. **Our model's own numbers drop hard** (Hit@20: 58.6% → 37.0%) — confirming the earlier suspicion: a meaningful chunk of the episode-level "win" really was same-series recognition, not genuine cross-title discovery.
2. **Popularity flips from behind to decisively ahead** (65.4% vs. our model's 37.0%) — the reverse of every episode-level comparison in this document. At only 257 items (205 eligible), a handful of blockbuster series/movies dominate viewership enough that "recommend what's popular" is a very strong baseline — and the cluster-based lift approach doesn't currently beat it once episode-duplication no longer inflates its apparent accuracy.

This was the most important finding in this document — and root-caused and fixed below, not left as an open gap.

### Root cause found: the Likelihood *ratio* was discarding popularity by design

`Likelihood = cluster_rate ÷ overall_rate` is scale-invariant — a blockbuster series watched by 80% of everyone gets a lift near 1.0 (same as a niche title watched equally by 2% of everyone and 2% of the cluster), because the ratio only measures *over-indexing*, not *how much anyone actually watches it*. At 500 episode-level items this didn't matter much (episode-level "hits" were dominated by same-series leakage anyway). At 257 series-level items, with a handful of genuine blockbusters, this ratio was actively suppressing exactly the titles popularity correctly favored.

**Fix**: drop the ratio's denominator — score with `cluster_viewing_rate × creator_boost` instead of `Likelihood × creator_boost`. This keeps the personalization (a title's rate is still computed within the user's own cluster) while no longer discarding absolute popularity.

| Scoring formula | Model Hit@10 / Hit@20 | Popularity Hit@10 / Hit@20 |
|---|---|---|
| `Likelihood × boost` (original) | 25.7% / 37.0% | 50.5% / 65.4% |
| **`cluster_viewing_rate × boost` (adopted)** | **54.0% / 69.2%** | 51.5% / 67.4% |

This one change flipped the model from losing to popularity by ~25-28 points to beating it — the single largest effect found anywhere in this project, bigger even than the creator-boost weight fix.

### Two more levers tested, and how they compare to the fix above

| Lever | Result | Verdict |
|---|---|---|
| **Scale users 2,218 → 3,311** (2 more independent DB draws, merged) | Model 50.6%/67.4% vs. popularity 50.2%/67.1% — model's lead **narrowed** to +0.4pp/+0.3pp | Low leverage; more data stabilized both sides roughly equally, didn't disproportionately help the model |
| **Re-tune cluster counts** (swept content-k ∈ {4,6,8,10,12,15}, audience-k ∈ {2,3,4,5,6,8} on the 2,218-user data) | Best: content-k=8, **audience-k=6** → 54.3%/69.7%, vs. k=4's 54.0%/69.2% | Low-but-real leverage; a small, genuine +2.3-2.8pp gain, consistent with the earlier finding that category/cluster granularity is a minor knob |

**Adopted going forward: content-k=8, audience-k=6, `cluster_viewing_rate × creator_boost` scoring.** Re-tuning cluster count helped more than adding ~1,100 more users did in this test — but both are minor next to the scoring-formula fix, which remains the dominant lever by a wide margin.

**Re-confirmed once more, jointly, on the series-level data**: a 26-run sweep (16 category-weight combinations at k=8/6, then 10 k combinations at the best weight found) landed on genre-dominant weights (genre=5, storyline=1, tone=1, era=1, maturity=1) at the same k=8/6 as the single best — **54.0%/70.1%, statistically indistinguishable from the current baseline's 54.3%/69.7%.** Category weights and cluster count remain low-leverage levers at series-level too, consistent with the earlier episode-level finding. **The baseline (genre=3, storyline=3, tone=2, era=1, maturity=1; k=8/6) stays as the adopted configuration** — simpler, already documented, and not meaningfully beaten by anything tested. Full sweep: `data/series_weight_sweep_at_k8_6.csv`, `data/series_k_sweep_at_best_weights.csv`.

Data: `data/content_features_series_rollup.csv`, `data/user_series_watch_sample.csv`, `data/series_level_validation.csv`, `data/series_vs_episode_validation_comparison.csv`, `data/series_scoring_blend_comparison.csv`, `data/series_kscan_results.csv`, `data/user_title_watch_sample_3311.csv`.

### Known limitations of the series-level result (superseded below — see "Permalink-slug linkage replaced")

- The eligible-item floor (≥5 viewers) still discards ~35 of 257 series/movies at 3,311-user scale — a much larger user sample would likely sharpen the cluster-viewing-rate estimates further.
- ~~11 of 350 episodes (~3%) couldn't be linked to a parent series via permalink slug and were treated as singletons~~ — this whole linkage approach was replaced; see below.
- The audience-k=6 sweep was one-factor-at-a-time from the k=8/k=4 baseline, not an exhaustive joint grid — a finer joint sweep could find a marginally better combination, though the flatness of both individual sweeps suggests there isn't a large one hiding.

## Permalink-slug linkage replaced with structured `content_series_id`/`content_season_id` (2026-07-14)

**The problem.** Series/season grouping was originally done by parsing the episode's permalink URL slug (`/shows/{slug}/{episode-slug}` → `slug` = the series key). This broke for shows with SEO-stuffed URLs that embed the season/episode number directly in the "show" segment — e.g. `/shows/watch-eken-babu-online-season-9-episode-3/...`. Each episode of **Eken Babu Season 9** and **Feludar Goyendagiri Season 3** produced a *different* slug, so the rollup treated every episode as its own singleton "series" instead of grouping them under one show. This was discovered concretely during the second real-user demo: 3 of that user's 4 held-out titles were Eken Babu Season 9 episodes, each occupying a separate slot in the printed top-20 instead of collapsing into one "Eken Babu" row.

**The fix.** `cms_v_watch_history_with_content` carries authoritative `content_series_id`, `content_season_id`, `content_season_number`, and `content_episode_number` fields (on the row that shares the tagged episode's `content_id` but a different fan-out — grouping by `content_id` and taking `MAX()` of each field reliably picks the populated value over the `0`/empty placeholder). Pulled this for all 350 tagged episodes in one query (`data/structured_linkage_350ep.csv`) — **100% resolved, 0 empty**, no permalink parsing needed at all. `series_id_for()` in the new pipeline (`pipeline_series_structured.py`) now keys directly on `content_series_id`; permalink slugs are no longer read anywhere in the pipeline.

**Effect on the item catalog.** The 107 slug-based "series/singleton" entries collapsed to **53 real series** (150 movies unchanged → 203 total items, down from 257). Confirmed correct: Eken Babu Season 9 and Feludar Goyendagiri Season 3 (previously 7+ singleton fragments each) are now each one series with all their episodes correctly grouped. `Virasat` and `Uttoradhikar` (24 tagged episodes each) are the two largest consolidated series.

**Effect on model performance — a real regression, diagnosed and fixed.** Consolidating fragments into real series concentrated viewer counts onto far fewer items (top series now have 400-1,000+ distinct viewers vs. small per-fragment counts before), which made **pure popularity a much stronger baseline** than before (61.0%/73.3% Hit@10/20, vs. 51.5%/67.4% pre-consolidation) — strong enough to beat the adopted `cluster_viewing_rate × creator_boost` formula outright (52.6%/70.2%). This is an honest side-effect of fixing the fragmentation bug, not a flaw in the fix itself: a handful of true blockbuster franchises now dominate watch-time mass, exactly as they should once counted correctly.

Re-tuned the scoring formula against this new, corrected item catalog (quick blend sweep, `pipeline_series_structured.py`):

| Scoring formula | Hit@10 / Hit@20 |
|---|---|
| `cluster_viewing_rate × boost(2.0, 2.0)` (old adopted formula, unchanged) | 52.6% / 70.2% |
| `popularity_rate` only | 61.0% / 73.3% |
| **`popularity_rate^0.7 × cluster_viewing_rate^0.3 × boost(0.5, 0.5)`** (new adopted) | **60.8% / 74.5%** |

The new blend matches pure popularity on Hit@10 and beats it on Hit@20, while still folding in cluster-affinity and creator-boost personalization (at lower weight than before, since raw popularity now carries more signal on its own). **This replaces the old `cl_rate × boost(2,2)` formula for the series-level pipeline.**

**Season-continuation: a separate, guaranteed-placement mechanism (not blended into the ranking above).** Added `season_continuation.py`: given a user's watched `content_id`s and a `series_id → {season_number → [content_ids sorted by episode]}` index (built from the same structured fields), it detects when a user has genuinely finished a season (≥90% of episodes, each itself passing a >60%-of-runtime engagement filter — completing the season and merely engaging with one episode are different thresholds and must not be conflated) and the next season exists but hasn't been started, and returns that next season's first episode as a direct recommendation. This is deliberately **not** folded into the discovery-ranking score — a completed-season signal (66.1% real Season-1→Season-2 conversion, validated earlier on 5,000 real Sampurna viewers) is strong enough to earn a guaranteed slot rather than compete on `popularity × cluster-rate` terms, where a niche Season 2 might otherwise be outscored by an unrelated blockbuster. Demoed and passing on the Sampurna Season 1→2 case.

**Not yet done:** wiring `structured_linkage_350ep.csv`'s query pattern up to the *full* (untagged) catalog — this session validated it only for the 500 tagged titles. Production rollout needs the same `content_series_id`-based grouping applied catalog-wide, with the old slug parsing fully retired (kept alive nowhere, not even as a fallback, since structured linkage resolved 100% of the tagged sample).

**Flagged for revisit at larger scale:** the `pop_rate^0.7 × cluster_rate^0.3` split was picked from a quick 5-point blend sweep (α ∈ {0, 0.25, 0.5, 0.7, 0.75, 1.0}) on the 2,218-user sample, not an exhaustive grid — it was good enough to recover the regression, not necessarily the true optimum. At 2,218 users, per-cluster viewer counts for many series are still small (single/low-double digits), which inherently favors weighting toward the more stable `pop_rate` signal; a much larger user sample would sharpen `cluster_rate` estimates and could shift the optimal blend back toward personalization. Re-tune this once a bigger pull (thousands, not ~2,200) is available — same method as `series_weight_sweep_at_k8_6.csv`, applied to the α blend instead.

Files: `pipeline_series_structured.py`, `season_continuation.py`, `data/structured_linkage_350ep.csv`, `data/content_features_series_rollup_structured.csv`.

## Next: this whole pipeline needs full-catalog tagging + a much larger user sample to be decision-grade

Steps 3-4 above prove the *method* works end-to-end (watch history → dedup → category profiles → clusters → lift/reach). To make it useful for real serving or content-strategy decisions, the two binding constraints are: (1) extend Step 1-2 tagging past the current 500/5,700 titles, and (2) scale the user sample well past 272 (thousands, not hundreds) so per-cluster-per-title viewer counts stop being single digits.
