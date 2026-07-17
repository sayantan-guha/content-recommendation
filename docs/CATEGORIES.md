# Programming Categories (v1 — bottom-up, discovered from 500-title sample)

Discovered via K-means clustering (k=8, chosen for interpretability — silhouette scores were flat across k=4-14, so k was picked for a business-usable category count, not a statistical peak) over each title's `storyline_tags` + `overall_tone_tags` multi-hot vector (58 dims: 42 storyline + 16 tone). Each cluster was named by inspecting its dominant tags, genre mix, and content-type split.

Soft mixture (the % columns in `data/content_categories_500.csv`) is a softmax over each title's distance to all 8 cluster centroids (temperature=0.35), not just the hard cluster label — mirroring the reference screenshot's "40% Category A, 26% Category B, ..." format. Average top-1 share is 53.5%, top-2 is 16.1%.

## The 8 categories

| Category | n (dominant) | Content mix | Defining storyline tags | Defining tone |
|---|---|---|---|---|
| **Family Betrayal & Social Drama** | 92 | 87% episode | betrayal, family_conflict, deception_disguise, harassment_accusation, legal_case_trial | dramatic, tense |
| **Suspense & Crime-Driven Thrillers** | 85 | 84% episode | life_threat, crime_investigation, murder_mystery, cover_up, betrayal | suspenseful, tense |
| **Supernatural Horror & Curses** | 69 | 91% episode | life_threat, supernatural_threat, curse_ritual, betrayal, kidnapping | dark, tense, intense, eerie |
| **Whodunit Detective Mysteries** | 59 | 81% episode | crime_investigation, murder_mystery, cover_up, family_secrets, missing_person | mysterious, tense, suspenseful |
| **Comedy of Errors & Farce** | 59 | 51% episode | comedy_of_errors, deception_disguise, mistaken_identity, ghost_haunting | comedic, chaotic |
| **Character-Driven Life Dramas** | 54 | 80% movie | family_conflict, self_discovery, forbidden_love, social_injustice | dramatic, emotional, bittersweet |
| **Secrets, Disguises & Revelations** | 44 | 66% episode | deception_disguise, family_secrets, mistaken_identity, missing_person, cover_up | mysterious, dramatic |
| **Feel-Good Life Stories** | 38 | 53% movie | self_discovery, reconciliation, unlikely_friendship, midlife_crisis | heartwarming, emotional, uplifting |

## Notes on how categories differ from each other (the non-obvious splits)

- **Whodunit Detective Mysteries vs. Suspense & Crime-Driven Thrillers** — both are crime/murder-heavy, but split on tone: the former is `mysterious`-dominant (cerebral, puzzle-solving — genres skew Detective/Mystery Drama), the latter is `suspenseful`+`tense`-dominant (danger/chase-driven — genres skew Thriller/Suspense/Adventure).
- **Family Betrayal & Social Drama vs. Character-Driven Life Dramas** — both are Drama-genre-heavy, but the former is almost entirely episodic serials with `legal_case_trial`/`harassment_accusation` (soap-style long-running conflict), while the latter is 80% movies with `self_discovery`/`forbidden_love` (contained, single-arc emotional stories).
- **Feel-Good Life Stories** unexpectedly absorbed Reality TV titles (8 of 38) alongside scripted comfort dramas — both share `heartwarming`/`self_discovery` tagging even though one is unscripted. Worth a human sanity-check if Reality TV should be a category of its own rather than blended in here.
- **Secrets, Disguises & Revelations** is the fuzziest category (lowest tag concentration, most genre-mixed) — it's catching titles with a "hidden truth" throughline that don't have a strong crime or comedy identity otherwise. Candidate for re-splitting once we go past 500 titles.

## Known limitations of this v1 taxonomy

- Fit on 500 of ~5,700 total catalog titles — uncommon genres (Action, Fantasy, Musical, Devotional, Sci-Fi) had only 1-3 examples in this sample, so they're absorbed into whichever cluster their tags happened to land nearest, not necessarily correctly.
- k=8 was a judgment call, not a statistically forced number — silhouette scores were roughly flat (0.11-0.14) across k=4 to k=14, meaning the tag space doesn't have sharply separated natural clusters. Categories here are usable macro-groupings, not ground truth.
- No human review pass has happened yet on category names or edge-case assignments (per the reference pattern's "LLM proposes tags, humans approve/correct" hybrid — this is the "LLM/algorithm proposes" half only).
- **v1 weighted every tag 1/0, storyline vs. tone dimensions equally per-tag** — superseded by v2 below, which found this leaves real signal (cast, director, genre) unused and lets generic high-frequency tags (e.g. `life_threat`) dominate distance as much as rare, distinctive ones.

## v2 — weighted properties + a held-out validation exercise (data/content_categories_500_v2.csv)

Prompted by a direct question: what if properties were weighted by how much they should matter, instead of every tag counting as 1/0? Tested against the same 265-user held-out validation used to check Step 3-4 (hold out one watched title per user, recompute their profile without it, rank all eligible unseen titles by the resulting cluster's Likelihood, check where the true title landed — see [AUDIENCE_CLUSTERS.md](AUDIENCE_CLUSTERS.md) for the full method).

**What was tried** (each property's multi-hot block L2-normalized before weighting, so the weight controls its actual share of influence, not just its raw tag-vocab size):

| Scheme | Properties in the clustering vector | Result vs. current v1 |
|---|---|---|
| A — v1 (current) | storyline + tone, binary, unweighted | baseline |
| B — all properties, equal weight | + genre, era, maturity, cast, director | better |
| C — all properties, weighted (cast/director high) | genre/storyline/cast/director = 3x, tone = 2x, era/maturity = 1x | better still on accuracy, **but**: |
| D — story-only, weighted | genre/storyline = 3x, tone = 2x, era/maturity = 1x, **cast/director excluded** | accuracy regressed vs. B/C |
| **E — v2 (adopted)** | D's categories **+ a separate cast/director affinity boost applied at ranking time**, not baked into clustering | **best of all, beats popularity baseline outright** |

**Why cast/director had to be pulled out of clustering**: scheme C's own clusters were inspected and two of the eight had collapsed into a single director's back-catalog — one cluster was 100% Soumen Halder titles (n=48), another 100% Joydeep Mukherjee (n=22-30) — because `director`/`cast` are near-unique identity fields (51/201 distinct values), so weighting them as heavily as `genre`/`storyline` let K-means cluster by "who made it" instead of "what kind of story it is." Dropping them (scheme D) fixed the silos (worst single-director concentration fell from 100% to 67%, and that 67% cluster is still fully coherent — all Social Drama) but also erased most of the accuracy gain, proving cast/director carry a real "loved one X-starrer, will like another" signal that just doesn't belong in the taxonomy itself.

**v2 recipe**: content clustering uses only `genre` (3x), `storyline_tags` (3x), `overall_tone_tags` (2x), `era_bucket` (1x), `maturity_tags` (1x) — `cast`/`director` are deliberately excluded here and instead applied as a ranking-time multiplier (see [AUDIENCE_CLUSTERS.md](AUDIENCE_CLUSTERS.md) Step 4.5).

### v2's 8 categories (k=8, same K-means setup as v1)

| Category | n | Content mix | Defining storyline | Defining tone |
|---|---|---|---|---|
| **Character-Driven Family Drama** | 168 | 66% episode | betrayal, family_conflict, deception_disguise, self_discovery, family_secrets | tense, dramatic, emotional |
| **Whodunit Detective Mysteries** | 71 | 87% episode | murder_mystery, crime_investigation, family_secrets, cover_up | suspenseful, mysterious, tense |
| **Feel-Good & Reconciliation Stories** | 56 | 57% movie | self_discovery, family_conflict, reconciliation, forbidden_love | emotional, dramatic, heartwarming |
| **Comedy of Errors & Farce** | 57 | 60% episode | comedy_of_errors, deception_disguise, domestic_conflict, missing_person | comedic, chaotic, mysterious |
| **Crime & Terror Suspense Thrillers** | 49 | 76% episode | crime_investigation, life_threat, terrorism, missing_person | tense, suspenseful, dark |
| **Revenge & Survival Thrillers** | 37 | 73% episode | life_threat, deception_disguise, survival, family_rivalry | tense, suspenseful, intense |
| **Supernatural Horror & Curses** | 35 | 86% episode | supernatural_threat, curse_ritual, ghost_haunting, kidnapping | eerie, dark, tense, ominous |
| **Family Legal & Social Drama** | 27 | 93% episode | family_conflict, harassment_accusation, legal_case_trial, domestic_conflict | dramatic, tense |

Every category now has a clean, single-genre or near-single-genre signature (e.g. "Family Legal & Social Drama" is 100% `genre_normalized == Social Drama`, "Supernatural Horror & Curses" is 94% Horror-family genres) — a direct effect of weighting `genre` at 3x alongside `storyline`.

### Held-out validation results (265 users, leave-one-out; full table in `data/feature_weighting_validation_comparison.csv`)

| Scheme | mean rank | Hit@10 | Hit@20 |
|---|---|---|---|
| A — v1 (current) | 72.9 | 11.5% | 21.1% |
| E — v2 (adopted) | **43.8** | **36.8%** | **49.4%** |
| popularity baseline (reference) | 55.8 | 28.0% | 38.7% |

v2 more than triples Hit@10 versus v1 and, unlike every category-only scheme tried, beats the naive popularity baseline outright.

### Known limitations of v2

- Same 500-of-5,700-title coverage and k=8-is-a-judgment-call caveats as v1 — this is a better feature-weighting scheme on the same tagging footprint, not a bigger one.
- Weight values (3x/2x/1x) were a reasoned starting point, not tuned via grid search — worth revisiting once the sample scales past a few hundred users.
- The cast/director ranking boost (1.0 per shared director, 0.3 per shared actor) is a hand-set heuristic, not learned — a natural next step once there's enough data to fit it.
