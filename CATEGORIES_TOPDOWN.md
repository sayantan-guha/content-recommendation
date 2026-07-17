# Programming Categories (v2 — top-down, using `genre_normalized` as the fixed category list)

Alternative to the bottom-up clustering approach in [CATEGORIES.md](CATEGORIES.md). Here the category list is **not discovered** — it's simply the existing `genre_normalized` field (41 distinct values in the 500-title sample), treated as the fixed taxonomy per the "top-down" option: *"you hand me a target category list, I map titles into it."*

## Method

- Categories = the 41 distinct `genre_normalized` values already in the data (`Drama`, `Thriller`, `Social Drama`, ... down to singletons like `Devotional`, `Action`).
- Per-category centroid = mean `storyline_tags` + `overall_tone_tags` multi-hot vector (58-dim) of all titles carrying that genre label.
- Each title's soft mixture = softmax over its (negative) distance to all 41 centroids, temperature tuned to 0.15 so avg top-2 share (16.1%) matches the bottom-up result (16.1%) — same calibration, for a fair side-by-side comparison. Avg top-1 share: 57.0% (vs. 53.5% bottom-up — comparable).
- Full per-title mixture: `data/content_categories_500_topdown.csv`. Side-by-side with bottom-up: `data/categories_comparison_500.csv`.

## Key finding: only 50.4% self-consistency

For every title, I checked whether its **own literal `genre_normalized` label** is also the category its tag profile is *closest* to. Only 50.4% of titles agree with their own label — the other ~half are tag-wise closer to a different genre's centroid. Examples:

| Title | Own genre label | Tag-closest category |
|---|---|---|
| Ranna Baati | Drama | Sports |
| Bariwali | Drama | Romance |
| Belashuru | Drama | Sports |
| Durga Sohay | Drama | Reality TV |
| Dwitiyo Purush | Thriller | Mystery Drama |

**Why:** `Drama` alone covers 168 of 500 titles (34% of the sample) — it's used as a catch-all default label in the CMS, not a coherent creative category. Its member titles' actual storyline/tone tags scatter across sports stories, romances, reality-adjacent content, psychological dramas, etc. Genre labels reflect how the CMS was populated, not how the content actually clusters by theme/tone.

## How the two taxonomies relate

Cross-tabbing bottom-up categories against their most common top-down (genre) constituents shows the bottom-up clusters are basically **coherent bundles of related genres**:

| Bottom-up category | Made of (top-down genres) |
|---|---|
| Whodunit Detective Mysteries | Mystery Drama, Mystery, Detective, Murder Mystery |
| Suspense & Crime-Driven Thrillers | Thriller, Suspense, Adventure, Mystery |
| Supernatural Horror & Curses | Horror, Revenge Thriller, Thriller, Mythological Horror |
| Family Betrayal & Social Drama | Social Drama, Drama, Political Thriller, Drama Thriller |
| Character-Driven Life Dramas | Romance, Musical Drama, Documentary, Drama |
| Comedy of Errors & Farce | Comedy, Horror Comedy, Romantic Comedy, Comedy Drama |
| Feel-Good Life Stories | Reality TV, Romance, Romantic, Romantic Comedy |
| Secrets, Disguises & Revelations | Drama Thriller, Comedy Drama, Drama, Musical Drama |

This is the expected relationship — bottom-up clustering effectively re-groups the fragmented genre field into fewer, tag-coherent macro-buckets, resolving exactly the granularity problem the raw `primary_genre_english` field has (212 raw values / ~41 even after normalization on this sample).

## Trade-offs: top-down (this) vs. bottom-up

| | Top-down (genre-as-category) | Bottom-up (tag clustering) |
|---|---|---|
| Category count | 41 (too many for a "Programming Category" row — most streaming UIs show 5-15) | 8 (usable) |
| Interpretability | Immediately familiar (it's just genre) | Needs naming/curation, but still human-readable |
| Internal coherence | Low — `Drama` (168 titles) is tag-incoherent; many categories are n=1-2 singletons with meaningless centroids | Higher — clusters were formed *from* tag similarity, so members share real thematic/tonal DNA |
| Business buy-in | Easy — matches existing CMS vocabulary | Requires sign-off on new category names |
| Fixes the known genre fragmentation problem (212 raw values, Bengali/English duplication, typos) | No — inherits it directly | Yes, indirectly (tags are genre-independent) |

**Recommendation:** genre-as-category is simplest to ship but doesn't actually solve the problem Step 2 exists to solve (a title→category mapping useful for downstream audience-affinity scoring) — `Drama` is too broad to be a useful "predicted programming category," and most other genre values are too small to be statistically meaningful buckets. The bottom-up categories are the better foundation for Step 3 onward; this file exists for comparison, not as the recommended path.
