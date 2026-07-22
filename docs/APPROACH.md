# Content Recommendation — Approach

> **Status note:** Steps 3–4 below (audience clustering, cluster-affinity scoring) describe the *original* design and were the production model through mid-2026. They've since been **retired and replaced by item-item collaborative filtering**, which strictly outperformed them on every held-out metric tested — see [README.md](../README.md)'s Status section and [src/recommender.py](../src/recommender.py) for the current three-tier production scorer (CF → content-similarity cold-start fallback → popularity fallback). Steps 1–2 (content tagging, Programming Categories) are still live — that same content vector now powers the cold-start fallback instead of audience-cluster scoring. This doc is kept as-is for the original design rationale; don't read it as describing current production.

Based on the reference pattern shown in two screenshots (a Netflix-style internal tool): a title-level "Product Creative Brief" showing tag-based **Predicted Programming Categories**, and an "Audiences to Design For" view showing, per audience segment, a category-mix breakdown plus per-title **Likelihood** (over-index) and **Viewers%** scores.

The pattern has two directions built on the same underlying data:
- **Content → Category**: given a title's tags, predict which Programming Categories it belongs to (a soft, percentage mixture, not a single label).
- **Audience → Title**: given an audience segment's viewing behavior, rank titles by how much more likely that segment is to watch them than the overall base.

Below are the steps to build this, with the techniques applicable at each one.

---

## Step 1 — Rich content tagging

Tag every title along multiple independent dimensions, not just genre:

| Dimension | Example values (from the reference) |
|---|---|
| `storyline` (multi-label) | celebrities, crime_solving, family_in_crisis, missing_person, uncovering_the_truth |
| `overall_tone` (multi-label) | emotional, suspenseful, tense |
| `core_genre` (single-label) | drama |
| `core_subgenre` | crime |
| `audience_age_band` | — |
| `primary_era` | 2020s |
| `language_spoken`, `country_of_origin` | — |

**Techniques:**
- Manual tagging by a content/taxonomy team (most reliable, most expensive).
- LLM-assisted extraction from synopsis/script text, with human review — scales much better than pure manual tagging, and is the practical way to backfill a whole catalog.
- Hybrid: LLM proposes tags, humans approve/correct — usually the best cost/quality tradeoff.

## Step 2 — Programming Category taxonomy + classifier

Define a fixed set of interpretable macro-categories (e.g. "Character Dramas," "Mystery & Crime Thrillers," "Provocative & Psychological Thriller") and build a mapping from a title's tag vector to a **soft distribution** over these categories (e.g. 40% Character Dramas, 26% Mystery & Crime Thrillers, ...).

**Techniques:**
- **Supervised multi-label classification** (logistic regression, gradient boosted trees, or a small neural net) if you have human-labeled examples to train on.
- **Unsupervised discovery**: cluster or topic-model (k-means, LDA) over the tag space across the whole catalog, then have humans name the resulting archetypes — useful when the category taxonomy doesn't exist yet.
- **LLM zero-/few-shot classification**: prompt an LLM with the tag vector (or synopsis directly) and a fixed category list, ask for a percentage mixture — fast to stand up, no training data required, but needs calibration checking.

## Step 3 — Audience / taste-cluster discovery (behavioral, not demographic)

Segment users by what they actually watch, not who they are.

**Techniques:**
- **K-means / GMM clustering** on each user's Programming Category (or genre) consumption profile — simple, interpretable, a good starting point.
- **Clustering on latent embeddings** from a matrix factorization model (e.g. ALS user factors) instead of raw category mix — captures more nuanced taste signal than category percentages alone, at the cost of interpretability.
- **Graph community detection** (e.g. Louvain) on a user-similarity graph built from shared-title co-viewing — an alternative to centroid-based clustering, tends to find more natural, non-spherical communities.

Each cluster gets its own "Predicted Programming Categories" profile — the aggregate category mix of what its members watch (e.g. one cluster is 81% Provocative & Psychological Thriller, another is a broad, mixed mainstream cluster).

## Step 4 — Per-cluster title affinity: Likelihood + Viewers%

For each (cluster, title) pair, compute:
- **Likelihood (over-index / lift)** = P(cluster watches title) / P(overall base watches title) — a simple, interpretable ratio, not a black-box score.
- **Viewers%** = reach = fraction of the cluster that has actually watched the title.

These two numbers together distinguish "everyone in this cluster loves it but few have found it yet" (high likelihood, low viewers%) from "already broadly seen by this cluster" (high viewers%) — exactly the two-column format in the reference screenshot.

**Techniques:**
- Plain conditional-probability arithmetic — no ML model needed here, which is the point: this stage is deliberately transparent and auditable.
- Apply a minimum-sample-size floor per (cluster, title) pair before ranking, to avoid small-cluster noise producing spurious high-Likelihood outliers.
- Consider Bayesian/Laplace smoothing for titles with very few observed watches.

## Step 5 — Serving: rows and ranking

- Precompute cluster assignments and per-cluster top-N title lists in batch (e.g. nightly).
- At request time, assign the active user to their cluster(s) — a user can have soft membership across more than one, not just a hard single assignment — and surface the corresponding Programming Category rows, titles ranked by Likelihood within each row.
- Blend in freshness/recency for new releases so the system doesn't only ever surface previously-established hits within a cluster (a cold-start-within-cluster problem).

## Step 6 — Reverse use case: content strategy / creative briefs

The same machinery answers a second question, upstream of any user ever watching anything: given a *pitched or in-production* title's tags, which Programming Categories would it land in, and by extension which audience clusters would it likely reach? This is what the "Finding Anamika" creative-brief screenshot shows — Step 1 + Step 2 applied to a title before it exists in any watch history, used for commissioning decisions rather than serving recommendations.

---

## Summary: technique choice by step

| Step | Lightweight option | More powerful option |
|---|---|---|
| Tagging | Manual | LLM-assisted + human review |
| Content → Category | Unsupervised clustering + naming | Supervised multi-label classifier |
| Audience clustering | K-means on category mix | Clustering on ALS/embedding latent factors |
| Title affinity | Raw lift/over-index | Same, with smoothing + significance testing |
| Serving | Static per-cluster batch lists | Real-time blend with session recency |
