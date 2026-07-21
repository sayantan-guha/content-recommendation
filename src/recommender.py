"""
Shared recommendation-model logic. Framework-agnostic -- used by backend/app.py
(FastAPI) and importable anywhere else that needs the model without a UI
dependency.

Three-tier scoring, replacing the old single cluster-based scorer (retired --
item-item CF strictly dominated it on every metric tested, see README/
src/pipeline_item_cf.py):

  1. Item-item CF (primary) -- for users with >=1 watched eligible title,
     score candidates by summed cosine-similarity to everything they've
     watched. This is src/pipeline_item_cf.py's validated approach, wired
     in here instead of just benchmarked standalone.
  2. Content-similarity cold-start fallback -- titles with <5 viewers have no
     CF co-occurrence signal regardless of which user is asking, so they're
     scored by cosine similarity to the user's content-tag profile instead
     (see cold_start_candidates below). Reserves ~10% of every slate.
  3. Popularity fallback -- for users with 0 watched eligible titles, CF has
     nothing to sum similarities over, so candidates are just ranked by raw
     popularity.

Deliberately NOT blended together (tested and rejected -- see
src/pipeline_item_cf.py's docstring: mixing in creator_boost or a
cluster/popularity score diluted CF's sharper signal). Each tier only
activates when the one above it has nothing to work with.

The movie/series type-affinity quota (apply_type_quota) is a final reordering
pass on top of whichever tier produced the ranked list -- agnostic to scoring
method.
"""
import ast
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans

DATA = str(Path(__file__).resolve().parent.parent / "data")
K_CONTENT = 8


def parse_list(s):
    if pd.isna(s):
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else [v]
    except Exception:
        return []


def norm_maturity(tags):
    return sorted(set(t.strip().lower() for t in tags))


def multihot(values, vocab):
    idx = {v: i for i, v in enumerate(vocab)}
    mat = np.zeros((len(values), len(vocab)))
    for i, lst in enumerate(values):
        for v in lst:
            if v in idx:
                mat[i, idx[v]] = 1.0
    return mat


def onehot(values, vocab):
    idx = {v: i for i, v in enumerate(vocab)}
    mat = np.zeros((len(values), len(vocab)))
    for i, v in enumerate(values):
        if v in idx:
            mat[i, idx[v]] = 1.0
    return mat


def l2_normalize_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def load_content_model():
    content = pd.read_csv(f"{DATA}/content_features_full_tagged.csv")
    content["_storyline"] = content["storyline_tags"].apply(parse_list)
    content["_tone"] = content["overall_tone_tags"].apply(parse_list)
    content["_maturity"] = content["maturity_tags"].apply(parse_list).apply(norm_maturity)
    content["_director"] = content["director_names"].apply(parse_list)
    content["_actor"] = content["actor_names"].apply(parse_list)
    content["item_id"] = content.apply(
        lambda r: f"movie::{r.content_id}" if r.content_type == "movie" else f"series::{r.content_id}",
        axis=1,
    )
    series_content = content.reset_index(drop=True)

    storyline_vocab = sorted(set(t for lst in series_content["_storyline"] for t in lst))
    tone_vocab = sorted(set(t for lst in series_content["_tone"] for t in lst))
    maturity_vocab = sorted(set(t for lst in series_content["_maturity"] for t in lst))
    genre_vocab = sorted(series_content["genre_normalized"].dropna().unique().tolist())
    era_vocab = sorted(series_content["era_bucket"].dropna().unique().tolist())

    blocks = {
        "genre": onehot(series_content["genre_normalized"], genre_vocab),
        "storyline": multihot(series_content["_storyline"], storyline_vocab),
        "tone": multihot(series_content["_tone"], tone_vocab),
        "era": onehot(series_content["era_bucket"], era_vocab),
        "maturity": multihot(series_content["_maturity"], maturity_vocab),
    }
    weights = {"genre": 3.0, "storyline": 3.0, "tone": 2.0, "era": 1.0, "maturity": 1.0}
    X = np.concatenate([l2_normalize_rows(blocks[b]) * weights[b] for b in blocks], axis=1)

    km = KMeans(n_clusters=K_CONTENT, random_state=42, n_init=10).fit(X)
    dists = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    logits = -dists / 0.35
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    mixture = ex / ex.sum(axis=1, keepdims=True)

    item_ids = series_content.item_id.values
    item_to_idx = {s: i for i, s in enumerate(item_ids)}
    director_sets = [set(x) for x in series_content["_director"]]
    actor_sets = [set(x) for x in series_content["_actor"]]

    struct = pd.read_csv(f"{DATA}/structured_linkage_full.csv").dropna(subset=["series_id"])
    episode_to_show_cid = dict(zip(struct.content_id, struct.series_id))
    movie_cids = set(content[content.content_type == "movie"].content_id)
    series_cids = set(content[content.content_type == "series"].content_id)

    def cid_to_item_id(cid):
        if cid in movie_cids:
            return f"movie::{cid}"
        show_cid = episode_to_show_cid.get(cid)
        if show_cid in series_cids:
            return f"series::{show_cid}"
        return None

    return {
        "series_content": series_content,
        "mixture": mixture,
        "item_to_idx": item_to_idx,
        "director_sets": director_sets,
        "actor_sets": actor_sets,
        "cid_to_item_id": cid_to_item_id,
        "content_type_arr": series_content["content_type"].values,
    }


def load_audience_model(model):
    watch_ep = pd.read_csv(f"{DATA}/user_title_watch_sample_5000.csv")
    watch_ep["item_id"] = watch_ep.content_id.apply(model["cid_to_item_id"])
    watch_ep = watch_ep.dropna(subset=["item_id"])
    watch = watch_ep.groupby(["user_id", "item_id"], as_index=False)["seconds_watched"].sum()
    watch["item_idx"] = watch.item_id.map(model["item_to_idx"])

    mixture = model["mixture"]

    def build_profile(idxs, secs):
        secs = np.array(secs, dtype=float)
        w = secs / secs.sum()
        return (mixture[idxs] * w[:, None]).sum(axis=0)

    item_viewer_counts = watch.groupby("item_idx").user_id.nunique()
    eligible_idx = set(item_viewer_counts[item_viewer_counts >= 5].index)
    overall_pop = watch.groupby("item_idx").user_id.nunique()

    user_counts = watch.groupby("user_id").size()
    eval_users = sorted(user_counts[user_counts >= 4].index.tolist())

    # item-item cosine-similarity matrix (binary co-occurrence), built once --
    # same construction as src/pipeline_item_cf.py
    all_uids = sorted(watch.user_id.unique())
    uid_to_row = {u: i for i, u in enumerate(all_uids)}
    n_items = len(mixture)
    rows_idx = watch.user_id.map(uid_to_row).values
    cols_idx = watch.item_idx.values
    ui = csr_matrix((np.ones(len(watch)), (rows_idx, cols_idx)), shape=(len(all_uids), n_items))
    ii = (ui.T @ ui).toarray()
    item_norm = np.sqrt(np.array(ui.multiply(ui).sum(axis=0)).flatten())
    item_norm[item_norm == 0] = 1.0
    sim = ii / (item_norm[:, None] * item_norm[None, :])
    np.fill_diagonal(sim, 0.0)

    return {
        "watch": watch,
        "sim": sim,
        "eligible_idx": eligible_idx,
        "overall_pop": overall_pop,
        "eval_users": eval_users,
        "build_profile": build_profile,
    }


def apply_type_quota(ranked, content_type_arr, type_counts, top_n):
    """Reorder `ranked` so the top_n slate matches the user's movie/series
    watch-history proportions, instead of taking a flat top-N cut.

    type_counts: e.g. {"movie": 7, "series": 3} from the user's watch history.
    Falls back to a plain top-N cut if the user has no typed history yet.
    """
    total = sum(type_counts.values())
    if total == 0:
        return ranked[:top_n]

    quotas = {t: (top_n * c) / total for t, c in type_counts.items()}
    floor_quotas = {t: int(q) for t, q in quotas.items()}
    remainder = top_n - sum(floor_quotas.values())
    # give leftover slots to the types with the largest fractional remainder
    fracs = sorted(quotas, key=lambda t: quotas[t] - floor_quotas[t], reverse=True)
    for t in fracs[:remainder]:
        floor_quotas[t] += 1

    buckets = {t: [] for t in type_counts}
    for idx in ranked:
        t = content_type_arr[idx]
        if t in buckets:
            buckets[t].append(idx)

    slate = []
    for t, q in floor_quotas.items():
        slate.extend(buckets.get(t, [])[:q])
    filled = set(slate)

    # if a type ran short of eligible candidates, backfill from the overall
    # ranked order (regardless of type) so the slate still reaches top_n
    if len(slate) < top_n:
        for idx in ranked:
            if idx not in filled:
                slate.append(idx)
                filled.add(idx)
            if len(slate) >= top_n:
                break

    # restore original relevance order within the quota-filled slate
    slate_set = set(slate)
    return [idx for idx in ranked if idx in slate_set][:top_n]


COLD_START_QUOTA_FRACTION = 0.1  # fraction of top_n reserved for cold-start (low-data) titles


def cold_start_candidates(model, mixture_profile, eligible_idx, watched, n):
    """Rank titles with too little watch data to trust CF/cluster signal (below
    the eligibility threshold) by content-tag similarity to the user's taste
    profile, instead of excluding them entirely. Lets brand-new/low-viewership
    titles surface without waiting for CF/popularity signal to accumulate.
    """
    if n <= 0:
        return []
    mixture = model["mixture"]
    cold_pool = [i for i in range(len(mixture)) if i not in eligible_idx and i not in watched]
    if not cold_pool:
        return []
    profile_norm = np.linalg.norm(mixture_profile) or 1.0
    item_vecs = mixture[cold_pool]
    item_norms = np.linalg.norm(item_vecs, axis=1)
    item_norms[item_norms == 0] = 1.0
    sims = (item_vecs @ mixture_profile) / (item_norms * profile_norm)
    order = np.argsort(-sims)
    return [cold_pool[i] for i in order[:n]]


def recommend_for_user(uid, held_out_idx, model, audience, top_n=10):
    watch = audience["watch"]
    sim = audience["sim"]
    eligible_idx = audience["eligible_idx"]
    overall_pop = audience["overall_pop"]
    build_profile = audience["build_profile"]

    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.item_idx != held_out_idx] if held_out_idx is not None else user_rows
    profile = build_profile(remaining.item_idx.values, remaining.seconds_watched.values)

    watched = set(user_rows.item_idx) - ({held_out_idx} if held_out_idx is not None else set())
    candidates = [i for i in eligible_idx if i not in watched]

    if len(remaining) == 0:
        # Tier 3: no watch history to score CF against -- fall back to popularity.
        pop = overall_pop.reindex(candidates, fill_value=0)
        ranked = pop.sort_values(ascending=False).index.tolist()
    else:
        # Tier 1: item-item CF -- sum similarity to everything the user has watched.
        cand_arr = np.array(candidates)
        scores = sim[remaining.item_idx.values][:, cand_arr].sum(axis=0)
        order = np.argsort(-scores)
        ranked = cand_arr[order].tolist()

    content_type_arr = model["content_type_arr"]
    type_counts = Counter(content_type_arr[i] for i in remaining.item_idx.values)

    cold_n = max(1, round(top_n * COLD_START_QUOTA_FRACTION)) if (top_n >= 5 and COLD_START_QUOTA_FRACTION > 0) else 0
    warm_n = top_n - cold_n
    top_warm = apply_type_quota(ranked, content_type_arr, type_counts, warm_n)
    top_cold = cold_start_candidates(model, profile, eligible_idx, watched, cold_n)
    top = top_warm + top_cold
    return top, ranked
