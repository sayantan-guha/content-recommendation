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
        "era_bucket_arr": series_content["era_bucket"].values,
    }


COMPLETION_THRESHOLD = 0.6  # validated on a 500-user real-timestamp sample: +8.8pp Hit@10, +10.7pp Hit@20


def load_audience_model(model):
    # Using the 1,100-user sample with real completion data as the sole watch
    # source (not the older 5,000-user sample, which has no run-length data)
    # so completion-rate filtering applies consistently to every user instead
    # of only the subset the two samples happened to overlap on. The 5,000-
    # user sample is stashed for now -- see README's Status section.
    watch_ep = pd.read_csv(f"{DATA}/user_watch_completion_sample_1100.csv")
    watch_ep["item_id"] = watch_ep.content_id.apply(model["cid_to_item_id"])
    watch_ep = watch_ep.dropna(subset=["item_id"])
    watch_ep["completion_pct"] = (watch_ep.seconds_watched / watch_ep.content_run_length_secs).clip(upper=2.0)
    watch_ep = watch_ep[watch_ep.completion_pct >= COMPLETION_THRESHOLD]
    watch_ep["created_at"] = pd.to_datetime(watch_ep["created_at"], utc=True, format="ISO8601")
    watch = watch_ep.groupby(["user_id", "item_id"], as_index=False).agg(
        seconds_watched=("seconds_watched", "sum"), last_watched_at=("created_at", "max")
    )
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


ERA_QUOTA_ENABLED = True  # experiment: see era-quota-experiment branch / README


def apply_era_quota(top_warm, ranked, era_arr, watched_idx, top_n):
    """Secondary quota pass, applied after apply_type_quota. CF and content
    similarity both naturally skew toward the catalog's 2020s-heavy bulk, so
    a user whose watch history has a real chunk of e.g. 1990s titles gets
    those crowded out even though same-era candidates score fine on their
    own similarity terms (shared genre/actor/director etc -- they just never
    reach the top of a slate dominated by newer titles). This nudges the
    already-built type-quota slate so each era represented in the user's
    watch history gets a proportional number of slots, by swapping the best
    still-unused same-era candidate (from `ranked`, so it's still relevance-
    ordered, not random) in for the slate's lowest-priority over-quota-era
    slot. Never invents an era the user hasn't actually watched.
    """
    if not len(watched_idx) or not top_n:
        return top_warm

    era_counts = Counter(era_arr[i] for i in watched_idx)
    total = sum(era_counts.values())
    quotas = {e: (top_n * c) / total for e, c in era_counts.items()}
    floor_quotas = {e: int(q) for e, q in quotas.items()}
    remainder = top_n - sum(floor_quotas.values())
    fracs = sorted(quotas, key=lambda e: quotas[e] - floor_quotas[e], reverse=True)
    for e in fracs[:remainder]:
        floor_quotas[e] += 1

    slate = list(top_warm)
    slate_set = set(slate)
    current_counts = Counter(era_arr[i] for i in slate)

    for era, target in floor_quotas.items():
        have = current_counts.get(era, 0)
        need = target - have
        if need <= 0:
            continue
        fill = [i for i in ranked if era_arr[i] == era and i not in slate_set][:need]
        if not fill:
            continue

        def is_over_quota(pos):
            e = era_arr[slate[pos]]
            return current_counts.get(e, 0) > floor_quotas.get(e, 0)

        removable = sorted(range(len(slate)), key=lambda pos: (not is_over_quota(pos), -pos))
        for f in fill:
            if not removable:
                break
            pos = removable.pop(0)
            removed_idx = slate[pos]
            slate[pos] = f
            slate_set.discard(removed_idx)
            slate_set.add(f)
            current_counts[era_arr[removed_idx]] -= 1
            current_counts[era] = current_counts.get(era, 0) + 1

    return slate


COLD_START_QUOTA_FRACTION = 0.1  # fraction of top_n reserved for cold-start (low-data) titles

# Creator (director/actor) overlap boost, cold-start fallback ONLY -- tested and
# REJECTED for the primary CF scorer (dilutes CF's already-sharp similarity
# signal, monotonically worse the stronger it's applied), but roughly DOUBLES
# the cold-start fallback's own Hit@10/20 in isolation (Hit@10 3.5% -> 7.7%,
# Hit@20 7.1% -> 13.0% at this strength): the fallback's pure content-tag
# similarity is comparatively weak/generic on its own (many mainstream titles
# saturate near cosine-similarity 1.0 after the K=8 mixture compression), so a
# real discriminating signal like shared cast/crew gives it something concrete
# to lean on where CF has no data at all. See README's Status section.
COLD_START_DIR_W = 0.3
COLD_START_ACTOR_W = 0.3


def content_based_ranking(model, mixture_profile, candidates, watched_idx=None):
    """Rank an arbitrary candidate pool by content-tag cosine similarity to the
    user's taste profile, boosted by director/actor overlap with watched_idx
    when provided. This is the shared scorer behind both the cold-start
    fallback (candidates = titles with too little CF data) and the low-
    signal edge cases in recommend_for_user (candidates = the same eligible
    pool CF would have used, when CF itself has nothing trustworthy to say).
    Returns (candidate_array, similarity_scores), NOT truncated to top-N.
    """
    cand_arr = np.array(candidates)
    if len(cand_arr) == 0:
        return cand_arr, np.array([])
    mixture = model["mixture"]
    profile_norm = np.linalg.norm(mixture_profile) or 1.0
    item_vecs = mixture[cand_arr]
    item_norms = np.linalg.norm(item_vecs, axis=1)
    item_norms[item_norms == 0] = 1.0
    sims = (item_vecs @ mixture_profile) / (item_norms * profile_norm)

    if watched_idx is not None and len(watched_idx) > 0:
        director_sets = model["director_sets"]
        actor_sets = model["actor_sets"]
        user_dirs = set().union(*(director_sets[i] for i in watched_idx))
        user_actors = set().union(*(actor_sets[i] for i in watched_idx))
        boost = np.array([
            1.0 + COLD_START_DIR_W * len(director_sets[c] & user_dirs)
            + COLD_START_ACTOR_W * len(actor_sets[c] & user_actors)
            for c in cand_arr
        ])
        sims = sims * boost

    return cand_arr, sims


def explain_recommendation(model, watched_idx, candidate_idx, overall_pop=None):
    """Human-readable reason a candidate was recommended, given the user's
    watched titles: genre match, storyline/tone tag overlap, shared director/
    actor, or high overall popularity when nothing else explains it. Used for
    UI display only -- doesn't affect ranking. Returns a list of short
    reason strings (empty if truly nothing overlaps and popularity is low).
    """
    series_content = model["series_content"]
    director_sets = model["director_sets"]
    actor_sets = model["actor_sets"]

    if len(watched_idx) == 0:
        return ["no watch history to compare against"]

    watched_genres = set(series_content.iloc[i].genre_normalized for i in watched_idx)
    watched_storyline, watched_tone = set(), set()
    for i in watched_idx:
        watched_storyline |= set(series_content.iloc[i]["_storyline"])
        watched_tone |= set(series_content.iloc[i]["_tone"])
    watched_dirs = set().union(*(director_sets[i] for i in watched_idx))
    watched_actors = set().union(*(actor_sets[i] for i in watched_idx))

    row = series_content.iloc[candidate_idx]
    reasons = []
    if row.genre_normalized in watched_genres:
        reasons.append(f"genre match ({row.genre_normalized})")
    storyline_ov = set(row["_storyline"]) & watched_storyline
    if storyline_ov:
        reasons.append(f"storyline overlap ({', '.join(list(storyline_ov)[:2])})")
    tone_ov = set(row["_tone"]) & watched_tone
    if tone_ov:
        reasons.append(f"tone overlap ({', '.join(list(tone_ov)[:2])})")
    dir_ov = director_sets[candidate_idx] & watched_dirs
    if dir_ov:
        reasons.append(f"same director ({', '.join(list(dir_ov)[:1])})")
    act_ov = actor_sets[candidate_idx] & watched_actors
    if act_ov:
        reasons.append(f"same actor ({', '.join(list(act_ov)[:1])})")
    if overall_pop is not None:
        pop = overall_pop.get(candidate_idx, 0)
        if pop >= overall_pop.quantile(0.9):
            reasons.append("high overall popularity")
    if not reasons:
        reasons.append("weak/generic content similarity only")
    return reasons


def cold_start_candidates(model, mixture_profile, eligible_idx, watched, n, watched_idx=None):
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
    cand_arr, sims = content_based_ranking(model, mixture_profile, cold_pool, watched_idx)
    order = np.argsort(-sims)
    return cand_arr[order[:n]].tolist()


# "Low watch history" turned out NOT to be a real edge case -- tested directly
# (src/experiments/cf_vs_content_breakeven_v2.py): CF beats content-similarity
# at every bucket from 1 remaining watched title up to 11+, no breakeven point
# exists (e.g. Hit@10 35.4% vs 10.4% at just 1 remaining title). A count-based
# MIN_WATCHED_FOR_CF threshold was tried here and REMOVED after this test --
# it would have actively hurt users with few-but-informative watches. The
# real trigger for falling back to content-similarity isn't "how many titles,"
# it's whether CF actually found any co-viewers at all -- see CF_SIGNAL_EPSILON.

# Edge case: "no similar users found" -- none of a user's watched titles have
# any real co-viewers among eligible candidates, so every candidate scores
# exactly 0 and argsort's result is arbitrary tie-broken noise, not a real
# ranking (verified on a real user: 3 of 4 watched titles had zero other
# viewers in our sample). Below this we treat CF as having produced no signal.
CF_SIGNAL_EPSILON = 1e-9

# Boosting CF scores by era match *before* ranking (instead of only fixing up
# the era mix afterwards via apply_era_quota) was tried and REJECTED: it
# dilutes CF's real co-viewer signal with a crude era prior, same failure
# mode as the rejected creator-boost-on-CF experiment. Monotonically worse
# the harder it's applied -- at boost weight 1.0, overall Hit@10 fell
# 45.6%->43.9% and Hit@10 on pre-2020s held-out titles specifically collapsed
# 51.2%->19.7%. apply_era_quota (a post-hoc slate reorder that never touches
# CF's own ranking) is the validated version -- see README's Status section.


def recommend_for_user(uid, held_out_idx, model, audience, top_n=10):
    watch = audience["watch"]
    sim = audience["sim"]
    eligible_idx = audience["eligible_idx"]
    overall_pop = audience["overall_pop"]
    build_profile = audience["build_profile"]

    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.item_idx != held_out_idx] if held_out_idx is not None else user_rows
    watched_idx = remaining.item_idx.values

    watched = set(user_rows.item_idx) - ({held_out_idx} if held_out_idx is not None else set())
    candidates = [i for i in eligible_idx if i not in watched]

    if len(remaining) == 0:
        # Edge case: zero watch history at all -- no profile to build, no CF,
        # no content-similarity possible either. Fall back to popularity.
        profile = None
        pop = overall_pop.reindex(candidates, fill_value=0)
        ranked = pop.sort_values(ascending=False).index.tolist()
    else:
        profile = build_profile(watched_idx, remaining.seconds_watched.values)
        cand_arr = np.array(candidates)
        scores = sim[watched_idx][:, cand_arr].sum(axis=0)

        if len(scores) == 0 or scores.max() <= CF_SIGNAL_EPSILON:
            # Edge case: "no similar users found" -- none of this user's
            # watched titles have any real co-viewers among eligible
            # candidates, so CF's ranking would be tie-broken noise, not
            # a real signal. Rank by content-similarity instead.
            _, sims = content_based_ranking(model, profile, cand_arr, watched_idx)
            order = np.argsort(-sims)
            ranked = cand_arr[order].tolist()
        else:
            order = np.argsort(-scores)
            ranked = cand_arr[order].tolist()

    content_type_arr = model["content_type_arr"]
    type_counts = Counter(content_type_arr[i] for i in watched_idx) if len(remaining) else Counter()

    cold_n = max(1, round(top_n * COLD_START_QUOTA_FRACTION)) if (top_n >= 5 and COLD_START_QUOTA_FRACTION > 0) else 0
    warm_n = top_n - cold_n
    top_warm = apply_type_quota(ranked, content_type_arr, type_counts, warm_n)
    if ERA_QUOTA_ENABLED and len(remaining):
        era_arr = model["era_bucket_arr"]
        top_warm = apply_era_quota(top_warm, ranked, era_arr, watched_idx, warm_n)
    top_cold = cold_start_candidates(model, profile, eligible_idx, watched, cold_n, watched_idx=watched_idx) if profile is not None else []
    top = top_warm + top_cold

    # Edge case: candidate pools ran dry (e.g. a power user has watched nearly
    # every eligible title, or the cold pool is empty) and the slate came up
    # short of top_n. Backfill from the full unwatched catalog by content-
    # similarity (or popularity, if there's no profile at all) rather than
    # silently returning fewer than top_n recommendations.
    if len(top) < top_n:
        filled = set(top) | watched
        remaining_pool = [i for i in range(len(model["mixture"])) if i not in filled]
        if remaining_pool:
            if profile is not None:
                cand_arr, sims = content_based_ranking(model, profile, remaining_pool, watched_idx)
                order = np.argsort(-sims)
                backfill = cand_arr[order].tolist()
            else:
                pop = overall_pop.reindex(remaining_pool, fill_value=0)
                backfill = pop.sort_values(ascending=False).index.tolist()
            top = top + backfill[: top_n - len(top)]

    return top, ranked
