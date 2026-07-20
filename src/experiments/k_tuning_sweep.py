"""
Grid-searches K_CONTENT (content-category clusters) and K_USER (audience
clusters) on the full-catalog model, evaluated via the same held-out LOO
methodology used throughout this project.

The content feature matrix (X) doesn't depend on K, so it's built once;
only the two KMeans fits are repeated per grid point. Still, the LOO
evaluation loop (one held-out trial per eligible user) is the expensive
part and dominates runtime -- this is a full re-run of the eval for every
grid point, not an approximation.

Usage:
    python3 src/experiments/k_tuning_sweep.py [path/to/watch_sample.csv]
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA = str(Path(__file__).resolve().parent.parent.parent / "data")
DIR_W, ACTOR_W = 0.5, 0.5
K_CONTENT_GRID = [6, 8, 10, 12]
K_USER_GRID = [4, 6, 8, 10]


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


content = pd.read_csv(f"{DATA}/content_features_full_tagged.csv")
content["_storyline"] = content["storyline_tags"].apply(parse_list)
content["_tone"] = content["overall_tone_tags"].apply(parse_list)
content["_maturity"] = content["maturity_tags"].apply(parse_list).apply(norm_maturity)
content["_director"] = content["director_names"].apply(parse_list)
content["_actor"] = content["actor_names"].apply(parse_list)
content["item_id"] = content.apply(
    lambda r: f"movie::{r.content_id}" if r.content_type == "movie" else f"series::{r.content_id}", axis=1
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


def fit_mixture(k_content):
    km = KMeans(n_clusters=k_content, random_state=42, n_init=10).fit(X)
    dists = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    logits = -dists / 0.35
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def evaluate(watch_ep, k_content, k_user):
    mixture = fit_mixture(k_content)
    watch = watch_ep.groupby(["user_id", "item_id"], as_index=False)["seconds_watched"].sum()
    watch["item_idx"] = watch.item_id.map(item_to_idx)

    def build_profile(idxs, secs):
        secs = np.array(secs, dtype=float)
        w = secs / secs.sum()
        return (mixture[idxs] * w[:, None]).sum(axis=0)

    profiles = {}
    for uid, rows_ in watch.groupby("user_id"):
        profiles[uid] = build_profile(rows_.item_idx.values, rows_.seconds_watched.values)
    all_uids = list(profiles.keys())
    P = np.array([profiles[u] for u in all_uids])
    scaler = StandardScaler().fit(P)
    Ps = scaler.transform(P)
    km_user = KMeans(n_clusters=k_user, random_state=42, n_init=10).fit(Ps)
    uid_to_cluster = dict(zip(all_uids, km_user.labels_))
    watch_c = watch.copy()
    watch_c["cluster"] = watch_c.user_id.map(uid_to_cluster)

    item_viewer_counts = watch.groupby("item_idx").user_id.nunique()
    eligible_idx = set(item_viewer_counts[item_viewer_counts >= 5].index)
    overall_pop = watch.groupby("item_idx").user_id.nunique()

    rng = np.random.default_rng(13)
    user_counts = watch.groupby("user_id").size()
    eval_users = user_counts[user_counts >= 4].index.tolist()

    holdout_choice = {}
    for uid in eval_users:
        rows_ = watch[watch.user_id == uid]
        choice = rows_.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
        holdout_choice[uid] = choice.item_idx

    ranks = []
    for uid in eval_users:
        held_idx = holdout_choice[uid]
        if held_idx not in eligible_idx:
            continue
        user_rows = watch[watch.user_id == uid]
        remaining = user_rows[user_rows.item_idx != held_idx]
        if len(remaining) == 0:
            continue
        profile = build_profile(remaining.item_idx.values, remaining.seconds_watched.values)
        profile_s = scaler.transform(profile.reshape(1, -1))
        d = np.linalg.norm(km_user.cluster_centers_ - profile_s, axis=1)
        assigned_cluster = int(np.argmin(d))

        mask_drop = (watch_c.user_id == uid) & (watch_c.item_idx == held_idx)
        wc = watch_c[~mask_drop]
        cl_viewers = wc[wc.cluster == assigned_cluster].groupby("item_idx").user_id.nunique()
        cl_size = wc[wc.cluster == assigned_cluster].user_id.nunique()

        watched = set(user_rows.item_idx) - {held_idx}
        candidates = [i for i in eligible_idx if i not in watched]
        cl_rate = (cl_viewers.reindex(candidates, fill_value=0) + 0.5) / (cl_size + 1.0)
        n_loo = wc.user_id.nunique()
        pop_rate = (overall_pop.reindex(candidates, fill_value=0) + 0.5) / (n_loo + 1.0)

        user_dirs = set().union(*(director_sets[i] for i in remaining.item_idx.values)) if len(remaining) else set()
        user_actors = set().union(*(actor_sets[i] for i in remaining.item_idx.values)) if len(remaining) else set()

        score = {}
        for c in candidates:
            boost = 1.0 + DIR_W * len(director_sets[c] & user_dirs) + ACTOR_W * len(actor_sets[c] & user_actors)
            score[c] = (pop_rate.get(c, 0.0) ** 0.7) * (cl_rate.get(c, 0.0) ** 0.3) * boost
        ranked = sorted(score, key=lambda k: score[k], reverse=True)
        ranks.append(ranked.index(held_idx) + 1)

    return {
        "n_eval": len(ranks),
        "mean_rank": float(np.mean(ranks)),
        "hit10": float(np.mean([r <= 10 for r in ranks])),
        "hit20": float(np.mean([r <= 20 for r in ranks])),
    }


if __name__ == "__main__":
    watch_sample = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/user_title_watch_sample_5000.csv"
    watch_ep = pd.read_csv(watch_sample)
    watch_ep["item_id"] = watch_ep.content_id.apply(cid_to_item_id)
    watch_ep = watch_ep.dropna(subset=["item_id"])
    print(f"watch sample: {watch_sample} ({watch_ep.user_id.nunique()} users)\n")

    print(f"{'k_content':>9s} {'k_user':>6s} {'n_eval':>7s} {'mean_rank':>10s} {'hit10':>7s} {'hit20':>7s}")
    results = []
    for k_content in K_CONTENT_GRID:
        for k_user in K_USER_GRID:
            res = evaluate(watch_ep, k_content, k_user)
            results.append((k_content, k_user, res))
            print(f"{k_content:9d} {k_user:6d} {res['n_eval']:7d} {res['mean_rank']:10.2f} "
                  f"{res['hit10']:7.3f} {res['hit20']:7.3f}")

    best = max(results, key=lambda r: r[2]["hit10"])
    print(f"\nBest by hit10: K_CONTENT={best[0]} K_USER={best[1]} -> "
          f"hit10={best[2]['hit10']:.3f} hit20={best[2]['hit20']:.3f}")
