"""
Resurrects the retired cluster-based scoring model (content clustering +
audience clustering + pop_rate/cluster_rate/creator_boost) SOLELY to test
whether the v3 storyline/tone tag rebuild + different feature weights/K
values improve it -- item-item CF is deliberately set aside for this test,
per request ("consider the only approach we have is cluster based approach").

Uses the same data as current production (data/user_watch_completion_sample_1100.csv,
completion-filtered at >=60%) and the same LOO methodology as eval_recommender.py,
so results are directly comparable to prior cluster-model numbers reported
in README.md history.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent.parent
DATA = str(REPO / "data")
sys.path.insert(0, str(REPO / "src"))
import recommender as rec  # only for cid_to_item_id / parse helpers structure

COMPLETION_THRESHOLD = 0.6


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

BLOCKS_RAW = {
    "genre": onehot(series_content["genre_normalized"], genre_vocab),
    "storyline": multihot(series_content["_storyline"], storyline_vocab),
    "tone": multihot(series_content["_tone"], tone_vocab),
    "era": onehot(series_content["era_bucket"], era_vocab),
    "maturity": multihot(series_content["_maturity"], maturity_vocab),
}
BLOCKS_NORM = {k: l2_normalize_rows(v) for k, v in BLOCKS_RAW.items()}

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


# ---------- watch data: same source + completion filter as current production ----------
watch_raw = pd.read_csv(f"{DATA}/user_watch_completion_sample_1100.csv")
watch_raw["item_id"] = watch_raw.content_id.apply(cid_to_item_id)
watch_raw = watch_raw.dropna(subset=["item_id"])
watch_raw["completion_pct"] = (watch_raw.seconds_watched / watch_raw.content_run_length_secs).clip(upper=2.0)
watch_raw = watch_raw[watch_raw.completion_pct >= COMPLETION_THRESHOLD]
watch = watch_raw.groupby(["user_id", "item_id"], as_index=False)["seconds_watched"].sum()
watch["item_idx"] = watch.item_id.map(item_to_idx)

item_viewer_counts = watch.groupby("item_idx").user_id.nunique()
ELIGIBLE_IDX = set(item_viewer_counts[item_viewer_counts >= 5].index)
overall_pop = watch.groupby("item_idx").user_id.nunique()
user_counts = watch.groupby("user_id").size()
eval_users = sorted(user_counts[user_counts >= 4].index.tolist())
print(f"catalog={len(series_content)} watch_rows={len(watch)} users={watch.user_id.nunique()} "
      f"eligible_items={len(ELIGIBLE_IDX)} eval_users={len(eval_users)}")

rng = np.random.default_rng(13)
holdout_choice = {}
for uid in eval_users:
    rows_ = watch[watch.user_id == uid]
    eligible_rows = rows_[rows_.item_idx.isin(ELIGIBLE_IDX)]
    if eligible_rows.empty:
        continue
    holdout_choice[uid] = eligible_rows.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0].item_idx


def build_feature_matrix(weights):
    return np.concatenate([BLOCKS_NORM[b] * weights[b] for b in BLOCKS_NORM], axis=1)


def ndcg_mrr(r):
    return (1.0 / np.log2(r + 1) if r <= 10 else 0.0, 1.0 / np.log2(r + 1) if r <= 20 else 0.0, 1.0 / r)


def run_config(weights, k_content, k_user, dir_w=0.5, actor_w=0.5, temp=0.35, seed=42):
    X = build_feature_matrix(weights)
    km = KMeans(n_clusters=k_content, random_state=seed, n_init=10).fit(X)
    dists = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    logits = -dists / temp
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    mixture = ex / ex.sum(axis=1, keepdims=True)

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
    km_user = KMeans(n_clusters=k_user, random_state=seed, n_init=10).fit(Ps)
    uid_to_cluster = dict(zip(all_uids, km_user.labels_))
    watch_c = watch.copy()
    watch_c["cluster"] = watch_c.user_id.map(uid_to_cluster)

    ranks, hit10, hit20, ndcg10s, ndcg20s, mrrs = [], [], [], [], [], []
    for uid, held_idx in holdout_choice.items():
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
        candidates = [i for i in ELIGIBLE_IDX if i not in watched]
        if held_idx not in candidates:
            continue
        cl_rate = (cl_viewers.reindex(candidates, fill_value=0) + 0.5) / (cl_size + 1.0)
        n_loo = wc.user_id.nunique()
        pop_rate = (overall_pop.reindex(candidates, fill_value=0) + 0.5) / (n_loo + 1.0)

        user_dirs = set().union(*(director_sets[i] for i in remaining.item_idx.values)) if len(remaining) else set()
        user_actors = set().union(*(actor_sets[i] for i in remaining.item_idx.values)) if len(remaining) else set()

        score = {}
        for c in candidates:
            boost = 1.0 + dir_w * len(director_sets[c] & user_dirs) + actor_w * len(actor_sets[c] & user_actors)
            score[c] = (pop_rate.get(c, 0.0) ** 0.7) * (cl_rate.get(c, 0.0) ** 0.3) * boost
        ranked = sorted(score, key=lambda k: score[k], reverse=True)
        r = ranked.index(held_idx) + 1
        ranks.append(r); hit10.append(r <= 10); hit20.append(r <= 20)
        n10, n20, mrr = ndcg_mrr(r)
        ndcg10s.append(n10); ndcg20s.append(n20); mrrs.append(mrr)

    return {
        "n_eval": len(ranks), "mean_rank": np.mean(ranks), "hit10": np.mean(hit10), "hit20": np.mean(hit20),
        "ndcg10": np.mean(ndcg10s), "ndcg20": np.mean(ndcg20s), "mrr": np.mean(mrrs),
    }


if __name__ == "__main__":
    DEFAULT_W = {"genre": 3.0, "storyline": 3.0, "tone": 2.0, "era": 1.0, "maturity": 1.0}

    print("\n=== WEIGHT SWEEP (K_CONTENT=8, K_USER=6 fixed) ===")
    weight_variants = {
        "default (3,3,2,1,1)": DEFAULT_W,
        "storyline-heavy (2,5,2,1,1)": {"genre": 2.0, "storyline": 5.0, "tone": 2.0, "era": 1.0, "maturity": 1.0},
        "tone-heavy (2,3,4,1,1)": {"genre": 2.0, "storyline": 3.0, "tone": 4.0, "era": 1.0, "maturity": 1.0},
        "genre-light (1,4,3,1,1)": {"genre": 1.0, "storyline": 4.0, "tone": 3.0, "era": 1.0, "maturity": 1.0},
        "balanced (1,1,1,1,1)": {"genre": 1.0, "storyline": 1.0, "tone": 1.0, "era": 1.0, "maturity": 1.0},
        "genre-only (5,0.1,0.1,0.1,0.1)": {"genre": 5.0, "storyline": 0.1, "tone": 0.1, "era": 0.1, "maturity": 0.1},
    }
    for label, w in weight_variants.items():
        r = run_config(w, 8, 6)
        print(f"[{label}] n_eval={r['n_eval']} mean_rank={r['mean_rank']:.1f} hit10={r['hit10']:.3f} "
              f"hit20={r['hit20']:.3f} ndcg10={r['ndcg10']:.3f} ndcg20={r['ndcg20']:.3f} mrr={r['mrr']:.3f}")

    print("\n=== K SWEEP (default weights fixed) ===")
    for k_content, k_user in [(8, 6), (10, 6), (12, 8), (14, 8), (16, 10), (6, 4), (20, 10)]:
        r = run_config(DEFAULT_W, k_content, k_user)
        print(f"[K_CONTENT={k_content}, K_USER={k_user}] n_eval={r['n_eval']} mean_rank={r['mean_rank']:.1f} "
              f"hit10={r['hit10']:.3f} hit20={r['hit20']:.3f} ndcg10={r['ndcg10']:.3f} ndcg20={r['ndcg20']:.3f} mrr={r['mrr']:.3f}")
