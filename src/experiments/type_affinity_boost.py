"""
Experiment: does adding an explicit movie/series type-affinity term to the
scoring formula improve recommendations?

Two candidate signals for a user's type preference are compared, at several
boost strengths, against the unmodified baseline model:

  - type_titlecount -- movie/series split by DISTINCT TITLES watched
  - type_watchtime  -- movie/series split by TOTAL SECONDS watched (structurally
                       biased toward series, since a series accumulates seconds
                       across many episodes -- included to quantify that bias)

Type-affinity boost: for a candidate of type t, boost_type = 1 + TYPE_W *
(pref_t - 0.5) * 2, where pref_t is the user's preference share for type t
(from remaining/training watch rows only, never the held-out item). This is
symmetric around a neutral 50/50 split (boost=1) and scales up to 1+TYPE_W
for a user who has watched only one type.

Result (2,218-user sample, held-out LOO validation):
  [baseline,        TYPE_W=0.0] hit10=0.604 hit20=0.737
  [type_titlecount, TYPE_W=0.5] hit10=0.604 hit20=0.737  (no measurable effect)
  [type_watchtime,  TYPE_W=0.5] hit10=0.602 hit20=0.738  (no measurable effect)
  [type_titlecount, TYPE_W=1.0] hit10=0.588 hit20=0.723  (hurts)
  [type_watchtime,  TYPE_W=1.0] hit10=0.575 hit20=0.706  (hurts more)
  [type_titlecount, TYPE_W=2.0] hit10=0.535 hit20=0.659  (hurts a lot)
  [type_watchtime,  TYPE_W=2.0] hit10=0.518 hit20=0.618  (hurts more)

Conclusion: at any weight strong enough to matter, an explicit type-affinity
term hurts held-out performance -- the existing 8-way category clustering
already implicitly separates movie-leaning vs. series-leaning content
(certain storylines/tones map naturally to one format or the other), so this
term is redundant with signal the model already has, and only crowds out
better-fitting candidates once weighted up. Watch-time is consistently worse
than title-count as the underlying signal, confirming that watch-time is a
noisier, episode-count-biased proxy for type preference. Not recommended for
production; see src/experiments/type_quota_slate.py for a different (slate
construction, not scoring) approach to the same question.
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA = str(Path(__file__).resolve().parent.parent.parent / "data")
K_CONTENT, K_USER = 8, 6
DIR_W, ACTOR_W = 0.5, 0.5


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


def fit_content_model():
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
    is_series = (series_content["content_type"] == "series").values

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
        "is_series": is_series,
        "cid_to_item_id": cid_to_item_id,
    }


def fit_audience_model(model, watch_sample_path):
    watch_ep = pd.read_csv(watch_sample_path)
    watch_ep["item_id"] = watch_ep.content_id.apply(model["cid_to_item_id"])
    watch_ep = watch_ep.dropna(subset=["item_id"])
    watch = watch_ep.groupby(["user_id", "item_id"], as_index=False)["seconds_watched"].sum()
    watch["item_idx"] = watch.item_id.map(model["item_to_idx"])

    mixture = model["mixture"]

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
    km_user = KMeans(n_clusters=K_USER, random_state=42, n_init=10).fit(Ps)
    uid_to_cluster = dict(zip(all_uids, km_user.labels_))
    watch_c = watch.copy()
    watch_c["cluster"] = watch_c.user_id.map(uid_to_cluster)

    item_viewer_counts = watch.groupby("item_idx").user_id.nunique()
    eligible_idx = set(item_viewer_counts[item_viewer_counts >= 5].index)
    overall_pop = watch.groupby("item_idx").user_id.nunique()

    user_counts = watch.groupby("user_id").size()
    eval_users = user_counts[user_counts >= 4].index.tolist()

    return {
        "watch": watch,
        "watch_c": watch_c,
        "scaler": scaler,
        "km_user": km_user,
        "eligible_idx": eligible_idx,
        "overall_pop": overall_pop,
        "eval_users": eval_users,
        "build_profile": build_profile,
    }


def type_pref_from_titlecount(is_series, idxs):
    n_series = sum(is_series[i] for i in idxs)
    n = len(idxs)
    return n_series / n if n else 0.5


def type_pref_from_watchtime(is_series, idxs, secs):
    secs = np.array(secs, dtype=float)
    series_secs = secs[[is_series[i] for i in idxs]].sum()
    total = secs.sum()
    return series_secs / total if total else 0.5


def type_boost(is_candidate_series, pref_series, type_w):
    pref_t = pref_series if is_candidate_series else (1 - pref_series)
    return 1.0 + type_w * (pref_t - 0.5) * 2


def evaluate(model, audience, variant, type_w, rng_seed=13):
    watch = audience["watch"]
    watch_c = audience["watch_c"]
    scaler = audience["scaler"]
    km_user = audience["km_user"]
    eligible_idx = audience["eligible_idx"]
    overall_pop = audience["overall_pop"]
    build_profile = audience["build_profile"]
    director_sets = model["director_sets"]
    actor_sets = model["actor_sets"]
    is_series = model["is_series"]

    rng = np.random.default_rng(rng_seed)
    eval_users = audience["eval_users"]
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

        if variant == "type_titlecount":
            pref_series = type_pref_from_titlecount(is_series, remaining.item_idx.values)
        elif variant == "type_watchtime":
            pref_series = type_pref_from_watchtime(is_series, remaining.item_idx.values, remaining.seconds_watched.values)
        else:
            pref_series = None

        score = {}
        for c in candidates:
            boost = 1.0 + DIR_W * len(director_sets[c] & user_dirs) + ACTOR_W * len(actor_sets[c] & user_actors)
            if pref_series is not None:
                boost *= type_boost(is_series[c], pref_series, type_w)
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
    watch_sample = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/user_title_watch_sample_2218.csv"
    model = fit_content_model()
    audience = fit_audience_model(model, watch_sample)
    print(f"watch sample: {watch_sample} ({audience['watch'].user_id.nunique()} users)")
    for variant, type_w in [("baseline", 0.0), ("type_titlecount", 0.5), ("type_watchtime", 0.5),
                             ("type_titlecount", 1.0), ("type_watchtime", 1.0),
                             ("type_titlecount", 2.0), ("type_watchtime", 2.0)]:
        res = evaluate(model, audience, variant, type_w)
        print(f"[{variant:16s} TYPE_W={type_w:.1f}] n_eval={res['n_eval']} mean_rank={res['mean_rank']:.1f} "
              f"hit10={res['hit10']:.3f} hit20={res['hit20']:.3f}")
