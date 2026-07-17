"""
Shared recommendation-model logic: same model as src/pipeline_full_catalog.py
(8 content-category clusters, 6 audience clusters, popularity^0.7 x cluster_rate^0.3
x creator_boost scoring). Framework-agnostic -- used by backend/app.py (FastAPI)
and importable anywhere else that needs the model without a UI dependency.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA = str(Path(__file__).resolve().parent.parent / "data")
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
    }


def load_audience_model(model):
    watch_ep = pd.read_csv(f"{DATA}/user_title_watch_sample_2218.csv")
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
    eval_users = sorted(user_counts[user_counts >= 4].index.tolist())

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


def recommend_for_user(uid, held_out_idx, model, audience, top_n=10):
    watch = audience["watch"]
    watch_c = audience["watch_c"]
    scaler = audience["scaler"]
    km_user = audience["km_user"]
    eligible_idx = audience["eligible_idx"]
    overall_pop = audience["overall_pop"]
    build_profile = audience["build_profile"]
    director_sets = model["director_sets"]
    actor_sets = model["actor_sets"]

    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.item_idx != held_out_idx] if held_out_idx is not None else user_rows
    profile = build_profile(remaining.item_idx.values, remaining.seconds_watched.values)
    profile_s = scaler.transform(profile.reshape(1, -1))
    d = np.linalg.norm(km_user.cluster_centers_ - profile_s, axis=1)
    assigned_cluster = int(np.argmin(d))

    mask_drop = (watch_c.user_id == uid) & (watch_c.item_idx == held_out_idx)
    wc = watch_c[~mask_drop] if held_out_idx is not None else watch_c
    cl_viewers = wc[wc.cluster == assigned_cluster].groupby("item_idx").user_id.nunique()
    cl_size = wc[wc.cluster == assigned_cluster].user_id.nunique()

    watched = set(user_rows.item_idx) - ({held_out_idx} if held_out_idx is not None else set())
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
    return ranked[:top_n], ranked
