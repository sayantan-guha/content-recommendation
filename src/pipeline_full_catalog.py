"""
Full-catalog discovery pipeline (movies + series scaled from the 500-title
sample to all 775 published titles: 472 movies + 303 series).

Structurally simpler than pipeline_series_structured.py: content_features_full_tagged.csv
already has ONE ROW PER SHOW (pulled directly from cms_v_series_latest), not one row
per episode -- so there's no need to roll up individually-tagged episodes into a
series entity via content_series_id grouping (that rollup was only necessary because
the original 500-title sample tagged individual EPISODES, not shows).

What's still needed: mapping a user's episode-level watch history (individual episode
content_ids) to the SHOW-level content_id used as the recommendation unit here. That's
what structured_linkage_full.csv's `series_id` column is for -- it already equals the
show-level content_id in content_features_full_tagged.csv's `series` rows (verified:
303/303 overlap).
"""
import pandas as pd
import numpy as np
import ast
from pathlib import Path
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA = str(Path(__file__).resolve().parent.parent / 'data')

content = pd.read_csv(f'{DATA}/content_features_full_tagged.csv')

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

content['_storyline'] = content['storyline_tags'].apply(parse_list)
content['_tone'] = content['overall_tone_tags'].apply(parse_list)
content['_maturity'] = content['maturity_tags'].apply(parse_list).apply(norm_maturity)
content['_director'] = content['director_names'].apply(parse_list)
content['_actor'] = content['actor_names'].apply(parse_list)

# item_id: the recommendation unit. Movies use their own content_id; series rows
# ALREADY are the show-level entity (one row per show), so they use their own
# content_id too -- unlike the 500-tagged pipeline, no series_id_for() rollup needed.
content['item_id'] = content.apply(
    lambda r: f"movie::{r.content_id}" if r.content_type == 'movie' else f"series::{r.content_id}",
    axis=1,
)

series_content = content.reset_index(drop=True)
print(f"full-catalog items: {len(series_content)} "
      f"({(series_content.content_type == 'movie').sum()} movies + "
      f"{(series_content.content_type != 'movie').sum()} series)")

STORYLINE_VOCAB = sorted(set(t for lst in series_content['_storyline'] for t in lst))
TONE_VOCAB = sorted(set(t for lst in series_content['_tone'] for t in lst))
MATURITY_VOCAB = sorted(set(t for lst in series_content['_maturity'] for t in lst))
GENRE_VOCAB = sorted(series_content['genre_normalized'].dropna().unique().tolist())
ERA_VOCAB = sorted(series_content['era_bucket'].dropna().unique().tolist())

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

BLOCKS = {
    'genre': onehot(series_content['genre_normalized'], GENRE_VOCAB),
    'storyline': multihot(series_content['_storyline'], STORYLINE_VOCAB),
    'tone': multihot(series_content['_tone'], TONE_VOCAB),
    'era': onehot(series_content['era_bucket'], ERA_VOCAB),
    'maturity': multihot(series_content['_maturity'], MATURITY_VOCAB),
}
WEIGHTS = {'genre': 3.0, 'storyline': 3.0, 'tone': 2.0, 'era': 1.0, 'maturity': 1.0}
X = np.concatenate([l2_normalize_rows(BLOCKS[b]) * WEIGHTS[b] for b in BLOCKS], axis=1)

K_CONTENT, K_USER = 8, 6
km = KMeans(n_clusters=K_CONTENT, random_state=42, n_init=10).fit(X)
dists = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
logits = -dists / 0.35
logits -= logits.max(axis=1, keepdims=True)
ex = np.exp(logits)
mixture = ex / ex.sum(axis=1, keepdims=True)

item_ids = series_content.item_id.values
item_to_idx = {s: i for i, s in enumerate(item_ids)}
DIRECTOR_SETS = [set(x) for x in series_content['_director']]
ACTOR_SETS = [set(x) for x in series_content['_actor']]
DIR_W, ACTOR_W = 0.5, 0.5

# content_id -> item_id: movies map to themselves; episodes map to their show via
# structured_linkage_full.csv's series_id (== the show's own content_id).
struct = pd.read_csv(f'{DATA}/structured_linkage_full.csv')
struct = struct.dropna(subset=['series_id'])
episode_to_show_cid = dict(zip(struct.content_id, struct.series_id))
movie_cids = set(content[content.content_type == 'movie'].content_id)
series_cids = set(content[content.content_type == 'series'].content_id)

def cid_to_item_id(cid):
    if cid in movie_cids:
        return f"movie::{cid}"
    show_cid = episode_to_show_cid.get(cid)
    if show_cid in series_cids:
        return f"series::{show_cid}"
    return None  # untagged/unlinked -- drop

# ---------- fit on the 2218-user training set ----------
watch_ep = pd.read_csv(f'{DATA}/user_title_watch_sample_2218.csv')
watch_ep['item_id'] = watch_ep.content_id.apply(cid_to_item_id)
watch_ep = watch_ep.dropna(subset=['item_id'])
watch = watch_ep.groupby(['user_id', 'item_id'], as_index=False)['seconds_watched'].sum()
watch['item_idx'] = watch.item_id.map(item_to_idx)
print(f"watch rows after mapping to full catalog: {len(watch)} "
      f"({watch.user_id.nunique()} users, {watch.item_idx.nunique()} distinct items watched)")

def build_profile_from_idx_secs(idxs, secs):
    secs = np.array(secs, dtype=float)
    w = secs / secs.sum()
    cat = mixture[idxs]
    return (cat * w[:, None]).sum(axis=0)

profiles = {}
for uid, rows_ in watch.groupby('user_id'):
    profiles[uid] = build_profile_from_idx_secs(rows_.item_idx.values, rows_.seconds_watched.values)
all_uids = list(profiles.keys())
P = np.array([profiles[u] for u in all_uids])
scaler = StandardScaler().fit(P)
Ps = scaler.transform(P)
km_user = KMeans(n_clusters=K_USER, random_state=42, n_init=10).fit(Ps)
uid_to_cluster = dict(zip(all_uids, km_user.labels_))
watch_c = watch.copy()
watch_c['cluster'] = watch_c.user_id.map(uid_to_cluster)

item_viewer_counts = watch.groupby('item_idx').user_id.nunique()
ELIGIBLE_IDX = set(item_viewer_counts[item_viewer_counts >= 5].index)

# ---------- held-out validation (same LOO methodology as the 500-tagged pipeline) ----------
rng = np.random.default_rng(13)
user_counts = watch.groupby('user_id').size()
eval_users = user_counts[user_counts >= 4].index.tolist()

holdout_choice = {}
for uid in eval_users:
    rows_ = watch[watch.user_id == uid]
    choice = rows_.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
    holdout_choice[uid] = choice.item_idx

overall_pop_full = watch.groupby('item_idx').user_id.nunique()
ranks, hit10, hit20 = [], [], []
pop_ranks, pop_hit10, pop_hit20 = [], [], []

for uid in eval_users:
    held_idx = holdout_choice[uid]
    if held_idx not in ELIGIBLE_IDX:
        continue
    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.item_idx != held_idx]
    if len(remaining) == 0:
        continue
    profile = build_profile_from_idx_secs(remaining.item_idx.values, remaining.seconds_watched.values)
    profile_s = scaler.transform(profile.reshape(1, -1))
    d = np.linalg.norm(km_user.cluster_centers_ - profile_s, axis=1)
    assigned_cluster = int(np.argmin(d))

    mask_drop = (watch_c.user_id == uid) & (watch_c.item_idx == held_idx)
    wc = watch_c[~mask_drop]
    cl_viewers = wc[wc.cluster == assigned_cluster].groupby('item_idx').user_id.nunique()
    cl_size = wc[wc.cluster == assigned_cluster].user_id.nunique()

    watched = set(user_rows.item_idx) - {held_idx}
    candidates = [i for i in ELIGIBLE_IDX if i not in watched]
    cl_rate = (cl_viewers.reindex(candidates, fill_value=0) + 0.5) / (cl_size + 1.0)
    n_loo = wc.user_id.nunique()
    pop_rate = (overall_pop_full.reindex(candidates, fill_value=0) + 0.5) / (n_loo + 1.0)

    user_dirs = set().union(*(DIRECTOR_SETS[i] for i in remaining.item_idx.values)) if len(remaining) else set()
    user_actors = set().union(*(ACTOR_SETS[i] for i in remaining.item_idx.values)) if len(remaining) else set()
    score = {}
    for c in candidates:
        boost = 1.0 + DIR_W * len(DIRECTOR_SETS[c] & user_dirs) + ACTOR_W * len(ACTOR_SETS[c] & user_actors)
        score[c] = (pop_rate.get(c, 0.0) ** 0.7) * (cl_rate.get(c, 0.0) ** 0.3) * boost
    ranked = sorted(score, key=lambda k: score[k], reverse=True)
    r = ranked.index(held_idx) + 1
    pool = len(ranked)
    ranks.append(r); hit10.append(r <= 10); hit20.append(r <= 20)

    pop = overall_pop_full.reindex(candidates, fill_value=0)
    pranked = pop.sort_values(ascending=False).index.tolist()
    pr = pranked.index(held_idx) + 1
    pop_ranks.append(pr); pop_hit10.append(pr <= 10); pop_hit20.append(pr <= 20)

print(f"\n[FULL-CATALOG] n_eval={len(ranks)} mean_rank={np.mean(ranks):.1f} "
      f"hit10={np.mean(hit10):.3f} hit20={np.mean(hit20):.3f}")
print(f"[POPULARITY baseline]  n_eval={len(pop_ranks)} mean_rank={np.mean(pop_ranks):.1f} "
      f"hit10={np.mean(pop_hit10):.3f} hit20={np.mean(pop_hit20):.3f}")

series_content.to_csv(f'{DATA}/content_features_full_catalog_rollup.csv', index=False)
print(f"\nsaved {DATA}/content_features_full_catalog_rollup.csv")
