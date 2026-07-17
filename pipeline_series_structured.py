import pandas as pd
import numpy as np
import ast
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA = '/Users/harisankarm/Documents/GIM/Hoichoi/content_recommendation/data'

content = pd.read_csv(f'{DATA}/content_features_500_tagged.csv')

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

# ---------- STRUCTURED linkage replaces permalink-slug parsing ----------
# content_series_id / content_season_id / content_season_number come directly from
# cms_v_watch_history_with_content (companion row), NOT from parsing the permalink URL.
# This fixes the SEO-stuffed-slug bug (e.g. Eken Babu S9, Feludar Goyendagiri S3) where
# episode numbers embedded in the URL slug caused each episode to look like its own series.
struct = pd.read_csv(f'{DATA}/structured_linkage_350ep.csv')
struct['content_id'] = struct['content_id'].astype(str)
sid_map = dict(zip(struct.content_id, struct.series_id))
season_id_map = dict(zip(struct.content_id, struct.season_id))
season_num_map = dict(zip(struct.content_id, struct.season_number))
episode_num_map = dict(zip(struct.content_id, struct.episode_number))

def series_id_for(row):
    if row.content_type == 'movie':
        return f"movie::{row.content_id}"
    sid = sid_map.get(row.content_id)
    if not sid or pd.isna(sid):
        return f"singleton::{row.content_id}"  # no structured linkage found -- true standalone
    return f"struct::{sid}"

content['series_id'] = content.apply(series_id_for, axis=1)
content['season_id'] = content['content_id'].map(season_id_map)
content['season_number'] = content['content_id'].map(season_num_map)
content['episode_number'] = content['content_id'].map(episode_num_map)

def most_common(vals):
    vals = [v for v in vals if pd.notna(v)]
    return Counter(vals).most_common(1)[0][0] if vals else None

# Real show titles from cms_v_series_latest, keyed by the raw (unprefixed) content_series_id.
# FIX: display_name was previously most_common(episode titles) -- since almost every episode
# in a series has a UNIQUE title (e.g. Eken Babu S9's episodes are "Panna Udhao", "Humkir Teer",
# ...), most_common() on all-distinct values just returns an arbitrary single episode's title,
# so recommendations showed as if they were individual one-off titles instead of "Eken Babu
# Season 9". Pulled real show titles directly from cms_v_series_latest instead.
show_titles = dict(zip(pd.read_csv(f'{DATA}/series_display_names.csv').series_id,
                        pd.read_csv(f'{DATA}/series_display_names.csv').show_title))

rows = []
for sid, grp in content.groupby('series_id'):
    if grp.content_type.iloc[0] == 'movie':
        display_name = grp.title_english.iloc[0]
    else:
        raw_series_id = sid.split('::', 1)[1] if sid.startswith('struct::') else None
        show_title = show_titles.get(raw_series_id)
        if show_title is None:
            # singleton:: (no structured linkage found) -- true standalone episode, keep its own title
            display_name = grp.title_english.iloc[0]
        else:
            season_numbers = sorted(set(int(n) for n in grp['season_number'].dropna().unique() if n > 0))
            display_name = f"{show_title} Season {season_numbers[0]}" if len(season_numbers) == 1 else show_title
    rows.append({
        'series_id': sid,
        'display_name': display_name,
        # recommendation-facing type is always 'movie' or 'series', NEVER 'episode' --
        # grp.content_type.iloc[0] is the raw underlying row type (e.g. 'episode' for
        # every non-movie row, even single-episode shows), which is an implementation
        # detail of how the row was sourced, not what should be shown to a user picking
        # from a recommendation list.
        'content_type': 'movie' if sid.startswith('movie::') else 'series',
        'genre_normalized': most_common(grp.genre_normalized),
        'era_bucket': most_common(grp.era_bucket),
        '_storyline': sorted(set(t for lst in grp['_storyline'] for t in lst)),
        '_tone': sorted(set(t for lst in grp['_tone'] for t in lst)),
        '_maturity': sorted(set(t for lst in grp['_maturity'] for t in lst)),
        '_director': sorted(set(t for lst in grp['_director'] for t in lst)),
        '_actor': sorted(set(t for lst in grp['_actor'] for t in lst)),
        'n_episodes_tagged': len(grp),
    })
series_content = pd.DataFrame(rows)
print(f"series-level items: {len(series_content)} "
      f"({(series_content.content_type=='movie').sum()} movies + "
      f"{(series_content.content_type!='movie').sum()} series/singletons)")

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

series_ids = series_content.series_id.values
sid_to_idx = {s: i for i, s in enumerate(series_ids)}
cid_to_series = dict(zip(content.content_id, content.series_id))
DIRECTOR_SETS = [set(x) for x in series_content['_director']]
ACTOR_SETS = [set(x) for x in series_content['_actor']]
DIR_W, ACTOR_W = 0.5, 0.5

# ---------- fit on the 2218-user training set (same watch data, re-rolled-up under structured series_id) ----------
watch_ep = pd.read_csv(f'{DATA}/user_title_watch_sample_2218.csv')
watch_ep = watch_ep[watch_ep.content_id.isin(cid_to_series)].copy()
watch_ep['series_id'] = watch_ep.content_id.map(cid_to_series)
watch = watch_ep.groupby(['user_id', 'series_id'], as_index=False)['seconds_watched'].sum()
watch['series_idx'] = watch.series_id.map(sid_to_idx)

def build_profile_from_idx_secs(idxs, secs):
    secs = np.array(secs, dtype=float)
    w = secs / secs.sum()
    cat = mixture[idxs]
    return (cat * w[:, None]).sum(axis=0)

profiles = {}
for uid, rows_ in watch.groupby('user_id'):
    profiles[uid] = build_profile_from_idx_secs(rows_.series_idx.values, rows_.seconds_watched.values)
all_uids = list(profiles.keys())
P = np.array([profiles[u] for u in all_uids])
scaler = StandardScaler().fit(P)
Ps = scaler.transform(P)
km_user = KMeans(n_clusters=K_USER, random_state=42, n_init=10).fit(Ps)
uid_to_cluster = dict(zip(all_uids, km_user.labels_))
watch_c = watch.copy()
watch_c['cluster'] = watch_c.user_id.map(uid_to_cluster)

series_viewer_counts = watch.groupby('series_idx').user_id.nunique()
ELIGIBLE_IDX = set(series_viewer_counts[series_viewer_counts >= 5].index)

# ---------- held-out validation (same LOO methodology as the adopted baseline) ----------
rng = np.random.default_rng(13)
user_counts = watch.groupby('user_id').size()
eval_users = user_counts[user_counts >= 4].index.tolist()

holdout_choice = {}
for uid in eval_users:
    rows_ = watch[watch.user_id == uid]
    choice = rows_.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
    holdout_choice[uid] = choice.series_idx

overall_pop_full = watch.groupby('series_idx').user_id.nunique()
ranks, hit10, hit20 = [], [], []
pop_ranks, pop_hit10, pop_hit20 = [], [], []

for uid in eval_users:
    held_idx = holdout_choice[uid]
    if held_idx not in ELIGIBLE_IDX:
        continue
    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.series_idx != held_idx]
    if len(remaining) == 0:
        continue
    profile = build_profile_from_idx_secs(remaining.series_idx.values, remaining.seconds_watched.values)
    profile_s = scaler.transform(profile.reshape(1, -1))
    d = np.linalg.norm(km_user.cluster_centers_ - profile_s, axis=1)
    assigned_cluster = int(np.argmin(d))

    mask_drop = (watch_c.user_id == uid) & (watch_c.series_idx == held_idx)
    wc = watch_c[~mask_drop]
    cl_viewers = wc[wc.cluster == assigned_cluster].groupby('series_idx').user_id.nunique()
    cl_size = wc[wc.cluster == assigned_cluster].user_id.nunique()

    watched = set(user_rows.series_idx) - {held_idx}
    candidates = [i for i in ELIGIBLE_IDX if i not in watched]
    cl_rate = (cl_viewers.reindex(candidates, fill_value=0) + 0.5) / (cl_size + 1.0)
    n_loo = wc.user_id.nunique()
    pop_rate = (overall_pop_full.reindex(candidates, fill_value=0) + 0.5) / (n_loo + 1.0)

    user_dirs = set().union(*(DIRECTOR_SETS[i] for i in remaining.series_idx.values)) if len(remaining) else set()
    user_actors = set().union(*(ACTOR_SETS[i] for i in remaining.series_idx.values)) if len(remaining) else set()
    score = {}
    for c in candidates:
        boost = 1.0 + DIR_W * len(DIRECTOR_SETS[c] & user_dirs) + ACTOR_W * len(ACTOR_SETS[c] & user_actors)
        # re-tuned after structured-linkage consolidation: consolidating fragmented
        # pseudo-series into real series concentrated viewer counts onto far fewer,
        # much more popular items, which made overall popularity a stronger signal
        # than cluster-rate alone. pop_rate^0.7 x cl_rate^0.3 recovered and slightly
        # beat pure popularity on hit20 while keeping personalization from cl_rate/boost.
        score[c] = (pop_rate.get(c, 0.0) ** 0.7) * (cl_rate.get(c, 0.0) ** 0.3) * boost
    ranked = sorted(score, key=lambda k: score[k], reverse=True)
    r = ranked.index(held_idx) + 1
    pool = len(ranked)
    ranks.append(r); hit10.append(r <= 10); hit20.append(r <= 20)

    pop = overall_pop_full.reindex(candidates, fill_value=0)
    pranked = pop.sort_values(ascending=False).index.tolist()
    pr = pranked.index(held_idx) + 1
    pop_ranks.append(pr); pop_hit10.append(pr <= 10); pop_hit20.append(pr <= 20)

print(f"\n[STRUCTURED-LINKAGE] n_eval={len(ranks)} mean_rank={np.mean(ranks):.1f} "
      f"hit10={np.mean(hit10):.3f} hit20={np.mean(hit20):.3f}")
print(f"[POPULARITY baseline]  n_eval={len(pop_ranks)} mean_rank={np.mean(pop_ranks):.1f} "
      f"hit10={np.mean(pop_hit10):.3f} hit20={np.mean(pop_hit20):.3f}")

series_content.to_csv(f'{DATA}/content_features_series_rollup_structured.csv', index=False)
print(f"\nsaved {DATA}/content_features_series_rollup_structured.csv")
