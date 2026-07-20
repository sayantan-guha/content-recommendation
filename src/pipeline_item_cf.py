"""
Item-item collaborative filtering: "users who watched X also watched Y",
scored by cosine similarity over the binary user-item watch matrix.

Benchmarked against the cluster-based model (src/pipeline_full_catalog.py) on
the same 5,000-user sample, same held-out LOO methodology, same eligible-item
pool (>=5 viewers). Result:

  [popularity baseline]              hit10=0.168  hit20=0.258
  [cluster model, popularity^0.3]    hit10=0.184  hit20=0.277  (+1.6pp lift over popularity)
  [item-item CF]                     hit10=0.296  hit20=0.398  (+12.8pp lift over popularity)

CF's lift over popularity is ~8x the cluster model's. The cluster model's
6 audience clusters concentrate 69% of users into just 2 "generic drama
viewer" blobs (verified: those clusters' item-popularity correlates 0.95+
with GLOBAL popularity), so its "personalized" score is barely distinct
from popularity for most users. CF sidesteps this entirely -- every user is
scored against their own specific co-viewers, not a cluster average.

Refinements tested and REJECTED (all made it worse -- see src/experiments/
cf_refinements.py): weighting co-occurrence by seconds_watched (no change),
shrinkage regularization (hurt -- the >=5-viewer floor already filters out
the low-count pairs shrinkage targets), blending in the creator_boost or
the cluster model's score (both hurt -- diluted CF's sharper signal with a
weaker one). Plain binary cosine similarity is the best-performing variant
found so far.

Known limitation (why this isn't a drop-in full replacement): CF only works
for titles with enough co-occurrence data (the same >=5-viewer floor used
for evaluation) and users with >=1 watched eligible title. New/cold-start
titles and near-zero-history users need a fallback (content-model or
popularity) -- see recommend_via_cf() in src/recommender.py for how the
three tiers are meant to compose. This script only fits + validates CF
itself.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA = str(Path(__file__).resolve().parent.parent / "data")


def parse_list(s):
    if pd.isna(s):
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else [v]
    except Exception:
        return []


content = pd.read_csv(f"{DATA}/content_features_full_tagged.csv")
content["item_id"] = content.apply(
    lambda r: f"movie::{r.content_id}" if r.content_type == "movie" else f"series::{r.content_id}",
    axis=1,
)
item_to_idx = {s: i for i, s in enumerate(content.item_id.values)}
n_items = len(content)
print(f"full-catalog items: {n_items} "
      f"({(content.content_type == 'movie').sum()} movies + "
      f"{(content.content_type != 'movie').sum()} series)")

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


watch_ep = pd.read_csv(f"{DATA}/user_title_watch_sample_5000.csv")
watch_ep["item_id"] = watch_ep.content_id.apply(cid_to_item_id)
watch_ep = watch_ep.dropna(subset=["item_id"])
watch = watch_ep.groupby(["user_id", "item_id"], as_index=False)["seconds_watched"].sum()
watch["item_idx"] = watch.item_id.map(item_to_idx)
print(f"watch rows after mapping to full catalog: {len(watch)} "
      f"({watch.user_id.nunique()} users, {watch.item_idx.nunique()} distinct items watched)")

all_uids = sorted(watch.user_id.unique())
uid_to_row = {u: i for i, u in enumerate(all_uids)}
n_users = len(all_uids)

item_viewer_counts = watch.groupby("item_idx").user_id.nunique()
ELIGIBLE_IDX = sorted(item_viewer_counts[item_viewer_counts >= 5].index)
ELIG_SET = set(ELIGIBLE_IDX)
overall_pop = watch.groupby("item_idx").user_id.nunique()

user_counts = watch.groupby("user_id").size()
eval_users = user_counts[user_counts >= 4].index.tolist()

# ---------- build the item-item cosine-similarity matrix once ----------
rows_idx = watch.user_id.map(uid_to_row).values
cols_idx = watch.item_idx.values
UI = csr_matrix((np.ones(len(watch)), (rows_idx, cols_idx)), shape=(n_users, n_items))
II = (UI.T @ UI).toarray()
item_norm = np.sqrt(np.array(UI.multiply(UI).sum(axis=0)).flatten())
item_norm[item_norm == 0] = 1.0
SIM = II / (item_norm[:, None] * item_norm[None, :])
np.fill_diagonal(SIM, 0.0)


def recommend_cf(watched_item_idxs, watched_all_idxs, top_n=10):
    """CF recommendation for a set of watched items. watched_all_idxs is
    excluded from candidates (everything the user has already seen)."""
    candidates = [i for i in ELIGIBLE_IDX if i not in watched_all_idxs]
    if not candidates or len(watched_item_idxs) == 0:
        return []
    cand_arr = np.array(candidates)
    scores = SIM[np.array(list(watched_item_idxs))][:, cand_arr].sum(axis=0)
    order = np.argsort(-scores)
    return cand_arr[order][:top_n].tolist()


# ---------- held-out validation (same LOO methodology as pipeline_full_catalog.py) ----------
rng = np.random.default_rng(13)
holdout_choice = {}
for uid in eval_users:
    rows_ = watch[watch.user_id == uid]
    choice = rows_.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
    holdout_choice[uid] = choice.item_idx

ranks, pop_ranks = [], []
for uid in eval_users:
    held_idx = holdout_choice[uid]
    if held_idx not in ELIG_SET:
        continue
    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.item_idx != held_idx]
    if len(remaining) == 0:
        continue
    watched_all = set(user_rows.item_idx) - {held_idx}
    candidates = [i for i in ELIGIBLE_IDX if i not in watched_all]
    if held_idx not in candidates:
        continue
    cand_arr = np.array(candidates)
    scores = SIM[remaining.item_idx.values][:, cand_arr].sum(axis=0)
    order = np.argsort(-scores)
    ranked = cand_arr[order].tolist()
    ranks.append(ranked.index(held_idx) + 1)

    pop = overall_pop.reindex(candidates, fill_value=0)
    pranked = pop.sort_values(ascending=False).index.tolist()
    pop_ranks.append(pranked.index(held_idx) + 1)

print(f"\n[ITEM-ITEM CF] n_eval={len(ranks)} mean_rank={np.mean(ranks):.1f} "
      f"hit10={np.mean([r <= 10 for r in ranks]):.3f} hit20={np.mean([r <= 20 for r in ranks]):.3f}")
print(f"[POPULARITY baseline]  n_eval={len(pop_ranks)} mean_rank={np.mean(pop_ranks):.1f} "
      f"hit10={np.mean([r <= 10 for r in pop_ranks]):.3f} hit20={np.mean([r <= 20 for r in pop_ranks]):.3f}")
