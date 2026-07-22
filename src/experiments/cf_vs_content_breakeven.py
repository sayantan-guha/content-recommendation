"""
Find the empirical breakeven point for MIN_WATCHED_FOR_CF: at each watch-
history-size bucket, compare CF-only vs content-similarity-only recovery of
a held-out title (same LOO methodology as eval_recommender.py), to see
where CF actually starts beating content-based ranking rather than assuming
a threshold.
"""
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
import recommender as rec

model = rec.load_content_model()
audience = rec.load_audience_model(model)

watch = audience["watch"]
sim = audience["sim"]
eligible_idx = audience["eligible_idx"]
build_profile = audience["build_profile"]

eval_users = audience["eval_users"]
rng = np.random.default_rng(13)
holdout_choice = {}
for uid in eval_users:
    rows_ = watch[watch.user_id == uid]
    eligible_rows = rows_[rows_.item_idx.isin(eligible_idx)]
    if eligible_rows.empty:
        continue
    holdout_choice[uid] = eligible_rows.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0].item_idx


def ndcg_mrr(r):
    return (1.0 / np.log2(r + 1) if r <= 10 else 0.0, 1.0 / np.log2(r + 1) if r <= 20 else 0.0, 1.0 / r)


# bucket by number of REMAINING watched titles after holdout (what the scorer
# actually sees), same variable recommend_for_user calls `remaining`
buckets = defaultdict(lambda: {"cf": [], "content": []})

for uid, held_idx in holdout_choice.items():
    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.item_idx != held_idx]
    n_remaining = len(remaining)
    if n_remaining == 0:
        continue
    watched_all = set(user_rows.item_idx) - {held_idx}
    candidates = [i for i in eligible_idx if i not in watched_all]
    if held_idx not in candidates:
        continue
    cand_arr = np.array(candidates)
    watched_idx = remaining.item_idx.values

    # CF-only rank
    cf_scores = sim[watched_idx][:, cand_arr].sum(axis=0)
    cf_order = np.argsort(-cf_scores)
    cf_ranked = cand_arr[cf_order].tolist()
    cf_rank = cf_ranked.index(held_idx) + 1

    # content-based-only rank (same helper used for the cold-start fallback / edge cases)
    profile = build_profile(watched_idx, remaining.seconds_watched.values)
    _, content_scores = rec.content_based_ranking(model, profile, cand_arr, watched_idx)
    content_order = np.argsort(-content_scores)
    content_ranked = cand_arr[content_order].tolist()
    content_rank = content_ranked.index(held_idx) + 1

    bucket = n_remaining if n_remaining <= 6 else ("7-10" if n_remaining <= 10 else "11+")
    buckets[bucket]["cf"].append(cf_rank)
    buckets[bucket]["content"].append(content_rank)


def summarize(ranks):
    if not ranks:
        return None
    hit10 = np.mean([r <= 10 for r in ranks])
    hit20 = np.mean([r <= 20 for r in ranks])
    mean_rank = np.mean(ranks)
    ndcgs = [ndcg_mrr(r) for r in ranks]
    ndcg10 = np.mean([n[0] for n in ndcgs])
    mrr = np.mean([n[2] for n in ndcgs])
    return {"n": len(ranks), "mean_rank": mean_rank, "hit10": hit10, "hit20": hit20, "ndcg10": ndcg10, "mrr": mrr}


order_keys = sorted([k for k in buckets if isinstance(k, int)]) + [k for k in ("7-10", "11+") if k in buckets]
print(f"{'bucket':>8} | {'n':>5} | {'CF hit10':>9} {'CF hit20':>9} {'CF mrr':>8} | {'Content hit10':>13} {'Content hit20':>13} {'Content mrr':>11} | winner")
for b in order_keys:
    cf_s = summarize(buckets[b]["cf"])
    ct_s = summarize(buckets[b]["content"])
    if not cf_s or not ct_s:
        continue
    winner = "CF" if cf_s["hit10"] > ct_s["hit10"] else ("content" if ct_s["hit10"] > cf_s["hit10"] else "tie")
    print(f"{str(b):>8} | {cf_s['n']:>5} | {cf_s['hit10']:>9.3f} {cf_s['hit20']:>9.3f} {cf_s['mrr']:>8.3f} | "
          f"{ct_s['hit10']:>13.3f} {ct_s['hit20']:>13.3f} {ct_s['mrr']:>11.3f} | {winner}")
