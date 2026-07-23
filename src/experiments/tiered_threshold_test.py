"""
Tests a user-proposed count-based tiering scheme against the current
production scorer (CF, blended with type/era quota + cold-start), using the
same held-out LOO methodology as eval_recommender.py's warm_item_recovery.

Proposed tiers (by remaining watched-title count after holdout):
    0            -> popularity only
    1-3          -> content-based + popularity blend
    4-10         -> content-based only
    11+          -> CF (same as production's primary scorer)

This directly contradicts what's already been tested and rejected in this
project: MIN_WATCHED_FOR_CF (a count-based CF/content-based switch) was
tried, found to be a net regression at every bucket down to 1 remaining
title (cf_vs_content_breakeven_v2.py), and removed; the leave-5-out re-test
(cf_vs_content_breakeven_leave5.py) confirmed it again with ~5x more trials
per bucket. This script re-checks that finding specifically for the
tiers proposed here, rather than relying on the earlier (differently
bucketed) results.

Run: python3 src/experiments/tiered_threshold_test.py
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
overall_pop = audience["overall_pop"]
build_profile = audience["build_profile"]
eval_users = audience["eval_users"]


def tier_for(n_remaining):
    if n_remaining == 0:
        return "0 (popularity)"
    if n_remaining <= 3:
        return "1-3 (content+pop blend)"
    if n_remaining <= 10:
        return "4-10 (content-based)"
    return "11+ (CF)"


def tiered_rank(watched_idx, secs, candidates_pool):
    """Returns the tiered-scheme's ranked candidate list for one user."""
    n = len(watched_idx)
    cand_arr = np.array(candidates_pool)

    if n == 0:
        pop = overall_pop.reindex(cand_arr, fill_value=0)
        return pop.sort_values(ascending=False).index.tolist()

    profile = build_profile(watched_idx, secs)
    _, cb_sims = rec.content_based_ranking(model, profile, cand_arr, watched_idx)

    if n <= 3:
        # combination of content-based and popularity: average each
        # signal's percentile rank so neither dominates on raw scale.
        cb_rank = np.argsort(np.argsort(-cb_sims)) / max(len(cb_sims) - 1, 1)
        pop_vals = overall_pop.reindex(cand_arr, fill_value=0).values.astype(float)
        pop_rank = np.argsort(np.argsort(-pop_vals)) / max(len(pop_vals) - 1, 1)
        blended = 0.5 * cb_rank + 0.5 * pop_rank
        order = np.argsort(blended)
        return cand_arr[order].tolist()

    if n <= 10:
        order = np.argsort(-cb_sims)
        return cand_arr[order].tolist()

    cf_scores = sim[watched_idx][:, cand_arr].sum(axis=0)
    order = np.argsort(-cf_scores)
    return cand_arr[order].tolist()


def ndcg_at_k(rank, k):
    return 1.0 / np.log2(rank + 1) if rank <= k else 0.0


rng = np.random.default_rng(13)
holdout_choice = {}
for uid in eval_users:
    rows_ = watch[watch.user_id == uid]
    eligible_rows = rows_[rows_.item_idx.isin(eligible_idx)]
    if eligible_rows.empty:
        continue
    holdout_choice[uid] = eligible_rows.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0].item_idx

buckets_tiered = defaultdict(list)
buckets_prod = defaultdict(list)

for uid, held_idx in holdout_choice.items():
    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.item_idx != held_idx]
    watched_idx = remaining.item_idx.values
    watched_all = set(user_rows.item_idx) - {held_idx}
    candidates = [i for i in eligible_idx if i not in watched_all]
    if held_idx not in candidates:
        continue

    tier = tier_for(len(watched_idx))

    tiered_ranked = tiered_rank(watched_idx, remaining.seconds_watched.values, candidates)
    if held_idx in tiered_ranked:
        buckets_tiered[tier].append(tiered_ranked.index(held_idx) + 1)

    _, prod_ranked = rec.recommend_for_user(uid, held_idx, model, audience, top_n=len(candidates) + 1)
    if held_idx in prod_ranked:
        buckets_prod[tier].append(prod_ranked.index(held_idx) + 1)


def summarize(ranks):
    if not ranks:
        return None
    hit10 = np.mean([r <= 10 for r in ranks])
    hit20 = np.mean([r <= 20 for r in ranks])
    mean_rank = np.mean(ranks)
    mrr = np.mean([1.0 / r for r in ranks])
    return {"n": len(ranks), "mean_rank": mean_rank, "hit10": hit10, "hit20": hit20, "mrr": mrr}


order = ["0 (popularity)", "1-3 (content+pop blend)", "4-10 (content-based)", "11+ (CF)"]
print(f"{'tier':>28} | {'n':>5} | {'tiered hit10':>12} {'tiered hit20':>12} {'tiered mrr':>10} | "
      f"{'prod hit10':>10} {'prod hit20':>10} {'prod mrr':>8} | winner")
for t in order:
    ts = summarize(buckets_tiered[t])
    ps = summarize(buckets_prod[t])
    if not ts or not ps:
        print(f"{t:>28} | (no eligible users in this bucket)")
        continue
    winner = "tiered" if ts["hit10"] > ps["hit10"] else ("production" if ps["hit10"] > ts["hit10"] else "tie")
    print(f"{t:>28} | {ts['n']:>5} | {ts['hit10']:>12.3f} {ts['hit20']:>12.3f} {ts['mrr']:>10.3f} | "
          f"{ps['hit10']:>10.3f} {ps['hit20']:>10.3f} {ps['mrr']:>8.3f} | {winner}")

all_tiered = [r for v in buckets_tiered.values() for r in v]
all_prod = [r for v in buckets_prod.values() for r in v]
ts, ps = summarize(all_tiered), summarize(all_prod)
print(f"\n{'OVERALL':>28} | {ts['n']:>5} | {ts['hit10']:>12.3f} {ts['hit20']:>12.3f} {ts['mrr']:>10.3f} | "
      f"{ps['hit10']:>10.3f} {ps['hit20']:>10.3f} {ps['mrr']:>8.3f}")
