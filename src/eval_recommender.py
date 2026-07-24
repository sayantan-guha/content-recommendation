"""
Evaluation harness for the shared recommender (src/recommender.py). Reuses the
same held-out leave-one-out (LOO) methodology as every prior pipeline script's
inline validation, but as a single reusable script with:

  1. Ranking-quality metrics beyond Hit@10/20: NDCG@10/20 and MRR, which reward
     *where* in the top-N the held-out title lands, not just whether it made
     the cut.
  2. A segmented view by task, since a single blended number hides where the
     model actually struggles:
       - "warm-item recovery" -- the standard LOO test (can we recover a title
         a user watched, from their remaining history?). Only titles with
         >=5 distinct viewers are eligible as LOO targets, since there's no
         reliable signal to test recovery of a title only 1-4 people watched.
       - "cold-start exposure" -- LOO can't test recovery of cold titles (too
         little data to hide-and-recover), so this instead measures whether
         the cold-start fallback (src/recommender.py's cold_start_candidates)
         actually gets low-data titles in front of users at all, comparing
         the fallback-enabled recommender against a fallback-disabled one.

Run: python3 src/eval_recommender.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recommender as rec


def ndcg_at_k(rank, k):
    return 1.0 / np.log2(rank + 1) if rank <= k else 0.0


def warm_item_recovery(model, audience, top_n=20):
    """Standard LOO test: hide one watched (eligible) title per user, rebuild
    their profile from the rest, rank all eligible candidates, see where the
    held-out title lands."""
    watch = audience["watch"]
    eligible_idx = audience["eligible_idx"]
    eval_users = audience["eval_users"]

    rng = np.random.default_rng(13)
    holdout_choice = {}
    for uid in eval_users:
        rows_ = watch[watch.user_id == uid]
        eligible_rows = rows_[rows_.item_idx.isin(eligible_idx)]
        if eligible_rows.empty:
            continue
        holdout_choice[uid] = eligible_rows.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0].item_idx

    ranks, hit10, hit20, ndcg10, ndcg20, mrr = [], [], [], [], [], []
    for uid, held_idx in holdout_choice.items():
        _, ranked, _ = rec.recommend_for_user(uid, held_idx, model, audience, top_n=top_n)
        if held_idx not in ranked:
            continue
        r = ranked.index(held_idx) + 1
        ranks.append(r)
        hit10.append(r <= 10)
        hit20.append(r <= 20)
        ndcg10.append(ndcg_at_k(r, 10))
        ndcg20.append(ndcg_at_k(r, 20))
        mrr.append(1.0 / r)

    return {
        "n_eval": len(ranks),
        "mean_rank": np.mean(ranks),
        "hit10": np.mean(hit10),
        "hit20": np.mean(hit20),
        "ndcg10": np.mean(ndcg10),
        "ndcg20": np.mean(ndcg20),
        "mrr": np.mean(mrr),
    }


def cold_start_exposure(model, audience, top_n=20, sample_size=500):
    """Does the cold-start fallback actually get low-data titles in front of
    users? Compares the shipped recommender (fallback on) against the same
    recommender with the fallback forced off, on a sample of users."""
    eligible_idx = audience["eligible_idx"]
    eval_users = audience["eval_users"]
    rng = np.random.default_rng(7)
    sample = rng.choice(eval_users, size=min(sample_size, len(eval_users)), replace=False)

    exposed_with_fallback, exposed_without_fallback = 0, 0
    for uid in sample:
        top, _, _ = rec.recommend_for_user(uid, None, model, audience, top_n=top_n)
        if any(i not in eligible_idx for i in top):
            exposed_with_fallback += 1

        orig_quota = rec.COLD_START_QUOTA_FRACTION
        rec.COLD_START_QUOTA_FRACTION = 0.0
        try:
            top_no_fallback, _, _ = rec.recommend_for_user(uid, None, model, audience, top_n=top_n)
        finally:
            rec.COLD_START_QUOTA_FRACTION = orig_quota
        if any(i not in eligible_idx for i in top_no_fallback):
            exposed_without_fallback += 1

    return {
        "n_users": len(sample),
        "pct_users_with_cold_title_fallback_on": exposed_with_fallback / len(sample),
        "pct_users_with_cold_title_fallback_off": exposed_without_fallback / len(sample),
    }


if __name__ == "__main__":
    model = rec.load_content_model()
    audience = rec.load_audience_model(model)

    print(f"catalog size: {len(model['series_content'])} titles, "
          f"eligible (warm, >=5 viewers): {len(audience['eligible_idx'])}, "
          f"eval users: {len(audience['eval_users'])}")

    warm = warm_item_recovery(model, audience, top_n=20)
    print(f"\n[WARM-ITEM RECOVERY] n_eval={warm['n_eval']} mean_rank={warm['mean_rank']:.1f} "
          f"hit10={warm['hit10']:.3f} hit20={warm['hit20']:.3f} "
          f"ndcg10={warm['ndcg10']:.3f} ndcg20={warm['ndcg20']:.3f} mrr={warm['mrr']:.3f}")

    cold = cold_start_exposure(model, audience, top_n=20)
    print(f"\n[COLD-START EXPOSURE] n_users={cold['n_users']} "
          f"with_fallback={cold['pct_users_with_cold_title_fallback_on']:.1%} "
          f"without_fallback={cold['pct_users_with_cold_title_fallback_off']:.1%}")
