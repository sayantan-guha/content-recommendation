"""
Experiment: instead of nudging scores (see type_affinity_boost.py), enforce a
user's movie/series split directly on the final slate. Movies and series are
ranked SEPARATELY by the existing baseline score, then interleaved with a
Bresenham-style largest-remainder method so that at every prefix length the
movie/series ratio matches the user's title-count split (computed from
remaining/training watch rows only, never the held-out item).

Two evaluation modes:
  - population: full 2,218-user held-out LOO test (one trial per user)
  - batch:      the N heaviest-history users, each tested over several
                held-out trials (their history is large enough to hold out
                many different titles), to see whether the quota's effect
                depends on how much watch history a user has

Results:

  [population, all eval users, N=1478 trials]
    baseline          hit10=0.604 hit20=0.737 mean_rank=18.4
    quota_titlecount  hit10=0.571 hit20=0.696 mean_rank=21.7   <- net WORSE

  [batch, 20 heaviest users (29-54 titles watched each), 20 trials/user]
    baseline          hit10=0.514 hit20=0.609 mean_rank=22.1
    quota_titlecount  hit10=0.524 hit20=0.622 mean_rank=24.6   <- net BETTER
    (7/20 users improved, 6/20 got worse, 7/20 unchanged -- mixed per-user,
     but the heavy-history subset nets positive where the full population
     nets negative)

Conclusion: quota-based slate construction is not uniformly wrong -- it
corrects real type mismatches when the underlying title-count ratio is
estimated from enough history to be reliable (as with these heavy users).
But most users in the full population have very few watched titles (the
eval-user floor is only >=4), so their ratio is noisy (a "40/60" split from
5 titles is really "2 vs 3"), and enforcing a hard quota on that noise
actively hurts more than it helps. If this is ever productionized, it should
be gated on watch-history depth (e.g. only apply once a user has watched
~15-20+ titles), not applied uniformly to every user.
"""
import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from type_affinity_boost import (
    DATA, DIR_W, ACTOR_W, fit_content_model, fit_audience_model, type_pref_from_titlecount,
)


def interleave_by_quota(movie_ranked, series_ranked, pref_series):
    """At position i, place a series title next iff series placed so far
    is behind round(pref_series * i) -- keeps the ratio correct at every
    prefix length, not just at the end."""
    out = []
    mi, si = 0, 0
    series_placed = 0
    i = 0
    while mi < len(movie_ranked) or si < len(series_ranked):
        target_series = round(pref_series * (i + 1))
        want_series = series_placed < target_series
        if want_series and si < len(series_ranked):
            out.append(series_ranked[si]); si += 1; series_placed += 1
        elif mi < len(movie_ranked):
            out.append(movie_ranked[mi]); mi += 1
        elif si < len(series_ranked):
            out.append(series_ranked[si]); si += 1; series_placed += 1
        i += 1
    return out


def rank_for_user(model, audience, uid, held_idx):
    """Returns (baseline_ranked, quota_ranked, pref_series) for one held-out trial."""
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

    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.item_idx != held_idx]
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

    baseline_ranked = sorted(score, key=lambda k: score[k], reverse=True)
    pref_series = type_pref_from_titlecount(is_series, remaining.item_idx.values)
    movie_ranked = sorted((c for c in candidates if not is_series[c]), key=lambda k: score[k], reverse=True)
    series_ranked = sorted((c for c in candidates if is_series[c]), key=lambda k: score[k], reverse=True)
    quota_ranked = interleave_by_quota(movie_ranked, series_ranked, pref_series)

    return baseline_ranked, quota_ranked, pref_series


def eval_population(model, audience, rng_seed=13):
    watch = audience["watch"]
    eligible_idx = audience["eligible_idx"]
    rng = np.random.default_rng(rng_seed)
    eval_users = audience["eval_users"]

    holdout_choice = {}
    for uid in eval_users:
        rows_ = watch[watch.user_id == uid]
        choice = rows_.sample(1, random_state=rng.integers(0, 1_000_000)).iloc[0]
        holdout_choice[uid] = choice.item_idx

    results = {"baseline": [], "quota_titlecount": []}
    for uid in eval_users:
        held_idx = holdout_choice[uid]
        if held_idx not in eligible_idx:
            continue
        user_rows = watch[watch.user_id == uid]
        if len(user_rows[user_rows.item_idx != held_idx]) == 0:
            continue
        baseline_ranked, quota_ranked, _ = rank_for_user(model, audience, uid, held_idx)
        results["baseline"].append(baseline_ranked.index(held_idx) + 1)
        results["quota_titlecount"].append(quota_ranked.index(held_idx) + 1)

    for variant, ranks in results.items():
        print(f"[{variant:16s}] n_eval={len(ranks)} mean_rank={np.mean(ranks):.1f} "
              f"hit10={np.mean([r <= 10 for r in ranks]):.3f} hit20={np.mean([r <= 20 for r in ranks]):.3f}")


def eval_batch(model, audience, n_users=20, n_trials=20, rng_seed=7):
    watch = audience["watch"]
    eligible_idx = audience["eligible_idx"]
    user_counts = watch.groupby("user_id").size().sort_values(ascending=False)
    batch_uids = user_counts.head(n_users).index.tolist()

    print(f"{'user_id':38s} {'#titles':>7s} {'pref_series':>11s} {'base_hit10':>10s} {'quota_hit10':>11s}")
    agg = {"baseline": [], "quota_titlecount": []}
    n_improved, n_worse, n_same = 0, 0, 0

    for uid in batch_uids:
        user_rows_all = watch[watch.user_id == uid]
        rng = np.random.default_rng(rng_seed)
        k = min(n_trials, len(user_rows_all))
        trial_idxs = rng.choice(user_rows_all.item_idx.values, size=k, replace=False)

        b_ranks, q_ranks = [], []
        for held_idx in trial_idxs:
            if held_idx not in eligible_idx:
                continue
            baseline_ranked, quota_ranked, pref_series = rank_for_user(model, audience, uid, held_idx)
            b_ranks.append(baseline_ranked.index(held_idx) + 1)
            q_ranks.append(quota_ranked.index(held_idx) + 1)

        if not b_ranks:
            continue
        b_hit10 = np.mean([r <= 10 for r in b_ranks])
        q_hit10 = np.mean([r <= 10 for r in q_ranks])
        pref_series_full = type_pref_from_titlecount(model["is_series"], user_rows_all.item_idx.values)
        print(f"{uid:38s} {len(user_rows_all):7d} {pref_series_full:11.2f} {b_hit10:10.2f} {q_hit10:11.2f}")
        agg["baseline"].extend(b_ranks)
        agg["quota_titlecount"].extend(q_ranks)
        if q_hit10 > b_hit10:
            n_improved += 1
        elif q_hit10 < b_hit10:
            n_worse += 1
        else:
            n_same += 1

    print(f"\nUsers where quota improved hit10: {n_improved}, worse: {n_worse}, same: {n_same} (of {len(batch_uids)})")
    for variant, ranks in agg.items():
        print(f"[{variant:16s}] mean_rank={np.mean(ranks):.1f} hit10={np.mean([r <= 10 for r in ranks]):.3f} "
              f"hit20={np.mean([r <= 20 for r in ranks]):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["population", "batch"], default="population")
    parser.add_argument("--watch-sample", default=f"{DATA}/user_title_watch_sample_2218.csv")
    parser.add_argument("--n-users", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=20)
    args = parser.parse_args()

    model = fit_content_model()
    audience = fit_audience_model(model, args.watch_sample)
    print(f"watch sample: {args.watch_sample} ({audience['watch'].user_id.nunique()} users)\n")

    if args.mode == "population":
        eval_population(model, audience)
    else:
        eval_batch(model, audience, n_users=args.n_users, n_trials=args.n_trials)
