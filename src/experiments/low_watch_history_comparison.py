"""
Manual side-by-side comparison of CF-only, content-based-only, and the
production blended technique (recommend_for_user -- CF + type/era quota +
cold-start backfill) for 10 real users with fewer than 8 total watched
titles, spanning watch-counts 1-7.

This is a companion to the earlier aggregate breakeven tests
(cf_vs_content_breakeven.py / _v2.py / leave5.py), which already showed CF
beats content-similarity at every bucket down to 1 remaining title on
Hit@10/20 averaged across hundreds of users. Those results say "on average,
don't switch." This script exists to let a human actually read the
recommendations for a handful of real low-history users side by side and
sanity-check that conclusion qualitatively, and to make it easy to try a
count-based threshold and see exactly which users' recommendations would
change if one were added.

Run: python3 src/experiments/low_watch_history_comparison.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "src"))
import recommender as rec

USERS = [
    "f21f6dd3-a0a7-43e7-9c6b-acbd22c32492",  # 1 watched title
    "992f20d3-eb6d-4445-9a28-3656bc94ea7e",  # 2
    "424c6911-1e61-4e95-bef1-09e860456c09",  # 2
    "c26c6c6e-e157-4d77-93e6-a98cc09ff133",  # 3
    "ea34c535-09cf-4e49-9479-c282f19b2c43",  # 4
    "070caab9-ef26-4c97-be4c-6cc999eb609f",  # 4
    "a419d8f3-3bbc-4211-a389-627eb103a607",  # 5
    "b9286548-856d-4c8a-8fdd-5a3c211ea522",  # 6
    "4e8e425e-e618-4147-98e7-7c932b39683c",  # 6
    "d9cbf5eb-3532-4067-9031-e5d10f4b967a",  # 7
]

TOP_N = 10


def titles(model, idxs):
    sc = model["series_content"]
    return [f"{sc.iloc[i].title_english} ({sc.iloc[i].content_type}/{sc.iloc[i].genre_normalized})" for i in idxs]


def main():
    model = rec.load_content_model()
    audience = rec.load_audience_model(model)
    watch = audience["watch"]
    sim = audience["sim"]
    eligible_idx = audience["eligible_idx"]
    build_profile = audience["build_profile"]
    sc = model["series_content"]

    for uid in USERS:
        user_rows = watch[watch.user_id == uid]
        watched_idx = user_rows.item_idx.values
        watched = set(watched_idx)
        n_watched = len(watched_idx)

        print("=" * 100)
        print(f"user {uid}  ({n_watched} watched titles)")
        print("-" * 100)
        print("Watched:")
        for t in titles(model, watched_idx):
            print(f"    {t}")

        candidates_eligible = [i for i in eligible_idx if i not in watched]
        cand_arr = np.array(candidates_eligible)

        # --- CF only ---
        cf_scores = sim[watched_idx][:, cand_arr].sum(axis=0)
        cf_signal_max = cf_scores.max() if len(cf_scores) else 0.0
        cf_order = np.argsort(-cf_scores)
        cf_top = cand_arr[cf_order[:TOP_N]]

        # --- Content-based only ---
        profile = build_profile(watched_idx, user_rows.seconds_watched.values)
        candidates_all = [i for i in range(len(model["mixture"])) if i not in watched]
        cb_cand_arr, cb_sims = rec.content_based_ranking(model, profile, candidates_all, watched_idx)
        cb_order = np.argsort(-cb_sims)
        cb_top = cb_cand_arr[cb_order[:TOP_N]]

        # --- Production blended technique ---
        prod_top, _ = rec.recommend_for_user(uid, None, model, audience, top_n=TOP_N)

        print(f"\n  CF signal strength (max co-viewer score): {cf_signal_max:.4f}"
              f"  {'[BELOW EPSILON -> would fall back to content-based]' if cf_signal_max <= rec.CF_SIGNAL_EPSILON else ''}")

        print(f"\n  CF-only top {TOP_N}:")
        for t in titles(model, cf_top):
            print(f"    {t}")

        print(f"\n  Content-based-only top {TOP_N}:")
        for t in titles(model, cb_top):
            print(f"    {t}")

        print(f"\n  PRODUCTION (blended) top {TOP_N}:")
        for t in titles(model, prod_top):
            print(f"    {t}")

        print()


if __name__ == "__main__":
    main()
