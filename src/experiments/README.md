# Experiments

Standalone, non-production comparisons of candidate scoring/ranking changes against the adopted model, using the same held-out LOO validation methodology as the rest of this project. Nothing here is imported by `backend/app.py` or the pipelines -- these are investigations, kept for reference.

| Script | Question | Verdict |
|---|---|---|
| [type_affinity_boost.py](type_affinity_boost.py) | Does adding an explicit movie/series type-affinity term to the *scoring formula* help? | No -- redundant with what the 8-way category clustering already implicitly captures; hurts once weighted enough to matter. Watch-time is a worse signal than title-count for this. |
| [type_quota_slate.py](type_quota_slate.py) | What if the user's movie/series split is enforced directly on the *final slate* (rank movies/series separately, interleave to match the ratio) instead of nudging scores? | Net negative across the full population (many users have too little watch history for the ratio to be reliable), but net **positive** on the subset of users with rich watch history (29+ titles) -- would need to be gated on watch-history depth to be worth productionizing. |

Run either directly:

```bash
python3 src/experiments/type_affinity_boost.py [path/to/watch_sample.csv]
python3 src/experiments/type_quota_slate.py --mode population
python3 src/experiments/type_quota_slate.py --mode batch --n-users 20 --n-trials 20
```
