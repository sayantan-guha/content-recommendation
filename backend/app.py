"""
Backend API serving the recommendation model to ui/app.py (and anything else).

Fits the content + audience model once at startup, then serves it over a few
read-only JSON endpoints. Run with:

    uvicorn backend.app:app --reload --port 8000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import recommender as rec

# low-watch-history-threshold-experiment branch only: restrict the picker to
# the same 10 users manually reviewed in
# src/experiments/low_watch_history_comparison.py (watch counts 1-7), so the
# CF-vs-content-based-vs-production comparison UI below has a fixed,
# already-analyzed set to click through instead of picking blind from 1075.
LOW_HISTORY_USERS = [
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

app = FastAPI(title="Hoichoi Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_state = {}


@app.on_event("startup")
def fit_model():
    model = rec.load_content_model()
    audience = rec.load_audience_model(model)
    _state["model"] = model
    _state["audience"] = audience


def _item_row(idx, why=None):
    row = _state["model"]["series_content"].iloc[idx]
    out = {
        "idx": int(idx),
        "title": row["title_english"],
        "type": row["content_type"],
        "genre": row["genre_normalized"],
        "storyline_tags": list(row["_storyline"]),
        "tone_tags": list(row["_tone"]),
        "director": list(row["_director"]),
        "actors": list(row["_actor"])[:3],
        "era": row["era_bucket"],
    }
    if why is not None:
        out["why"] = why
    return out


@app.get("/users")
def list_users():
    # Restricted to the 10 manually-reviewed low-watch-history users for
    # this experiment branch's threshold comparison -- see LOW_HISTORY_USERS.
    return {"users": LOW_HISTORY_USERS}


@app.get("/users/{uid}/history")
def watch_history(uid: str):
    # No 404 for an unrecognized/zero-history uid -- recommend_for_user
    # handles that case (popularity fallback), so an empty history list is a
    # valid, renderable response rather than an error.
    watch = _state["audience"]["watch"]
    rows = watch[watch.user_id == uid].sort_values("last_watched_at", ascending=False)
    return {"uid": uid, "history": [_item_row(i) for i in rows.item_idx.tolist()]}


@app.get("/users/{uid}/recommendations")
def recommendations(uid: str, top_n: int = 10, held_out_idx: int = None):
    watch = _state["audience"]["watch"]
    model = _state["model"]
    audience = _state["audience"]
    top, full_ranked = rec.recommend_for_user(uid, held_out_idx, model, audience, top_n=top_n)

    watched_idx = watch[watch.user_id == uid].item_idx.tolist()
    overall_pop = audience["overall_pop"]
    result = {
        "uid": uid,
        "recommendations": [
            _item_row(i, why=rec.explain_recommendation(model, watched_idx, i, overall_pop)) for i in top
        ],
    }
    if held_out_idx is not None:
        rank = full_ranked.index(held_out_idx) + 1 if held_out_idx in full_ranked else None
        result["held_out"] = {"idx": held_out_idx, "rank": rank}
    return result


@app.get("/users/{uid}/compare")
def compare_techniques(uid: str, top_n: int = 10):
    """CF-only, content-based-only, and the production blended technique,
    side by side for the same user -- the manual low-watch-history threshold
    comparison from src/experiments/low_watch_history_comparison.py, exposed
    over the API so it can be eyeballed in the UI instead of a terminal dump.
    """
    model = _state["model"]
    audience = _state["audience"]
    watch = audience["watch"]
    sim = audience["sim"]
    eligible_idx = audience["eligible_idx"]
    overall_pop = audience["overall_pop"]
    build_profile = audience["build_profile"]

    user_rows = watch[watch.user_id == uid]
    watched_idx = user_rows.item_idx.values
    watched = set(watched_idx)

    def why(i):
        return rec.explain_recommendation(model, watched_idx, i, overall_pop)

    if len(watched_idx) == 0:
        cf_rows, cb_rows = [], []
    else:
        candidates_eligible = [i for i in eligible_idx if i not in watched]
        cand_arr = np.array(candidates_eligible)
        cf_scores = sim[watched_idx][:, cand_arr].sum(axis=0)
        cf_signal_max = float(cf_scores.max()) if len(cf_scores) else 0.0
        cf_order = np.argsort(-cf_scores)
        cf_top = cand_arr[cf_order[:top_n]]
        cf_rows = [_item_row(i, why=why(i)) for i in cf_top]

        profile = build_profile(watched_idx, user_rows.seconds_watched.values)
        candidates_all = [i for i in range(len(model["mixture"])) if i not in watched]
        cb_cand_arr, cb_sims = rec.content_based_ranking(model, profile, candidates_all, watched_idx)
        cb_order = np.argsort(-cb_sims)
        cb_top = cb_cand_arr[cb_order[:top_n]]
        cb_rows = [_item_row(i, why=why(i)) for i in cb_top]

    prod_top, _ = rec.recommend_for_user(uid, None, model, audience, top_n=top_n)
    prod_rows = [_item_row(i, why=why(i)) for i in prod_top]

    return {
        "uid": uid,
        "n_watched": len(watched_idx),
        "cf_signal_max": cf_signal_max if len(watched_idx) else 0.0,
        "below_cf_epsilon": (cf_signal_max <= rec.CF_SIGNAL_EPSILON) if len(watched_idx) else True,
        "cf": cf_rows,
        "content_based": cb_rows,
        "production": prod_rows,
    }


@app.get("/health")
def health():
    return {"status": "ok", "users_loaded": len(_state.get("audience", {}).get("eval_users", []))}
