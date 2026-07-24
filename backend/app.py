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
# a fixed, already-analyzed set of users instead of picking blind from 1075.
# Second batch -- 10 fresh users (not overlapping the first 1-7 watch-count
# batch), spanning watch counts 5-10, for continued manual review.
LOW_HISTORY_USERS = [
    "5eaee6b1-4d49-4966-b973-02fe920f16ca",  # 5 watched titles
    "70bdb36a-9c79-4ebb-975c-f52255a1eac0",  # 5
    "f2cd168d-c675-4cb8-8c91-608ce2ca478b",  # 6
    "a06d8343-05b5-47d7-aefc-3d2c82078dcf",  # 6
    "ad07f91b-48f4-4ef4-9558-fc1c9c004205",  # 7
    "92077b8d-3809-41eb-8631-94fac46ed993",  # 7
    "6f2122bc-5143-4576-9536-77d51e25855a",  # 8
    "599ea337-7e35-47fa-9336-689befc5418e",  # 8
    "262c50df-ac05-40ec-9907-d5463a074f53",  # 9
    "b2007c79-dc77-4567-9b28-e656a7ff36b6",  # 10
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
        # CF's actual signal is co-viewing, not genre/tag/cast overlap -- use
        # explain_cf_recommendation (which of the user's watched titles drove
        # this candidate's CF score) instead of the content-based explainer,
        # so the "why" here is honest about what actually ranked it.
        cf_rows = [
            _item_row(i, why=rec.explain_cf_recommendation(model, sim, watched_idx, i) or ["Weak co-viewing signal"])
            for i in cf_top
        ]

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
