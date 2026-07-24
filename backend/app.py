"""
Backend API serving the recommendation model to ui/app.py (and anything else).

Fits the content + audience model once at startup, then serves it over a few
read-only JSON endpoints. Run with:

    uvicorn backend.app:app --reload --port 8000
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import recommender as rec

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
        # _actor is already capped to the first 2 (lead actors) in
        # load_content_model -- no separate truncation here, so what's
        # displayed always matches what the "why" explanation actually
        # compared against (previously a display-only [:3] slice could show
        # different actors than the ones a "same actor" reason referenced).
        "actors": list(row["_actor"]),
        "era": row["era_bucket"],
    }
    if why is not None:
        out["why"] = why
    return out


@app.get("/users")
def list_users():
    # Every user with any watch data, not just eval_users (>=4 watches) --
    # a strictly larger pick list for the UI dropdown. recommend_for_user
    # works for any uid regardless of this list (even ones with zero rows
    # here), so this only affects what's easy to browse-select.
    watch = _state["audience"]["watch"]
    return {"users": sorted(watch.user_id.unique().tolist())}


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
    _, full_ranked = rec.recommend_for_user(uid, held_out_idx, model, audience, top_n=top_n)

    # full_ranked is sorted purely by similarity/CF score, before the
    # type/era quota reordering and cold-start backfill get applied (those
    # optimize for slate composition, not strict relevance order) -- sliced
    # directly so what's returned is genuinely sorted by similarity.
    top = full_ranked[:top_n]

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


@app.get("/health")
def health():
    return {"status": "ok", "users_loaded": len(_state.get("audience", {}).get("eval_users", []))}
