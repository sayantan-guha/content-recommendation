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


def _item_row(idx):
    row = _state["model"]["series_content"].iloc[idx]
    return {
        "idx": int(idx),
        "title": row["title_english"],
        "type": row["content_type"],
        "genre": row["genre_normalized"],
    }


@app.get("/users")
def list_users():
    return {"users": _state["audience"]["eval_users"]}


@app.get("/users/{uid}/history")
def watch_history(uid: str):
    watch = _state["audience"]["watch"]
    rows = watch[watch.user_id == uid]
    if rows.empty:
        raise HTTPException(404, "user not found or has no eligible watch history")
    return {"uid": uid, "history": [_item_row(i) for i in rows.item_idx.tolist()]}


@app.get("/users/{uid}/recommendations")
def recommendations(uid: str, top_n: int = 10, held_out_idx: int = None):
    watch = _state["audience"]["watch"]
    if uid not in set(watch.user_id):
        raise HTTPException(404, "user not found or has no eligible watch history")
    top, full_ranked = rec.recommend_for_user(uid, held_out_idx, _state["model"], _state["audience"], top_n=top_n)
    result = {"uid": uid, "recommendations": [_item_row(i) for i in top]}
    if held_out_idx is not None:
        rank = full_ranked.index(held_out_idx) + 1 if held_out_idx in full_ranked else None
        result["held_out"] = {"idx": held_out_idx, "rank": rank}
    return result


@app.get("/health")
def health():
    return {"status": "ok", "users_loaded": len(_state.get("audience", {}).get("eval_users", []))}
