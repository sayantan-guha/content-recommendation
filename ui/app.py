"""
Sample UI for the Hoichoi content-recommendation model.

Loads the full-catalog pipeline (same model as src/pipeline_full_catalog.py:
8 content-category clusters, 6 audience clusters, popularity^0.7 x cluster_rate^0.3
x creator_boost scoring), lets you pick a real user, and shows:
  - their watch-history genre breakdown
  - top 10 recommendations (movies + series, single merged list)
  - a held-out validation check: hide one watched title, see if the model
    would have surfaced it back in the top 10/20

Run with: streamlit run ui/app.py
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA = str(Path(__file__).resolve().parent.parent / "data")
K_CONTENT, K_USER = 8, 6
DIR_W, ACTOR_W = 0.5, 0.5

st.set_page_config(page_title="Hoichoi Recommender — Sample UI", layout="wide")


def parse_list(s):
    if pd.isna(s):
        return []
    try:
        v = ast.literal_eval(s)
        return v if isinstance(v, list) else [v]
    except Exception:
        return []


def norm_maturity(tags):
    return sorted(set(t.strip().lower() for t in tags))


def multihot(values, vocab):
    idx = {v: i for i, v in enumerate(vocab)}
    mat = np.zeros((len(values), len(vocab)))
    for i, lst in enumerate(values):
        for v in lst:
            if v in idx:
                mat[i, idx[v]] = 1.0
    return mat


def onehot(values, vocab):
    idx = {v: i for i, v in enumerate(vocab)}
    mat = np.zeros((len(values), len(vocab)))
    for i, v in enumerate(values):
        if v in idx:
            mat[i, idx[v]] = 1.0
    return mat


def l2_normalize_rows(mat):
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


@st.cache_resource(show_spinner="Fitting content model (categories + clusters)...")
def load_content_model():
    content = pd.read_csv(f"{DATA}/content_features_full_tagged.csv")
    content["_storyline"] = content["storyline_tags"].apply(parse_list)
    content["_tone"] = content["overall_tone_tags"].apply(parse_list)
    content["_maturity"] = content["maturity_tags"].apply(parse_list).apply(norm_maturity)
    content["_director"] = content["director_names"].apply(parse_list)
    content["_actor"] = content["actor_names"].apply(parse_list)
    content["item_id"] = content.apply(
        lambda r: f"movie::{r.content_id}" if r.content_type == "movie" else f"series::{r.content_id}",
        axis=1,
    )
    series_content = content.reset_index(drop=True)

    storyline_vocab = sorted(set(t for lst in series_content["_storyline"] for t in lst))
    tone_vocab = sorted(set(t for lst in series_content["_tone"] for t in lst))
    maturity_vocab = sorted(set(t for lst in series_content["_maturity"] for t in lst))
    genre_vocab = sorted(series_content["genre_normalized"].dropna().unique().tolist())
    era_vocab = sorted(series_content["era_bucket"].dropna().unique().tolist())

    blocks = {
        "genre": onehot(series_content["genre_normalized"], genre_vocab),
        "storyline": multihot(series_content["_storyline"], storyline_vocab),
        "tone": multihot(series_content["_tone"], tone_vocab),
        "era": onehot(series_content["era_bucket"], era_vocab),
        "maturity": multihot(series_content["_maturity"], maturity_vocab),
    }
    weights = {"genre": 3.0, "storyline": 3.0, "tone": 2.0, "era": 1.0, "maturity": 1.0}
    X = np.concatenate([l2_normalize_rows(blocks[b]) * weights[b] for b in blocks], axis=1)

    km = KMeans(n_clusters=K_CONTENT, random_state=42, n_init=10).fit(X)
    dists = np.linalg.norm(X[:, None, :] - km.cluster_centers_[None, :, :], axis=2)
    logits = -dists / 0.35
    logits -= logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    mixture = ex / ex.sum(axis=1, keepdims=True)

    item_ids = series_content.item_id.values
    item_to_idx = {s: i for i, s in enumerate(item_ids)}
    director_sets = [set(x) for x in series_content["_director"]]
    actor_sets = [set(x) for x in series_content["_actor"]]

    struct = pd.read_csv(f"{DATA}/structured_linkage_full.csv").dropna(subset=["series_id"])
    episode_to_show_cid = dict(zip(struct.content_id, struct.series_id))
    movie_cids = set(content[content.content_type == "movie"].content_id)
    series_cids = set(content[content.content_type == "series"].content_id)

    def cid_to_item_id(cid):
        if cid in movie_cids:
            return f"movie::{cid}"
        show_cid = episode_to_show_cid.get(cid)
        if show_cid in series_cids:
            return f"series::{show_cid}"
        return None

    return {
        "series_content": series_content,
        "mixture": mixture,
        "item_to_idx": item_to_idx,
        "director_sets": director_sets,
        "actor_sets": actor_sets,
        "cid_to_item_id": cid_to_item_id,
    }


@st.cache_resource(show_spinner="Fitting audience clusters over watch history...")
def load_audience_model(_model):
    watch_ep = pd.read_csv(f"{DATA}/user_title_watch_sample_2218.csv")
    watch_ep["item_id"] = watch_ep.content_id.apply(_model["cid_to_item_id"])
    watch_ep = watch_ep.dropna(subset=["item_id"])
    watch = watch_ep.groupby(["user_id", "item_id"], as_index=False)["seconds_watched"].sum()
    watch["item_idx"] = watch.item_id.map(_model["item_to_idx"])

    mixture = _model["mixture"]

    def build_profile(idxs, secs):
        secs = np.array(secs, dtype=float)
        w = secs / secs.sum()
        return (mixture[idxs] * w[:, None]).sum(axis=0)

    profiles = {}
    for uid, rows_ in watch.groupby("user_id"):
        profiles[uid] = build_profile(rows_.item_idx.values, rows_.seconds_watched.values)
    all_uids = list(profiles.keys())
    P = np.array([profiles[u] for u in all_uids])
    scaler = StandardScaler().fit(P)
    Ps = scaler.transform(P)
    km_user = KMeans(n_clusters=K_USER, random_state=42, n_init=10).fit(Ps)
    uid_to_cluster = dict(zip(all_uids, km_user.labels_))
    watch_c = watch.copy()
    watch_c["cluster"] = watch_c.user_id.map(uid_to_cluster)

    item_viewer_counts = watch.groupby("item_idx").user_id.nunique()
    eligible_idx = set(item_viewer_counts[item_viewer_counts >= 5].index)
    overall_pop = watch.groupby("item_idx").user_id.nunique()

    user_counts = watch.groupby("user_id").size()
    eval_users = sorted(user_counts[user_counts >= 4].index.tolist())

    return {
        "watch": watch,
        "watch_c": watch_c,
        "scaler": scaler,
        "km_user": km_user,
        "eligible_idx": eligible_idx,
        "overall_pop": overall_pop,
        "eval_users": eval_users,
        "build_profile": build_profile,
    }


def recommend_for_user(uid, held_out_idx, model, audience, top_n=10):
    watch = audience["watch"]
    watch_c = audience["watch_c"]
    scaler = audience["scaler"]
    km_user = audience["km_user"]
    eligible_idx = audience["eligible_idx"]
    overall_pop = audience["overall_pop"]
    build_profile = audience["build_profile"]
    director_sets = model["director_sets"]
    actor_sets = model["actor_sets"]

    user_rows = watch[watch.user_id == uid]
    remaining = user_rows[user_rows.item_idx != held_out_idx] if held_out_idx is not None else user_rows
    profile = build_profile(remaining.item_idx.values, remaining.seconds_watched.values)
    profile_s = scaler.transform(profile.reshape(1, -1))
    d = np.linalg.norm(km_user.cluster_centers_ - profile_s, axis=1)
    assigned_cluster = int(np.argmin(d))

    mask_drop = (watch_c.user_id == uid) & (watch_c.item_idx == held_out_idx)
    wc = watch_c[~mask_drop] if held_out_idx is not None else watch_c
    cl_viewers = wc[wc.cluster == assigned_cluster].groupby("item_idx").user_id.nunique()
    cl_size = wc[wc.cluster == assigned_cluster].user_id.nunique()

    watched = set(user_rows.item_idx) - ({held_out_idx} if held_out_idx is not None else set())
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
    ranked = sorted(score, key=lambda k: score[k], reverse=True)
    return ranked[:top_n], ranked


# ---------------------------------------------------------------------------
# hoichoi brand design system tokens (from the official design-system file):
# gradient -60deg #d20820 -> #6d0550, Outfit (headers) + Manrope (body),
# dark OTT surfaces, pillow/pill radii, brand badge + button styles.
# ---------------------------------------------------------------------------
HC_GRADIENT = "linear-gradient(-60deg,#d20820 0%,#6d0550 100%)"
BADGE_STYLES = {
    "series": "background:#191919;color:#fff;",
    "movie": "background:#f5f5f5;color:#2a2a2a;",
}


def inject_css():
    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=Manrope:wght@400;500;600;700&display=swap');
        :root {
            --hc-red: #d20820; --hc-velvet: #6d0550;
            --hc-gradient: __GRADIENT__;
            --hc-soot: #191919; --hc-dark-grey: #2a2a2a; --hc-mid-grey: #888888;
            --hc-light-gray: #cccccc; --hc-off-white: #f5f5f5; --hc-white: #ffffff;
            --hc-success: #1a8754; --hc-warning: #e6a817;
        }
        #MainMenu, footer, header {visibility: hidden;}
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
        .block-container {padding-top: 1rem; max-width: 100%; background: #fafafa;}
        [data-testid="stAppViewContainer"] { background: #fafafa; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e8e8e8; }
        [data-testid="stMarkdownContainer"] p { color: #444; }

        .hoichoi-nav {
            display:flex; align-items:center; gap:14px;
            padding: 0.4rem 0 1.2rem 0; border-bottom: 1px solid #e8e8e8; margin-bottom: 1.4rem;
        }
        .hoichoi-logo {
            font-family:'Outfit',sans-serif; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.03em;
            background: var(--hc-gradient);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .hoichoi-tag {
            font-family:'Outfit',sans-serif; font-size: 0.7rem; font-weight:600; color:#999;
            border:1px solid #e0e0e0; border-radius:9999px; padding: 4px 14px;
            letter-spacing:0.04em; text-transform:uppercase;
        }
        .hero {
            border-radius: 24px; padding: 2.4rem; margin-bottom: 1.8rem;
            background: __GRADIENT__;
            position: relative; overflow: hidden; box-shadow: 0 8px 32px rgba(210,8,32,0.2);
        }
        .hero::after {
            content:""; position:absolute; top:-60px; right:-60px; width:220px; height:220px;
            border-radius:88px; background:rgba(255,255,255,0.05);
        }
        .hero-inner { position: relative; z-index: 1; max-width: 640px; }
        .hero-eyebrow {
            font-family:'Outfit',sans-serif; color:rgba(255,255,255,0.65); font-weight:600; font-size:0.75rem;
            letter-spacing:0.18em; text-transform:uppercase;
        }
        .hero-title { font-family:'Outfit',sans-serif; font-size: 2.4rem; font-weight: 800; color:#fff; margin: 0.4rem 0; letter-spacing:-0.03em;}
        .hero-meta { font-family:'Manrope',sans-serif; color:rgba(255,255,255,0.8); font-size:0.95rem; margin-bottom: 1.2rem;}
        .hero-btn {
            display:inline-block; font-family:'Outfit',sans-serif; padding: 0.65rem 1.7rem; border-radius: 9999px;
            font-weight:700; font-size:0.85rem; background:#fff; color:var(--hc-soot); margin-right:10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }
        .rail-title {
            font-family:'Outfit',sans-serif; font-size: 1.05rem; font-weight:700; color:#191919;
            margin: 1.6rem 0 0.8rem 0; display:flex; align-items:center; gap:9px;
        }
        .rail-title::before { content:""; width:3px; height:20px; background:var(--hc-gradient); border-radius:2px; }
        .rail-scroll { display:flex; gap:14px; overflow-x:auto; padding-bottom: 10px; }
        .poster-card {
            flex: 0 0 auto; width: 168px; border-radius: 14px; overflow:hidden;
            background: #ffffff; box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }
        .poster-banner {
            height: 78px; position:relative; background: __GRADIENT__;
        }
        .poster-badge {
            position:absolute; bottom:8px; left:10px; font-family:'Outfit',sans-serif;
            font-size:0.62rem; font-weight:700; padding:2px 8px; border-radius:9999px;
            letter-spacing:0.05em; text-transform:uppercase;
        }
        .poster-watched {
            position:absolute; top:8px; right:8px; font-size:0.62rem; font-weight:700;
            background:var(--hc-success); color:#fff; padding:2px 7px; border-radius:9999px;
        }
        .poster-rank {
            position:absolute; top:8px; left:8px; font-family:'Outfit',sans-serif; font-size:0.65rem;
            font-weight:800; color:#fff; background:rgba(0,0,0,0.4); padding:1px 7px; border-radius:9999px;
        }
        .poster-body { padding: 10px 12px 14px; }
        .poster-title {
            font-family:'Outfit',sans-serif; font-weight:700; font-size:0.88rem; color:#191919; line-height:1.2;
            overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
        }
        .poster-genre { font-family:'Manrope',sans-serif; font-size:0.72rem; color:var(--hc-mid-grey); margin-top:3px;}
        .verdict-hit { background:#f0fdf4; border:1px solid var(--hc-success); color:#146c43; padding:0.8rem 1.1rem; border-radius:14px; font-weight:600; font-family:'Manrope',sans-serif;}
        .verdict-close { background:#fff9ec; border:1px solid var(--hc-warning); color:#8a6300; padding:0.8rem 1.1rem; border-radius:14px; font-weight:600; font-family:'Manrope',sans-serif;}
        .verdict-miss { background:#fff5f5; border:1px solid var(--hc-red); color:#a3051d; padding:0.8rem 1.1rem; border-radius:14px; font-weight:600; font-family:'Manrope',sans-serif;}
        </style>
        """
    st.markdown(css.replace("__GRADIENT__", HC_GRADIENT), unsafe_allow_html=True)


def poster_card(title, subtitle, badge_type, rank=None, watched=False):
    badge_css = BADGE_STYLES.get(badge_type, BADGE_STYLES["movie"])
    rank_html = f'<div class="poster-rank">#{rank}</div>' if rank else ""
    watched_html = '<div class="poster-watched">✓ watched</div>' if watched else ""
    return (
        f'<div class="poster-card">'
        f'<div class="poster-banner">{rank_html}{watched_html}'
        f'<div class="poster-badge" style="{badge_css}">{badge_type}</div>'
        f'</div>'
        f'<div class="poster-body">'
        f'<div class="poster-title">{title}</div>'
        f'<div class="poster-genre">{subtitle}</div>'
        f'</div>'
        f'</div>'
    )


def render_rail(title, items):
    st.markdown(f'<div class="rail-title">{title}</div>', unsafe_allow_html=True)
    cards = "".join(
        poster_card(it["title"], it["subtitle"], it["badge_type"], it.get("rank"), it.get("watched", False))
        for it in items
    )
    st.markdown(f'<div class="rail-scroll">{cards}</div>', unsafe_allow_html=True)


def main():
    inject_css()

    model = load_content_model()
    audience = load_audience_model(model)
    series_content = model["series_content"].set_index("item_id")
    eval_users = audience["eval_users"]

    st.markdown(
        """
        <div class="hoichoi-nav">
            <div><span class="hoichoi-logo">hoichoi</span><span class="hoichoi-tag">Recommendation Engine — Internal Demo</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Pick a viewer")
        uid = st.selectbox("user_id", eval_users, index=0)
        st.markdown("---")
        st.markdown(
            "**Model:** `popularity_rate^0.7 × cluster_rate^0.3 × creator_boost`\n\n"
            "8 content categories · 6 audience clusters, fit on the full 775-title catalog.\n\n"
            "**Held-out test:** hide one watched title, rebuild the profile from the "
            "rest, and check whether the model would've surfaced it back in the top 10/20."
        )

    watch = audience["watch"]
    user_rows = watch[watch.user_id == uid]
    watched_idx = user_rows.item_idx.tolist()
    watched_items = series_content.iloc[watched_idx]

    # Hero: the user's own top recommendation, styled like the hoichoi.tv banner.
    top10, _ = recommend_for_user(uid, None, model, audience, top_n=10)
    hero_idx = top10[0]
    hero = series_content.iloc[hero_idx]
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-inner">
                <div class="hero-eyebrow">Top pick for this viewer</div>
                <div class="hero-title">{hero['title_english']}</div>
                <div class="hero-meta">{hero['content_type'].title()} • {hero['genre_normalized']}</div>
                <span class="hero-btn">▶ Watch Now</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Rail 1: continue watching / watch history
    history_items = [
        {
            "title": row["title_english"],
            "subtitle": f"{row['content_type'].title()} • {row['genre_normalized']}",
            "badge_type": row["content_type"],
        }
        for _, row in watched_items.iterrows()
    ]
    render_rail(f"Continue Watching ({len(history_items)} titles)", history_items)

    # Rail 2: top 10 recommendations
    rec_items = [
        {
            "title": series_content.iloc[idx]["title_english"],
            "subtitle": f"{series_content.iloc[idx]['content_type'].title()} • {series_content.iloc[idx]['genre_normalized']}",
            "badge_type": series_content.iloc[idx]["content_type"],
            "rank": rank,
            "watched": idx in watched_idx,
        }
        for rank, idx in enumerate(top10, start=1)
    ]
    render_rail("Recommended For You", rec_items)

    # Held-out validation, styled as a small verdict banner + genre breakdown.
    st.markdown('<div class="rail-title">Held-out validation</div>', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        held_choice = st.selectbox(
            "Hide this title and see if the model recommends it back:",
            options=watched_idx,
            format_func=lambda i: series_content.iloc[i]["title_english"],
        )
        held_top10, full_ranked = recommend_for_user(uid, held_choice, model, audience, top_n=10)
        rank = (full_ranked.index(held_choice) + 1) if held_choice in full_ranked else None
        held_title = series_content.iloc[held_choice]["title_english"]
        if rank is None:
            st.markdown(
                f'<div class="verdict-miss">⚠ \'{held_title}\' wasn\'t in the eligible candidate pool.</div>',
                unsafe_allow_html=True,
            )
        elif rank <= 10:
            st.markdown(
                f'<div class="verdict-hit">✅ HIT — \'{held_title}\' would rank #{rank} (top 10)</div>',
                unsafe_allow_html=True,
            )
        elif rank <= 20:
            st.markdown(
                f'<div class="verdict-close">〜 CLOSE — \'{held_title}\' would rank #{rank} (top 20)</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="verdict-miss">❌ MISS — \'{held_title}\' would rank #{rank}</div>',
                unsafe_allow_html=True,
            )
    with col2:
        genre_counts = watched_items["genre_normalized"].value_counts()
        st.caption("Viewer's genre breakdown")
        st.bar_chart(genre_counts)


if __name__ == "__main__":
    main()
