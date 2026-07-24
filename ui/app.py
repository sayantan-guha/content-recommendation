"""
Sample UI for the Hoichoi content-recommendation model.

Thin client over the backend API (backend/app.py), which fits the full-catalog
recommendation model and serves it as JSON. This app just renders the
recommendations for a picked viewer -- no model internals shown.

Run the backend first:  uvicorn backend.app:app --port 8000
Then the UI:             streamlit run ui/app.py
"""
from collections import Counter
from io import BytesIO

import base64
import matplotlib.pyplot as plt
import streamlit as st

from common import api_get, inject_css, render_rail

st.set_page_config(page_title="Hoichoi Recommender — Sample UI", layout="wide")


def type_composition_chart_b64(history):
    """Donut chart of movie vs series share, as a base64 PNG (so it can be
    embedded in a flex wrapper for precise alignment against the hero card)."""
    if not history:
        return '<div style="text-align:center; color:#9c8d94; font-size:0.85rem; padding:2.4rem 0;">No watch history yet</div>'
    counts = Counter(it["type"] for it in history)
    labels = [t.title() for t in counts]
    values = list(counts.values())
    # soft pastel takes on the brand red/velvet hues
    colors = ["#f6a6b2", "#c9a7e0"]
    total = sum(values)

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    fig.patch.set_alpha(0.0)
    wedges, _, autotexts = ax.pie(
        values,
        labels=labels,
        colors=colors[: len(values)],
        autopct=lambda p: f"{p:.0f}%",
        startangle=90,
        pctdistance=0.72,
        labeldistance=1.14,
        wedgeprops={"width": 0.42, "edgecolor": "#fafafa", "linewidth": 1.2},
        textprops={"fontfamily": "sans-serif", "fontsize": 8.5, "color": "#5b4a52", "fontweight": "600"},
    )
    for at in autotexts:
        at.set_color("#4a3b42")
        at.set_fontweight("bold")
        at.set_fontsize(8.5)

    # center label: total titles watched
    ax.text(0, 0.10, f"{total}", ha="center", va="center",
            fontsize=14, fontweight="bold", color="#4a3b42", fontfamily="sans-serif")
    ax.text(0, -0.16, "titles", ha="center", va="center",
            fontsize=7, color="#9c8d94", fontfamily="sans-serif")

    ax.set_aspect("equal")
    buf = BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.05, dpi=200)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="width:300px; display:block; margin:0 auto;">'


def main():
    inject_css()

    try:
        api_get("/health")
    except Exception:
        st.error(
            f"Can't reach the backend API. Start it with:\n\n"
            "`uvicorn backend.app:app --port 8000`"
        )
        st.stop()

    eval_users = api_get("/users")["users"]

    nav_col, picker_col = st.columns([2, 1])
    with nav_col:
        st.markdown(
            """
            <div class="hoichoi-nav" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">
                <span class="hoichoi-logo">hoichoi</span>
                <span class="hoichoi-tag">Recommendation Engine</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    # st.selectbox instead of a free-text combobox: typing filters/searches
    # its options live, it has a built-in clear ("x") button, and any pick
    # reruns the app immediately -- no separate blur/Enter step needed,
    # unlike a text_input (which Streamlit only ever commits on blur/Enter).
    # Trade-off: it can only pick users already in the list below (now every
    # user with any watch data, not just eval_users -- see backend /users),
    # not a genuinely new/arbitrary user_id.
    # index=None is what makes the widget clearable at all (a plain int
    # index never has a "cleared" state to return to) -- pre-seeding
    # session_state before creating the widget is what still gives it a
    # real default selection on first load instead of starting blank.
    if "uid_picker" not in st.session_state:
        st.session_state["uid_picker"] = eval_users[0]

    with picker_col:
        st.markdown('<div class="viewer-picker">', unsafe_allow_html=True)
        uid = st.selectbox(
            "Pick a viewer",
            eval_users,
            index=None,
            placeholder="Pick a viewer...",
            label_visibility="collapsed",
            key="uid_picker",
        ) or ""
        st.markdown("</div>", unsafe_allow_html=True)

    if not uid:
        st.info("Pick a viewer above to see their recommendations.")
        st.stop()

    st.markdown('<hr style="border-color:#e8e8e8; margin: 0.6rem 0 1.4rem;">', unsafe_allow_html=True)

    rec_resp = api_get(f"/users/{uid}/recommendations", top_n=10)
    top10 = rec_resp["recommendations"]
    history = api_get(f"/users/{uid}/history")["history"]

    # Hero: the user's own top recommendation, with the watch-mix donut alongside it.
    hero = top10[0]
    hero_col, chart_col = st.columns([2.4, 1])
    with hero_col:
        st.markdown(
            f"""
            <div class="hero">
                <div class="hero-inner">
                    <div class="hero-eyebrow">Top pick for this viewer</div>
                    <div class="hero-title">{hero['title']}</div>
                    <div class="hero-meta">{hero['type'].title()} • {hero['genre']}</div>
                    <span class="hero-btn">▶ Watch Now</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with chart_col:
        chart_img = type_composition_chart_b64(history)
        st.markdown(
            f"""
            <div class="watch-mix-wrap">
                <div class="rail-title" style="justify-content:center; margin:0;">Watch Mix</div>
                {chart_img}
            </div>
            """,
            unsafe_allow_html=True,
        )

    rec_items = [
        {
            "title": it["title"],
            "subtitle": f"{it['type'].title()} • {it['genre']}",
            "badge_type": it["type"],
            "genre": it["genre"],
            "era": it.get("era"),
            "storyline_tags": it.get("storyline_tags", []),
            "tone_tags": it.get("tone_tags", []),
            "director": it.get("director", []),
            "actors": it.get("actors", []),
            "why": it.get("why", []),
        }
        for it in top10
    ]
    render_rail(
        "Recommended For You", rec_items,
        more_link="pages/1_All_Recommendations.py", more_label="View More →",
    )

    history_items = [
        {
            "title": it["title"],
            "subtitle": f"{it['type'].title()} • {it['genre']}",
            "badge_type": it["type"],
            "watched": True,
            "genre": it["genre"],
            "era": it.get("era"),
            "storyline_tags": it.get("storyline_tags", []),
            "tone_tags": it.get("tone_tags", []),
            "director": it.get("director", []),
            "actors": it.get("actors", []),
        }
        for it in history
    ]
    render_rail(f"Watched History ({len(history_items)})", history_items, cta_label="↺ Watch Again")


if __name__ == "__main__":
    main()
