"""
Full recommendation list ("View More") for the viewer picked on the main
page. A real Streamlit page (not a session_state view-toggle), so it gets
its own URL and the browser's native back button returns to the main page
exactly as it would for any other site.
"""
import streamlit as st

from common import api_get, inject_css, render_grid

st.set_page_config(page_title="All Recommendations — Hoichoi Recommender", layout="wide")

inject_css()

uid = st.session_state.get("uid_picker")

st.markdown(
    """
    <div class="hoichoi-nav" style="border-bottom:none; margin-bottom:0; padding-bottom:0;">
        <span class="hoichoi-logo">hoichoi</span>
        <span class="hoichoi-tag">Recommendation Engine</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.page_link("app.py", label="← Back", icon=None)

if not uid:
    st.info("Pick a viewer on the main page first.")
    st.stop()

st.markdown('<hr style="border-color:#e8e8e8; margin: 0.6rem 0 1.4rem;">', unsafe_allow_html=True)

TOP_N = 30
rec_resp = api_get(f"/users/{uid}/recommendations", top_n=TOP_N)
recs = rec_resp["recommendations"]

st.markdown(
    f'<div class="rail-title">All Recommendations for {uid} ({len(recs)})</div>',
    unsafe_allow_html=True,
)

rec_items = [
    {
        "title": it["title"],
        "subtitle": f"{it['type'].title()} • {it['genre']}",
        "badge_type": it["type"],
        "rank": i + 1,
        "genre": it["genre"],
        "era": it.get("era"),
        "storyline_tags": it.get("storyline_tags", []),
        "tone_tags": it.get("tone_tags", []),
        "director": it.get("director", []),
        "actors": it.get("actors", []),
        "why": it.get("why", []),
    }
    for i, it in enumerate(recs)
]
render_grid(rec_items)
