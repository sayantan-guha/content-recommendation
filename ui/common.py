"""
Shared UI building blocks for app.py and pages/*.py -- API client, hoichoi
brand CSS, and poster-card/rail rendering. Split out so the "view more"
recommendations page can reuse the exact same look without duplicating it.
"""
import os

import requests
import streamlit as st

API_BASE = os.environ.get("HC_RECS_API", "http://localhost:8000")

# ---------------------------------------------------------------------------
# hoichoi brand design system tokens (from the official design-system file):
# gradient -60deg #d20820 -> #6d0550, Outfit (headers) + Manrope (body),
# light-theme surfaces, pillow/pill radii, brand badge + button styles.
# ---------------------------------------------------------------------------
HC_GRADIENT = "linear-gradient(-60deg,#d20820 0%,#6d0550 100%)"
# Exact tokens from the "Badges" component in the uploaded design system:
# SERIES -> #191919/#fff, FILM -> #f5f5f5/#2a2a2a.
BADGE_STYLES = {
    "series": ("SERIES", "background:#191919;color:#fff;"),
    "movie": ("MOVIE", "background:#f5f5f5;color:#2a2a2a;"),
}


@st.cache_data(ttl=300)
def api_get(path, **params):
    resp = requests.get(f"{API_BASE}{path}", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


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
        .viewer-picker div[data-baseweb="select"] > div {
            border-radius: 9999px !important; border-color:#e0e0e0 !important;
            font-family:'Manrope',sans-serif; background:#fff;
        }
        .viewer-picker label { display:none; }
        .hero {
            border-radius: 24px; padding: 2.4rem; margin-bottom: 1.8rem; box-sizing:border-box;
            min-height: 320px;
            background: __GRADIENT__;
            position: relative; overflow: hidden; box-shadow: 0 8px 32px rgba(210,8,32,0.2);
        }
        .watch-mix-wrap {
            display:flex; flex-direction:column; justify-content:center; align-items:center;
            gap: 1rem; min-height: 320px; margin-bottom: 1.8rem;
        }
        .hero::after {
            content:""; position:absolute; top:-60px; right:-60px; width:220px; height:220px;
            border-radius:88px; background:rgba(255,255,255,0.05);
        }
        .hero-inner { position: relative; z-index: 1; max-width: 640px; }
        .hero-eyebrow {
            font-family:'Outfit',sans-serif; color:rgba(255,255,255,0.5); font-weight:700; font-size:10px;
            letter-spacing:0.15em; text-transform:uppercase;
        }
        .hero-title { font-family:'Outfit',sans-serif; font-size: 40px; font-weight: 800; color:#fff; margin: 0.4rem 0; letter-spacing:-0.03em;}
        .hero-meta { font-family:'Manrope',sans-serif; color:rgba(255,255,255,0.8); font-size:0.95rem; margin-bottom: 1.2rem;}
        .hero-btn {
            display:inline-block; font-family:'Outfit',sans-serif; padding: 11px 24px; border-radius: 50px;
            font-weight:700; font-size:12px; background:#fff; color:var(--hc-soot); margin-right:10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1); cursor:pointer;
        }
        .rail-title {
            font-family:'Outfit',sans-serif; font-size: 18px; font-weight:700; color:#191919;
            margin: 1.6rem 0 0.8rem 0; display:flex; align-items:center; gap:9px; letter-spacing:-0.02em;
        }
        .rail-title::before { content:""; width:3px; height:22px; background:var(--hc-gradient); border-radius:2px; flex-shrink:0; }
        .rail-scroll { display:flex; gap:14px; overflow-x:auto; padding-bottom: 10px; }
        .poster-grid { display:flex; flex-wrap:wrap; gap:14px; padding-bottom: 10px; }
        .poster-card {
            flex: 0 0 auto; width: 232px; border-radius: 14px; overflow:hidden;
            background: #ffffff; box-shadow: 0 2px 10px rgba(0,0,0,0.06);
        }
        .poster-card-more {
            display:flex; align-items:center; justify-content:center; text-align:center;
            flex-direction:column; gap:10px; cursor:pointer;
            background: #fdf2f4;
            border: 1.5px dashed var(--hc-red);
        }
        .poster-more-label {
            font-family:'Outfit',sans-serif; font-weight:700; font-size:13px; color:var(--hc-red);
        }
        .poster-banner {
            height: 100px; position:relative; background: __GRADIENT__;
        }
        .poster-badge {
            position:absolute; bottom:8px; left:10px; font-family:'Outfit',sans-serif;
            font-size:9px; font-weight:700; padding:3px 8px; border-radius:4px;
            letter-spacing:0.06em; text-transform:uppercase;
        }
        .poster-watched {
            position:absolute; top:8px; right:8px; font-size:0.62rem; font-weight:700;
            background:var(--hc-success); color:#fff; padding:2px 7px; border-radius:9999px;
        }
        .poster-rank {
            position:absolute; top:8px; left:8px; font-family:'Outfit',sans-serif; font-size:0.65rem;
            font-weight:800; color:#fff; background:rgba(0,0,0,0.4); padding:1px 7px; border-radius:9999px;
        }
        .poster-body { padding: 12px; }
        .poster-title {
            font-family:'Outfit',sans-serif; font-weight:700; font-size:13px; color:#191919; line-height:1.2;
            margin-bottom:2px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
        }
        .poster-genre { font-family:'Manrope',sans-serif; font-size:11px; color:#888888; line-height:1.5; margin-bottom:8px;}
        .poster-details {
            font-family:'Manrope',sans-serif; font-size:11px; color:#666666; line-height:1.65;
            margin-bottom:9px; padding-top:7px; border-top:1px solid #f0f0f0;
        }
        .poster-details b { color:#444444; font-weight:700; }
        .poster-why {
            font-family:'Manrope',sans-serif; font-size:11px; color:#1e6b3c; line-height:1.5;
            background:#eefaf1; border-radius:8px; padding:6px 8px; margin-bottom:9px;
        }
        .poster-cta {
            display:inline-block; font-family:'Outfit',sans-serif; font-size:9px; font-weight:600; color:#fff;
            background: var(--hc-gradient); border-radius:50px; padding:5px 14px;
        }
        .verdict-hit { background:#f0fdf4; border:1px solid var(--hc-success); color:#146c43; padding:0.8rem 1.1rem; border-radius:14px; font-weight:600; font-family:'Manrope',sans-serif;}
        .verdict-close { background:#fff9ec; border:1px solid var(--hc-warning); color:#8a6300; padding:0.8rem 1.1rem; border-radius:14px; font-weight:600; font-family:'Manrope',sans-serif;}
        .verdict-miss { background:#fff5f5; border:1px solid var(--hc-red); color:#a3051d; padding:0.8rem 1.1rem; border-radius:14px; font-weight:600; font-family:'Manrope',sans-serif;}
        .api-status { font-family:'Manrope',sans-serif; font-size:0.7rem; color:#999; margin-bottom:0.6rem; }
        div[data-testid="stImage"] { display:flex; justify-content:center; }
        </style>
        """
    st.markdown(css.replace("__GRADIENT__", HC_GRADIENT), unsafe_allow_html=True)


def poster_card(title, subtitle, badge_type, rank=None, watched=False, cta_label="▶ Watch Now", details=None, why=None):
    badge_label, badge_css = BADGE_STYLES.get(badge_type, BADGE_STYLES["movie"])
    rank_html = f'<div class="poster-rank">#{rank}</div>' if rank else ""
    watched_html = '<div class="poster-watched">✓ watched</div>' if watched else ""

    details_html = ""
    if details:
        lines = []
        if details.get("storyline_tags"):
            lines.append(f"<b>Storyline:</b> {', '.join(details['storyline_tags'])}")
        if details.get("actors"):
            lines.append(f"<b>Actors:</b> {', '.join(details['actors'])}")
        if details.get("director"):
            lines.append(f"<b>Director:</b> {', '.join(details['director'])}")
        if details.get("tone_tags"):
            lines.append(f"<b>Tone:</b> {', '.join(details['tone_tags'])}")
        if details.get("era"):
            lines.append(f"<b>Era:</b> {details['era']}")
        if lines:
            details_html = f'<div class="poster-details">{"<br>".join(lines)}</div>'

    why_html = f'<div class="poster-why">{" • ".join(why)}</div>' if why else ""

    return (
        f'<div class="poster-card">'
        f'<div class="poster-banner">{rank_html}{watched_html}'
        f'<div class="poster-badge" style="{badge_css}">{badge_label}</div>'
        f'</div>'
        f'<div class="poster-body">'
        f'<div class="poster-title">{title}</div>'
        f'<div class="poster-genre">{subtitle}</div>'
        f'{why_html}{details_html}'
        f'<span class="poster-cta">{cta_label}</span>'
        f'</div>'
        f'</div>'
    )


def render_rail(title, items, cta_label="▶ Watch Now", more_link=None, more_label="View More"):
    """more_link, if given, is a "pages/xxx.py?query=string" target -- appends
    a dashed "view more" tile at the end of the rail linking to it (a real
    Streamlit page navigation, so the browser back button returns here)."""
    st.markdown(f'<div class="rail-title">{title}</div>', unsafe_allow_html=True)
    cards = "".join(
        poster_card(
            it["title"], it["subtitle"], it["badge_type"],
            it.get("rank"), it.get("watched", False), cta_label,
            details=it, why=it.get("why"),
        )
        for it in items
    )
    st.markdown(f'<div class="rail-scroll">{cards}</div>', unsafe_allow_html=True)
    if more_link:
        st.page_link(more_link, label=more_label, icon="➡️")


def render_grid(items, cta_label="▶ Watch Now"):
    """Same poster cards as render_rail, but wrapped (not horizontal-scroll)
    -- used by the "view more" full-list page."""
    cards = "".join(
        poster_card(
            it["title"], it["subtitle"], it["badge_type"],
            it.get("rank"), it.get("watched", False), cta_label,
            details=it, why=it.get("why"),
        )
        for it in items
    )
    st.markdown(f'<div class="poster-grid">{cards}</div>', unsafe_allow_html=True)
