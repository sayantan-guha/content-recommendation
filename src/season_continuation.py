"""
Season-continuation recommender.

Recommends the NEXT SEASON of a show (as a series-level entity, e.g. "Sampurna
Season 2" -- not a specific episode content_id) whenever a user has watched any of
the previous season and hasn't yet started the next one. Its output is meant to be
merged into the SAME single ranked list as regular discovery recommendations (movies
and other series) -- there is no separate "continuation rail"; this just produces a
few series/movie-level entries that get folded into one combined top-N list, usually
placed first since "you're partway through a show, here's what's next" is a stronger
signal than a popularity/cluster-affinity guess.

Product decisions baked in here (as requested):
- No completion-percentage gate. Earlier this required ~90% of the previous season to
  be finished before suggesting the next one; that's been removed. Watching even a
  single genuinely-engaged episode of a season is enough to suggest the next season
  exists -- "did you start this show" rather than "did you finish it."
- The recommendation is always a SERIES-level entity (a season of a show), never a
  bare episode -- consistent with discovery recommendations, which are also
  series/movie-level, not episode-level.

Uses the structured `content_series_id` / `content_season_id` / `content_season_number`
fields from cms_v_watch_history_with_content, NOT permalink slug parsing -- the same
structured linkage used for series/season grouping across the whole pipeline (see
pipeline_series_structured.py and AUDIENCE_CLUSTERS.md for why the slug approach was
replaced).
"""
import pandas as pd


def build_season_index(structured_watch_df):
    """
    structured_watch_df: one row per (content_id) with columns
        content_id, series_id, season_id, season_number, episode_number
    covering the FULL catalog (not just tagged titles) -- this should come from a
    query against cms_v_watch_history_with_content / cms_v_videos_latest joined on
    content_series_id, grouped by content_id with max() on the season/episode fields
    (see structured_linkage.csv build in this session for the query pattern).

    Returns: series_id -> {season_number -> [content_id, ...]} sorted by episode_number,
    used to find "the next season" and how many episodes it has.
    """
    idx = {}
    for series_id, grp in structured_watch_df.groupby('series_id'):
        seasons = {}
        for season_number, sgrp in grp.groupby('season_number'):
            seasons[season_number] = sgrp.sort_values('episode_number').content_id.tolist()
        idx[series_id] = seasons
    return idx


def recommend_next_seasons(user_watched_content_ids, content_to_series, season_index,
                            show_titles=None, min_episodes_watched=1):
    """
    user_watched_content_ids: set of content_ids the user has GENUINELY watched (each
        entry should already have passed a per-episode engagement filter, e.g.
        seconds_watched > 0.6 * content_run_length_secs, before being passed in here).
    content_to_series: content_id -> series_id.
    season_index: output of build_season_index().
    show_titles: optional series_id -> real show title (e.g. from cms_v_series_latest,
        see data/series_display_names.csv) -- used to build a proper "Show Season N"
        display_name. Falls back to the raw series_id if not provided.
    min_episodes_watched: minimum number of a season's episodes the user must have
        genuinely watched before the NEXT season is suggested. Deliberately NOT a
        completion-percentage threshold -- default 1, i.e. any real engagement with
        the season is enough. (This previously required ~90% completion; that gate
        has been removed per product direction -- "watched the previous season"
        means started it, not necessarily finished it.)

    Returns a list of dicts: {series_id, display_name, content_type, watched_season,
    next_season, episode_count} -- one per series where the user has started (but not
    finished, and not yet started) the next season. `content_type` is always 'series'
    (never 'episode'), so this can be merged directly into the same list format as
    discovery-ranked movie/series candidates -- no separate rail, no episode-level
    entries.
    """
    show_titles = show_titles or {}
    watched_by_series = {}
    for cid in user_watched_content_ids:
        sid = content_to_series.get(cid)
        if sid is None:
            continue
        watched_by_series.setdefault(sid, set()).add(cid)

    recommendations = []
    for series_id, watched_cids in watched_by_series.items():
        seasons = season_index.get(series_id)
        if not seasons:
            continue
        season_numbers = sorted(s for s in seasons if s > 0)  # 0 = unassigned/placeholder
        for sn in season_numbers:
            episodes = seasons[sn]
            watched_in_season = watched_cids & set(episodes)
            if len(watched_in_season) < min_episodes_watched:
                continue
            next_sn = sn + 1
            if next_sn not in seasons:
                continue
            next_episodes = seasons[next_sn]
            already_started_next = bool(watched_cids & set(next_episodes))
            if already_started_next:
                continue
            raw_sid = series_id.split('::', 1)[1] if '::' in series_id else series_id
            show_title = show_titles.get(raw_sid, raw_sid)
            recommendations.append({
                'series_id': series_id,
                'display_name': f"{show_title} Season {next_sn}",
                'content_type': 'series',
                'watched_season': sn,
                'next_season': next_sn,
                'episode_count': len(next_episodes),
            })
    return recommendations


def merge_into_single_list(continuation_recs, discovery_ranked, top_n=10):
    """
    Combine season-continuation recommendations with the regular discovery-ranked
    list into ONE single list -- no separate rail. Continuation entries are placed
    first (a user partway through a show is a stronger, more specific signal than a
    popularity/cluster-affinity guess), followed by discovery entries, deduped by
    series_id, truncated to top_n total. Both continuation and discovery entries are
    already series/movie-level (never bare episodes), so the merged list is uniform.

    continuation_recs: output of recommend_next_seasons().
    discovery_ranked: list of dicts with at least {series_id, display_name,
        content_type, genre_normalized} in discovery-ranked order.
    """
    seen = set()
    merged = []
    for rec in continuation_recs:
        if rec['series_id'] in seen:
            continue
        seen.add(rec['series_id'])
        merged.append(rec)
    for rec in discovery_ranked:
        if len(merged) >= top_n:
            break
        if rec['series_id'] in seen:
            continue
        seen.add(rec['series_id'])
        merged.append(rec)
    return merged[:top_n]


if __name__ == '__main__':
    # Demo using the real Sampurna Season-1-completion example validated earlier in
    # this project (5,000 real users, 66.1% went on to watch Season 2).
    season_index = {
        '42c19725-9d73-4bb8-8fd6-d6ef6228958c': {
            1: ['s1e1', 's1e2', 's1e3', 's1e4', 's1e5', 's1e6'],
            2: ['s2e1', 's2e2', 's2e3', 's2e4', 's2e5', 's2e6'],
        }
    }
    content_to_series = {f's1e{i}': '42c19725-9d73-4bb8-8fd6-d6ef6228958c' for i in range(1, 7)}
    content_to_series.update({f's2e{i}': '42c19725-9d73-4bb8-8fd6-d6ef6228958c' for i in range(1, 7)})
    show_titles = {'42c19725-9d73-4bb8-8fd6-d6ef6228958c': 'Sampurna'}

    user_finished = {'s1e1', 's1e2', 's1e3', 's1e4', 's1e5', 's1e6'}  # finished S1, hasn't touched S2
    recs = recommend_next_seasons(user_finished, content_to_series, season_index, show_titles)
    print(recs)
    assert recs == [{
        'series_id': '42c19725-9d73-4bb8-8fd6-d6ef6228958c',
        'display_name': 'Sampurna Season 2',
        'content_type': 'series',
        'watched_season': 1,
        'next_season': 2,
        'episode_count': 6,
    }]
    print("OK: user who finished Season 1 gets 'Sampurna Season 2' recommended (series-level, not an episode).")

    # updated behavior: NO completion cap -- a single genuinely-watched episode of
    # Season 1 is now enough to suggest Season 2 exists (previously this required
    # ~90% completion and would have returned nothing).
    user_barely_started = {'s1e1'}  # watched just 1 of 6 Season 1 episodes
    recs_barely = recommend_next_seasons(user_barely_started, content_to_series, season_index, show_titles)
    assert recs_barely == [{
        'series_id': '42c19725-9d73-4bb8-8fd6-d6ef6228958c',
        'display_name': 'Sampurna Season 2',
        'content_type': 'series',
        'watched_season': 1,
        'next_season': 2,
        'episode_count': 6,
    }]
    print("OK: user who watched just 1 of 6 Season 1 episodes still gets Season 2 recommended (no 90% cap).")

    # merge demo: continuation entry + a couple of ordinary discovery entries, as ONE list
    discovery_ranked = [
        {'series_id': 'movie::abc', 'display_name': 'Some Movie', 'content_type': 'movie', 'genre_normalized': 'Drama'},
        {'series_id': 'struct::xyz', 'display_name': 'Some Other Show Season 1', 'content_type': 'series', 'genre_normalized': 'Thriller'},
    ]
    merged = merge_into_single_list(recs, discovery_ranked, top_n=10)
    assert merged[0]['display_name'] == 'Sampurna Season 2'
    assert merged[1]['display_name'] == 'Some Movie'
    assert merged[2]['display_name'] == 'Some Other Show Season 1'
    print("OK: merged single list places the continuation entry first, followed by discovery entries -- one list, no separate rail.")
