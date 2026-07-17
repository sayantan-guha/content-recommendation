# Storyline & Tone Tag Taxonomy (v1)

Closed vocabulary for LLM-based tag extraction from `synopsis_text`. Built from reviewing 30 stratified sample titles (see `data/sample_for_tagging_30.csv`). Each title gets 2-4 `storyline_tags` and 1-3 `overall_tone_tags`, chosen only from these lists (no free-form additions).

## Storyline tags (42)

| Category | Tags |
|---|---|
| Crime & mystery | `crime_investigation`, `murder_mystery`, `missing_person`, `cover_up` |
| Thriller & action | `terrorism`, `espionage_undercover`, `fugitive_manhunt`, `kidnapping`, `heist` |
| Family & relationships | `family_conflict`, `family_secrets`, `paternity_question`, `domestic_conflict`, `in_law_tension` |
| Romance | `forbidden_love`, `love_triangle`, `arranged_marriage`, `extramarital_affair`, `reconciliation` |
| Horror & supernatural | `supernatural_threat`, `ghost_haunting`, `curse_ritual`, `final_battle` |
| Social & political | `social_injustice`, `political_unrest`, `harassment_accusation`, `class_divide`, `gender_empowerment` |
| Deception & betrayal | `deception_disguise`, `betrayal`, `scam_con`, `manipulation` |
| Comedy structure | `mistaken_identity`, `comedy_of_errors`, `ensemble_multiple_storylines`, `family_rivalry` |
| Personal journey | `self_discovery`, `midlife_crisis`, `unlikely_friendship` |
| Legal & stakes | `legal_case_trial`, `life_threat`, `survival` |

## Tone tags (16)

`tense`, `suspenseful`, `dark`, `comedic`, `dramatic`, `emotional`, `mysterious`, `bittersweet`, `heartwarming`, `chaotic`, `empowering`, `uplifting`, `ominous`, `eerie`, `intense`, `tragic`

## Notes

- v1 vocabulary derived from a 30-title sample across 8 genres — expect a small number of new tags to be needed once all 500 titles are processed (uncommon genres like Action, Fantasy, Musical weren't in the sample).
- If a title needs a storyline that doesn't fit any tag, flag it rather than force-fitting — used as a signal for a v2 vocabulary update.
