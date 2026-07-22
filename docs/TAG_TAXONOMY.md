# Storyline & Tone Tag Taxonomy (v2)

Closed vocabulary for LLM-based tag extraction from `synopsis_text`. Each title gets 2-4 `storyline_tags` and 1-3 `overall_tone_tags`, chosen only from these lists (no free-form additions).

**v2 update (branch `retag-storyline-tone-v2`):** the v1 vocabulary was built from a 30-title pilot sample that, by its own documented caveat below, didn't include niche genres like Sci-Fi, Fantasy, Musical, Sports, or Mythological/Devotional. Once the catalog grew to 775 titles, those genres existed but had no matching tags — titles were getting force-fit into generic tags like `family_conflict`/`self_discovery` that missed what was actually distinctive about them (e.g. a Sports Drama about an underdog athlete tagged the same as a family melodrama). Checked directly: 25 titles across these niche genres were re-read and re-tagged with 8 new storyline tags + 3 new tone tags added below (marked **v2**); the other ~750 titles (well-served by v1) were left untouched. See [DATA_AND_METHODOLOGY.md](DATA_AND_METHODOLOGY.md) for how this fits into the current pipeline.

## Storyline tags (50: 42 v1 + 8 v2)

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
| **Sports & competition (v2)** | `sports_underdog_journey` |
| **Music & art (v2)** | `musical_passion_pursuit` |
| **Sci-fi & speculative (v2)** | `time_travel_reincarnation`, `rogue_experiment_conspiracy`, `epic_quest_exploration` |
| **Mythology & the divine (v2)** | `divine_miracle_intervention`, `mythological_gods_demons_war`, `occult_exorcism` |

## Tone tags (19: 16 v1 + 3 v2)

`tense`, `suspenseful`, `dark`, `comedic`, `dramatic`, `emotional`, `mysterious`, `bittersweet`, `heartwarming`, `chaotic`, `empowering`, `uplifting`, `ominous`, `eerie`, `intense`, `tragic`, **`epic` (v2)**, **`whimsical` (v2)**, **`awe_inspiring` (v2)**

## Notes

- v1 vocabulary derived from a 30-title sample across 8 genres — this gap (uncommon genres like Action, Fantasy, Musical weren't in the sample) is exactly what the v2 update above addresses.
- If a title needs a storyline that doesn't fit any tag, flag it rather than force-fitting — used as a signal for the next vocabulary update.
- `time_travel_reincarnation` covers both literal time travel (sci-fi) and past-life reincarnation (mythological/drama) — deliberately merged since both titles found in this catalog treat "displacement across time/lives" as the same narrative engine, not two separate concerns worth separate tags at this catalog size.
- `mythological_gods_demons_war` is distinct from the existing `final_battle` (v1): `final_battle` is a generic climax beat any genre can have, this new tag is specifically for stories where the central premise *is* a divine-vs-demonic conflict (Ramayana/Mahishasura-style epics), not just any story that happens to end in a fight.
