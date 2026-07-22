# Storyline & Tone Tag Taxonomy (v3 — full rebuild)

Closed vocabulary for LLM-based tag extraction from `synopsis_text`. Each title gets 2-4 `storyline_tags` and 1-3 `overall_tone_tags`, chosen only from these lists (no free-form additions).

**v3 is a from-scratch rebuild, not an extension of v1/v2.** The v1 vocabulary (42 storyline + 16 tone tags) was designed from a 30-title pilot sample; v2 patched 8 tags onto it for genres the pilot missed. Both were built when the catalog was much smaller. With the catalog at 775 titles, this rebuild designed the vocabulary entirely from scratch — a stratified sample spanning every genre in the current catalog, not reusing any v1/v2 tag names — then re-tagged all 775 titles against it from scratch, not incrementally patched.

**Scale of the rebuild:** all 775 titles tagged (dispatched across 12 parallel tagging passes of ~65 titles each, same closed-vocabulary methodology as before — read the synopsis, pick only from the fixed lists below). 100% vocabulary compliance verified programmatically (zero tags used outside the lists, zero empty tag rows). Every one of the 65 storyline tags and 16 tone tags below is actually used at least once in the final tagging (no dead tags). `tag_low_confidence` (thin/generic synopses, e.g. one-line blurbs, test entries) is at 87/775 — comparable to v1/v2's terminal rate, confirming these are genuine thin-content cases rather than a vocabulary-fit problem.

## Storyline tags (65)

| Category | Tags |
|---|---|
| Crime & investigation | `disappearance_investigation`, `serial_predator_pattern`, `cold_case_reopened`, `true_crime_confession`, `wrongful_accusation_clearing`, `undercover_impersonation`, `stolen_fortune_conspiracy`, `courtroom_battle`, `power_underworld_struggle`, `criminal_past_resurfacing` |
| Mystery & suspense | `isolated_group_secrets`, `staged_death_puzzle`, `suicide_or_murder_ambiguity`, `gaslight_conspiracy_doubt`, `immortal_witness_revelation`, `obsessive_intrusion` |
| Action & justice | `corrupt_system_takedown`, `vigilante_retribution`, `gang_rivalry_clash`, `freedom_movement_struggle` |
| Adventure & quest | `hidden_treasure_hunt`, `legendary_place_quest`, `exotic_expedition_mystery`, `wilderness_rescue_chase` |
| Horror & supernatural | `deity_curse_deliverance`, `tantric_exorcism_battle`, `haunted_dwelling_mystery`, `folklore_come_alive` |
| Sci-fi & speculative | `covert_experiment_conspiracy`, `accidental_time_displacement`, `lost_invention_pursuit` |
| Mythology & devotion | `divine_epic_retelling`, `faith_and_devotion_journey` |
| Family & relationships | `controlling_parent_dynamic`, `sibling_bond_strain`, `long_awaited_reunion`, `family_under_siege`, `estranged_family_confession`, `family_opposed_romance` |
| Romance | `cross_border_love_pursuit`, `meet_cute_mishap`, `love_triangle_friendship_test`, `dating_app_mixup`, `multi_couple_breakup_saga`, `forced_cohabitation_trial`, `timeless_love_nostalgia` |
| Comedy structure | `eccentric_local_hero`, `taboo_relationship_experiment`, `trickster_takeover`, `house_sharing_mishaps`, `festive_surprise_reunion` |
| Personal journey & drama | `creative_passion_vs_duty`, `addiction_spiral_redemption`, `body_image_insecurity`, `grief_and_loss_processing`, `dying_art_companionship`, `unlikely_ally_bond`, `abuse_survivor_justice`, `ambition_marriage_strain`, `musical_legacy_pursuit`, `underdog_athletic_ambition`, `small_town_dreamer_makeover` |
| Social & documentary | `cultural_documentary_profile`, `classic_literary_adaptation`, `stand_up_comedy_showcase`, `communal_dining_conversation`, `ensemble_womens_survival` |

## Tone tags (16)

`gritty`, `foreboding`, `wistful`, `tender`, `frenetic`, `triumphant`, `unsettling`, `wry`, `poignant`, `gripping`, `playful`, `reverent`, `melancholic`, `cathartic`, `nostalgic`, `sardonic`

## Notes

- This vocabulary was designed by directly reading a stratified sample of ~130 titles across every genre in the current 775-title catalog, then locked before tagging began — same closed-vocabulary discipline as v1/v2, applied at the current catalog's actual scale and genre spread instead of a 30-title pilot.
- If a title needs a storyline that doesn't fit any tag, flag it rather than force-fitting — used as a signal for the next vocabulary update.
- **Known architectural limitation this rebuild does NOT fix:** the K=8 content-category clustering compresses each title's raw tag vector down to an 8-dimensional softmax "mixture" before any downstream similarity scoring happens (`src/recommender.py`'s cold-start fallback, and the retired cluster model both operate on this compressed vector, not the raw tags). Spot-checked directly: for well-served mainstream genres, this compression makes many unrelated titles score ~0.99-1.0 cosine-similar to each other regardless of tag quality — a ceiling on how much any tag vocabulary improvement can sharpen mainstream-genre similarity. Where the new vocabulary demonstrably *does* help is niche/thin genres: e.g. "Golondaaj" (a sports underdog film) previously matched top-6 with unrelated Documentary/Social Drama titles at similarity=1.0 under the old vocabulary; under this rebuild it correctly surfaces "East Bengaler Chhele" (another sports underdog film) as its closest match. The fix is genre-specific richness, not a fix to the compression ceiling itself.
