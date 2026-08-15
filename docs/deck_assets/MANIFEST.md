# Deck Assets Manifest

Consolidated from `docs/` (originals left in place — see "Copy, not move" below). Every file listed here was opened and visually inspected this round, not just filename-checked. Nothing was regenerated or edited — defects found are flagged, not fixed, per this round's scope.

**Copy, not move**: originals remain in `docs/`. Three generator scripts write directly to `docs/` as their canonical, documented output path (`src/generate_deck_visuals.py`, `src/data/verify_pipeline.py`, and `evaluate_holdout.py` per `README.md:174`) — moving the originals away would break future re-runs of those scripts. Everything under `deck_assets/` is a copy for one-stop access.

---

## comparison/

| File | Shows | Feedback point | Dimensions | Aspect ratio | Notes |
|---|---|---|---|---|---|
| `chart_v1_vs_v2_comparison.png` | Two-panel bar chart: IoU by category (near-identical v1/v2) + false-positive area % by category (real, modest improvement) | **Comparison slides** | 1785×832 | 2.15:1 | Wider than 16:9 (1.78:1) — will letterbox or need trimming margins to fill a slide cleanly. Verified renders correctly, no defects found. |

## confusion_matrix/

| File | Shows | Feedback point | Dimensions | Aspect ratio | Notes |
|---|---|---|---|---|---|
| `chart_confusion_matrix.png` | Category × outcome heatmap, v2 scene counts (identical to v1) | **More visuals** (also supports comparison slides) | 1183×658 | 1.80:1 | Closest of all assets to native 16:9 — should drop into a slide with minimal adjustment. Verified renders correctly, no defects found. |

## region_map/

| File | Shows | Feedback point | Dimensions | Aspect ratio | Notes |
|---|---|---|---|---|---|
| `chart_region_accuracy.png` | v2 pass rate by region, sorted, with n/total labels | **Area of difference / different places** | 1330×776 | 1.71:1 | Close to 16:9. Verified renders correctly, no defects found. This is the direct visual for this feedback point — see "Gaps" below, it was the only one with zero visual coverage before last round. |

## scene_overlays/

Curated selection — 5 of the 14 available `holdout_v1/v2_viz_*.png` files, not all of them (v1 and v2 are visually near-identical for most scenes since their predictions barely differ; kept both versions only for the one scene where they genuinely diverge).

| File | Shows | Feedback point | Dimensions | Aspect ratio | Notes |
|---|---|---|---|---|---|
| `holdout_v2_viz_oil_00000.png` | SAR / ground truth / prediction, oil scene, IoU 0.955 (near-perfect) | **More visuals** | 2250×750 | 3.00:1 | Best "correct detection" example. **Title text ("Prediction (V2)" + IoU/Dice line) is clipped at the top edge** — a layout bug in the source generator (`evaluate_holdout.py`), present in all 14 holdout viz files, not specific to this one. Very wide aspect ratio (3:1) — will need significant letterboxing or cropping to fit 16:9. |
| `holdout_v2_viz_oil_00002.png` | Same, oil scene, IoU 0.259 (partial/imperfect) | **More visuals** | 2250×750 | 3.00:1 | Honest "not every detection is clean" example. Same title-clipping defect as above. Same wide-aspect caveat. |
| `holdout_v2_viz_lookalike_00001.png` | Same, lookalike scene, IoU 0.000 — large false-positive region | **More visuals** (also supports the lookalike-problem narrative) | 2250×750 | 3.00:1 | Clean SAR panel (no swath artifact — see below), dramatic and legible false positive. Same title-clipping defect. Same wide-aspect caveat. |
| `holdout_v1_viz_no_oil_00001.png` | Same, no_oil scene under **v1** — perfect suppression (IoU 1.000) | **More visuals** — this pair is the strongest "before/after" visual asset in the whole set | 2250×750 | 3.00:1 | Pair with the v2 file below: v1 correctly suppresses this scene, v2 doesn't. Same title-clipping defect. Same wide-aspect caveat. |
| `holdout_v2_viz_no_oil_00001.png` | Same scene under **v2** — false positive appears (IoU 0.000) | **More visuals** | 2250×750 | 3.00:1 | Same title-clipping defect. Same wide-aspect caveat. |

**Not included, and why**: `oil_00001` (redundant with `oil_00000`, both near-perfect), `lookalike_00000` and `no_oil_00000` (both show a real artifact — the SAR VV panel has a white triangular no-data wedge from the source scene's rotated satellite swath geometry, not a bug, but not clean for a slide without a caption explaining it), and the v1 copies of every scene except `no_oil_00001` (v1 and v2 are visually indistinguishable for those scenes — keeping both is redundant).

## architecture/

**Gap — no image exists.** See `architecture/GAP.md`. `docs/system_architecture.md` has a Mermaid diagram written as markdown text, not a rendered image file, and no other architecture visual exists anywhere in the repo (confirmed by a full repo-wide image search, not assumed). If an architecture slide is wanted, the Mermaid block needs to be rendered and exported separately — not done this round (inventory/consolidation only).

---

## Gaps against the professor's three feedback points

| Feedback point | Covered? |
|---|---|
| Better comparison slides | **Yes** — `comparison/chart_v1_vs_v2_comparison.png`, plus `confusion_matrix/chart_confusion_matrix.png` |
| Area of difference / different places | **Yes** — `region_map/chart_region_accuracy.png` |
| More visuals overall | **Yes** — all of the above plus `scene_overlays/` |

All three feedback points now have at least one supporting visual. The one **notable gap found this round that isn't one of the three named feedback points**: no rendered system-architecture diagram exists anywhere, despite `system_architecture.md` having been corrected content-wise in an earlier round. Flagging since a deck about a technical system will likely want one even though the professor didn't name it explicitly.

## Defects found this round (flagged, not fixed — inventory-only scope)

1. **`docs/data_inspection.png`** (not copied into `deck_assets/` — this is a pipeline-verification diagnostic, not deck material): the VH Polarization panel renders completely blank/black. Traced to the underlying synthetic data generator, not a plotting bug — worth knowing if this file is ever considered for a slide, but it isn't part of this manifest's curated set.
2. **Title-clipping across all 14 `holdout_v1/v2_viz_*.png` files** (5 of which are in `scene_overlays/`): the prediction panel's title text is cut off at the top edge of the image in every single one. Consistent, reproducible, affects the whole set equally — a fixable layout issue in `evaluate_holdout.py`'s figure generation, not something wrong with any individual scene's data.
3. **Rotated-swath white padding** in `lookalike_00000` and `no_oil_00000`'s SAR panels (both versions) — real source-imagery geometry, not a bug, but visually reads as broken without context. Excluded from the curated `scene_overlays/` selection for this reason.
4. **Aspect ratio mismatch**: all `holdout_*_viz_*.png` files are 3:1 (very wide, flat strips) vs. a 16:9 slide's 1.78:1 — they will need noticeable letterboxing or cropping to use directly. The three chart PNGs are much closer to 16:9 already.

None of the above were fixed this round — inventory and consolidation only, per instruction.
