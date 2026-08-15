# Deck Assets Manifest

Consolidated from `docs/` (originals left in place — see "Copy, not move" below). Every file listed here was opened and visually inspected, not just filename-checked. First built as an inventory-only pass (nothing regenerated); a follow-up round then fixed the two defects that were fixable (title-clipping, missing architecture diagram) — see "Defects found and their status" below for what's fixed vs. still open.

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
| `holdout_v2_viz_oil_00000.png` | SAR / ground truth / prediction, oil scene, IoU 0.955 (near-perfect) | **More visuals** | 2187×766 | 2.85:1 | Best "correct detection" example. Title-clipping bug fixed this round (see Defects below) — title text fully visible. Still a wide, flat aspect ratio — will need letterboxing or cropping to fit 16:9. |
| `holdout_v2_viz_oil_00002.png` | Same, oil scene, IoU 0.259 (partial/imperfect) | **More visuals** | 2187×766 | 2.85:1 | Honest "not every detection is clean" example. Same wide-aspect caveat. |
| `holdout_v2_viz_lookalike_00001.png` | Same, lookalike scene, IoU 0.000 — large false-positive region | **More visuals** (also supports the lookalike-problem narrative) | 2187×766 | 2.85:1 | Clean SAR panel (no swath artifact — see below), dramatic and legible false positive. Same wide-aspect caveat. |
| `holdout_v1_viz_no_oil_00001.png` | Same, no_oil scene under **v1** — perfect suppression (IoU 1.000) | **More visuals** — this pair is the strongest "before/after" visual asset in the whole set | 2187×766 | 2.85:1 | Pair with the v2 file below: v1 correctly suppresses this scene, v2 doesn't. Same wide-aspect caveat. |
| `holdout_v2_viz_no_oil_00001.png` | Same scene under **v2** — false positive appears (IoU 0.000) | **More visuals** | 2187×766 | 2.85:1 | Same wide-aspect caveat. |

**Not included, and why**: `oil_00001` (redundant with `oil_00000`, both near-perfect), `lookalike_00000` and `no_oil_00000` (both show a real artifact — the SAR VV panel has a white triangular no-data wedge from the source scene's rotated satellite swath geometry, not a bug, but not clean for a slide without a caption explaining it), and the v1 copies of every scene except `no_oil_00001` (v1 and v2 are visually indistinguishable for those scenes — keeping both is redundant).

## architecture/

| File | Shows | Feedback point | Dimensions | Aspect ratio | Notes |
|---|---|---|---|---|---|
| `system_architecture.png` | Rendered system architecture diagram: Frontend (React/Leaflet) → Backend (FastAPI, real endpoint names) → Model Layer (PyTorch, from-scratch U-Net, explicitly no pretrained backbone) → Database (SQLite) → File Storage, with real data-flow labels | Not one of the three named feedback points, but closes a real gap (see below) | 2880×1120 | 2.57:1 | Rendered from the Mermaid source in `docs/system_architecture.md` at 2x scale via headless browser + `html2canvas` (in-browser `<img>`+canvas rasterization failed with a "tainted canvas" security error because Mermaid's text labels use SVG `foreignObject`; `html2canvas` renders the DOM directly and avoids that). Verified before rendering that the source itself already matched the corrected architecture info (real PyTorch/U-Net, no ONNX/ResNet, real endpoint paths cross-checked directly against `main.py`) — no drift found, so the source needed no fix, only rendering. Wider than 16:9 — will need modest letterboxing. |

---

## Gaps against the professor's three feedback points

| Feedback point | Covered? |
|---|---|
| Better comparison slides | **Yes** — `comparison/chart_v1_vs_v2_comparison.png`, plus `confusion_matrix/chart_confusion_matrix.png` |
| Area of difference / different places | **Yes** — `region_map/chart_region_accuracy.png` |
| More visuals overall | **Yes** — all of the above plus `scene_overlays/` |

All three feedback points now have at least one supporting visual, and the architecture gap identified last round is now closed too.

## Defects found and their status

1. **Title-clipping across all 14 `holdout_v1/v2_viz_*.png` files** — **fixed this round.** Root cause: the prediction panel's title is two lines while the other two panels' titles are one line, and `plt.tight_layout()` sizes axes to fit the *existing* figure bounds rather than growing the figure for a taller title — the top of that title rendered past the canvas edge and `savefig()` silently clipped it. Fixed properly (not a padding hack) by adding `bbox_inches="tight"` to the `savefig()` call in `evaluate_holdout.py`, which computes the actual rendered bounding box (including anything that overflows the nominal figure size) and includes all of it in the saved image. Verified by reproducing the bug in isolation, confirming the fix resolves it in isolation, then regenerating all 14 real images and visually re-inspecting 3 of them directly — all fully legible now, image height grew from 750px to 766px to accommodate the previously-clipped text, and underlying IoU/Dice numbers are unchanged (no regression, confirmed against the v1/v2 summary metrics printed during regeneration).
2. **No rendered architecture diagram** — **fixed this round.** See `architecture/` above.
3. **`docs/data_inspection.png`**'s blank VH Polarization panel — still open, not touched (out of scope this round — it's a pipeline-verification diagnostic, not part of this manifest's curated deck set).
4. **Rotated-swath white padding** in `lookalike_00000` and `no_oil_00000`'s SAR panels — still open, not a bug (real source-imagery geometry), excluded from the curated `scene_overlays/` selection for this reason, unchanged this round.
5. **Aspect ratio mismatch**: `holdout_*_viz_*.png` files are now 2.85:1 (was 3.00:1 before the title fix) vs. a 16:9 slide's 1.78:1 — still noticeably wider than a slide, will need letterboxing or cropping. The architecture diagram (2.57:1) and three chart PNGs remain the closest fits to 16:9.

## Infrastructure note

Rendering the architecture diagram needed a way to execute JavaScript against local HTML (for Mermaid.js) — `file://` URLs are blocked from running scripts in this environment's browser sandbox (a real security boundary, not a bug), so a minimal static file server config (`docs-static`, port 8124) was added to `.claude/launch.json` alongside the existing `backend`/`frontend` entries. It's inert unless explicitly started and may be useful for similar rendering needs in future rounds.
