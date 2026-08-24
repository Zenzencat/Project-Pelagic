# Project Pelagic: Status & Handoff Document

This document captures the current status of the project, including recent resolutions, verified assumptions, training runs, evaluation metrics, and remaining open issues. Rewritten in full during a health-check pass (session 5) after drift was found across four prior sessions' worth of edits — prior versions of this file had several confirmed fixes missing entirely (the caching fix, inference-path consolidation, coordinate/GSD fix, DSen2-CR scoping) and one section whose "Resolved" framing was no longer accurate. If you're reading this cold, this file should now be trustworthy end to end.

---

## Recent Resolutions

### 1. Part II File Count Discrepancy
* **Status**: Resolved (documentation error, not a data bug).
* Part II (Zenodo Record ID `8253899`) consists of `Mask_no_oil` (685 files) and `Mask_lookalike` (685 files), totaling **1,370 scenes**. The combined Part I + Part II training/validation dataset contains **2,570 scenes** (1,200 from Part I + 1,370 from Part II). README.md, ai_technique.md, and download_sample.py reflect the corrected totals.

### 2. Hard-Fail Guard in `get_real_dataloaders()`
* **Status**: Resolved.
* Added `allow_verification_fallback: bool = False` to `get_real_dataloaders()` in `src/data/dataset.py`. If matching GeoTIFF images for masks are not found and the parameter is `False`, the loader raises a `RuntimeError` at construction time instead of silently falling back to synthetic mock data. `verify_real_pipeline.py` passes `allow_verification_fallback=True` since its purpose is local pipeline-shape verification, not real training.

### 3. Smart dB/Linear-Scale Preprocessing — `main.py` only, NOT the shared preprocessing module
* **Status**: Resolved in `src/api/main.py`'s `/api/predict`. **Not** propagated to `src/data/preprocess.py::run_full_preprocessing()` — see Open Issue #1 below, found during this session's health check.
* `/api/predict` checks `np.any(image_raw < 0)` to distinguish real dB-scale Sentinel-1 imagery from linear-scale synthetic data and runs the correct calibration path for each. This fixed a real zero-confidence segmentation failure when real imagery was run through the synthetic-only preprocessor.
* This exact fix was never applied to the shared `run_full_preprocessing()` used by `get_dataloaders()` (the synthetic-only training path) and by `get_real_dataloaders()`'s verification-fallback branch — that function still unconditionally assumes dB-scale input. See Open Issue #1.

### 4. U-Net v2 Retraining with Scheduler and Oversampling
* **Status**: Completed.
* `kaggle_kernel/train_kaggle.py` v13, three changes vs. the v1 script that produced `model_real_best.pt`:
  1. **Cosine Annealing Scheduler**: `1e-3` decaying to `1e-6` over 15 epochs — fixed v1's wild epoch-to-epoch Val IoU oscillation (0.27–0.70).
  2. **Epoch-Level Global Metric Pooling**: validation IoU/Dice now pool intersection/union across the whole epoch instead of averaging noisy per-batch ratios.
  3. **Lookalike Hard-Negative Oversampling**: background patch sampling prob 0.005 for lookalike vs. 0.002 for no_oil (2.5x), ~71.4% of the negative background patch pool.
* Completed on a Tesla T4 GPU. Checkpoints: `model_real_v2_best.pt`, `model_real_v2_final.pt`. Log: `kaggle_output/training_v2_log.csv`.

### 5. Dataloader Duplication & Drift Between `dataset.py` and `train_kaggle.py`
* **Status**: Resolved.
* `dataset.py`'s local loader now uses the same category-balanced patch sampling as `train_kaggle.py` (via a shared `category` parameter threading through `src/data/preprocess.py::extract_patches_balanced()`). Both files carry an explicit cross-reference banner since `train_kaggle.py` can't import `src/data/` directly (Kaggle kernels must be single-file) — if you change sampling rates or hard-fail behavior in one, mirror it in the other.

### 6. `scene_id` Cache Bug in `/api/predict`
* **Status**: Resolved.
* `scene_id` has a `UNIQUE` constraint on the `detections` table. Re-predicting an already-seen scene under a *different* checkpoint used to silently return the stale cached row instead of re-running inference — the exact reason the demo DB needed manual resets between checkpoint swaps. Fixed via a `checkpoint_hash` column (MD5 of the loaded checkpoint file, migration-safe `ALTER TABLE`): `predict()` now compares the cached row's hash to the currently-loaded checkpoint and overwrites in place on mismatch, while still short-circuiting on a genuine same-checkpoint repeat call (no wasted recompute). Verified via an injected-staleness test.

### 7. Whole-Image vs. Tiled Inference Consolidated
* **Status**: Resolved.
* `/api/predict` used to run the full 2048x2048 scene through the U-Net in one forward pass; `evaluate_holdout.py` tiled into 256x256 patches matching the training regime. Diagnosed a `no_oil_00001` false positive that looked like a live-vs-eval discrepancy (70% confidence live vs. a "minor" 41px offline finding) — the two methods actually agree closely (36 vs. 41 active px, same location); the real story is a genuine **v2-only model regression** (v1 max prob 0.04 at that pixel, v2 max prob 0.83), not a code-path bug. Still consolidated both call sites into `src/inference.py::run_tiled_inference()` so they can't silently drift apart again.

### 8. Fake-Coordinate and Mock-GSD Bugs Fixed
* **Status**: Resolved.
* `/api/predict` used to display every detection at a fixed fake Gulf-of-Thailand coordinate (9.0N, 100.5E) and convert pixel distances using an arbitrary `0.00015` deg/pixel display constant, regardless of which scene was analyzed. Every real holdout GeoTIFF carries real embedded WGS84 georeferencing (`ModelTiepointTag`/`ModelPixelScaleTag`, confirmed ~10 m/pixel) that was simply never read. `/api/predict` now uses each scene's real georeferencing (`src/analysis/geoutils.py`), falling back to the old fixed point only for synthetic scenes with no real-world location. Verified end-to-end in the browser (map actually re-centers and re-renders per scene), and the 5-scene demo lineup re-confirmed with zero regression (see Demo Lineup section below).
* This also surfaced that **the 30 holdout scenes are not all Eastern Mediterranean** — real coordinates span 6 distinct regions (Eastern Mediterranean, Red Sea, Gulf of Mexico, Baltic Sea, Western Mediterranean, Sunda Strait). See Region Findings below.
* `checkpoint_comparison_summary`/`holdout_per_scene_results` now use each scene's real, **latitude-corrected** pixel area (a flat km/degree constant would be wrong by up to ~2x between the Baltic and Sunda Strait scenes — longitude-degree distance shrinks by `cos(latitude)`) instead of the old mock-scale-derived, `_UNVERIFIED`-flagged figures from an earlier pass.
* **Known remaining gap, not fixed**: the frontend's "ขนาดคราบ" (slick size) badge in `App.jsx` is still a hardcoded fake value (`8.5`/`14.5` km²) for every scene — never wired to real geojson-derived area. Flagged, not in scope for any session so far.

### 9. Mock AIS Layer Replaced with Live Sentinel-1 Fetch + Real GFW Attribution
* **Status**: Resolved.
* `/api/predict` used to attribute every detection to 2 hardcoded fake vessels (`Petro Express`, `Nakhon Fishery 21`) inserted directly in `main.py` — flagged as fake in an earlier round. Replaced with a genuinely additive **live mode**: a new `POST /api/live/fetch` endpoint (bbox + date range in, real Sentinel-1 imagery + real nearby-vessel attribution out), wired into the frontend as a new "Live Fetch" panel alongside (not replacing) the 4 verified demo scenes.
* **Real credentials were already present in `.env`** (not documented as such anywhere — `scripts/validate_credentials.py` only checked CDS/CDSE previously, and had never been run against the GFW token). Both CDSE and GFW credentials verified genuinely valid this session; `validate_credentials.py` now checks all three (CDS/CDSE/GFW).
* **CDSE integration pivoted mid-implementation based on a real test result**: the original plan was to download a raw Sentinel-1 SAFE product via CDSE's OData/zipper download service and calibrate it manually (parsing the product's calibration LUT XML). A live test showed this project's `CDSE_CLIENT_ID`/`CDSE_CLIENT_SECRET` are **Sentinel Hub OAuth2 client credentials** — they authenticate against CDSE's shared Keycloak token endpoint fine (`src/analysis/cdse_auth.py`), but the raw-download service rejects that token with HTTP 401 `"Token audience not allowed"` (error `DAT-ZIP-609`); it needs a different credential type (CDSE portal username/password) not present here. Pivoted to CDSE's **Sentinel Hub Process API** instead (`src/data/cdse_fetch.py`), verified live and working with the same credentials: it returns a real, already-orthorectified, already-`SIGMA0_ELLIPSOID`-calibrated GeoTIFF for an exact bbox, with real `ModelTiepointTag`/`ModelPixelScaleTag` values `src/analysis/geoutils.py` already parses unmodified. Strictly simpler than the original plan (no GDAL, no manual GCP handling, no LUT XML parsing, no multi-GB download — a live 0.2°×0.2° request completed in ~25s end-to-end including inference).
* **Real acquisition timestamp**: sourced from CDSE's public (no-auth) OData catalog search (`src/analysis/sentinel1_revisit_check.py::search_same_track_passes()`, reused as-is, extended with `id`/`end` fields) — `ContentDate.Start`/`End`, stored verbatim as `detections.acquisition_start_utc`/`acquisition_end_utc`. Never defaulted; `/api/live/fetch` returns a `SKIPPED` envelope if no product matches the given bbox/date range.
* **Real GFW attribution** (`src/analysis/gfw_client.py`): queries GFW's real `public-global-presence:latest` dataset (the all-vessel AIS presence dataset — deliberately not `public-global-fishing-effort`, which is fishing-only) via the v3 4Wings Report API, `group-by=MMSI`, for the acquisition day. Verified live against a real Singapore Strait scene (2026-08-11): returned real vessel records (MMSI, ship name, lat/lon, day-level timestamp) directly, no separate vessel-ID resolution call needed. Radius is a request parameter (default 10km, per the task's spec).
* **End-to-end verified live, not just at the SKIPPED gates** (both are exercised — see `src/data/verify_live_pipeline.py`): a real `POST /api/live/fetch` call against the Singapore Strait bbox returned `status: "OK"`, a real non-null acquisition timestamp, and 5 real candidate nearby vessels, rendered correctly in the browser (map re-centered, vessel markers + tooltips, honest `OK:` status banner). Confirmed via `get_page_text`/network-request inspection rather than a pixel screenshot — this session's browser tooling could not composite a frame (`"Browser pane is not displayed"`) despite the page rendering and functioning correctly; noted here rather than silently omitted.
* **DB migration**: `detections` gained `source`, `acquisition_start_utc`, `acquisition_end_utc`, `cdse_product_id`, `vessel_attribution_status`, `vessel_search_radius_km` (see `docs/db_schema.md`). One-time cleanup on migration: all pre-existing `nearby_vessels` rows deleted (100% legacy mock data — there's no real timestamp for any pre-existing detection to have run real attribution against) and every pre-existing detection's `vessel_attribution_status` set to `skipped_no_timestamp`. The frontend now shows that reason honestly instead of silently rendering "0 vessels."
* **Known, disclosed limitations for a future round to verify further**:
  1. GFW's 4Wings Report endpoint is day-resolution, not per-ping — "closest available AIS timestamp to acquisition" is honestly day-level, disclosed in `gfw_client.py`'s docstring and the UI copy, not hidden.
  2. The live-fetched Singapore Strait test scene scored ~0.79 confidence (a detection) over what is very likely open, oil-free shipping-lane water — plausible given this project's already-documented finding that the v2 checkpoint false-positives on real open water in general (see the Demo Lineup section below); this is a known model-generalization limitation, not a live-pipeline bug, but worth keeping in mind before using live mode in a presentation.
  3. Sentinel Hub Process API's synchronous endpoint caps output raster size (`MAX_OUTPUT_PX = 2048` in `cdse_fetch.py`) — a very large requested bbox is silently downsampled below the project's real ~10m GSD rather than rejected. Fine for a demo-scale AOI, would need chunking for a much larger one.
* **Not in scope / explicitly untouched**: the 4 cached demo scenes' own predict/cache-hit flow (beyond the one-time vessel-status backfill), `init_db()`'s pre-existing Gulf-of-Thailand/Andaman-Sea seed-mock detections (already excluded from the demo UI, and outside this task's literal "the 2 hardcoded mock_vessels in main.py" scope).

### 10. Live-Fetch Preprocessing Audit (double-calibration check) + Required GFW Attribution
* **Status**: Audited, no code defect found (the previous round's `already_calibrated` flag was already correct) — verified explicitly this round with real numbers, not assumed. GFW attribution text added.
* **Audit**: Sentinel Hub Process API's `SIGMA0_ELLIPSOID` output (used by `/api/live/fetch`, see Resolution #9) is already real, calibrated linear sigma0 — not raw uncalibrated digital numbers. `src/data/preprocess.py::preprocess_for_prediction()`'s linear-scale branch normally applies `calibrate_to_sigma()` (a DN→power squaring step meant for uncalibrated data), which would double-calibrate an already-calibrated live scene. Confirmed `src/api/main.py`'s live-fetch endpoint already calls `preprocess_for_prediction(image_raw, already_calibrated=True)`, which skips that squaring step — this was in fact already fixed during the previous round's implementation, not left as a latent bug; this round's job was to verify it, not to discover and fix it fresh.
* **Real verification, not assumption**: fetched a live Singapore Strait scene (2026-08-11) and inspected the pipeline stage-by-stage.
  * Pre-normalization sigma0→dB: VV min/max/mean = -50.00 / 40.23 / **-15.36 dB**, VH = -50.00 / 37.40 / **-30.91 dB**. A ~15.5dB VV−VH gap is the textbook real relationship for C-band ocean/land backscatter (cross-pol VH is always substantially weaker than co-pol VV) — physically sane, not garbage.
  * Final normalized [0,1] output (the correct `already_calibrated=True` path): VV mean 0.408 (0.03% floored at 0, 4.69% saturated at 1), VH mean 0.118 (71.2% floored at 0, 0.45% saturated at 1) — meaningful spread, not degenerate.
  * **Direct comparison against the wrong path** (`already_calibrated=False`, i.e. double-calibrating): VV floor-saturation jumps from 0.03% → 74.59%, VH from 71.2% → 89.2% — confirms double-calibration would measurably crush the signal, and confirms the current code correctly avoids it.
  * Cross-checked against real holdout scenes run through the same function's dB branch (`oil_00000`, `oil_00001`, `no_oil_00000`): those also show heavy VV floor-saturation (98.4–99.95%) as an existing, pre-existing characteristic of this pipeline (unrelated to live-fetch, unchanged by this round) — so the live path's statistical shape is consistent with, not an outlier from, the established real-data pipeline.
* **Conclusion**: no mismatch requiring a fix. `normalize_image()`'s fixed `[-25, 0]` dB clip bounds hold for Process API's pre-calibrated output — both are real sigma0 dB, the same physical quantity the model was trained on.
* **Required Global Fishing Watch attribution added** (GFW API Terms of Use, Section 3): `frontend/src/App.jsx` now renders a linked "Powered by Global Fishing Watch." credit (→ https://globalfishingwatch.org) in two places, both conditional on real vessel data actually being displayed (not a buried static footer): directly under the candidate-vessel list in the details panel, and on the map's floating layer-control card whenever the AIS overlay is on and showing real markers. **This same line must also appear on any presentation slide/deck that shows the vessel-attribution feature** — noted here so it isn't dropped when deck content is assembled: **"Powered by Global Fishing Watch." (https://globalfishingwatch.org)**.
* **Verification screenshots**: `docs/deck_assets/live_fallback/` (corrected path — this entry previously said `docs/screenshots/round14_live/`, which was never the real location) — live-fetch panel before a fetch, a real completed live detection with its real acquisition timestamp, the real GFW vessel attribution result (or honest empty state, whichever was true at capture time) with the "Powered by Global Fishing Watch" credit visible in frame. Presentation-day fallback if live network/CDSE access fails, same standard as the Round 9 demo fallback screenshots. Re-captured in Resolution #11/#12 below after both bugs found in those screenshots were fixed.

### 11. Land-Sea Masking for Live-Fetched Detections
* **Status**: Resolved.
* **The bug**: Round 14's live-fetch screenshots showed the oil-slick mask covering real Singapore land districts (Jurong West, Bukit Batok, Queenstown) — a known SAR oil-spill detection failure mode (dark backscatter over land can resemble the same low-backscatter signature as an oil slick) that the pipeline never explicitly guarded against. Confirmed directly, not assumed: re-running the exact same Singapore Strait live fetch and cross-referencing the raw predicted mask against `global_land_mask` showed **27.1% of the raw predicted "oil" pixels (715,798 of 2,641,623) were on land**.
* **Fix**: new `src/analysis/landmask.py::strip_land_pixels()`, wired into both `/api/predict` and `/api/live/fetch` in `src/api/main.py`, called on the raw binary mask **before** contour extraction (`mask_to_polygons`), confidence-score averaging, and the saved mask PNG — so the fix is at the raster/pixel level, not a frontend-only visual clip, and the reported slick area (km²) and confidence score are corrected too, not just the map drawing. Skipped for `/api/predict`'s synthetic-scene placeholder-coordinate fallback path (no real lat/lon there to check land/sea against).
* **Dataset choice**: `global-land-mask` (PyPI) — a ~1km-resolution (verified directly against its bundled `.npz`: 21600×43200 grid, 1/120deg per cell) global land/ocean raster derived from GSHHG, pure numpy + a ~2.5MB bundled array, no network call and no heavier geospatial stack (shapely/geopandas/fiona) needed beyond what this project already depends on. `strip_land_pixels()` on a full 2048×2048 scene runs in ~0.15s.
* **Known limitation, found and documented while sanity-checking the dataset before wiring it in**: land reclaimed more recently than GSHHG's data vintage is still classified as water — e.g. Singapore's Marina Bay (1.281N, 103.859E) reads `is_land=False`. Cross-checked this wasn't silently masking the fix's own verification: the originally-reported districts (Jurong West, Bukit Batok, Queenstown) and the live scene's actual port/reclaimed coastline (Pasir Panjang, Keppel Terminal, HarbourFront, Tanjong Pagar — all real, non-recent reclamation) were spot-checked individually against real coordinates and all correctly read `is_land=True`; a rendered screenshot that initially looked like it still covered these areas turned out to be OSM label placement sitting near, not on, the actual (correctly masked) coastline once zoomed in — verified by reading the actual saved mask pixel value at each real coordinate (all `0`/clear), not by eyeballing the rendered map.
* **Verified via a real live-fetch call**, same Singapore Strait bbox as Round 14 (`1.10,103.70` to `1.30,103.90`, `2026-08-05`–`2026-08-15`): response `detail` now reads *"Land-sea mask removed 715,798 of 2,641,623 raw predicted oil px (27.1%) that fell on land."*, shown live in the UI banner, not just logged. New screenshots in `docs/deck_assets/live_fallback/` show the corrected polygon hugging the real coastline (confirmed at zoom: small islands like Pulau Semakau correctly excluded as holes in the mask; the whole northern land mass Jurong West→Queenstown clear of any mask).
* **Cached demo scenes confirmed unaffected**: ran `strip_land_pixels()` directly against all 4 cached demo scenes' saved mask PNGs (`oil_00000/1/3/4`) — `land_px_fraction_in_scene = 0.0` for all four (zero land anywhere in any of their scene bboxes, not just their polygons), `oil_px_removed_on_land = 0` for all four. Confirms these curated scenes are genuinely 100% open water and this fix has zero effect on them, verified directly rather than assumed from curation history.
* **Not yet done**: `checkpoint_comparison_summary`/holdout evaluation metrics were not re-run through this mask — this round only touched the live API's inference-serving path (`/api/predict`, `/api/live/fetch`), not `evaluate_holdout.py`'s standalone metric computation. If land-masking should also correct the held-out IoU/Dice numbers reported elsewhere in this doc, that's a separate, not-yet-done pass.

### 12. GFW Vessel Position Precision & Map Marker Stacking Fixed
* **Status**: Resolved.
* **The bug**: Round 14 screenshots showed 5 attributed vessels with 3 of them at an identical 0.0km distance, and only 2 of 5 markers visible on the map.
* **Root cause, confirmed against the real raw GFW API response, not assumed**: GFW's 4Wings Report endpoint (`public-global-presence:latest`, `spatial-resolution: HIGH`) reports each vessel's position as its **0.01-degree grid-cell center** (verified directly against real response rows — every `lat`/`lon` value lands on an exact 0.01deg multiple), not a precise AIS ping. Three real, distinct vessels (Marina Genesis, Cast Lion, Torm Dover — different MMSIs, genuinely present) legitimately shared one grid cell that day, so their haversine distance from that identical cell center really was identical — a correctly-computed distance to an honestly-coarse position, **not** a silently-failed calculation defaulting to 0. On the map, `CircleMarker`s drawn at that identical lat/lon rendered exactly on top of each other, so only the topmost of each co-located group was visible/clickable — the literal cause of "only 2 of 5 markers."
* **Also found and fixed while investigating**: `group-by: MMSI` groups by grid-cell-per-day, not one row per vessel overall — a vessel crossing more than one 0.01deg cell on the acquisition day produced multiple raw rows under the same MMSI. `get_nearby_vessels()` now deduplicates by MMSI (keeping the closest-to-centroid entry) so one real vessel can't occupy two of the returned candidate slots.
* **Fix**: `src/analysis/gfw_client.py` now computes and returns `position_resolution_m` per vessel (the real grid-cell diagonal size at that latitude, ~1572m for the Singapore Strait test) alongside the unchanged, correctly-computed `distance_meters`. New `nearby_vessels.position_resolution_m` DB column (migration-safe `ALTER TABLE`). Frontend (`frontend/src/App.jsx`):
  * `formatVesselDistance()` shows `"< 1.6 กม. (ระยะกริด AIS)"` instead of a falsely precise `"0.0 กม."` whenever the real distance is smaller than the real grid-cell size — the underlying number is never fabricated, just not over-presented as more precise than it is.
  * `spreadColocatedVessels()` spreads markers that share an identical reported position into a small ring around that shared point purely for on-screen visibility (real lat/lon/distance used everywhere else — list, tooltip — are unchanged), with the tooltip disclosing "ตำแหน่งโดยประมาณ — N ลำ ใช้กริด AIS เดียวกัน" (approximate position, N vessels sharing this AIS grid cell) whenever it applies. Spread radius tuned to `0.15 × position_resolution_m`: an earlier `0.35` value was found, by re-inspecting a real capture, to let two *adjacent* grid cells' spread rings reach far enough to overlap and hide a marker again — 0.15 keeps every group's ring comfortably inside its own cell.
* **Verified via a real live-fetch call**: same Singapore Strait bbox, real response showed 5 vessels (Marina Genesis, Cast Lion, Torm Dover, Ayu 28, TB Mega Daya 35), all correctly deduplicated by MMSI, `position_resolution_m` populated (~1572m). Screenshot of the sidebar vessel list (`docs/deck_assets/live_fallback/04_vessel_attribution_panel.png`) shows all 5 with the honest `"< 1.6 กม. (ระยะกริด AIS)"` label; screenshot of the map (`03_map_and_gfw_attribution.png`) shows 5 distinct, individually-visible markers (confirmed by cropping and re-inspecting the marker cluster region directly, not just glancing at the full frame) where Round 14 showed 2.

### 13. Land-Sea Mask Resolution Gap Fixed (Real OSM Island Refinement) + Expanded Live-Fallback Screenshots
* **Status**: Resolved.
* **The bug**: a live-fetch screenshot taken after Resolution #11 still showed the oil-slick mask covering Pulau Semakau (a real Singapore island) despite the land-sea mask being in place.
* **Root cause, measured directly**: `global_land_mask`'s raster (Resolution #11) is a fixed ~930m-spaced global grid. A fine test grid sampled over Semakau's own real bounding box (Nominatim-confirmed: `1.1890651–1.2172813N, 103.7577792–103.7813209E`) found only **32.8%** of sample points reading as land — real island area genuinely falls between grid points at this resolution, small islands and narrow coastal features being hit hardest.
* **Tried the task's suggested fix first, measured it, rejected it with evidence**: swapped in Natural Earth's 10m land + minor-islands vector polygons (downloaded, clipped to the real Singapore Strait scene bbox with `shapely.clip_by_rect`). Result: Natural Earth's layers together account for only **7.62%** of the oil pixels the raster method already flags as land (vs. the raster's own 27.10%), and **neither Natural Earth layer contains Pulau Semakau at all**. "10m" in Natural Earth's name is a cartographic map scale (1:10,000,000), not a ground-resolution guarantee — its generalization is coarser than useful at a single ~20km SAR scene's scale. Vector-via-Natural-Earth would have been a regression, not a fix; this was measured before being ruled out, not assumed.
* **Real fix**: new `src/analysis/osm_coastline.py` queries the live OSM Overpass API for real `natural=coastline` ways scoped to the scene's own bbox (a small per-request query, not a bulk download), reconstructs **closed rings** (islands) from the raw way segments via endpoint-matching (verified against Semakau specifically: 11 separate raw way segments correctly assemble into one real 361-point closed ring matching its known real bounding box), and rasterizes them at the scene's own full pixel resolution (not a fixed global grid) using `cv2.fillPoly`'s even-odd rule. `src/analysis/landmask.py::strip_land_pixels()` now unions this with the existing raster classification — real OSM data refines islands the raster handles poorly; the raster remains the fallback for continental coastline (deliberately not attempted here — see the module's docstring for why bbox-edge ring-closing was ruled out) and for whenever OSM/Overpass is unreachable (never fails the request over a missing refinement; verified this fallback path actually engaged for real during testing when Overpass rate-limited a rapid run of test calls with HTTP 429/504).
* **Found and fixed two real bugs along the way, not assumed away**:
  1. Overpass returns HTTP 406 to `requests`' default User-Agent — fixed by sending a descriptive one, per Overpass/OSM API etiquette (found live, not from docs).
  2. GFW's presence API returns `null` (not `[]`) for a region/date with **zero** vessel presence — `gfw_client.py` previously crashed on this (`TypeError: 'NoneType' object is not iterable`, a real HTTP 500) until hit by a real query over a genuinely quiet remote location (see below). Fixed to treat `null` as empty.
* **Re-measured on the same Singapore Strait repro case, real numbers**: raw oil px unchanged at 2,641,623; land-pixels-removed went from **715,798 (27.10%)** (Resolution #11 baseline) to **781,193 (29.57%)**, of which **65,395 px** are attributable specifically to the new OSM island refinement (i.e. pixels the raster alone still missed). Visually confirmed in a real in-browser capture: Pulau Semakau now renders in its real OSM land color, fully excluded from the mask (`docs/deck_assets/live_fallback/06_singapore_landmask_fixed_map.png`, vs. `03_map_and_gfw_attribution.png` before). Separately confirmed the continental HarbourFront/Alexandra mainland coast is materially unchanged by this fix (67,959 → 67,904 residual oil px in that crop, a 0.08% difference) — expected, since that fix is deliberately scoped to closed island rings; that stretch of coast was already correctly handled by Resolution #11's raster mask.
* **Verified the fix generalizes, not just on the original repro case**: re-ran end-to-end against two more real, different live-fetch locations (see screenshot table below) — Stockholm Archipelago, Sweden (thousands of small skerries: 49.1% land removed, 85,063 px via OSM refinement — even more than Singapore) and the Chonos Archipelago, Chile (remote Patagonian fjords: 67.4% land removed). Both real Sentinel-1 products, real CDSE fetches, real OSM queries.
* **Cached demo scenes confirmed still unaffected**: re-ran the new raster+OSM hybrid directly against all 4 cached demo scenes (`oil_00000/1/3/4`) — `land_px_fraction_in_scene = 0.0` and `oil_px_removed_on_land = 0` for all four, same as Resolution #11's check. (Two of the four scenes' Overpass calls actually hit rate-limit errors — HTTP 504/429 — during this verification, real confirmation that the "OSM unreachable → fall back to raster, never fail the request" path works, not just a theoretical claim.)
* **Known scope limitation, documented in `osm_coastline.py`**: an island's OSM outer coastline ring is rasterized as fully land, including any genuinely separate inner water body (e.g. a lagoon) unless that inner shore is itself separately tagged `natural=coastline` (most enclosed reservoirs are tagged `natural=water` instead, not queried here). For Pulau Semakau specifically this means its internal reservoir cells are over-masked as land too — a minor conservative simplification, not a regression from the pre-fix state (which covered the *entire* island, reservoir included, as "oil").
* **Expanded `docs/deck_assets/live_fallback/` screenshot set** (`05`-`10`, originals `01`-`04` untouched): the original set was all one Singapore Strait bbox reused across debugging rounds. Added, in one real run of `scripts/capture_live_variety_screenshots.js`: the fixed Singapore case (`05`/`06`), a genuinely different real location with a different coastline profile (Stockholm Archipelago, `07`/`08`), a real honest few/no-vessels result (Chonos Archipelago returned **zero** AIS vessels for its real date — kept as the genuine result, not forced or swapped for a busier location; finding this is what surfaced the GFW `null`-handling bug above), and a labeled standalone close-up of a real 5-vessel marker cluster proving Resolution #12's marker-spread fix stays legible outside a full-map screenshot (`10`). Full per-file description: `docs/deck_assets/MANIFEST.md`.

### 14. Fixed-Weight Slick Stroke Misread as an Unlabeled Layer on Fragmented Scenes
* **Status**: Resolved.
* **The bug**: a report of a yellow/orange filled area on the live-fetch map that looked like a second, unlabeled layer grouped under "AIS Overlay." Traced first: the map only ever renders 4 shape types (`frontend/src/App.jsx`), and the yellow/orange filled polygon is unambiguously the post-mask oil-slick contour (`selectedDet.geojson_mask.coordinates`, already through `strip_land_pixels()` server-side) — it already had its own `showSlick`-gated, separately-labeled `คราบน้ำมัน (Oil Slick Mask)` legend row, unchanged since the very first scaffold commit. No mislabeling or state-sharing bug with AIS existed in the code.
* **Real root cause, found on the second pass**: land-sea masking (Resolution #11/#13) carves land out of the raw prediction, which shatters a scene into many small disjoint water-region fragments — confirmed with real counts, not estimated: cached demo scenes range 9-228 fragments, but live-fetch scenes (which always carry land-mask fragmentation) range **138-692** — Singapore itself sits at 492, not the "single-digit" originally guessed. At the fixed 3px stroke weight every fragment was drawn with regardless of count, a dense cluster of small fragments (Stockholm's 692, packed into a fraction of the scene) reads as a solid yellow mass at normal zoom, easily mistaken for a second layer.
* **Considered dissolving fragments (`shapely.unary_union`) first, tested it, rejected it with evidence**: ran it against a real 692-fragment Stockholm result — count went from 692 to **726** (up, not down), because these are genuinely disjoint real water regions separated by land with nothing overlapping to merge; a few inputs needed `buffer(0)` cleanup which split rather than joined. Dissolving cannot fix a "too many real separate shapes" problem, only a "spurious overlapping duplicates" one, which this isn't.
* **Fix**: `frontend/src/App.jsx`'s new `getSlickStrokeWeight(nFragments)` scales the contour's stroke width down as fragment count grows (`weight: 3` at ≤50 fragments down to `weight: 1` above 400), computed per-detection from the real `geojson_mask.coordinates.length` already in the API response — no backend change, purely a rendering-density fix. Verified visually: re-captured the same real Stockholm live-fetch (`docs/deck_assets/live_fallback/08_stockholm_archipelago_map.png`) — individual fragment boundaries are now legible as thin outlines around visible purple fill instead of a fused yellow web. Confirmed no regression on simple scenes: `oil_00000` (29 fragments) still renders at the original bold `weight: 3`.

### 15. Fragment Boundary Accuracy Audit (Round 17) — Pass, No Fix Needed
* **Status**: Audited, no defect found. Sub-pixel deviation confirmed with real measurements, not assumed from the pipeline design.
* **The question**: Rounds 15/16 confirmed the land-sea mask correctly *excludes* land (measured pixel-removal rates) and that whole islands are no longer covered. Left open: do the individual polygon fragment *boundaries* actually trace the real coastline accurately, or just approximately (e.g. following the coarser ~1km `global_land_mask` raster grid instead of the finer OSM vector data where the two overlap)?
* **Method**: re-fetched real OSM `natural=coastline` data live from Overpass for the exact Stockholm Archipelago and Chonos Archipelago bboxes used in Round 16's `capture_live_variety_screenshots.js` test (same code path: `assemble_closed_rings`), picked 3 real, isolated, small-to-medium closed-ring islands per location (spans 380m–1271m, 126–989 real vector points each — not toy shapes), rasterized each with the exact same function and pixel scale (`rasterize_closed_rings`, `PIXEL_SCALE_DEG = 8.983152841195218e-05`, the same GSD `src/data/cdse_fetch.py` requests for every real live fetch) the production pipeline uses, extracted the rasterized boundary via `cv2.findContours`, and measured each boundary pixel's real-meter distance back to the original (un-rasterized) OSM vector ring it came from. Reproducible script: `scripts/audit_fragment_boundary_accuracy.py`; raw Overpass responses cached at `output/round17_{stockholm,chonos}_coastline.json`.
* **Real numbers, 6 islands across 2 real, geographically opposite locations** (Baltic Sea, ~59.3N vs. Patagonian fjords, ~45.3S — different hemisphere, different coastline character, ruling out a location-specific artifact):

  | Location | Island span | Vector pts | Mean deviation | Max deviation | Pixel scale (lat × lon) |
  |---|---:|---:|---:|---:|---|
  | Stockholm | 1271 m | 989 | 3.67 m | 10.47 m | 10.00 m × 5.12 m |
  | Stockholm | 502 m | 657 | 4.09 m | 11.20 m | 10.00 m × 5.08 m |
  | Stockholm | 722 m | 593 | 3.28 m | 9.59 m | 10.00 m × 5.12 m |
  | Chonos | 920 m | 422 | 4.40 m | 11.16 m | 10.00 m × 7.04 m |
  | Chonos | 380 m | 135 | 4.26 m | 11.44 m | 10.00 m × 7.03 m |
  | Chonos | 570 m | 126 | 5.11 m | 11.23 m | 10.00 m × 7.04 m |

  Mean deviation is consistently 3.3–5.1 m; max deviation across all six real islands never exceeds ~11.4 m — i.e. never more than one pixel diagonal (~11.2–12.2 m at these two latitudes). Also checked for a systematic directional bias from the `int32` truncation in `rasterize_closed_rings`' pixel-coordinate cast (a plausible source of a consistent, one-sided offset rather than random noise) — found none; the fractional-pixel truncation offset averages close to unbiased, not skewed toward zero in one direction.
* **Visual confirmation**: `output/round17_boundary_overlay_idx897.png` (whole-island overlay) and `output/round17_boundary_overlay_idx897_zoom.png` (tight ~40×40px / ~400m×200m crop) plot the rasterized fragment boundary directly against the real OSM vector line for one Stockholm island. The rasterized boundary hugs the vector line everywhere, including through narrow inlets and headlands — the visible pattern is the expected pixel-grid "staircase" quantization of a smooth line, not an offset, gap, or overshoot.
* **Conclusion**: **Pass.** Fragment boundary tracing is accurate to within one pixel (~5–11 m) of the real coastline for OSM-refined closed-ring islands — which, per Round 14's fragment-count diagnosis, make up the large majority of small fragments in dense archipelago scenes (Stockholm's 692, Chonos's 138). This is already at the model's own native ~10m GSD; there is no coarser-grid artifact to fix, and no visible-at-normal-zoom deviation. **No fix attempted or needed**, per this round's explicit scope (a sub-pixel discrepancy isn't worth the effort or risk of touching working masking code).
* **Known, already-documented scope boundary this audit did not re-test**: mainland/continental coastline fragments (open, non-closed OSM paths) still rely solely on the coarser ~1km `global_land_mask` raster by Round 16's deliberate design (bbox-edge ring-closing was tested and rejected then, not reattempted here) — this audit's pixel-level accuracy finding applies specifically to the OSM-refined closed-ring island case, which is what the task asked about and what dominates fragment count in archipelago scenes.

---

## Demo Lineup — 4 Oil-Detection Scenes Only (Updated in the presentation-prep sessions after Session 5)

The live demo dropped from 5 scenes to 4, oil-detection only. `no_oil_00004` was
in the lineup as the "clean water, no oil" example, but a later round fixing
the map's fixed-zoom framing (it made every detected slick icon-sized
regardless of real area) also exposed that `no_oil_00004`'s real coordinates
are inland, not water. Investigating every `no_oil` holdout scene as a
replacement found a harder problem: across all 10 `no_oil` scenes,
**genuinely open water and the live v2 checkpoint correctly outputting ~0%
confidence never co-occur** — the two confirmed-water scenes (`no_oil_00000`,
`no_oil_00001`) both false-positive live (~67.8%, ~66.6%), every scene the
model correctly suppresses is on land, and one file (`no_oil_00005`) is a
broken all-zero array. Full investigation, including how each scene was
checked (raw SAR content + reverse geocoding): `docs/deck_assets/MANIFEST.md`.
Rather than show a land scene mislabeled as water, or a live false positive,
the demo is now oil-detection scenes only:

| Scene | Confidence | Polygons | Real coordinates |
|---|---|---|---|
| oil_00000 | 97.5969% | 29 | 35.5307N, 34.7851E |
| oil_00001 | 90.9043% | 27 | 35.4087N, 34.9938E |
| oil_00003 | 96.3351% | 9 | 20.1616N, 38.2182E |
| oil_00004 | 95.2886% | 228 | 35.8224N, 35.0308E |

Values match the last-confirmed numbers exactly (zero regression), and
coordinates are real and vary correctly per scene (confirmed independently
against `docs/holdout_scene_coordinates.json`). Confirmed in-browser:
switching scenes re-centers/re-zooms the Leaflet map to fit the real detected
polygon's bounding box (not a fixed zoom) and re-renders overlay geometry,
not just the API response.

**Also fixed since this table was first written**: the "ขนาดคราบ" (slick
size) badge described as a hardcoded fake `8.5`/`14.5` km² value in Recent
Resolution #8 and Open Issue #2 below is no longer accurate — it's since been
wired to a real shoelace-formula area computed from each detection's own
GeoJSON polygon (same lat-corrected km/deg method used elsewhere in this
repo). Left the original entries below as-is rather than rewriting session
history; treat this note as the current source of truth on that specific
item.

---

## Training Logs Comparison (Val IoU Stability)

* **Val IoU Peaks**: v2 achieved a higher peak Validation IoU (**`0.7424`** at epoch 11) vs. v1 (**`0.6965`** at epoch 12).
* **Metric Stability**: v1's Validation IoU fluctuated wildly epoch-to-epoch (e.g. `0.5010` → `0.6965` → `0.6439` in epochs 11-13); v2's stabilized above `0.70` from epoch 6 onward.

| Epoch | v1 Val Loss | v1 Val IoU | v2 Val Loss | v2 Val IoU | v2 Val Dice |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.725177 | 0.422358 | 0.506773 | 0.641398 | 0.781526 |
| 2 | 1.124958 | 0.274703 | 0.926747 | 0.267861 | 0.422540 |
| 3 | 0.497706 | 0.585735 | 0.428006 | 0.690064 | 0.816613 |
| 4 | 1.122395 | 0.606193 | 0.393626 | 0.711528 | 0.831454 |
| 5 | 0.396908 | 0.673934 | 0.509283 | 0.686476 | 0.814095 |
| 6 | 0.556104 | 0.496347 | 0.367070 | **`0.727504`** | 0.842260 |
| 7 | 0.529022 | 0.534189 | 0.405825 | **`0.706110`** | 0.827743 |
| 8 | 0.479961 | 0.638172 | 0.398910 | **`0.712574`** | 0.832167 |
| 9 | 0.646530 | 0.515526 | 0.351514 | **`0.721042`** | 0.837913 |
| 10 | 0.466078 | 0.674395 | 0.350150 | **`0.733229`** | 0.846084 |
| 11 | 0.697020 | 0.501025 | 0.329764 | **`0.742467`** | 0.852202 |
| 12 | 0.372412 | 0.696502 | 0.347648 | **`0.739894`** | 0.850505 |
| 13 | 0.478718 | 0.643939 | 0.352208 | **`0.739612`** | 0.850318 |
| 14 | 0.403184 | 0.669000 | 0.351726 | **`0.740577`** | 0.850956 |
| 15 | 0.550864 | 0.670618 | 0.346164 | **`0.741993`** | 0.851890 |

---

## Held-Out Test Set Comparison (v1 vs v2, 30 scenes)

| Category | Version | IoU | Dice | Avg FP pixels/scene | Avg FP % of scene | Avg FP area (km², real GSD) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Oil | v1 | 0.7263 | 0.8044 | 99,575.4 | 2.374% | 8.948 |
| Oil | v2 | 0.7239 | 0.8039 | 89,587.6 | 2.136% | 8.113 |
| No Oil | v1 | 0.7000 | 0.7000 | 300.4 | 0.007% | 0.024 |
| No Oil | v2 | 0.7000 | 0.7000 | 64.8 | 0.002% | 0.005 |
| Lookalike | v1 | 0.0000 | 0.0000 | 1,733,578.4 | 41.332% | 148.354 |
| Lookalike | v2 | 0.0000 | 0.0000 | 1,571,115.3 | 37.458% | 134.461 |

Full source data: `docs/checkpoint_comparison_summary.{json,csv}`, `docs/holdout_per_scene_results.{json,csv}`.

### Confusion matrix (outcome buckets — identical between v1 and v2)

| Category | Correct | Partial | False Positive | False Negative |
|---|:---:|:---:|:---:|:---:|
| Oil (n=10) | 8 | 2 | 0 | 0 |
| No Oil (n=10) | 7 | – | 3 | – |
| Lookalike (n=10) | 0 | – | 10 | – |

Full source: `docs/confusion_matrix_breakdown.{json,csv}`.

### Lookalike Analysis — corrected framing
* **Result**: Lookalike IoU/Dice are exactly `0.0000` across **all 10/10 scenes in both v1 and v2** — this is universal, not "some scenes," and every one of the 10 has at least one false-positive pixel.
* **The real per-scene picture is more nuanced than "still fails everywhere"**: false-positive area ranges from 0.02% to **81.4%** of scene area across the 10 scenes (worse at the tail than an earlier ~70-80% estimate), highly bimodal — 4/10 scenes are catastrophic (45-81%), 2/10 are near-clean (<3%).
* **v1→v2 did produce a real, if modest, improvement**: average false-positive area dropped from 41.3% to 37.5% of scene area (148.4 → 134.5 km² using real, latitude-corrected GSD) — genuine, but far more modest than an earlier "10-45% per-scene" framing implied (that range described the spread across scenes, not the typical improvement).
* **Region matters as much as category** (new finding this round, previously not visible anywhere in the results): pass rate under v2 varies sharply by region, not just by oil/no_oil/lookalike:

  | Region | v2 pass rate |
  |---|:---:|
  | Eastern Mediterranean (bulk of the set) | 11/15 |
  | Sunda Strait | 3/3 |
  | Red Sea | 1/7 |
  | Gulf of Mexico | 0/2 |
  | Western Mediterranean | 0/2 |
  | Baltic Sea | 0/1 |

  Full source: `docs/region_coverage.{json,csv}`. Small per-region sample sizes (n=1-2 for three of the six) mean the 0/2, 0/2, 0/1 results should be read as suggestive, not conclusive on their own — but Eastern Med vs. Red Sea (15 and 7 scenes respectively, both reasonably sized) is a real, sizeable gap worth investigating.
* **Implication unchanged**: simple background-patch oversampling alone is insufficient to separate lookalikes from real oil given identical SAR backscatter-reduction signatures. Distinguishing them needs either structural context (geometry, e.g. length-width ratio) or external metadata (wind speed, multi-temporal shape stability, optical imagery) — see the ERA5/multi-temporal/DSen2-CR investigations below.
* **Isolating the oversampling change's specific effect** (with vs. without, same checkpoint epoch, same eval path) requires a new ablation training run — **blocked pending explicit user go-ahead**, not started.

---

## New Capability Investigation: ERA5 Wind Cross-Check & Multi-Temporal SAR Comparison

* **Status**: Blocked on missing data, code written and ready pending real inputs.
* **Goal**: reduce the lookalike false-positive problem by post-processing detections against (1) ERA5 10m wind speed at acquisition time — oil slicks only show a dark SAR signature within ~1.5-6 m/s — and (2) whether a same-track Sentinel-1 pass exists nearby in time, to check if the detected shape drifted (real oil) or stayed static (lookalike-likely).
* **No acquisition timestamp exists anywhere for this dataset**, confirmed by: (1) no `DateTime` tag or any date-bearing tag in the GeoTIFFs themselves; (2) querying the Zenodo API directly for all three record IDs (8346860 / 8253899 / 13761290) — each deposit is exhaustively just the image `.7z` + mask `.7z` archives, no metadata/sidecar file. This is a property of the original Zenodo deposit, not something a Kaggle mirror stripped out. Without a timestamp there's no way to pick an ERA5 timestep or anchor a "nearby pass" search for any of these 30 scenes as currently downloaded.
* **Sentinel-1 revisit cadence verified current, not assumed**: the 2026 constellation reconfiguration (Sentinel-1C + Sentinel-1D) completed 2026-06-24, restoring a ~6-day nominal same-track revisit — not the ~12-day figure from the prior single-satellite gap.
* **Real, credential-free finding**: Copernicus Data Space Ecosystem's catalog *search* API needs no auth (only downloads do). A real 12-month Sentinel-1 GRD coverage-density check against all 30 scenes' actual coordinates (`docs/sentinel1_coverage_density.json`) confirms real regional variance (57-724 passes/year, mean 274) — every scene has substantial nearby archive density, so the multi-temporal feature has real potential once timestamps exist.
* **Built, not yet run**: `src/analysis/era5_wind_check.py` and `src/analysis/sentinel1_revisit_check.py` — real fetch/classify/compare logic against the documented CDS and CDSE APIs. Standalone scripts only, not wired into `/api/predict` or the DB schema.
* **Credential prep done**: `.env.example` documents the exact env vars each service needs and where to get them; `scripts/validate_credentials.py` does a real minimal auth check against both CDS and CDSE (no data download) — verified its skip-path against this repo's current unconfigured environment.
* **What's still needed**: a free CDS API key, free CDSE OAuth2 client credentials, and — the real blocker — per-scene acquisition timestamps from a source outside this repo. Being chased separately (contacted the companion paper's corresponding author, DOI `10.1016/j.marpolbul.2024.116549`, awaiting reply as of this writing). Not a repo problem to keep digging at further.

---

## New Capability Investigation: SAR-Optical Cloud-Removal Fusion (DSen2-CR)

* **Status**: Scoped, not built. Nothing downloaded/kept, nothing run.
* **Goal**: a third lookalike-discrimination signal — reconstruct a cloud-free Sentinel-2 optical view of each scene (DSen2-CR, Meraner et al. ISPRS 2020 Best Paper, github.com/ameraner/dsen2-cr, pretrained) as a second modality, since lookalikes and real oil can be SAR-identical but visually different in optical color/texture.
* **Feasibility findings from reading the actual code** (not the README's word for it):
  * The maintainer's own README admits they can't currently replicate/test the code themselves.
  * Confirmed exact input spec: 13-band Sentinel-2 L1C + 2-band Sentinel-1, 128x128 crops, channels-first. SAR clipping is band-specific/asymmetric (`[-25,0]` VV, `[-32.5,0]` VH dB) — differs from this project's own uniform `[-25,0]` clip, a real adaptation point. Cloud mask is derived automatically, not a required separate input.
  * All 4 Google Drive pretrained checkpoints verified genuinely downloadable (real HDF5 signatures, correct headers, ~72.4MB each — matches a ~19M-parameter architecture) — not just that the link returns HTTP 200, which alone is not reliable (Drive returns 200 for restricted links too). Downloaded then immediately deleted, not kept.
  * License confirmed GPLv3 by reading the actual file — usable for a school project.
  * The README's "PyTorch implementation" is incomplete: imports `base_model.BaseModel`, which doesn't exist anywhere in the repo. No PyTorch-format pretrained weights are published either — pretrained inference is locked to the old TensorFlow 1.15/Keras 2.2.4/Python 3.7 stack (confirmed still pip-installable but Python-3.7-only), which needs its own isolated environment (can't coexist with this project's own venv).
  * GPU requirement for inference genuinely unverified — flagged as an honest estimate (grounded in parameter count vs. this project's own U-Net's known CPU runtime), not a guess dressed up as confirmed.
  * Confirmed Sentinel-2 access needs no credentials beyond what's already prepped for Sentinel-1 (same CDSE account) — validated with a real search against this project's actual scene coordinates.
* **My recommendation** (see `docs/cloud_removal_scoping.md` for full reasoning): integrate as a **post-hoc lookalike-discrimination stage on top of the existing v2 U-Net**, not retrained input channels — because the timestamp problem scales completely differently between the two (30 holdout scenes vs. ~2,570 training scenes if baked into retraining).
* **Time estimate once timestamps + credentials exist**: roughly 1-2 weeks, with most uncertainty in the discrimination-logic step, not the plumbing (~3-5 days, moderate confidence).
* Design sketch: `src/analysis/cloud_removal_fusion.py`. Real, tested fetch/preprocessing logic where verifiable; the actual DSen2-CR inference call is a deliberate `NotImplementedError` stub (subprocess boundary into a separate environment), not a fake pass-through.

---

## Open Issues

### 1. `run_full_preprocessing()` has no dB/linear-scale branching — found this session
* **Status**: Open, not fixed. Found during this session's health-check test run, not previously documented anywhere.
* `src/data/preprocess.py::run_full_preprocessing()` (used by `get_dataloaders()` — the synthetic-only training path — and by `get_real_dataloaders()`'s verification-fallback branch) unconditionally assumes dB-scale input, unlike `main.py`'s `/api/predict` (Recent Resolution #3), which correctly branches on `np.any(image_raw < 0)`.
* **Confirmed with a direct test**: raw synthetic data from `generate_synthetic.py` is small positive linear values (0.0001-0.30). Running it through `run_full_preprocessing()` produces a preprocessed patch value range of **exactly `[1.0, 1.0]`** — completely saturated, zero information content.
* **Does not affect the real v1/v2 checkpoints** — those were trained exclusively via `train_kaggle.py` on real dB-scale Kaggle data, where the "always assume dB" logic is correct by design (Kaggle never has synthetic fallback data, per the hard-fail guard).
* **Does undermine three verification scripts' actual purpose**, even though all three currently report "success":
  * `verify_pipeline.py` — passes its own shape/range assertions (`[0,1]` bound-check) while actually feeding degenerate flat-`1.0` patches through the pipeline. The assertions check bounds, not information content, so this was never caught.
  * `src/verify_training.py` — the resulting 2-epoch training run collapses to all-zero predictions. The script's own comment attributes this to "common for short runs on small synthetic sets due to random initialization" — given the confirmed flat-`1.0` input, the more likely real cause is that the training data has no signal to learn from at all, not the epoch count.
  * `verify_real_pipeline.py` — same saturated `[1.0, 1.0]` image range observed when run in verification-fallback mode (its normal mode locally, since real images aren't downloaded here).
* **Not fixed as part of this session** — this is a report-only health check per this round's scope. Worth a small, well-scoped fix in a future round (mirror `main.py`'s branching into `run_full_preprocessing()`).

### 2. Frontend slick-size badge is hardcoded fake data
* **Status**: Open, flagged not fixed (see Recent Resolution #8).
* `frontend/src/App.jsx`'s "ขนาดคราบ" (slick size) badge shows a hardcoded `8.5`/`14.5` km² for every single detection regardless of the real geojson-derived area. Not wired to any real data at any point. Low priority relative to other open items, but worth knowing before presenting the demo — a sharp-eyed viewer could notice every detection shows the same area.

### 3. Issue 5 — Lookalike oversampling ablation
* **Status**: Blocked, pending explicit user go-ahead. Not started.
* Isolating the lookalike-oversampling change's real effect (with vs. without, same checkpoint epoch, same eval path) requires a new training run — v1 vs. v2 differ in three things simultaneously (LR scheduler, pooled-IoU, oversampling rate), not a clean ablation. Budget the same order of GPU time as the original v2 training run.

### 4. Acquisition-timestamp gap (ERA5 / multi-temporal SAR / DSen2-CR)
* **Status**: Blocked on a human path outside this repo, being chased separately. See the two "New Capability Investigation" sections above for full detail — not repeated here to avoid the file drifting out of sync with itself again.
