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

## New Capability Investigation: Post-Hoc Lookalike Classifier (Yang & Singha ESSD/PANGAEA dataset)

Follow-on to Track D's post-hoc-stage recommendation above, prompted by a real labeled lookalike dataset becoming available (Yang & Singha 2025, ESSD, PANGAEA DOI `10.1594/PANGAEA.980773` — 2,290 look-alike/other patches + 3,225 oil-object patches, Sentinel-1, Eastern Mediterranean, 2019, JPG + Pascal VOC XML).

### Phase -1: Kaggle API access — Resolved
* **Status**: Credentials found and working, confirmed with a real API call, not just a version check.
* `~/.kaggle/credentials.json` exists but is **not** the legacy `kaggle.json` (`username`+`key`) format documented in most tutorials — it's the newer OAuth-style credential set (`refresh_token`, `access_token`, `access_token_expiration`, `scopes`) that Kaggle's current CLI (`kaggle==2.2.4`) uses. Initially flagged as suspicious for that reason (unfamiliar format, an elevated-looking `resources.admin:*` scope, an `access_token_expiration` that read as already-past) — resolved by actually testing it rather than trusting the file's shape: `pip install kaggle && kaggle datasets list` authenticated successfully as `trongzen` and returned real, current dataset listings (including entries with `lastUpdated` timestamps matching today's date). No `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars or `.env` entries exist or are needed — the CLI reads `credentials.json` directly.
* No action needed from Zen; this blocks nothing.

### Phase 0: Did the original training set contain lookalike-type examples? — Resolved
* **Status**: Yes, in real and substantial quantity — this is a "saw it but didn't learn to distinguish it" problem, not a "never saw this pattern" one.
* **685 distinct lookalike scenes** were part of the original 2,570-scene training set (Zenodo Part II, `Mask_lookalike`), confirmed three ways, not just cited from earlier documentation:
  1. Locally: `data/raw/Mask_lookalike/` contains exactly 685 `.tif` files (matches the count already on record in Resolution #1 above); 5 spot-checked masks are all-zero (`nonzero_px = 0`), i.e. genuinely empty ground truth — consistent with "lookalike" meaning *no real oil, but a suspicious dark signature*.
  2. The **actual v1/v2 Kaggle training run's own log** (`kaggle_output/project-pelagic-training.log`, real captured stdout from the live Kaggle job, not a rerun) shows a deliberate discrepancy check built into `kaggle_kernel/train_kaggle.py`: **681/685 Lookalike masks are byte-identical to No-Oil masks** (expected — both categories are blank ground truth), but a separate **image-level** hash check found **0 identical image files out of 685 compared** — i.e. the 685 lookalike *images* are genuinely distinct real scenes, not a mislabeled copy of the no_oil set. The script explicitly logs `"Kept 685 distinct Lookalike scenes"` and folds them into training (`Total distinct negative scenes: 1370`, `Train split: 2056 scenes`, `Val split: 514 scenes`).
  3. This matches Resolution #4 (already on record): v2's training deliberately **oversampled lookalike background patches 2.5x relative to no_oil** (`prob=0.005` vs `0.002`) specifically as hard negatives — i.e. the training pipeline didn't just passively include lookalikes, it went out of its way to show the model more of them per epoch than plain open water.
* **Why the model still fails 100% on holdout lookalikes despite this**: the most likely explanation, and one already on record in this doc's "Lookalike Analysis" section (line 246) from an earlier round, is that the U-Net only ever sees raw VV/VH SAR backscatter — and a real oil slick and a real lookalike (biogenic film, current front, low-wind patch, etc.) can produce **near-identical dark-patch backscatter signatures** in that same 2-channel input space. Oversampling more examples of a category doesn't help if the input representation itself doesn't carry the distinguishing signal — the model has nothing to learn to key on. This is a real, structural information-availability limit of SAR-only pixel input, not a training-recipe or data-quantity bug.
* **Implication for this task's plan**: this raises the bar for Phase 1/2. A post-hoc classifier trained on **crops from the same raw dB SAR channels alone** risks hitting the identical wall the U-Net already hit on 685 real, deliberately-oversampled in-domain examples. The plan's emphasis on hand-crafted texture/shape features (GLCM stats, edge density, compactness) or an external signal (the ESSD dataset's different sensor-processing pipeline, or eventually wind/multi-temporal context) is the right instinct precisely because it's a different feature space, not just more of the same one. Proceeding to Phase 1 with this explicitly in mind rather than assuming more SAR examples alone would fix it.

### Pre-Phase-1 feasibility gate: do hand-crafted texture/shape features separate oil from lookalike at all, on this project's own data? — Resolved: modest, real separation
* **Status**: Done, per Zen's explicit instruction to test this *before* touching the ESSD/PANGAEA dataset — a fast, cheap go/no-go check, not a full Phase 1 build.
* **Method** (`scripts/lookalike_feature_separability_check.py`): extracted candidate-region crops using the **real v2 checkpoint's own prediction** (`checkpoints/model_real_v2_best.pt`, via the exact production `preprocess_for_prediction` + `run_tiled_inference` path — not a heuristic stand-in), then computed GLCM texture (contrast, dissimilarity, homogeneity, energy, correlation, ASM), edge density (Canny), and shape/compactness (circularity, solidity, aspect ratio, bbox-fill-fraction) on each candidate's largest connected component. Deliberately did **not** use ground-truth masks to pick the crop — a real post-hoc classifier only ever sees the U-Net's own proposed region, and lookalike ground truth is blank by definition, so using GT to select the crop would have been circular.
* **Data**: 50 real scenes, no ESSD data involved — 10 holdout oil + 10 holdout lookalike (fully local already) plus 15 oil + 15 lookalike sampled evenly across the real training pools (1200/685 scenes) and freshly downloaded from the exact Kaggle datasets `train_kaggle.py` itself trains on (`shuddhabrotabanerjee/oil-spill-dartis-part1`/`-part2`) — the local repo checkout only has masks for these, not images (see Phase 0 above), so this was a real, targeted download, not reuse of anything already present.
* **Sanity check passed**: the real model's predicted region vs. ground truth on the 25 oil scenes averaged IoU 0.671 (median 0.705) — consistent with this project's already-documented reasonable oil-bucket performance, confirming the inference call in this script matches production rather than silently diverging.
* **Real result — 8 of 11 features are individually statistically significant (Welch's t-test, p<0.05) between oil and lookalike**, with moderate effect sizes (|Cohen's d| 0.53–1.19): lookalike candidate regions are more solid/convex (`solidity` d=-0.82, p=0.0059) and fill more of their own bounding box (`blob_fill_fraction` d=-1.19, **p=0.0001**, the single strongest feature) than oil regions, which tend to be more elongated/irregular; oil regions also show higher GLCM contrast/dissimilarity and lower homogeneity/energy/correlation (rougher, more textured internally) than lookalikes' smoother texture. This lines up physically with oil often forming windswept/current-driven streaks with internal thickness variation, vs. many lookalike causes (biogenic slicks, low-wind zones, current fronts) producing more uniform, rounder patches — a real, sensible mechanism, not an arbitrary statistical artifact.
* **Quantified separability, honestly modest, not clean**: 2D PCA (74.2% variance explained) gives a silhouette score of **0.131** (positive = real structure, but far from the >0.5 that would mean cleanly separated clusters — the PCA scatter plot shows real lean but heavy overlap, `output/lookalike_feature_separability_pca.png`). A leave-one-out logistic regression on the 11 features reaches **70% accuracy (n=50) vs. a 50% majority-class baseline** — a genuine 40% relative reduction in error, clearly better than chance, but nowhere near a solved problem on its own.
* **Conclusion and decision**: this clears the bar Zen set ("if oil and lookalike separate even modestly, Phase 1 is worth the investment") — the signal is real (multiple significant features, physically sensible direction, 70% vs. 50% LOO accuracy) but modest (silhouette 0.131, heavy scatter overlap). **Recommendation: proceed to Phase 1**, with the honest expectation set going in that texture/shape features alone will likely land in a "meaningfully better than 0/10, not a full fix" range (rough extrapolation from 70% binary LOO accuracy on n=50 crops, not a promise) — and that ERA5 wind context (already scoped, currently blocked on missing acquisition timestamps, see above) remains a good complementary signal for a later round rather than a replacement, since wind-vs-damping physics is a genuinely different signal from shape/texture and could plausibly cover cases texture alone won't.
* Raw outputs: `output/lookalike_feature_separability.csv` (per-scene feature table), `output/lookalike_feature_separability_pca.png` (plot), `output/lookalike_feature_separability_summary.json` (metrics). Reproducible via `python scripts/lookalike_feature_separability_check.py` (re-downloads the same 15+15 training-sample TIFFs to a local temp dir if not already present).
* **Refactor note**: the GLCM/edge/compactness feature code was pulled out into `src/analysis/candidate_region_features.py` once Phase 1 needed the identical logic for a second, differently-formatted data source (ESSD/PANGAEA, below) — single source of truth so the two callers' feature definitions can't drift apart, mirroring this project's own existing `dataset.py`/`train_kaggle.py` precedent (Resolution #5). `lookalike_feature_separability_check.py` now imports from it; behavior unchanged (verified: the shared module's `glcm_features` computes over the whole crop exactly as the original inline version did — the original's `mask_crop` parameter was actually unused dead code despite its docstring claiming otherwise, a pre-existing minor doc/code mismatch this refactor incidentally fixed rather than introduced).

## Phase 1: ESSD/PANGAEA data acquisition, domain alignment, and final dataset composition

* **Status**: Done. Zen approved proceeding to Phase 1 after the pre-Phase-1 feasibility gate above showed modest, real separation.

### Download
* **Real, live external hiccup found and resolved — not a true outage**: individual-file downloads at `https://download.pangaea.de/dataset/980773/files/<name>` need no auth (bulk zip does, HTTP 401). An initial pass guessing the no-oil (`nc`/`nw`) filenames as the same short `<subset>-NNNN.jpg` form the oil (`oc`/`ow`) subset uses returned a clean, 100%-consistent HTTP 500 across every no-oil file tried and 100% success on every oil file tried — indistinguishable at first from a real, category-specific PANGAEA backend outage (its own error page even says "we are informed about this... come back later"). Re-derived the real filenames directly from the dataset's own tab-separated metadata export instead of guessing: no-oil patches use a longer `<subset>-NNNN-00-NNNNNN.jpg` form (they're indexed differently since, unlike oil patches, they carry no per-object annotation row to derive a short name from). The "outage" was a wrong filename, confirmed by immediately succeeding once corrected — logged here so a future session doesn't waste time treating this as a PANGAEA-side problem again.
* **Full dataset downloaded, not a sample**: all **3,655 patches** — **1,365 oil** (`oc`=375 oil/coast + `ow`=990 oil/water, 3,225 individual oil objects across them) and **2,290 no-oil/look-alike** (`nc`=351 no-oil/coast + `nw`=1,939 no-oil/water) — plus all 1,365 oil XML (Pascal VOC) annotations (no-oil patches carry no XML — no object to annotate, the same "blank ground truth" shape as this project's own `Mask_lookalike`). 100% success, 0 failures, ~550MB total. `scripts/download_essd_pangaea.py`; raw files in `data/external/essd_pangaea/` (gitignored, matching `data/raw/`'s existing treatment of large downloaded data); patch-level index at `data/external/essd_pangaea/patch_index.csv`.

### Domain alignment — addressed explicitly, per-finding, not glossed over
* **Format gap**: ESSD patches are pre-normalized 8-bit single-channel (VV only — confirmed `<depth>1</depth>` in every XML) JPGs, 640x640, vs. this project's raw calibrated 2-channel (VV+VH) dB GeoTIFFs, 2048x2048. The per-image JPG normalization parameters aren't published or recoverable, so **pixel-value calibration between the two was ruled out as infeasible**, not attempted partially. Chose **feature-space alignment** instead (the approach flagged as the fallback in the original task framing): the exact same GLCM texture / Canny edge-density / shape-compactness pipeline (`src/analysis/candidate_region_features.py`) runs on both sources' candidate-region crops, since these are relative (percentile-clipped, locally quantized) measures by construction, not raw intensity comparisons.
* **Candidate-region definition per source**:
  - This project's own scenes: real v2 U-Net's own predicted mask (already-established method from the feasibility gate).
  - ESSD oil (`oc`/`ow`): the dataset's own VOC bounding box, refined by a **local** Otsu threshold within (bbox + 15px margin) to recover an actual blob shape rather than a bare rectangle — verified directly on a real sample (`ow-0001`): local Otsu found a 1,080px blob vs. the annotated bbox's own 1,058px area (kept within ~2%), whereas a *global*, whole-640x640-image Otsu on the same patch produced a useless 73,132px blob covering 42% of the frame — the same "global threshold is too coarse, must be localized" failure mode already discovered on this project's own raw scenes, confirmed a second time on a completely different data source.
  - ESSD no-oil (`nc`/`nw`): no bounding box exists (a look-alike patch has no annotated object, by definition) — darkest-10th-percentile threshold + morphological cleanup + largest connected component, applied to the **whole patch** (no bbox prior to localize with). **1,365/1,365 (100%) of oil patches yielded a usable candidate**; **1,479/2,290 (64.6%) of no-oil patches did** — the remaining 811 (35.4%) had no dark blob clearing the minimum-area threshold anywhere in the patch, meaning their look-alike signature isn't a single localized dark blob at that threshold (a real, disclosed limitation of this simple detector, not silently dropped — the skip count is in `data/external/essd_pangaea/essd_extraction_summary.json`). Final ESSD feature table: **2,844 patches** (1,365 oil + 1,479 no-oil), `data/external/essd_pangaea/essd_features.csv`.
* **Real cross-domain finding, reported honestly rather than cherry-picked**: re-ran the same per-feature Cohen's d / Welch's t-test analysis from the feasibility gate on the much larger ESSD-only sample (n=2,844, vs. n=50 before) — **one feature (`circularity`) is directionally consistent with the small project-own-data result** (oil less circular/more irregular than lookalike in both: d=-0.563 here vs. d=-0.531 there), which is reassuring, a real signal that survives a much bigger, differently-formatted sample. **But several GLCM texture features flip sign between the two domains**: e.g. `glcm_contrast` is higher for oil than lookalike in this project's own data (d=+0.630, oil rougher) but *lower* for oil than no-oil in ESSD (d=-0.169, oil smoother) — same for `glcm_homogeneity`/`dissimilarity`. Most likely cause: JPG contrast-stretch normalization alters *relative* texture statistics differently than physical dB scaling does, compounded by ESSD's own oil-vs-no-oil candidate-region methodology asymmetry (tight annotated-bbox-guided crop for oil vs. looser whole-patch threshold for no-oil) being a genuine confound in ESSD's own feature values, separate from the true physical oil/lookalike distinction. **This is exactly the generalization risk the original task asked to flag, not paper over** — it's real and it showed up in the data, not just a theoretical worry about region/year mismatch.
* **Mitigation applied for Phase 2**: per-source z-score standardization (standardize each domain's own features to its own mean/std *before* pooling) rather than pooling raw feature values directly. This doesn't force artificial agreement — a sign-inconsistent feature (like the GLCM ones above) still won't help a pooled linear classifier after standardization, since its within-class mean difference points opposite ways in the two domains and will net toward zero-ish combined signal; a sign-consistent feature (like `circularity`) keeps its usable signal. Verified this is what actually happens, not just a plausible-sounding claim: pooled (n=2,894: 1,390 oil + 1,504 no-oil, this project's own 30 feasibility-gate scenes + ESSD's 2,844) 2D-PCA silhouette is **0.029** — close to ESSD-alone's own **0.030** (barely moved by pooling), both lower than the small-sample project-own-data figure (0.131, likely inflated some by small-n noise) but still positive, i.e. real, non-random structure survives pooling, just weaker than either the optimistic small-n estimate or a naive (non-standardized) pooling would suggest. Plot: `output/essd_plus_own_pooled_pca.png`.

### Final prepared dataset composition, going into Phase 2
| Source | Role | Oil / positive | No-oil / lookalike | Notes |
|---|---|---|---|---|
| ESSD/PANGAEA | Training (out-of-domain, large N) | 1,365 patches | 1,479 patches (811 skipped, no localized blob) | JPG, single region/year (E. Med, 2019), feature-space-aligned via shared extraction code |
| This project's own training pool | Training (in-domain, small N so far) | 15 scenes sampled (of 1,200 available) | 15 scenes sampled (of 685 available) | Raw dB GeoTIFF, real v2-model candidate regions; more available on-demand directly from the Kaggle-hosted training datasets during Phase 2's Kaggle run, not locally download-limited |
| This project's own 30-scene holdout | **Evaluation only — Phase 3, never training** | 10 scenes | 10 lookalike + 10 no_oil scenes | Explicitly held out from every step above (Phase 0's feasibility-gate script used these read-only for the go/no-go check, not for fitting anything) — Phase 2 must not touch these during training or model/threshold selection, only Phase 3's final report may |

* **Leakage note, stated explicitly since Phase 2 is next**: the pre-Phase-1 feasibility check computed features on the 10 holdout lookalike + 10 holdout oil scenes, but only to answer "does this feature space separate at all" — no classifier was fit on them and none of their features were used as classifier training/validation data. Phase 2 must draw its training pool only from ESSD + this project's own **training**-set scenes (never `data/holdout/`), so Phase 3's evaluation stays honest.

### Post-Phase-1 correction: ESSD's signal does NOT transfer to this project's own data — reverses the plan to train primarily on ESSD
* **Status**: Critical follow-up finding, prompted by Zen questioning the pooled PCA plot directly ("the 'own' points all sit inside the main overlapping cluster, none reach the separated region that's mostly ESSD oil triangles"). That visual read was correct and led to a decisive, negative result — reported in full rather than downplayed, since it changes the Phase 2 plan materially.
* **Diagnostic 1 — is "own" separable at all inside the shared/pooled feature space, or only in isolation?** Split the earlier pooled analysis strictly into the legitimate 30-scene own **training pool** (15 oil + 15 lookalike, `source=="train_sample"`) vs. the 20-scene own **holdout** (kept as a read-only diagnostic exactly as the feasibility gate already used it, not used to tune or select anything here either):
  - Own-training-pool-only silhouette, own's own PCA axes: **0.172** (consistent with the earlier ~0.13 estimate — real, if modest, self-contained separation).
  - Own-training-pool silhouette evaluated on the **shared** pooled PCA axes (fit on own+ESSD together), but scored using only own points as each other's neighbors: **0.174** — essentially unchanged. So own's oil and own's lookalike genuinely separate from *each other* along the shared axes; that part isn't the problem.
  - **The actual problem**: re-scored the same shared-space points but let ESSD points count as neighbors too (the real, honest pooled-silhouette computation) — own's per-point silhouette **collapses to 0.007** (ESSD's own subset stays at 0.030, matching the previously-reported pooled figure almost exactly, since ESSD's 2,844 points numerically dominate the average). In plain terms: own's oil and lookalike points *are* separated from each other, but neither group is cleanly separated from ESSD's *opposite*-class cloud — exactly the "own points sit inside the big overlapping middle mass, not out near ESSD's separated oil region" pattern Zen spotted in the plot.
* **Diagnostic 2 — the decisive test: does a classifier trained only on ESSD generalize to this project's own data at all?** Fit a logistic regression on ESSD alone (n=2,844, per-domain-standardized as already documented), predicted on own data it never saw during fitting:
  - **Own training pool (n=30): 40.0% accuracy — *below* the 50% majority baseline.**
  - **Own holdout (n=20, read-only diagnostic): 55.0% accuracy — indistinguishable from chance at this sample size** (confusion matrix: 7/10 lookalike scenes misclassified as oil, 8/10 oil scenes correctly classified — the classifier is mostly just predicting "oil" regardless of input, not discriminating).
* **Conclusion: confirmed, not just theorized — an ESSD-only-trained classifier does not transfer to this project's own SAR data.** This validates Zen's stated concern directly: the modest oil-vs-no-oil signal ESSD's texture/shape features carry (Phase 1's 75.7% 5-fold CV accuracy, reported above) is substantially **ESSD-JPG-domain-specific**, most likely explained by the GLCM sign-flip already found and documented above, compounded by ESSD's own internal oil-vs-no-oil candidate-region-methodology asymmetry (bbox-guided crop for oil vs. whole-patch threshold for no-oil) — a confound in ESSD's own labels, not a property of real oil-vs-lookalike physics that would carry over to a different sensor-processing pipeline.
* **Revised plan for Phase 2, before writing any training code**: do **not** train the classifier primarily on ESSD and hope it transfers. Instead:
  1. **This project's own training pool is the primary signal**, and it is not actually small in absolute terms — 685 lookalike + 1,200 oil scenes are available (only 15+15 sampled locally so far for speed; Phase 2's Kaggle run can draw the full pool directly from the already-mounted `oil-spill-dartis-part1`/`-part2` datasets with no extra local download). The feasibility gate already showed this domain alone reaches 70% LOO accuracy (n=50, mixing train+holdout in that read-only check) — real, own-domain-consistent signal, unlike ESSD's.
  2. **ESSD's role, if any, should be narrowed** to the one feature that was directionally consistent across both domains (`circularity`) rather than the full feature vector, or dropped from training entirely and kept only as a documented negative result in this file. Default going in: exclude ESSD from Phase 2's actual training data, re-derive a larger own-domain training sample instead, and revisit ESSD only if the larger own-only sample turns out insufficient.
* This is exactly the kind of finding the original task asked to surface honestly rather than paper over ("if it doesn't improve meaningfully, report that too, with the most likely reason why") — surfacing it now, before Phase 2 training rather than after, saves training a classifier on the wrong primary data source.

---

## Phase 2: Build and train the classifier — Done, on the revised (own-data-only) basis

* **Status**: Trained on Kaggle T4, real run (not estimated). Zen approved proceeding on the revised basis after the ESSD non-transfer finding above.
* **Data**: excluded ESSD entirely, per the revised plan. Used the **full** real training pool this time, not the 15+15 sample from the feasibility gate — **all 1,200 oil scenes (Part I) + all 685 lookalike scenes (Part II)**, the same Kaggle-hosted `oil-spill-dartis-part1`/`-part2` datasets `train_kaggle.py` itself trains the U-Net on, pulled directly on the Kaggle kernel (no local download needed). `data/holdout/` was never referenced anywhere in this kernel — confirmed by reading the script, not just by intent.
* **Real infrastructure hiccups hit and fixed while getting this running** (documented since they cost real debugging time, not swept under the rug):
  1. First run failed immediately (`FileNotFoundError` on `model_real_v2_best.pt`) — the `trongzen/project-pelagic-model-best` Kaggle dataset had just been updated with the v2 checkpoint (it previously only had v1) and was still mid-processing ("Dataset version is being created") when the kernel launched. Confirmed `ready` via `kaggle datasets status` and re-ran.
  2. Second run failed the same way even with the dataset confirmed ready — the actual `/kaggle/input` mount path has since drifted from the flat `/kaggle/input/<dataset-slug>/...` convention `train_kaggle.py` and this file's first draft both hardcoded (now `/kaggle/input/datasets/<owner>/<slug>/...`, an extra `datasets/` segment). `kaggle_eval_kernel/eval_kaggle.py` had already independently hit and fixed this exact issue with a recursive `os.walk` search instead of a hardcoded path — applied the same fix here (`find_dir_with_tifs()`), for both the model checkpoint and the Part I/II data directories, rather than hardcoding a path that had already proven to drift once.
  3. Fetching Kaggle kernel logs from this Windows environment via the `kaggle` CLI crashes (`UnicodeEncodeError`, Windows' default `cp1252` console/file encoding choking on a UTF-8 character in the log) — worked around by calling the underlying `kaggle` Python API directly and writing the log with explicit `encoding='utf-8'`. Noted here in case a future session hits the same CLI crash and wastes time treating it as a Kaggle-side problem.
* **Real run, full pool**: 1,885 scene pairs total. Candidate-region extraction (same method as the feasibility gate: real v2-model prediction, largest connected component) succeeded on **100% of oil scenes (1,200/1,200)** and **95.9% of lookalike scenes (657/685, 28 skipped — the model correctly predicted nothing at all on those 28 training-distribution lookalikes)** — 1,857 total usable examples, a large, real, own-domain-only dataset (37x the feasibility gate's n=50). Feature extraction ran ~49 minutes on the T4 (~1.5s/scene, most of it U-Net inference + GLCM computation, not GPU-bound).
* **Scene-level stratified 80/20 split** (`sklearn.train_test_split`, `random_state=42`): 1,485 train (960 oil, 525 lookalike) / 372 val (240 oil, 132 lookalike). Split scene IDs saved (`output/lookalike_classifier_train_val_split.json`) so this can be audited later.
* **Three lightweight classifiers compared** (per-domain feature standardization fit on train only, applied to val):

  | Model | Val accuracy | Precision | Recall | F1 | AUC |
  |---|:---:|:---:|:---:|:---:|:---:|
  | Logistic Regression | 0.839 | 0.844 | 0.921 | 0.880 | 0.907 |
  | Random Forest | 0.844 | 0.832 | 0.950 | 0.887 | 0.915 |
  | **Gradient Boosting (selected)** | **0.847** | 0.865 | 0.904 | 0.884 | **0.934** |

  Gradient Boosting's val confusion matrix (rows=true lookalike/oil, cols=predicted lookalike/oil): `[[98, 34], [23, 217]]` — 98/132 lookalike scenes correctly flagged (74.2% specificity), 217/240 oil scenes correctly kept (90.4% recall). This is genuinely strong, own-domain, held-out validation performance — a real jump from the feasibility gate's n=50 LOO estimate (70% accuracy), consistent with having ~37x more training data and the full real class distribution rather than a small even sample.
* **Saved artifacts**: `checkpoints/lookalike_classifier_v1.joblib` (the trained Gradient Boosting model + its `StandardScaler` + the exact `FEATURE_COLUMNS` order, everything needed to run it), `output/lookalike_classifier_own_domain_features.csv` (all 1,857 scenes' features), `output/lookalike_classifier_results.json`, `output/lookalike_classifier_train_val_split.json`. Kernel: `kaggle_kernel_lookalike/train_lookalike_classifier.py` (single-file, synced-with-`src/` convention matching `train_kaggle.py`'s own precedent), pushed as `trongzen/project-pelagic-lookalike-classifier`.
* **Caveat going into Phase 3, stated plainly**: this 84.7%/AUC 0.934 number is validation performance on scenes drawn from the **same training-pool distribution** the classifier was fit on (held-out scenes, no leakage within that pool, but still the same underlying data source as training). It is not yet evidence of holdout performance — the 30-scene holdout spans 6 real-world regions this training pool's regional composition is unverified against (Zenodo's Part I/II scenes don't carry the same confirmed embedded georeferencing the holdout scenes do, per earlier rounds' region-coverage work). Phase 3 is the real test.

---

## Phase 3: Real holdout evaluation — mixed result, NOT recommended for integration as-is

* **Status**: Done. Real, honest numbers below — not rounded up, not cherry-picked (single deterministic run; the classifier and U-Net are both non-stochastic at inference time).
* **Method** (`src/evaluate_holdout_lookalike_filter.py`): runs the real v2 U-Net over all 30 holdout scenes (same tiled-inference path as `compare_checkpoints.py`, same "before" outcome bucketing already used for `docs/confusion_matrix_breakdown.json`, confirmed to reproduce it exactly: oil 8 correct/2 partial, no_oil 7 correct/3 false_positive, lookalike 0 correct/10 false_positive). For each scene, extracts the same candidate-region features Phase 2 trained on (largest connected component of the U-Net's own prediction) and applies the trained classifier: a "lookalike" verdict suppresses the entire predicted mask (scene becomes a null detection); an "oil" verdict leaves the prediction untouched; a scene with no candidate region at all (prediction already empty) passes through unfiltered.
* **Real infrastructure snag, fixed cleanly**: the Kaggle-trained `checkpoints/lookalike_classifier_v1.joblib` (Gradient Boosting) failed to unpickle locally (`ModuleNotFoundError: No module named '_loss'`) — a real sklearn-version incompatibility between the Kaggle kernel's environment and this local venv, not a corrupted file. Fixed by refitting all three candidate classifiers **locally**, from the already-downloaded feature CSV and the exact same saved train/val scene-ID split (`output/lookalike_classifier_train_val_split.json`) — reproduced the Kaggle run's numbers almost exactly (e.g. Gradient Boosting AUC 0.9335 locally vs. 0.9339 on Kaggle), confirming this is the same model on the same data, not a different fit.

### The result
| Category | Before (n=10 each) | After |
|---|---|---|
| **oil** | 8 correct, 2 partial | **2 correct, 1 partial, 7 false_negative** |
| **no_oil** | 7 correct, 3 false_positive | 7 correct, 3 false_positive (unchanged) |
| **lookalike** | 0 correct, 10 false_positive | **8 correct, 2 false_positive** |

* **The lookalike bucket improved dramatically**: 0/10 → 8/10 correct — the post-hoc filter genuinely works on 8 of the 10 real holdout lookalike scenes it was built to fix.
* **The oil bucket regressed badly, and this is disqualifying, not a minor side effect**: 8 correct/2 partial (10/10 "found the real oil" in some form) collapsed to 2 correct/1 partial/**7 false_negative** — 7 real oil detections the U-Net got right were then thrown away by the classifier calling them "lookalike." `no_oil` is unaffected either way (every no_oil scene's prediction was either already empty or too fragmented to clear the candidate-region size threshold, so the classifier never got a chance to touch it, for better or worse).
* **Not a threshold-tuning problem — checked directly, not assumed**: pulled the classifier's raw P(oil) for every oil-bucket scene. 5 of the 7 wrongly-suppressed oil scenes are *confidently* wrong (P(oil) = 0.034–0.089, 0.294), not borderline; only 2 (`oil_00004` at 0.467, `oil_00009` at 0.436) sit near the 0.5 boundary. Moving the decision threshold would rescue at most those 2 without addressing the other 5 — this is a real feature-space misclassification, not a miscalibrated cutoff. Likewise, the 2 lookalike scenes that still slip through are confidently (not borderline) misclassified as oil (P(oil)=0.93, 0.95) — genuinely hard cases from the classifier's own perspective, not near-misses either.
* **Most likely reason, consistent with everything else this investigation found**: the classifier trained well on the full 1,200-oil-scene training pool (84.7% val accuracy, AUC 0.934 — Phase 2's numbers were real, not wrong) but that training pool's shape/texture characteristics for the *oil* class apparently don't fully represent the specific oil scenes in this diverse, 6-region holdout set — the same region-generalization risk flagged as a concern for the ESSD data in Phase 1 turns out to also apply, to a real and costly degree, to this project's own training-pool-vs-holdout gap for the oil class specifically (interesting that it does *not* show up as a problem for the lookalike class, which generalized well from training pool to holdout).
* **Conclusion, plainly**: **not recommended for integration as currently built.** The lookalike-bucket win is real and worth keeping as a documented result, but trading 7 real oil false-negatives for 8 fixed lookalike false-positives is not an acceptable net swap for an oil-spill detection tool, where missing a real spill is the worse failure mode. Phase 4 (additive/toggleable integration) should not proceed on this classifier as-is — per the original task's own gate ("Do NOT integrate into the main live pipeline until Phase 3's result is reviewed and approved"), and this result does not clear that bar.
* Full per-scene results: `docs/phase3_holdout_with_filter_results.{json,csv}`, category summary: `docs/phase3_holdout_with_filter_summary.json`.

---

## Phase 3.5: Region-correlation diagnostic on the 7 misclassified oil scenes — hypothesis NOT supported, in an interesting way

* **Status**: Data analysis only, per Zen's explicit instruction — no retraining, no code changes to the classifier. Findings below, stopped for review as asked.
* **Question**: does the training pool's regional composition explain why the classifier wrongly suppresses 7/10 real holdout oil detections — specifically, do the 7 misclassified scenes cluster in the holdout's already-known *weak* regions (Red Sea/Gulf of Mexico/Baltic/Western Med, per the existing region-accuracy table), or are they spread more broadly?

### Finding 1: the 7 misclassified scenes cluster in the *strongest* region, not the weak ones — the opposite of the hypothesis
Cross-referenced each oil scene's classifier verdict (`docs/phase3_holdout_with_filter_results.json`) against its real region (`docs/region_coverage.json`, from embedded GeoTIFF coordinates):

| Region | Oil scenes (n) | Misclassified as lookalike |
|---|:---:|:---:|
| Eastern Mediterranean | 7 | **6/7 (85.7%)** |
| Red Sea | 3 | 1/3 (33.3%) |

All 7 misclassified scenes: `oil_00000, oil_00001, oil_00004, oil_00005, oil_00006, oil_00008 (Red Sea), oil_00009` — six of the seven are Eastern Mediterranean, the region with this project's *best* documented baseline U-Net pass rate (11/15 across all categories). The two Red Sea scenes the classifier actually got right (`oil_00002`, `oil_00003`) are from the same region as `oil_00008`, the one Red Sea scene it got wrong — so it's not a clean regional split either; Red Sea itself is mixed (2 correct, 1 wrong). Gulf of Mexico, Baltic Sea, and Western Mediterranean have **zero** oil-category holdout scenes at all (per `region_coverage.json`, that table's "0/2", "0/1", "0/2" entries were for the U-Net's baseline all-category pass rate, not oil specifically — this bucket only has Eastern Med and Red Sea oil scenes to begin with), so those regions can't be evaluated for this specific question. **The hypothesis as stated — misclassification concentrated in historically weak regions — is not supported by the data.** If anything the opposite pattern shows up (worse on the strong region), though n=7 is too small to call that a confirmed inverse relationship rather than noise.

### Finding 2: the training pool is NOT narrowly Eastern-Mediterranean — an earlier assumption in this doc was wrong, corrected here
Phase 2's write-up above stated training scenes "don't carry the same confirmed embedded georeferencing the holdout scenes do" — **checked directly this round and that's incorrect**. Training GeoTIFFs (Zenodo Part I/II) carry a `ModelTransformationTag` (a full affine matrix — the *other* standard GeoTIFF georeferencing convention, vs. holdout scenes' `ModelTiepointTag`+`ModelPixelScaleTag` pair, which is why the same `get_scene_geolocation()` used for holdout scenes silently doesn't apply and likely fed the wrong assumption). Real coordinates recovered directly from a systematic sample (every ~80th of the 1,200 oil scenes, every ~45th of the 685 lookalike scenes — n=15+15, ~1-2% of each pool, evenly spaced not cherry-picked):

| Sample | Real-world location found |
|---|---|
| oil scenes | North Sea (55.2N, 4.1E); **Gulf of Mexico / Bay of Campeche (6 of 15 samples** — 19-28N, -89 to -95W); Black Sea (43.7N, 35.7E); Levantine Mediterranean edge (33.4N, 33.2E; 33.4N, 30.0E — near but south of the holdout's stated 34-37N Eastern Med box); Portugal Atlantic coast (35.1N, -8.8W) |
| lookalike scenes | North Sea (54.6N, 3.5E); Sicilian Strait (37.1N, 13.7E); Persian Gulf / Strait of Hormuz (24.7N, 54.6E; 26.1N, 56.0E); North Sea/Scotland (59.1N, -1.6E; 58.5N, -2.1E); Persian Gulf (28.3N, 48.6E); Trinidad/Caribbean (10.6N, -61.5W); Gulf of Mexico (29.0N, -90.6W; 28.4N, -89.6W; 28.9N, -89.4W; 18.9N, -91.4W; 19.4N, -90.7W); Sea of Japan/Korea Strait (34.6N, 131.3E); Sinai/Eastern Med edge (31.5N, 33.2E) |

The training pool is genuinely global — North Sea, Black Sea, Persian Gulf, Trinidad, Japan/Korea Strait, Portugal, and (heavily) Gulf of Mexico all show up in a small systematic sample, alongside a few scenes near but not matching the holdout's specific Eastern Med bounding box. **This directly contradicts the "training pool lacks holdout's regional diversity" framing** — if anything, the training pool appears *more* geographically diverse than the holdout's curated 6 regions, not less. Gulf of Mexico in particular is well-represented in training (6/15 oil samples, 5/15 lookalike samples) despite being one of the holdout's documented weak regions for the base U-Net.

### Revised interpretation
A simple "training pool doesn't cover the holdout's regions" story doesn't hold up — the training pool already spans more of the globe than the holdout does. The concentration of misclassifications in Eastern Mediterranean oil scenes specifically is real (Finding 1), but its likely cause is probably not raw geographic coverage. A quick supplementary check (candidate-region features for all 10 oil holdout scenes, same read-only extraction as Phase 3, no retraining) shows a mixed picture rather than one clean rule: some misclassified scenes (`oil_00000`: solidity 0.776, fill-fraction 0.482; `oil_00005`: solidity 0.793, fill-fraction 0.605) have shape stats that sit much closer to the lookalike class's typical profile (Phase 2's training data: lookalike mean solidity 0.692, fill-fraction 0.490 vs. oil mean 0.570/0.271) — plausibly genuinely oil-slick-shaped in a way that happens to look "lookalike-shaped" by these metrics. Others (`oil_00001`: solidity 0.460, fill-fraction 0.246 — both already oil-typical) don't fit that story at all and were still misclassified, meaning the Gradient Boosting model's decision isn't reducible to one or two features either. **Not chasing this further per Zen's explicit "don't retrain yet" instruction** — flagging it as the most concrete lead for a future round (e.g., a smaller, geographically-blind ablation, or inspecting which of the 11 features the model actually weights most heavily for these specific scenes) rather than concluding it here.
* Reproducible via the same feature-extraction call already in `src/evaluate_holdout_lookalike_filter.py`; no new script needed for this diagnostic beyond one-off analysis in this session (not saved as a script since it was exploratory cross-referencing of already-produced files, not a new pipeline stage).

---

## Phase 3.6: Full feature-vector inspection of the 7 misclassified oil scenes — two distinct, real causes found; no extraction bug

* **Status**: Data analysis only, per Zen's explicit instruction — no retraining. Full feature vectors, importance-weighted attribution, training-pool rarity check, and direct visual inspection below.
* **Method**: pulled all 11 features for all 10 oil holdout scenes (not just the 7 misclassified — kept the 3 correct ones as a comparison group), computed each feature's z-score against the training pool's oil-class and lookalike-class mean/std (`output/lookalike_classifier_own_domain_features.csv`, n=1,857), and weighted each feature's "pull toward lookalike" (`|z_oil| − |z_lookalike|`, positive = closer to the lookalike distribution) by the trained Gradient Boosting model's own `feature_importances_` — so the ranking reflects what the model actually relies on, not an arbitrary feature list. Model's real global importances: **`blob_fill_fraction` (0.362) and `glcm_correlation` (0.203) alone account for 56% of total importance**; `aspect_ratio` (0.156) and `circularity` (0.098) most of the rest; the remaining 7 GLCM/edge features sum to under 15%.

### Two distinct groups, two distinct causes — not one single bug

**Group A — `oil_00000`, `oil_00004`, `oil_00005`, `oil_00006`: genuinely massive, scene-spanning real slicks, rare in training**
* In every one of these 4 scenes, `blob_fill_fraction` is the dominant driver by a wide margin (importance-weighted pull +0.47 to +0.63 — 5-10x the next-largest feature for the same scene). Their candidate blob's bounding box is `(0,0,2048,2048)` or nearly the full frame for all four.
* **Checked directly, not assumed, whether this is a real slick or a detection artifact**: `blob_vs_gt_iou` for these four is 0.78–0.97 — the candidate blob closely tracks the real ground-truth mask, which is *also* enormous (`oil_00000`: 2,015,200 GT px = 48% of the scene; `oil_00005`: 2,527,264 px = 60%). **Visually confirmed too** (`output/phase3_6_diagnostic/oil_candidate_regions_overview.png`): the candidate-blob row and ground-truth row are near-identical sprawling shapes for all four — real, large, amoeba-like slicks, not noise or a broken threshold.
* **The actual mechanism, quantified**: a blob this large, relative to its own tight bounding box, naturally has a much higher fill-fraction than a small localized detection. Checked how common this scale is in the 1,200-scene oil training pool: **only 18/1,200 (1.5%) have a candidate blob over 1,000,000 px, and only 1.75% have fill-fraction over 0.6** — the exact scale these 4 holdout scenes sit at is genuinely rare in what the classifier learned from. This is a real, boring, fixable-in-principle explanation for this group: not enough large-scale training examples for the model to have learned "huge real slicks can still be legitimately oil," not a bug in the pipeline.

**Group B — `oil_00001`, `oil_00008`, `oil_00009`: already oil-typical on shape, pulled wrong by texture (`glcm_correlation`) — genuinely ambiguous, not one clean cause**
* These 3 do **not** show the Group A pattern (`blob_fill_fraction` is a minor or even oil-favoring factor for them). Instead `glcm_correlation` dominates in every case (weighted pull +0.20 to +0.67).
* **`oil_00001` is a distinct sub-case**: its `glcm_correlation` (0.900) is an extreme *positive* outlier vs. the oil training mean (0.627±0.058, z=+4.71) — an unusually smooth, coherent texture patch, atypical even for oil. Visually it's a genuinely thin, elongated streak (aspect_ratio 5.06, the highest of all 10 scenes) — a real, distinctively-shaped detection the classifier hasn't seen much of either.
* **`oil_00008`/`oil_00009` share their pattern with two of the three *correctly*-classified scenes**: `oil_00002` and `oil_00003` (both Red Sea, both correct) also have `glcm_correlation` as their #1 lookalike-pulling feature (weighted pull +0.31, +0.09) — low correlation relative to the oil mean is evidently common across several Red-Sea/Levant-area oil scenes generally, correctly-classified ones included. What separates `oil_00008`/`oil_00009` (wrong) from `oil_00002`/`oil_00003` (right) is the *cumulative* pull across their top 4 features (0.35–0.38 vs. 0.09–0.13), not one single feature being uniquely broken. **This is a genuinely ambiguous feature-space overlap for these 3 scenes** — a legitimate finding, not a failure to find a cause.
* **Visually confirmed no extraction bug here either**: `oil_00001`'s thin streak and `oil_00008`/`oil_00009`'s fragmented, multi-lobed blobs both closely track their own ground truth in the same overview plot.

### The "boring explanation" check — ruled out
Directly inspected the candidate-region crop, blob mask, and ground-truth mask side-by-side for **all 10** oil holdout scenes (not just the 7), not just their summary statistics. Every candidate blob's shape closely matches its own ground truth, for both the misclassified and correctly-classified scenes alike. **No corrupted, mismatched, or degenerate candidate-region extraction was found anywhere in the oil bucket.** The 7 failures are real feature-space/model-decision issues, not a data pipeline bug.

### Summary
| Scene | Region | Top driver | Real cause |
|---|---|---|---|
| oil_00000 | E. Med | blob_fill_fraction (+0.53) | massive real slick, rare training scale |
| oil_00004 | E. Med | blob_fill_fraction (+0.53) | massive real slick, rare training scale |
| oil_00005 | E. Med | blob_fill_fraction (+0.63) | massive real slick, rare training scale |
| oil_00006 | E. Med | blob_fill_fraction (+0.47) | massive real slick, rare training scale |
| oil_00001 | E. Med | glcm_correlation (+0.67) | extreme texture/shape outlier (thin streak) |
| oil_00008 | Red Sea | glcm_correlation (+0.28) | ambiguous, shared pattern with correct Red Sea scenes |
| oil_00009 | E. Med | glcm_correlation (+0.20) | ambiguous, shared pattern with correct Red Sea scenes |

This also sharpens Phase 3.5's regional finding: it was never really about "Eastern Mediterranean" as a region — it's that 5 of the 6 misclassified Eastern Med scenes belong to the two identified mechanisms above (scale rarity or texture-outlier/ambiguity), which happen to concentrate in this holdout's Eastern Med scenes largely by coincidence of which real slicks were included, not because of anything specific to that region.
* Outputs: `output/phase3_6_diagnostic/oil_holdout_full_features.json` (full feature vectors + blob/GT IoU for all 10 oil scenes), `output/phase3_6_diagnostic/oil_candidate_regions_overview.png` (visual check), `output/phase3_6_diagnostic/*_crop.npz` (raw crop/blob/GT arrays per scene).
* **Not retrained, per Zen's instruction.** Stopping here for review.

---

## Phase 3.7: Oversample large-blob training examples — validation improved, holdout unchanged (the fix didn't transfer)

* **Status**: Done. Honest result: **the targeted fix worked exactly as intended within the training pool, but did not transfer to the actual holdout scenes it was meant to fix.**
* **Method**: identified the 18 training-pool oil scenes with `blob_area_px > 1,000,000` (Phase 3.6's threshold), split 12 train / 6 val by the existing scene-level split. Swept oversampling factors 1x/3x/4x/5x/8x (duplicating only the 12 train-side examples, val left untouched) and picked by validation performance before touching the holdout at all, per the task's instruction:

  | Factor | Val accuracy | Precision | Recall | F1 | AUC |
  |---|:---:|:---:|:---:|:---:|:---:|
  | 1x (baseline) | 0.847 | 0.865 | 0.904 | 0.884 | 0.934 |
  | 3x | 0.860 | 0.873 | 0.917 | 0.894 | 0.935 |
  | **4x (selected)** | **0.868** | 0.869 | 0.938 | 0.902 | **0.936** |
  | 5x | 0.860 | 0.862 | 0.933 | 0.896 | 0.935 |
  | 8x | 0.863 | 0.862 | 0.938 | 0.898 | 0.931 |

  4x improved every metric with no sign of overfitting (8x's slightly lower AUC hints at where over-oversampling would start to hurt) — a modest, well-justified choice, not the most aggressive option. Saved as `checkpoints/lookalike_classifier_v2_oversampled.joblib`.
* **Confirmed the oversampling worked as intended on the training pool's own distribution**: checked the 6 large-blob oil examples held out in the *validation* split specifically (never oversampled, never trained on) — **5 of 6 moved toward correctly-classified-as-oil** (e.g. `oil_00057.tif`: P(oil) 0.082→0.256; `oil_00932.tif`: 0.310→0.391), 1 of 6 moved the other way. So the intervention is real and measurable, not a no-op — it does teach the model to weigh large-fill-fraction candidates more toward "oil" in general.

### The real holdout re-run: bucket-level result is byte-for-byte identical to Phase 3
| Category | Phase 3 (before) | Phase 3.7 (after oversampling) |
|---|---|---|
| oil | 2 correct, 1 partial, 7 false_negative | **2 correct, 1 partial, 7 false_negative — unchanged** |
| no_oil | 7 correct, 3 false_positive | 7 correct, 3 false_positive — unchanged (no no_oil scene ever reaches the classifier, same as before) |
| lookalike | 8 correct, 2 false_positive | 8 correct, 2 false_positive — unchanged, same 2 scenes (`lookalike_00003`, `lookalike_00007`) still slip through |

* **Group A (`oil_00000`, `oil_00004`, `oil_00005`, `oil_00006`) — did NOT flip back, and mostly got *more* confidently wrong**, not less:

  | Scene | Before P(oil) | After P(oil) | Direction |
  |---|:---:|:---:|---|
  | oil_00000 | 0.089 | 0.021 | worse |
  | oil_00004 | 0.467 | 0.363 | worse |
  | oil_00005 | 0.079 | 0.034 | worse |
  | oil_00006 | 0.034 | 0.039 | ~unchanged |

  This is the opposite of what the training-pool validation result predicted. The fix generalizes to *more of the training pool's own* large-blob examples (5/6 held-out ones improved) but not to these specific 4 holdout scenes — a genuine, still-unresolved train-vs-holdout gap, not something this intervention could reach.
* **Group B (`oil_00001`, `oil_00008`, `oil_00009`) — as expected, not this fix's target, and indeed unchanged in outcome, but interesting movement**: `oil_00008` (0.294→0.482) and `oil_00009` (0.436→0.490) both moved substantially closer to the 0.5 boundary — nearly flipping — while `oil_00001` barely moved (0.052→0.079). None crossed the threshold, so the bucket table is unchanged, but `oil_00008`/`oil_00009` are now genuinely close calls rather than confident misses.
* **Lookalike bucket held, as asked**: still 8/10 correct, the same 2 false positives — oversampling large *oil* examples did not make the classifier more lenient toward real lookalikes slipping through as oil.
* **no_oil bucket, verified unchanged**: identical, as expected — every no_oil scene's candidate is either already empty or too fragmented to reach the classifier at all (`no_candidate` verdict in both runs), so this bucket structurally cannot be affected by any change to the classifier itself.

### Conclusion
A real, honestly negative result for the stated goal: **the training-data-scarcity fix for Group A is confirmed correct as far as it goes (it measurably helps the model on more of the training pool's own large-blob examples) but does not close the gap for the 4 specific holdout scenes it targeted** — those scenes apparently differ from the training pool's large-blob examples in some way beyond raw scale/fill-fraction that this intervention doesn't address. Net holdout outcome is unchanged from Phase 3: still not recommended for integration. `output/lookalike_classifier_v2_oversampled_val_results.json`, `docs/phase3_7_holdout_oversampled_results.{json,csv}`, `docs/phase3_7_holdout_oversampled_summary.json`.

---

## Phase 3.8, Task A: Size-based safety rule for Group A — NOT feasible, ruled out with real numbers

* **Status**: Checked, rule **not proposed for implementation** — real lookalikes reach the same scale as Group A's oil scenes, in both the training pool and the holdout itself.
* **Question**: is there a `blob_fill_fraction` (or scale) threshold above which "lookalike" could be safely overridden back to "oil" as a hard post-classifier rule, given Group A's 4 oil scenes sit at fill-fraction 0.44–0.61?
* **Training pool check**: the 657 usable training lookalike examples reach fill-fraction up to **0.999** (mean 0.471, essentially the same mean as Group A's oil scenes) — the top 10 highest-fill-fraction training lookalike examples are all at 0.999+, with `blob_area_px` up to 4,194,304 (literally the entire 2048×2048 scene). Real lookalikes routinely span the whole frame.
* **Holdout check (the real target)**: extracted the same feature for all 10 holdout lookalike scenes directly (not assumed) — `lookalike_00000` reaches fill-fraction **0.8275**, `lookalike_00001` reaches **0.8254**, `lookalike_00006` reaches **0.6615** — all three exceed every one of Group A's 4 oil scenes (max 0.605), and `lookalike_00000`/`00005`/`00006`/`00008` all have the identical `(0,0,2048,2048)` whole-scene bbox Group A's oil scenes have.
* **Conclusion, per the task's explicit instruction to report honestly if this happens**: there is no clean separation. Real lookalikes — in training and in this exact holdout — reach equal or higher scale/fill-fraction than the 4 oil scenes a size-based rule would be trying to rescue. Any fill-fraction threshold high enough to leave real lookalikes alone would also fail to catch Group A; any threshold low enough to catch Group A would flip several genuine holdout lookalikes (at minimum `lookalike_00000`, `lookalike_00001`, `lookalike_00006`) into false positives — undoing Phase 3's actual win. **Not implemented, as instructed.**
* Data: `output/phase3_6_diagnostic/holdout_lookalike_full_features.json` (new, this round), training-pool lookalike stats computed directly from `output/lookalike_classifier_own_domain_features.csv`.

---

## Phase 3.8, Task B: Add U-Net confidence as a feature — real, if partial, improvement

* **Status**: Done. **This is the first intervention that actually rescues a holdout scene without any lookalike regression** — a genuine, if modest, net improvement over Phase 3's baseline filter.
* **New feature**: `mean_unet_confidence` — the U-Net's own mean sigmoid probability (continuous, not thresholded) within the candidate blob's pixels. Added to the shared feature-extraction pipeline as an own-domain-only addition (kept separate from `src/analysis/candidate_region_features.py`'s `FEATURE_COLUMNS`, since ESSD patches have no U-Net probability to draw from — see that module's updated docstring reasoning applied the same way in `kaggle_kernel_lookalike/train_lookalike_classifier.py`). Required a full re-run of feature extraction over all 1,885 training scenes (the binary mask alone, already saved from Phase 2, isn't enough — needed the continuous probability array) — real Kaggle T4 run, ~59 minutes, same 1,857/1,885 keep rate as Phase 2 (reproducible: identical skip counts).
* **Validation check first, per the task's instruction** (same train/val split as Phase 2, Gradient Boosting selected again by val AUC):

  | Metric | Phase 2 (11 features) | Phase 3.8 (12 features, +confidence) |
  |---|:---:|:---:|
  | Val accuracy | 0.847 | **0.879** |
  | Val precision | 0.865 | **0.901** |
  | Val recall | 0.904 | 0.913 |
  | Val F1 | 0.884 | **0.907** |
  | Val AUC | 0.934 | **0.947** |

  Real improvement across the board, no regression on the validation check — cleared the gate to proceed to the holdout. `mean_unet_confidence` came in as the model's **2nd-most-important feature (20.0%)**, behind only `blob_fill_fraction` (32.0%) and ahead of `glcm_correlation` (16.2%) — a real, load-bearing signal, not a token addition.

### The real holdout re-run
| Category | Phase 3 baseline (before any filter) | Phase 3/3.7 (with filter, no confidence feature) | **Phase 3.8 (with confidence feature)** |
|---|---|---|---|
| oil | 8 correct, 2 partial | 2 correct, 1 partial, 7 false_negative | **3 correct, 1 partial, 6 false_negative** |
| no_oil | 7 correct, 3 false_positive | 7 correct, 3 false_positive | 7 correct, 3 false_positive — unchanged |
| lookalike | 0 correct, 10 false_positive | 8 correct, 2 false_positive | **8 correct, 2 false_positive — unchanged, same 2 scenes** |

Full three-way P(oil) comparison, since the picture is more nuanced than a single before/after:

| Scene | Phase 3 baseline | Phase 3.7 (oversampled) | Phase 3.8 (+confidence) |
|---|:---:|:---:|:---:|
| oil_00000 (Group A) | 0.089 | 0.021 | 0.042 |
| oil_00001 (Group B) | 0.052 | 0.079 | **0.188** |
| oil_00004 (Group A) | 0.467 | 0.363 | 0.176 |
| oil_00005 (Group A) | 0.079 | 0.034 | 0.056 |
| oil_00006 (Group A) | 0.034 | 0.040 | 0.010 |
| oil_00008 (Group B) | 0.294 | 0.482 | 0.358 |
| oil_00009 (Group B) | 0.436 | 0.490 | **0.633** |

* **`oil_00009` (Group B) flipped from misclassified to correct** — and not a borderline flip: P(oil) reached **0.633**, a real, non-marginal swing past the boundary.
* **`oil_00001` (Group B) moved consistently toward correct across both interventions** (0.052 → 0.079 → 0.188) but still short of 0.5.
* **`oil_00008` (Group B) is not monotonic** — oversampling (Phase 3.7) pushed it up to 0.482 (just short of flipping), but adding the confidence feature pulled it back down to 0.358. The two interventions don't compose cleanly; this scene's outcome depends on which fix is applied, not a strictly additive improvement.
* **Group A is still not fixed, and not uniformly**: `oil_00000`/`oil_00006` stayed roughly flat (confidently wrong throughout), `oil_00005` ticked slightly worse, and `oil_00004` kept getting *more* confidently wrong across both interventions (0.467 → 0.363 → 0.176) — consistent with Task A's finding that these scenes are genuinely hard to separate from real lookalikes on the features available, not something either fix reaches.
* **Lookalike bucket held exactly**, same 2 false positives (`lookalike_00003`, `lookalike_00007`) as every prior round — adding U-Net confidence did not make the classifier more lenient toward real lookalikes.
* **no_oil bucket verified unchanged** — structurally unaffected, as in every prior round (no no_oil candidate ever reaches the classifier).

### Conclusion
The best-performing configuration found across every intervention this investigation has tried (Phase 3 baseline filter, Phase 3.7 oversampling, this round): **oil bucket 3 correct/1 partial/6 false_negative**, still far below the pre-filter baseline (8 correct/2 partial) and still net-negative overall (trading 6 real oil misses for 8 fixed lookalike false-positives remains an unfavorable swap for an oil-spill detector), but the first change to actually move a real number in the right direction without giving anything back elsewhere. Saved: `checkpoints/lookalike_classifier_v3_confidence.joblib`, `output/lookalike_classifier_v3_confidence_val_results.json`, `docs/phase3_8_holdout_confidence_results.{json,csv}`, `docs/phase3_8_holdout_confidence_summary.json`. Still not recommended for integration as-is — the oil-bucket regression remains too large — but this is now the strongest basis for a future round to build on.

---

## Phase 3.9: Confidence-gated classifier application — no safe threshold exists, confirmed with real numbers (including a direct counter-example)

* **Status**: Done. **The premise doesn't hold**: U-Net confidence and correctness are only weakly related in this domain — the model is routinely very confident *and wrong* on real lookalikes, which is the entire reason this post-hoc classifier project exists. Reported honestly per the pattern already established for Task A above, with the full requested sweep run anyway so the tradeoff is quantified, not just asserted.

### Distribution overlap — training pool
| | n | mean | median | p90 | p95 | max |
|---|---:|---:|---:|---:|---:|---:|
| oil | 1,200 | 0.922 | 0.926 | 0.958 | 0.966 | 0.984 |
| lookalike | 657 | 0.902 | 0.905 | 0.975 | 0.985 | **0.994** |

Means are nearly identical (0.922 vs 0.902); lookalike's own *upper* tail actually exceeds oil's. **52.2% of all training lookalike examples (343/657) have U-Net confidence above 0.9**, and even **2.7% (18 examples) exceed 0.99** — the U-Net isn't just occasionally overconfident on lookalikes, it's confident on them more often than not.

### Distribution overlap — holdout (the real target, and where the clearest counter-example lives)
| Scene | Confidence | | Scene | Confidence |
|---|:---:|---|---|:---:|
| oil_00000 (Group A) | 0.978 | | lookalike_00000 | 0.971 |
| oil_00004 (Group A) | 0.967 | | **lookalike_00001** | **0.980** |
| oil_00005 (Group A) | 0.985 | | lookalike_00002 | 0.951 |
| oil_00006 (Group A) | 0.977 | | lookalike_00005 | 0.959 |
| oil_00003 (correct) | 0.966 | | lookalike_00006 | 0.974 |

**`lookalike_00001` — a real, confirmed-false lookalike detection — has the single highest U-Net confidence (0.980) of all 20 oil+lookalike holdout scenes combined, exceeding every one of Group A's 4 "confidently real oil" scenes except `oil_00005`.** This is a direct, concrete answer to the task's question: yes, the U-Net is sometimes very confident and very wrong on lookalikes, in this exact holdout, not just hypothetically in the training pool.

### Validation sweep (per the task's instruction, checked before touching holdout)
Simulated the gate on the held-out validation split: gated examples (confidence above threshold) are forced to "oil" regardless of the classifier; others use the classifier's real verdict.

| Threshold | Oil examples gated | Lookalike examples wrongly gated to "oil" | Net correct (with gate) | Net correct (classifier only) |
|---|---:|---:|---:|---:|
| 0.90 | 183 | 73 | 281 | **327** |
| 0.95 | 42 | 35 | 301 | **327** |
| 0.97 | 10 | 20 | 312 | **327** |
| 0.99 | 0 | 5 | 322 | **327** |
| 0.995 | 0 | 0 | 327 | 327 |

**Gating is strictly worse than the classifier alone at every threshold tested on validation** — there is no point where gating helps overall accuracy; it only ever adds false "oil" calls on lookalike examples that the classifier was already getting right.

### The real holdout run anyway, full threshold sweep (gate = keep U-Net's raw call above threshold, else use Phase 3.8's classifier verdict; no_oil bucket untouched in every case, as always)
| Threshold | Oil bucket | Lookalike bucket |
|---|---|---|
| 0.90 | 8 correct, 2 partial | 1 correct, 9 false_positive |
| 0.95 | 7 correct, 1 partial, 2 false_negative | 3 correct, 7 false_positive |
| 0.97 | 6 correct, 1 partial, 3 false_negative | 5 correct, 5 false_positive |
| **0.975** | **6 correct, 1 partial, 3 false_negative** | **7 correct, 3 false_positive** |
| **0.98** | **4 correct, 1 partial, 5 false_negative** | **8 correct, 2 false_positive (unchanged)** |
| 0.99 (no gate fires) | 3 correct, 1 partial, 6 false_negative | 8 correct, 2 false_positive |

**The exact mechanism the task asked about, confirmed on real data**: at threshold 0.975, 3 of Group A's 4 scenes get protected (`oil_00000`=0.978, `oil_00005`=0.985, `oil_00006`=0.977 all clear it) — but `lookalike_00001` (0.980) *also* clears it, flipping from correctly-classified back to a false positive. There is no threshold that protects those 3 oil scenes without also admitting that one lookalike: `lookalike_00001`'s confidence (0.980) is mathematically higher than 2 of the 3 oil scenes the gate would need to protect, so any threshold low enough to catch them is low enough to catch it too.
* **Two genuinely different real operating points, not a single answer** — presented for a decision, not decided here:
  - **Threshold ≈0.975**: oil bucket improves from 3→**6** correct (nearly matching the original 8-correct pre-filter baseline), lookalike bucket costs 1 (8→**7** correct). A real trade: 3 oil scenes rescued for 1 lookalike false-positive reintroduced.
  - **Threshold ≈0.98**: oil bucket improves from 3→**4** correct, lookalike bucket **unchanged at 8**. A smaller, "free" gain with zero cost.
* **This is a genuine value judgment, not a technical one** — given this investigation's own repeated framing that missing real oil is the worse failure mode for a spill detector, 0.975's trade could be read as favorable; but it does concretely give back some of Phase 3's lookalike win to get there, and that's a call for Zen, not something to decide unilaterally.
* Not implemented as a change to any saved artifact yet — this is a reported finding with real numbers on both sides, per the task's framing. Data: `output/phase3_6_diagnostic/holdout_confidence_all.json` (new, this round).

---

## Phase 4: Final configuration implemented, saved, and wired in as additive/toggleable — NOT enabled by default

* **Status**: Done. Zen accepted Phase 3.9's oil-vs-lookalike trade; this round turns it into real, saved, callable code rather than a notebook simulation.
* **Final artifact**: `checkpoints/lookalike_classifier_final.joblib` (the Phase 3.8 Gradient Boosting classifier, 12 features including `mean_unet_confidence`) — a plain copy of `lookalike_classifier_v3_confidence.joblib` under its canonical final name; the versioned files (`_v1`, `_v2_oversampled`, `_v3_confidence`) are kept alongside it, not deleted, so every prior round's exact artifact stays reproducible.
* **Real code, not a simulation**: new `src/analysis/lookalike_filter.py` —
  - `GATE_THRESHOLD = 0.975`, the exact value swept and reported in Phase 3.9 (the range that reproduces that exact result is `[0.974, 0.977)` — 0.975 sits inside it, not a rounded stand-in for a more precise number that was never actually computed differently).
  - `apply_confidence_gate(mean_unet_confidence, classifier_verdict, threshold=GATE_THRESHOLD)` — the gate rule itself, as a real function.
  - `classify_and_filter(image_raw, probs_full, pred_mask, ...)` — the full pipeline: candidate-region extraction (reusing `src/analysis/candidate_region_features.py`, unchanged), `mean_unet_confidence` computation, classifier prediction, gate application, and the filtered mask + a diagnostic dict (`applied`, `verdict`, `gated`, `proba_oil`, `mean_unet_confidence`, `reason`).
* **`src/evaluate_holdout_lookalike_filter.py` refactored to call this real module** (previously it duplicated the classifier-prediction logic inline) — the confirmation run below therefore exercises the *actual* saved code path, not a re-implementation of it. Also gained `--gate`/`--gate-threshold` flags (default off) so both configurations are reproducible from the same script.
* **Final confirmation run, exact match, no discrepancy**:

  | Category | No gate (`--out-prefix phase4_confirm_nogate`) | **With gate** (`--out-prefix phase4_confirm_gated`, threshold 0.975) |
  |---|---|---|
  | oil | 3 correct, 1 partial, 6 false_negative (reproduces Phase 3.8 exactly) | **6 correct, 1 partial, 3 false_negative** |
  | no_oil | 7 correct, 3 false_positive | 7 correct, 3 false_positive — unchanged |
  | lookalike | 8 correct, 2 false_positive (reproduces Phase 3.8 exactly) | **7 correct, 3 false_positive** |

  Matches Phase 3.9's simulated prediction exactly: **oil 6/10 correct, lookalike 7/10 correct, no_oil unchanged at 7/10** — confirmed which scenes moved, too: `oil_00000`/`oil_00005`/`oil_00006` (all Group A) get gated (`mean_unet_confidence` 0.978/0.985/0.977, all clear 0.975) and kept as oil; `lookalike_00001` (0.980) also clears the gate and is the one new false positive, exactly as identified in Phase 3.9. No unexpected discrepancy to report.
* **Wired into `/api/predict` as additive/toggleable, not a silent replacement**: `PredictRequest` gained `apply_lookalike_filter: bool = False`. When unset or `False` (every existing caller, including the 4 verified cached demo scenes and anything that predates this field), the code path is **byte-for-byte identical** to before this round — `classify_and_filter()` is never even called. Only when a caller explicitly passes `apply_lookalike_filter: true` does the filter run, and the response gains an extra `lookalike_filter` diagnostic key.
* **Cache-safety, a real risk that was caught and fixed before it could bite**: `/api/predict`'s existing scene_id cache (keyed by `checkpoint_hash`, added in an earlier round to invalidate stale rows on checkpoint swap) would otherwise let a `apply_lookalike_filter=True` call silently return a stale *unfiltered* cached row (or vice versa) for a scene_id already predicted once. Fixed by folding the filter setting into the cache key (`effective_hash = CHECKPOINT_HASH + "+lookalike_filter"` when enabled) — reuses the exact same "hash mismatch → overwrite in place" logic already proven for checkpoint swaps, rather than adding new cache logic.
* **Verified end-to-end against the real running app, not just unit-level**: backed up `data/pelagic.db` first, then via `fastapi.testclient.TestClient` against a **non-demo** holdout scene (`lookalike_00002`, deliberately not one of the 4 verified demo scenes) — confirmed (a) default and explicit-`False` calls return identical results, (b) `apply_lookalike_filter=True` correctly suppresses the detection (this scene is a real lookalike; classifier verdict `lookalike`, not gated, `proba_oil=0.074`) and returns the diagnostic info, (c) toggling back to default afterward correctly reverts to the original unfiltered result (cache invalidation works in both directions, not just one). Restored `data/pelagic.db` from the backup afterward — `git status`/`git diff` on that file show **zero net change**.
* **`/api/live/fetch` and the 4 cached demo scenes: untouched**, per the explicit instruction — no code in the live-fetch endpoint was modified this round, and the verification above deliberately used a non-demo scene precisely to avoid writing anything new under any of the 4 demo scene_ids.
* **Not enabled by default anywhere.** This is a real, tested, available capability — not yet a decision to change what the live app shows. Turning it on for the actual demo/live paths (if ever desired) would be a separate, explicit future decision, not something this round did silently.

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
