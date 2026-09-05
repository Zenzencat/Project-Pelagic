# Project Pelagic: Status & Handoff

Updated 2026-09-05 after resuming `prompt/pelagic-loop.md` from Task 1's
unfinished verification. The requested `prompts/` directory does not exist.
This file is the current source of truth; earlier model investigations remain in
[status_history.md](status_history.md).

## Current environment — verified in this continuation

- `.env`, `~/.cdsapirc`, and process credentials `CDSAPI_URL`, `CDSAPI_KEY`,
  `CDSE_CLIENT_ID`, `CDSE_CLIENT_SECRET`, `GFW_TOKEN` are absent. Historical
  successful credential checks do not establish availability here.
- Application networking now works: real CDSE OData requests returned catalog
  metadata. The previous session's DNS failure is no longer a current blocker.
- Python is 3.14.6. Optional dependencies were installed in disposable
  `/tmp/pelagic-verification-py314`, using system site packages. Added cdsapi
  0.7.7, xarray 2026.7.0, netCDF4 1.7.4 and global-land-mask 1.0.0; the resolver
  installed NumPy 2.5.2 there. Main Python dependencies were not changed.
- `frontend/npm ci --ignore-scripts --no-audit --no-fund` succeeded against the
  existing lockfile. Vite production build succeeds. Chromium 145.0.7632.6 and
  FastAPI TestClient work; prior dependency/browser/HTTP blockers are resolved.
- No satellite pixels, wind record or GFW vessel data were fetched in this
  continuation. Catalog access does not verify authenticated Process/CDS access.

## Acceptance checkpoint and next action

Tasks 1–3 have implemented optional API/UI paths, passing local checks, and a
real credential blocker for remaining external verification. Task 1's skipped
NetCDF test and unverified HTTP/browser behavior have now been checked.

**First remaining end-to-end acceptance check:** confirm original live
Sentinel-1 inference and optional ERA5 retrieval against a real live-fetched
scene. Requires CDSE credentials plus CDS credentials and dataset license
acceptance. Then verify a real different-date SAR comparison and SAR/optical
pair. Do not infer these passes from the local fixtures or public catalog.

Task 4 remains stopped at its previously documented viability hard stop; no
DSen2-CR integration or inference is claimed. The optional experiment was not
restarted in this continuation.

## Task 1 — ERA5 wind evidence

**Implemented; local NetCDF/HTTP/browser checks pass. Real CDS check BLOCKED.**

- `src/analysis/era5_wind_check.py::get_wind_evidence()` requests ERA5 hourly
  single levels u10/v10 on a 0.25° grid. Uses the nearest UTC hour (including
  rollover), requires that exact returned hour, selects a nearby spatial cell,
  checks units/finite values/ambiguity and computes `hypot(u10, v10)`.
  Signed and 0–360 longitude axes are supported.
- `include_era5` defaults false on `POST /api/live/fetch`. Uses catalog
  acquisition time and the fetched GeoTIFF scene center, not a separate wind
  measurement for each slick polygon. Missing configuration is `not_configured`;
  retrieval failures/timeouts are `unavailable`, with no numeric wind value.
  CDS retrieval runs in a separate process with a 60-second budget.
- Evidence persists in `detections.supplementary_json` and appears in detection
  details. The UI displays speed/time/grid point or an explicit reason. Old
  rows do not acquire guessed evidence. No wind-derived oil verdict is added.
- The formerly skipped NetCDF test now writes and reads a real NetCDF file
  containing **synthetic test components**, checks year rollover and speed,
  and rejects wrong hours, distant cells and nonfinite values. CDS is mocked.
- New HTTP regression uses actual FastAPI routing, validation and JSON responses,
  a disposable SQLite database, and the real missing-CDS adapter. Checks optional
  defaults, location/time propagation, persistence, and invalid request HTTP 422.
  Satellite fetch/inference are explicit doubles in that regression.
- ERA5 remains hourly and coarse (~28 km north-south per 0.25° interval), not a
  sensor at the SAR detection. CDS queues can exceed the local budget.

## Task 2 — different-date Sentinel-1 evidence

**Implemented; local checks and real catalog selection pass. Pixels BLOCKED.**

- Shared bounded/paginated OData search, existing CDSE auth and Process transport,
  and `_analyze_live_scene` reuse primary preprocessing/U-Net/land-mask/contours.
  `include_temporal` defaults false; window defaults ±30 days (1–90).
- Candidates require valid identity/time, another UTC date, IW GRDH VV/VH and
  full footprint coverage. Known matching orbit track/direction rank first.
  At most two candidates are tried. No result is `no_match`; catalog, fetch or
  inference failure is `unavailable`. Duplicate pixel hashes are rejected.
- Primary selection honors the requested inclusive date range. Process fetching
  checks all source identities/dates, full validity and GeoTIFF bbox. Ambiguous
  mosaics or mismatched sources are rejected. Strict source validation has only
  been tested with doubles, not authenticated real Process responses.
- Primary/comparison metadata retain identity, dates, bbox, source validation,
  pixel hash, count, confidence and fixed-stretch VV/segmentation previews.
  Live empty masks produce empty geometry; rectangular bboxes use separate x/y
  contour scales. Holdout behavior is unchanged.
- Browser check renders distinct dated SAR previews using test graphics and
  explicit fixture responses. No persistence/change verdict is introduced.
- Real public catalog selection for bbox `[1.1, 103.7, 1.3, 103.9]`, primary
  window 2026-08-05 through 2026-08-15, selected product
  `d514853f-fc94-4799-b40e-c44e81412d0b`, acquired
  `2026-08-10T11:24:44.116503Z`. Its name ends in `_COG.SAFE`;
  compatibility with Process source identifiers is still unverified.
- ±30-day search returned 26 catalog rows and 8 eligible alternatives. First
  ranked was `a69cadd2-a4fb-4066-a7a8-b2e421ce77f3`, acquired
  `2026-08-22T11:24:44.593430Z`, matching relative orbit 171/ASCENDING.
  Saved names, footprints, attributes and selection results are in
  [live_catalog_verification.json](live_catalog_verification.json).
- Limits: 1000 catalog rows, conservative full coverage, at most two attempts,
  and 2048-pixel output cap. Orbit/sea state/resampling differences prevent
  treating visual differences as a validated oil classification.

## Task 3 — Sentinel-2 L2A visual supplement

**Implemented; local checks and real catalog no-match pass. RGB pair BLOCKED.**

- `include_optical` defaults false; window ±10 days (1–30), maximum tile cloud
  percentage 20 (0–100). Reuses OData, live bearer token and Process transport.
- Filters documented `S2MSI2A` product type, finite cloud cover, date and complete
  footprint coverage; ranks cloud cover before temporal proximity. No usable
  product returns `no_match`; catalog/render failure returns `unavailable`.
- Requests B04/B03/B02 with fixed 2.5x display gain and dataMask alpha. Validates
  returned product identity, dates, cloud metadata, dimensions and full validity.
  Attempts at most two candidates. No fusion, cloud removal or oil label.
- API/storage/UI retain separate SAR/optical dates, products, bbox, cloud
  percentage and RGB preview. Browser fixtures verify dated side-by-side images,
  image loading, status displays and optional request flags.
- Real OData search around the selected SAR time returned 5 L2A products; none
  met the default coverage/cloud/date criteria. The actual evidence function
  returned `no_match` without invoking Process. Recorded in the catalog artifact.
  This proves honest no-match behavior for this query, not a successful RGB fetch.
- Cloud percentage is tile-wide; even a selected scene need not have a cloud-free
  crop. Strict coverage/source checks can reject useful partial scenes. No real
  SAR/optical geographic or visual alignment check has been performed here.

## Task 4 — isolated DSen2-CR viability

**Previously STOPPED at Checkpoint 1. Not retried in this continuation.**

Prior bounded investigation read `docs/cloud_removal_scoping.md` and the official
repository instructions. Target stack: Python 3.7, TensorFlow GPU 1.15.0,
Keras 2.2.4, NumPy 1.17, h5py 2.10.0. No official weights were found locally.
No import, instantiation, weight load or inference pass was established.

Previous attempted commands and failures (historical, not new checks):

- `rtk proxy timeout 15s docker image inspect tensorflow/tensorflow:1.15.0-gpu-py3 --format '{{.Id}}'`
  failed with Docker socket permission denied; image availability was not proven.
- `rtk proxy timeout 15s git clone --depth 1 https://github.com/ameraner/dsen2-cr.git /tmp/pelagic-dsen2cr-viability`
  failed with `Could not resolve host: github.com`, exit 128. Network availability
  has since changed; this is no longer evidence of a current DNS blocker.
- Python 3.7/conda were absent from PATH. After a default-cache permission failure,
  `rtk proxy uv --cache-dir /tmp/pelagic-uv-cache python find 3.7 --no-python-downloads --offline`
  returned no interpreter, exit 2.

The hard-stop outcome concerns environment/bootstrap, not evidence the model is
broken. No TensorFlow/Keras dependencies were added to Pelagic. Keep deferred
while completing real-data verification of the higher-priority tracks.

## Verification in this continuation

- `rtk proxy /tmp/pelagic-verification-py314/bin/python -m pytest tests -q`:
  **30 passed, no skips**. Includes NetCDF selection, HTTP transport/persistence,
  optional sources, catalog ranking, source/pixel/bbox validation, failures and
  the actual v2 U-Net on a **synthetic calibrated 256×256 VV/VH raster**.
  Land masking and external fetching in API tests remain explicit doubles.
- The disposable dependency stack emitted nine warnings: one NumPy binary-size
  warning on NetCDF import and eight NumPy shape-deprecation warnings in
  NetCDF/tifffile adapters. Tests pass, but this is not a clean production-stack
  compatibility certification. No project dependency pins were changed.
- `rtk proxy python scripts/verify_supplementary_ui.py`: **passed** in Chromium
  at 1366×768. Checks available/not_configured/unavailable/skipped/no_match,
  wind values and absence when missing, separate acquisition dates, preview
  loading, default opt-out, toggled request flags, and no page exceptions.
  All API responses and graphics are explicitly test-only. No backend is used.
- In `frontend/`, `rtk proxy npm run build`: passed; `rtk proxy npm run lint`:
  exit 0 with one pre-existing unused `loading` variable warning in App.jsx.
- `rtk proxy python -m compileall -q src tests scripts/verify_supplementary_ui.py`:
  passed. Whitespace/diff review completed. No changes to database, checkpoints
  or frontend lockfile; this continuation adds verification and documentation.
- **External data verified:** public CDSE catalog metadata only, as described
  above. **Not verified:** authenticated Process/CDS/GFW, real satellite pixels,
  actual wind, external inference, actual SAR/optical visual correspondence.
- Reproduction steps and remaining acceptance checks:
  [supplementary_verification.md](supplementary_verification.md).

## Existing application and model state

FastAPI `src/api/main.py` → CDSE catalog/auth/Process → calibrated VV/VH TIFF →
`preprocess_for_prediction(..., already_calibrated=True)` → shared tiled U-Net →
land masking/contours → SQLite → React/Leaflet. Optional evidence follows primary
inference and persists with details. Four cached demos coexist with live fetch.

Default checkpoint is `model_real_v2_best.pt`; v1 is `model_real_best.pt`.
`best_model.pth` / `latest.pth` are synthetic/older artifacts, not substitutes.
Historical Singapore Strait/GFW verification and earlier live improvements are
in status_history.md; those authenticated checks were not repeated here.

## Evaluation and post-hoc lookalike findings

The 30-scene holdout spans six regions. Both real checkpoints have lookalike
IoU/Dice 0 on all 10 lookalike scenes. Mean false-positive scene area falls from
41.3% to 37.5% (148.4 to 134.5 km²). README's 10–45% range describes individual
scenes, not the average. Scheduler/pooled metrics also differ, so this is not
an isolated oversampling ablation. These historical evaluations were not rerun.

| Category | Baseline v2 outcomes | Opt-in filter + 0.975 gate |
|---|---|---|
| Oil | 8 correct, 2 partial | 6 correct, 1 partial, 3 missed |
| No oil | 7 correct, 3 false positives | unchanged |
| Lookalike | 0 correct, 10 false positives | 7 correct, 3 false positives |

Sources: checkpoint_comparison_summary, holdout_per_scene_results,
confusion_matrix_breakdown, region_coverage, phase4_confirm_* in docs.

- ESSD/PANGAEA JPG texture features did not transfer to raw-dB SAR; rejected.
  Do not retry without reading the Phase 1 findings in status_history.md.
- Own-domain classifier used the 1,200 oil / 685 lookalike pool, 1,857 usable
  candidates. Large-blob oversampling did not rescue holdout oil; the confidence
  gate helped validation but still loses real oil. Filter remains disabled by
  default, opt-in only on `/api/predict`.
- `candidate_region_features.py` is the shared feature source. Kaggle/local
  sklearn skew may require local refit from saved CSVs and train/val split.
- Zenodo training/holdout acquisitions still have no timestamps. This blocks
  matching historical scenes to supplementary sources; live catalog observations
  do have timestamps. Do not repeat the completed timestamp investigation.

## Remaining pre-existing issues and corrected status drift

1. **Resolved, merged into this branch (`fix/synthetic-preprocessing`,
   commit `b0ea525`).** `run_full_preprocessing()` previously treated synthetic
   linear input as dB and saturated it. It now delegates to
   `preprocess_for_prediction()`, matching `/api/predict`'s existing
   dB/linear branching. `tests/test_preprocess.py` (6 passed) covers both
   formulas, HWC/CHW handling, patch alignment and balanced sampling.
2. **Resolved, merged into this branch (`fix/no-fabricated-empty-masks`,
   commit `e8170db`).** `/api/predict` no longer manufactures a polygon for
   empty masks; it now preserves an empty GeoJSON `coordinates` array, matching
   the live path, and repairs legacy cached rows via the `+empty_geometry_v1`
   cache-recipe suffix. `tests/test_predict_empty_masks.py` (5 passed) covers
   this. Empty DB initialization still seeds mock records (unchanged).
3. The previous status incorrectly said the slick-size badge was hardcoded.
   Inspected App.jsx already uses `calcSlickAreaKm2` on GeoJSON coordinates;
   this predates the current continuation. No badge code was changed here.
4. An oversampling ablation requires a separate authorized GPU training run.