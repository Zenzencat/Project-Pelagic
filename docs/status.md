# Project Pelagic: Status & Handoff

Updated 2026-09-05 for the preview-storage handoff in
`prompt/pelagic_handoff.md` (the requested `prompts/` directory does not exist).
The branch was fast-forwarded to `origin/main` at `7c06a8b` during this session,
including the four upstream supplementary-evidence QC fixes. Earlier research
and historical verification remain in [status_history.md](status_history.md).

2026-09-06: `prompt/pelagic-credential-verification.md` §5 (GFW AIS attribution)
closed out — see "GFW AIS attribution" under Live credential verification.

## Current environment and scope

- User-supplied CDSE client credentials, GFW token and `CDSAPI_URL` are configured
  in the ignored local `.env`, matching `/home/papajittan/Downloads/env`, with
  permissions 0600. Values are not reproduced
  in documentation. `CDSAPI_KEY` is still absent; real ERA5 access also requires
  dataset-license acceptance. The CDSE OAuth2 client credentials were
  authenticated on 2026-09-05 (see "Live credential verification" below); a full
  real `/api/live/fetch` completed end-to-end.
- Python is 3.14.6. The current disposable environment is
  `/tmp/pelagic-preview-verify.wExiDS`, using system site packages plus xarray
  2026.7.0, netCDF4 1.7.4 and global-land-mask 1.0.0. Its resolver installed NumPy
  2.5.2. The older `/tmp/pelagic-verification-py314` no longer exists.
  No project dependency pins or lockfiles were changed.
- Real v2/v1 checkpoints and synthetic TIFFs are present. The requested
  `oil_00000`, `no_oil_00004` and `lookalike_00000` holdout TIFFs are absent.
- The preview-storage task moved previews out of new database writes; DSen2-CR
  is out of scope. CDSE Process product-name matching and credential
  verification were completed separately on 2026-09-05 (real authenticated
  Process response + full live fetch — see "Live credential verification").

## Preview storage

New live detections store PNGs in
`data/raw/live/previews/<detection_id>_<kind>.png`. The existing `data/raw/`
gitignore rule covers them. The two SAR preview encoders in `src/api/main.py`
and the optical encoder in `src/data/sentinel2_optical.py` keep their in-memory
data-URI interfaces. After the primary INSERT commits and the real detection ID
exists, `src/api/preview_storage.py` converts the known preview slots to files.

- Original SAR, original overlay, comparison SAR, comparison overlay and optical
  RGB use separate filenames. The same JSON field names now hold
  `/api/previews/<filename>` references.
- `GET /api/previews/{filename}` serves PNGs with generated-name validation and
  symlink/path checks. This follows the existing API namespace; there was no
  existing generated-asset serving route to reuse.
- `frontend/src/SupplementaryEvidence.jsx` renders SAR overlays and optical RGB.
  App.jsx supplies its existing backend base URL so relative references resolve
  to the backend. Old inline data URIs continue to render. Missing or evicted
  images show “Preview unavailable.”
- PNG writes are atomic. Invalid previews and disk failures remove the affected
  new inline field and preserve metadata with an explicit preview error.
  An outer guard preserves primary success if the storage boundary raises.
  The upstream serialization/SQLite degradation behavior remains in place.
- `MAX_PREVIEW_FILES = 500`, near the top of the storage module, configures
  oldest-first eviction after a batch, breaking mtime ties by filename.
  Only matching regular preview PNGs are candidates. Cleanup is best effort;
  filesystem errors can temporarily prevent the cap from being met.
- This is a file-count limit, not a total-byte limit or whole-detection retention
  policy. Historical references can return 404 after eviction. Raw scene TIFFs
  and segmentation masks retain their existing cache behavior.
- Historical SQLite rows are never migrated on read or startup. The actual
  tracked database in this checkout has no `supplementary_json` column yet;
  all application imports used disposable databases to avoid even a schema
  mutation of that file.

## Current verification

- Full suite: `rtk proxy /tmp/pelagic-preview-verify.wExiDS/bin/python -m pytest tests -q`:
  **61 passed, no skips**, including ten preview-storage tests and the real v2
  U-Net on a synthetic calibrated raster. External services are test doubles.
- The isolated dependency stack emitted 22 warnings: a NetCDF/NumPy binary-size
  warning and shape-deprecation warnings from NetCDF/tifffile. Passing tests
  do not establish production-stack compatibility.
- A deterministic HTTP live fixture with all five previews measured exact
  persisted UTF-8 JSON sizes of **2,741,694 → 1,535 bytes** for the same evidence
  before/after preview conversion. These are test graphics, not a real
  satellite detection's measured payload.
- All five preview GET bodies match their original PNG bytes. Tests verify
  old inline JSON remains byte-identical, invalid previews are removed,
  oldest-first eviction, unrelated-file preservation, path/symlink rejection,
  and HTTP primary success despite disk-write or unexpected storage failures.
- `rtk proxy npm run build` and `rtk proxy npm run lint` in `frontend/` pass;
  lint is clean with no warnings.
- `rtk proxy python scripts/verify_supplementary_ui.py` passes in Chromium at
  1366×768 against the local Vite server, covering available/not_configured/
  unavailable/skipped/no_match, new backend URLs, legacy inline images,
  preview 404 fallback, optional request flags and no page exceptions.
- Astra planned the change, Luna implemented it and Sol independently validated
  the implementation. No commit or push was made. A pre-pull WIP stash remains
  as a recovery copy.

The actual tracked database remains **10,395,648 bytes before and after**.
SHA-256 before and after:
`0cb14d4afcac644bbc144d59bf103550f6b792f3c234058a42207aa9f782b326`.
There was no historical cleanup or migration.

All six requested `POST /api/predict` combinations were attempted with actual
FastAPI routing and a disposable database: `oil_00000`, `no_oil_00004`,
`lookalike_00000`, each with `apply_lookalike_filter` false and true. All returned
HTTP 404 because the TIFFs are missing. Real holdout inference regression is
therefore **blocked**, not passed. `/api/predict` does not create supplementary
previews; the separate live HTTP fixture verifies the changed storage path.

Build/lint and Chromium checks use local code. Browser responses and graphics
are explicit fixtures, not satellite observations. No authenticated external
API verification is claimed. Reproduction steps and retention details are in
[supplementary_verification.md](supplementary_verification.md).

## Supplementary evidence capabilities and external gaps

### ERA5 wind

Implemented as an optional `include_era5` flag, default false, on
`POST /api/live/fetch`. `get_wind_evidence()` in
`src/analysis/era5_wind_check.py` requests hourly u10/v10, selects the nearest
UTC hour (including rollover), validates time/location/units/finite values and
computes `hypot(u10, v10)`. Signed and 0–360 longitude axes are supported.

The requested point is the fetched scene center at the real catalog acquisition
time. ERA5's 0.25° grid is roughly 28 km north-south, not a measurement at each
SAR pixel or slick polygon. Retrieval runs in a separate process with a
60-second budget; CDS queues can exceed it. Missing configuration reports
`not_configured`; failures report `unavailable`, without a numeric wind value.
No wind-derived oil verdict is produced.

Local NetCDF checks write real files containing synthetic test components;
CDS retrieval is mocked. Real ERA5 retrieval remains blocked on a CDS key and
license acceptance.

### Different-date Sentinel-1 comparison

Implemented as optional `include_temporal`, default false, with a default
±30-day window (allowed 1–90). Reuses the catalog/auth/Process transport and
`_analyze_live_scene` preprocessing, U-Net, land masking and contour pipeline.

Candidates require another UTC acquisition date, IW GRDH VV/VH, valid identity
and full footprint coverage. Matching known orbit track/direction rank first.
The upstream QC fix deduplicates sensing times so COG and non-COG product rows
do not count as separate revisits. At most two candidates are attempted.
Duplicate pixel hashes, ambiguous/mismatched sources and invalid bboxes are
rejected. Catalog searches are bounded to 1000 rows; output is capped at
2048 pixels. No match and unavailable are explicit outcomes.

Historical public catalog verification for bbox `[1.1, 103.7, 1.3, 103.9]`
selected primary `d514853f-fc94-4799-b40e-c44e81412d0b`, acquired
`2026-08-10T11:24:44.116503Z`. The saved search has 26 rows and eight eligible
product rows representing **four alternative acquisitions**, not eight
revisits. First-ranked alternative:
`a69cadd2-a4fb-4066-a7a8-b2e421ce77f3`,
`2026-08-22T11:24:44.593430Z`, relative orbit 171/ASCENDING.
The other dates are July 29, September 3 and July 17, 2026.
See [live_catalog_verification.json](live_catalog_verification.json).

Dates, source metadata, fixed-stretch SAR previews and segmentation are
presented as evidence, without persistence or oil classification.
Orbit/sea-state/resampling differences remain limitations.

### Live credential verification — real authenticated Process response (2026-09-05)

Checked against a **real authenticated Sentinel Hub Process response**, not
just static-scoped code. CDSE OAuth2 client credentials in `.env`
authenticated successfully (`validate_credentials.py` CDSE check: PASS).

Probe: `POST` to `sh.dataspace.copernicus.eu/api/v1/process` for bbox
`[1.1, 103.7, 1.3, 103.9]`, selected primary
`d514853f-fc94-4799-b40e-c44e81412d0b`
(`S1D_IW_GRDH_1SDV_20260810T112444_20260810T112509_004062_007671_C8DC_COG.SAFE`).

- **`_COG` product-name matching: no mismatch.** The Process API reports the
  tile under the identical identifier the catalog uses:
  `sentinel1ProductId` = `"S1D_..._C8DC_COG.SAFE"`. After the `.SAFE` strip both
  sides are `S1D_..._C8DC_COG`; `verify_sources` accepts it. The earlier concern
  that Process would return a non-COG identifier is disproven against a real
  response.
- **A different, real blocker was found and fixed.** With the sub-second catalog
  `product['start']` (`...T11:24:44.116503Z`) passed verbatim as the Process
  `timeRange.from`, the API returned **zero tiles / all-zero pixels** — every
  live fetch would fail at `verify_sources` ("lacks source tiles"). Sentinel Hub
  filters on whole-second acquisition timestamps (`date: "2026-08-10T11:24:44Z"`),
  so a sub-second `from` bound falls after the scene and excludes it.
  `fetch_scene_geotiff` now floors `from` / ceils `to` to whole seconds.
  `verify_sources`' name and date checks are unchanged and still reject a wrong
  product (neighbouring slices are ~25 s away). Regression test
  `tests/test_observations.py::test_process_identity_real_cog_product` uses the
  verbatim captured name/ID pair; `test_fetch_validates_pixels_and_interval` now
  uses a sub-second interval and asserts the widening. Suite: 61 passed, 1
  skipped.
- **Full live end-to-end (real, not cached): SUCCESS.** `POST /api/live/fetch`
  for the same bbox/date range returned `status: OK`, detection id 31
  (`live_ff555771a909`), persisted to `data/pelagic.db` with `supplementary_json`.
  `verify_sources` passed (`provenance.status = "verified"`, tile id matches),
  tiled U-Net inference ran (CPU — see note), confidence `0.7475`,
  200,682 predicted oil px after land masking (421,139 of 621,821 raw px removed
  as land, 39,937 via OSM island refinement), pixel SHA-256
  `41e98adc…43bd44b`. Acquisition timestamp and bbox in the response are the real
  fetched-scene values (`2026-08-10T11:24:44.116503Z` … `T11:25:09.115202Z`),
  not placeholders. GFW attribution ran (`ok`, 5 candidate vessels within 10 km)
  — since re-verified in its own pass, see "GFW AIS attribution" below.
- Note: the default checkpoint loads on a 4 GB CUDA device but `run_tiled_inference`
  OOMs on a full 2048² live scene there; the end-to-end run above forced CPU
  inference (`CUDA_VISIBLE_DEVICES=""`, ~124 s). This is an environment resource
  limit, not a code defect, and is out of scope for this check.
- Not verified here (separate prompt items): ERA5 wind, multi-temporal SAR pair,
  Sentinel-2 optical.

### GFW AIS attribution — real vessel-data verification (2026-09-06)

Closes `prompt/pelagic-credential-verification.md` §5: confirm the nearby-vessel
attribution returns real GFW AIS data, not the removed hardcoded `mock_vessels`
and not a fixture. `GFW_TOKEN` present in `.env`; `validate_credentials.py` GFW
check: **PASS** (real `GET /v3/datasets/public-global-presence:latest`, HTTP 200).

**Verified against a real external service — `POST /api/live/fetch`, full endpoint.**
Bbox `[1.10, 103.70, 1.30, 103.90]`, `date_from=2026-08-05`, `date_to=2026-08-15`,
`radius_km=10.0`, run three times against a locally-served backend (CPU inference).
All three returned `status: OK` on the real primary product
`S1D_IW_GRDH_1SDV_20260810T112444_20260810T112509_004062_007671_C8DC_COG.SAFE`
(real acquisition `2026-08-10T11:24:44.116503Z … T11:25:09.115202Z`), detection
ids 32/33/34, and exercised the real GFW path:

- The GFW step issues `POST https://gateway.api.globalfishingwatch.org/v3/4wings/report`
  with params `spatial-resolution=HIGH`, `temporal-resolution=DAILY`,
  `datasets[0]=public-global-presence:latest`, `date-range=2026-08-10,2026-08-11`,
  `format=JSON`, and a JSON body `{geojson: <10 km buffer polygon around 1.20,
  103.80>, group-by: MMSI}`.
- **What came back for the scene's own acquisition day (2026-08-10): honest
  `empty`.** HTTP 200 with `{"total":1,"entries":[{}]}` — the outer entry dict
  carries no dataset key, GFW's real shape for a box/date with zero recorded
  presence. `get_nearby_vessels` returned `status: "empty"`, zero vessels,
  detail "No AIS vessel presence found within 10.0km on 2026-08-10";
  `vessel_attribution_status` stored as `empty`. No vessel row was fabricated.
  (Runs 32 and 34; run 33 hit a transient `ConnectionResetError` from the GFW
  gateway and stored `error` — also honest, no fabricated data. The report
  endpoint intermittently resets connections; a retry succeeds.)
- 2026-08-10 appears to be a **gap in GFW's `public-global-presence` coverage**,
  not a real absence of ships in the Singapore Strait: the identical query for
  **2026-08-09** returns 5 real candidates (KST SUPER, PSA HULK CS04, KST KIJANG,
  PILOT GP01, FORCE) and for **2026-08-11** returns 5 real candidates (SC6336G,
  PILOT GP54, PILOT GP47, PILOT GP53, PILOT 12). The dataset's advertised
  `endDate` is `2026-09-02`. The 2026-09-05 detection-31 run recorded `ok` for
  this same date/box, so GFW's data for 2026-08-10 changed between then and now.

**Real vessel data is genuine GFW AIS, not a fixture.** The same
`get_nearby_vessels` code path, same bbox, for 2026-08-11 receives a real
`public-global-presence:v4.0` payload of **3080 rows** — real MMSIs, ship names,
flags (SGP/MYS/PHL/MLT/BLZ/MNG), vessel types and entry/exit timestamps
(e.g. MMSI 249082000 = CMA CGM ALHAMBRA, MLT-flag container ship; MMSI 563250600
= MAJESTIC HARMONY, SGP passenger ferry). The removed mock names
(`Petro Express`, `Nakhon Fishery 21`) appear nowhere. Known behavioural note,
not a fabrication and out of scope to change here: when the raw payload has
thousands of rows, the top-5-by-distance candidates all land in the query's
centre 0.01° grid cell and report an identical `distance_meters ≈ 0.3` with
`position_resolution_m ≈ 1572` — consistent with the documented grid-cell-centre
positioning (`src/analysis/gfw_client.py` docstring), though a spatially spread
sample would be more informative; flagged for a possible later refinement.

**Unit-tested locally (separate from the above):** `tests/test_gfw_client.py`
(new, 16 tests) covers `gfw_client.py`'s parsing, distance sort, MMSI dedup,
radius filter, `top_n` cap, the empty-entry and `null`-rows (Chonos-style) cases,
and every honest-failure path (missing token → `skipped_no_credentials`;
401/403 → `error`; 5xx → `error`; network exception → `error`; unparseable body
→ `error`) — all asserting no vessel is ever fabricated. GFW responses there are
synthetic fixtures built to the real JSON shape and labelled as such; they are
not real API output. Previously `gfw_client.py` had no direct tests
(`tests/conftest.py` stubs `get_nearby_vessels` for every API test).

### Sentinel-2 L2A optical supplement

Implemented as optional `include_optical`, default false. Default search window
is ±10 days (1–30), tile cloud threshold 20% (0–100). Reuses CDSE catalog/auth
and Process transport. Requires L2A identity, date, finite cloud metadata and
full footprint coverage; ranks cloud cover before temporal proximity.

Requests B04/B03/B02 with fixed 2.5× display gain and dataMask alpha, validates
returned sources/dates/bounds/full validity, and tries at most two candidates.
Separate optical/SAR dates and provenance remain visible. No fusion, cloud
removal or automated oil classification is added.

Historical public catalog search found five L2A products, none meeting the
default criteria; the evidence function returned `no_match` without Process
access. This proves that query's no-match behavior, not a real RGB fetch.
Whole-tile cloud percentage cannot guarantee a cloud-free crop, and strict
coverage checks may reject useful partial scenes. Actual SAR/optical visual
alignment has not been established.

### DSen2-CR

**Stopped for good by explicit user decision; do not retry without new
instruction.** No integration, import, model instantiation, weights load or
inference was established.

The investigated stack was Python 3.7, TensorFlow GPU 1.15.0, Keras 2.2.4,
NumPy 1.17 and h5py 2.10.0. Earlier DNS/Docker-access failures later recovered,
but the official Dockerfile's `nvidia/cuda:9.0-cudnn7-devel` base image was
absent from Docker Hub. An alternative TensorFlow image was pulled, but remained
untested beyond that before the user stopped work for lack of training/evaluation
resources. Scratch repository and image were removed; no TensorFlow/Keras
dependencies were added to Pelagic.
See [cloud_removal_scoping.md](cloud_removal_scoping.md) and historical reports.

## Existing application, model and QC state

FastAPI → CDSE catalog/auth/Process → calibrated VV/VH TIFF →
`preprocess_for_prediction(..., already_calibrated=True)` → tiled U-Net →
land mask/contours → SQLite → React/Leaflet. Optional evidence follows primary
inference. Four cached demo records coexist with live mode, but local holdout
TIFF availability must be checked before rerunning them.

Default checkpoint: `model_real_v2_best.pt`; v1: `model_real_best.pt`.
`best_model.pth` and `latest.pth` are synthetic/older artifacts.
The latest upstream QC fixes preserve independent latitude/longitude scales
through geolocation, land masking and contours, deduplicate comparison sensing
times, correct stale docs/Kaggle preprocessing, and prevent supplementary
persistence failures from failing primary detection.

Other resolved behavior remains:
`run_full_preprocessing()` delegates to prediction preprocessing for dB/linear
consistency; Kaggle's standalone duplicate must stay synchronized.
Empty predictions do not manufacture polygons, including legacy cache repair
through `+empty_geometry_v1`. The slick-size badge uses
`calcSlickAreaKm2()` on real detection geometry. Empty database initialization
still seeds explicit mock demo records.

## Evaluation and post-hoc lookalike findings

Historical evaluation was not rerun in this session. The 30-scene holdout spans
six regions. Both real checkpoints have lookalike IoU/Dice zero on all ten
lookalike scenes. Mean false-positive scene area falls from **41.3% to 37.5%**
(148.4 to 134.5 km²). README's 10–45% range describes individual scenes, not the
typical improvement. Scheduler/pooled metrics differ, so this is not an isolated
oversampling ablation.

| Category | Baseline v2 | Opt-in filter + 0.975 gate |
|---|---|---|
| Oil | 8 correct, 2 partial | 6 correct, 1 partial, 3 missed |
| No oil | 7 correct, 3 false positives | unchanged |
| Lookalike | 0 correct, 10 false positives | 7 correct, 3 false positives |

Sources: checkpoint_comparison_summary, holdout_per_scene_results,
confusion_matrix_breakdown, region_coverage and phase4_confirm_* in docs.

- The classifier trades lost real oil detections for fewer lookalike alarms and
  remains disabled by default; only `/api/predict` exposes its opt-in flag.
- ESSD/PANGAEA JPG-domain texture did not transfer to raw-dB SAR; rejected.
  Do not retrain on it without reading the historical Phase 1 finding.
- Own-domain features used 1,200 oil / 685 lookalike scenes and 1,857 usable
  candidates. Large-blob oversampling did not rescue holdout oil. Kaggle sklearn
  version skew can require local refit from saved features and train/val split.
- `candidate_region_features.py` remains the shared feature definition source.
- Zenodo training/holdout scenes have no acquisition timestamps. Matching them
  to external temporal/wind/optical sources remains blocked; do not repeat that
  completed investigation. Live catalog observations do have timestamps.
- An oversampling ablation needs a separately authorized GPU training run.

## Next action

Restore the three required holdout TIFFs to complete the handoff's six real
inference checks. The Process source-name check in
`prompt/pelagic-credential-verification.md` is **done** (real authenticated
response, `_COG` name matches, sub-second `timeRange` bug fixed, full live fetch
completed — see "Live credential verification"). §5 GFW AIS attribution is
**done** (real `/api/live/fetch` runs, real GFW 4Wings query, honest `empty` for
the scene's date, real vessels confirmed on adjacent dates, new
`tests/test_gfw_client.py` — see "GFW AIS attribution"). Still open from that
prompt: a real different-date SAR pair and Sentinel-2 optical behavior. ERA5
additionally needs `CDSAPI_KEY` and license acceptance.
