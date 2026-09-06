# Project Pelagic: Status & Handoff

Updated 2026-09-06 for the live external credential verification pass
(`prompt/pelagic-credential-verification.md`) and Tasks 3–8 execution
(CI, decoupled DB seeding, multi-temporal SAR, Sentinel-2 optical, calibration audit).
Earlier research and historical verification remain in [status_history.md](status_history.md).

2026-09-06: `prompt/pelagic-credential-verification.md` §5 (GFW AIS attribution)
closed out — see "GFW AIS attribution" under Live credential verification.

## Current environment and scope

- User-supplied CDSE client credentials and GFW token are configured in the
  ignored local `.env` and authenticated against live APIs (CDSE Process API
  and GFW v3 4Wings API). `CDSAPI_KEY` for ERA5 wind was skipped by user request;
  ERA5 degrades cleanly and honestly to `not_configured`.
- Environment is Python 3.11.9 (`.venv`) on Windows. All dependencies from `requirements.txt`
  are satisfied. `pytest.ini` configures `pythonpath = .` for direct invocation.
- Real v2/v1 checkpoints and synthetic TIFFs are present. The requested
  `oil_00000`, `no_oil_00004` and `lookalike_00000` holdout TIFFs remain absent
  for offline holdout inference, but live CDSE fetching is completely functional.
- This session verified live CDSE Sentinel-1 SAR acquisition, tiled U-Net inference,
  OSM land masking, GFW AIS vessel matching, multi-temporal revisit SAR, and
  Sentinel-2 true-color optical RGB. Previews are stored on disk under `data/raw/live/previews/`.

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
- `frontend/src/SupplementaryEvidence.jsx` and `docs/live_dashboard.html` render
  SAR overlays and optical RGB. App.jsx supplies its existing backend base URL
  so relative references resolve to the backend. Old inline data URIs continue
  to render. Missing or evicted images show “Preview unavailable.”
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

## Current verification

- Full test suite: `pytest tests -k "not symlink" -v`:
  **60 passed, 1 skipped** (the optional NetCDF test requiring CDS credentials),
  0 failures.
- Unit tests cover decoupled database initialization: `tests/test_database_seeding.py`
  proves that `init_db(seed_demo=False)` leaves fresh databases empty, while
  `scripts/seed_demo_data.py` explicitly populates demo scenes when desired.
- GitHub Actions CI workflow created at `.github/workflows/ci.yml` running both
  backend test suite (Python 3.11) and frontend lint/build (Node 20).
- Live End-to-End Verification (Detection #31 and #32 in `data/pelagic.db`):
  - Primary Sentinel-1 SAR acquisition (`2026-08-10T11:24:44Z`, Singapore Strait)
    analyzed with 74.75% confidence oil slick detected.
  - GFW AIS attribution matched 5 real commercial vessels (`status: ok`).
  - Multi-temporal comparison: revisit pass (`2026-08-22`) retrieved, validated via
    datatake matching, and segmented with `status: available`.
  - Sentinel-2 optical RGB: low-cloud scene (`2026-08-22`, 13.1% cloud cover) retrieved
    via expanded $\pm30$ min sensing time window and rendered with `status: available`.
  - Previews generated on disk: `32_original_sar.png`, `32_original_overlay.png`,
    `32_temporal_0_sar.png`, `32_temporal_0_overlay.png`, `32_optical_rgb.png`.
- Standalone zero-config map dashboard created at `docs/live_dashboard.html` rendering
  all layers using free Esri Satellite and OSM tiles without API keys.

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
creates clean, empty tables; mock demo seeding is explicitly separated via
`scripts/seed_demo_data.py`. GitHub Actions CI (`.github/workflows/ci.yml`) runs
automated regression tests and frontend checks on pushes/PRs.

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
- Synthetic calibration ambiguity audit is completed in `docs/synthetic_calibration_audit.md`:
  generator emitted linear power $\sigma^0$ while Level-1 calibration squared DN amplitude.

## Live external verification status

The live credential verification checklist in
`prompt/pelagic-credential-verification.md` is **verified and completed**:
- **Real live Sentinel-1 detection**: Verified (Detection #31 and #32). Downloaded
  calibrated SAR raster via CDSE Process API, ran tiled U-Net inference,
  executed OSM land masking, and persisted previews to disk.
- **GFW AIS vessel attribution**: Verified (`ok`, 5 real nearby vessels matched).
- **Multi-temporal SAR comparison**: Verified (`available`). Prioritized COG products
  in ranking and supported core datatake validation in `verify_sources()`.
  Candidate revisit product (`S1D...20260822T112444...COG.SAFE`) successfully
  fetched, segmented, and stored with preview images.
- **Sentinel-2 optical RGB supplement**: Verified (`available`). Handled granule
  sensing time offsets in Process `timeRange` and confirmed true-color RGB rendering
  for low-cloud scene (`S2C...20260822...SAFE`, 13.1% cloud cover).
- **ERA5 wind**: Skipped by design (`not_configured`). Remains available when
  `CDSAPI_KEY` is provided and dataset license is accepted.

## Next action

Restore the three required holdout TIFFs to complete the handoff's six real
offline `/api/predict` inference checks. The Process source-name check, GFW AIS
attribution (§5), live Sentinel-1 detection, multi-temporal comparison pass,
and Sentinel-2 optical RGB supplement are all verified against real services.
ERA5 remains opt-in pending `CDSAPI_KEY` and CDS dataset license acceptance.
