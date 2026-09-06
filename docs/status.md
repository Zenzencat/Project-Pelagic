# Project Pelagic: Status & Handoff

Updated 2026-09-06 for the live external credential verification pass
(`prompt/pelagic-credential-verification.md`) and Tasks 3–8 execution
(CI, decoupled DB seeding, multi-temporal SAR, Sentinel-2 optical, calibration audit).
Earlier research and historical verification remain in [status_history.md](status_history.md).

## Current environment and scope

- User-supplied CDSE client credentials and GFW token are configured in the
  ignored local `.env` and **authenticated and verified** against live APIs.
  `CDSAPI_KEY` for ERA5 wind was skipped by user request; ERA5 degrades cleanly
  and honestly to `not_configured`.
- Python is 3.11.9 on Windows (`.venv`).
  All dependencies from `requirements.txt` are satisfied.
  `pytest.ini` configures `pythonpath = .` for direct invocation.
- Real v2/v1 checkpoints and synthetic TIFFs are present. The requested
  `oil_00000`, `no_oil_00004` and `lookalike_00000` holdout TIFFs remain absent
  for offline holdout inference, but live CDSE fetching is completely functional.
- This session verified live CDSE Sentinel-1 SAR acquisition, tiled U-Net inference,
  OSM land masking, real GFW AIS vessel matching, multi-temporal revisit SAR, and
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

This historical catalog result is not an authenticated pixel-fetch test.
The primary name ends in `_COG.SAFE`; real Process source-identifier
compatibility remains unverified. Dates, source metadata, fixed-stretch SAR
previews and segmentation are presented as evidence, without persistence or
oil classification. Orbit/sea-state/resampling differences remain limitations.

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
offline `/api/predict` inference checks. The live satellite, comparison, optical,
and vessel pipelines are completely verified and working. ERA5 remains opt-in
pending `CDSAPI_KEY` and CDS dataset license acceptance.
