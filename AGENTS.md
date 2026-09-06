# AGENTS.md — Context for AI Models Working in This Repo

This file orients any AI coding assistant (Claude, GPT, Gemini, etc.) picking up
this repo cold. It is a map, not the source of truth — **`docs/status.md` is
the authoritative, up-to-date account of what's resolved, what's broken, and
what's blocked.** Read that file before making claims about project state.

## What this project is

Project Pelagic (SWU Prasarnmit, AI Engineering Track final project) is a
proof-of-concept SAR maritime oil-slick detector: Sentinel-1 C-band radar
imagery (VV/VH) → U-Net semantic segmentation → detected slick polygons +
nearby AIS vessels rendered on a Leaflet map dashboard.

- **Backend**: FastAPI (`src/api/main.py`), SQLite (`data/pelagic.db`)
- **Model**: U-Net (`src/models/unet.py`), PyTorch, trained on Kaggle (T4 GPU)
- **Frontend**: Vite + React + Leaflet (`frontend/`)
- **Training data**: 3-part Sentinel-1 SAR Oil Spill Dataset (Trujillo-Acatitla
  et al.), Zenodo record IDs `8346860` / `8253899` / `13761290`

## Where to look first

| Question | File |
|---|---|
| "What's currently broken / open / blocked?" | [`docs/status.md`](docs/status.md) — rewritten in full during a session-5 health check specifically because earlier versions had drifted from reality. Trust this over any other doc's framing. |
| "How do I run this locally?" | [`README.md`](README.md) — Quick Start Guide section |
| "What's the model architecture / training recipe?" | [`docs/ai_technique.md`](docs/ai_technique.md), `kaggle_kernel/train_kaggle.py` |
| "What's the system architecture?" | [`docs/system_architecture.md`](docs/system_architecture.md) |
| "What's the DB schema?" | [`docs/db_schema.md`](docs/db_schema.md) |
| "What are the eval numbers?" | `docs/checkpoint_comparison_summary.{json,csv}`, `docs/holdout_per_scene_results.{json,csv}`, `docs/confusion_matrix_breakdown.{json,csv}`, `docs/region_coverage.{json,csv}` |
| "What's the post-hoc lookalike filter and does it work?" | [`docs/status.md`](docs/status.md) Phases 1–4; `docs/phase4_confirm_*.{json,csv}` for the final holdout numbers; `src/analysis/lookalike_filter.py` for the code |

## Known trap: README vs. status.md disagree on one number

README's "Lookalike False Alarm Suppression Analysis" section frames the v1→v2
improvement as a "10% to 45%" per-scene pixel reduction. `docs/status.md`
(session 5, more recent and more carefully verified) corrects this: the
*average* false-positive area dropped from 41.3% to 37.5% of scene area —
real, but more modest than the README's framing implies. The 10-45% range in
the README describes spread across individual scenes, not the typical
improvement. **Prefer status.md's framing** if asked about this metric, and
flag the discrepancy rather than silently repeating either figure.

## Structural notes worth knowing before editing

- **`src/data/preprocess.py::run_full_preprocessing()` now delegates
  normalization to `preprocess_for_prediction()`**, so synthetic linear-scale
  inputs take the same calibration path as `/api/predict` while existing dB
  inputs retain their prior behavior. Deterministic regression tests cover the
  independent numerical formulas, patch alignment, balanced sampling, random
  state restoration, and synthetic dataloader integration. See status.md
  Resolution #3 and the resolved Open Issue #1 entry.
- **`kaggle_kernel/train_kaggle.py` cannot import `src/data/`** (Kaggle
  kernels must be single-file), so its preprocessing and patch-sampling logic
  is a manual duplicate of `src/data/dataset.py` / `preprocess.py`. Both files
  carry an explicit cross-reference banner — if you change sampling rates,
  hard-fail behavior, or dB-scale handling in one, mirror it in the other or
  they will drift. Re-synced 2026-09-05 for the `preprocess_for_prediction()`
  delegation above; verified byte-identical to the repo version on real
  holdout dB data, so `checkpoints/model_real_v2_best.pt` still corresponds to
  what this kernel produces.
- **`frontend/src/App.jsx`'s slick-size badge ("ขนาดคราบ") is real.** It goes
  through `calcSlickAreaKm2()` on the detection's own GeoJSON coordinates. An
  earlier version of this file claimed it was a hardcoded `8.5`/`14.5` km²
  mock; that mock was replaced and the note was left stale. See status.md's
  "Remaining pre-existing issues and corrected status drift" item 3.
- **Live mode exists alongside the 4 cached demo scenes**: `POST /api/live/fetch`
  fetches a real Sentinel-1 scene from Copernicus Data Space Ecosystem (via
  the Sentinel Hub Process API, not a raw SAFE download — see `docs/status.md`
  Recent Resolution #9 for why) and attributes it to real nearby AIS vessels
  via the Global Fishing Watch API (`src/analysis/gfw_client.py`). This
  replaced the old hardcoded `mock_vessels` in `main.py`. `CDSE_CLIENT_ID` /
  `CDSE_CLIENT_SECRET`, `GFW_TOKEN`, and `CDSAPI_KEY` were all verified authenticated
  and functional in 2026-09-06 live end-to-end tests (Detections #31, #32, and #36).
  ERA5 10m wind vector retrieval returned 4.76 m/s at 11:00 UTC for Detection #36.
- **Checkpoints**: `checkpoints/model_real_best.pt` (v1) and
  `checkpoints/model_real_v2_best.pt` (v2, current default) are the real
  trained models; `best_model.pth` / `latest.pth` are older/synthetic-run
  artifacts — don't assume they're interchangeable.
- **Lookalike detection is the unsolved core problem**: SAR backscatter
  reduction from lookalikes (wind shadows, biogenic films) is
  indistinguishable from real oil at the pixel level. Both checkpoints score
  `0.0000` IoU/Dice on all 10 lookalike holdout scenes — v2's oversampling
  reduces false-positive area but doesn't come close to solving it. The
  supplementary-evidence sources below (ERA5 wind, different-date SAR,
  Sentinel-2 optical) are now built and wired into the **live** path, but they
  cannot be applied to the holdout/training scenes at all: **this dataset has
  no acquisition timestamps anywhere**, confirmed against the Zenodo deposits
  directly. SAR-optical fusion via DSen2-CR was stopped for good by explicit
  decision. See status.md for full detail — don't re-investigate this from
  scratch.
- **Supplementary evidence on `POST /api/live/fetch`** (added 2026-09-05,
  Tasks 1–3 of `prompt/pelagic-loop.md`): three opt-in sources, all defaulting
  to `false`, all surfacing evidence only — **none of them produces an oil,
  lookalike, or persistence verdict**, and that is deliberate. Keep it that way.
  - `include_era5` → `src/analysis/era5_wind_check.py::get_wind_evidence()`,
    ERA5 10 m u/v on a 0.25° grid via CDS. Needs `CDSAPI_URL`/`CDSAPI_KEY`
    (or `~/.cdsapirc`) *plus* dataset-license acceptance.
  - `include_temporal` → `_temporal_evidence()` in `main.py`, using
    `src/analysis/sentinel1_revisit_check.py::select_comparison_products()`.
    Reuses the primary fetch/inference path via `_analyze_live_scene()`. Verified
    live on 2026-08-22 revisit pass (`f2d1b36f..._COG.SAFE`).
  - `include_optical` → `src/data/sentinel2_optical.py`, Sentinel-2 L2A
    true-colour RGB with a cloud threshold. Verified live on 2026-08-22 low-cloud
    pass (13.1% cloud cover).
  - Shared plumbing lives in `src/data/observation_catalog.py` (public OData
    search + footprint validation) and `src/data/sentinel_process.py` (Process
    transport + returned-source verification). Packaging normalization (`verify_sources()`)
    and granule sensing time windowing ($\pm30$ min) are verified.
  - Results persist in `detections.supplementary_json` and come back on
    `GET /api/detections/{id}` as `supplementary`; legacy rows are `NULL` and
    surface as `{}`. Rendered by `frontend/src/SupplementaryEvidence.jsx` and
    `docs/live_dashboard.html`. New live PNG previews are saved by `src/api/preview_storage.py`
    under ignored `data/raw/live/previews/` and served at `/api/previews/{filename}`.
    Existing inline preview rows remain unchanged. `MAX_PREVIEW_FILES` controls
    oldest-first eviction; an evicted preview returns 404 and the UI reports it
    unavailable.
  - **Every status here is honest by design** (`available` / `no_match` /
    `unavailable` / `not_configured` / `skipped`). Never substitute a
    placeholder number for a missing measurement.
  - **Live credential verification checklist**: Completed for CDSE Sentinel-1,
    Sentinel-2 optical, and GFW AIS. Recorded in `docs/status.md` and `docs/last_live_fetch_result.json`.
- **Database initialization decoupled**: `init_db(seed_demo=False)` by default
  creates clean empty tables. Explicit demo seeding is handled by `scripts/seed_demo_data.py`.
- **Automated CI**: `.github/workflows/ci.yml` runs full backend pytest suite (Python 3.11)
  and frontend lint/build (Node 20). `pytest.ini` configures `pythonpath = .` so bare `pytest` works.
- **Synthetic calibration audit**: Fully documented in `docs/synthetic_calibration_audit.md`.
- **Post-hoc lookalike classifier (Phases 1–4, `src/analysis/lookalike_filter.py`)**:
  a downstream Gradient Boosting classifier + U-Net-confidence gate that judges
  each candidate detection oil-vs-lookalike. On the 30-scene holdout it fixes
  the lookalike bucket (0→7 correct) but costs real oil detections (10→7
  found) — **a net-negative trade for a spill detector, so it is NOT enabled
  by default**. It exists only as an opt-in `apply_lookalike_filter` flag on
  `POST /api/predict` (default `False` = byte-identical to before it existed).
  An ESSD/PANGAEA external dataset was tried as a training source and
  **explicitly rejected** — its texture signal is JPG-domain-specific and does
  not transfer to this project's raw-dB SAR (status.md Phase 1). Don't retry
  training on ESSD without reading that finding.
- **`src/analysis/candidate_region_features.py`** is the single source of truth
  for the GLCM/edge/shape feature definitions, shared by the feasibility-check
  script, the ESSD script, and the live filter — mirror any change across all
  callers (same discipline as `dataset.py` ↔ `train_kaggle.py`).
- **`.joblib` classifiers trained on Kaggle may not unpickle locally**
  (`ModuleNotFoundError: No module named '_loss'`) — an sklearn version skew,
  not a corrupt file. Refit locally from
  `output/lookalike_classifier_own_domain_features*.csv` + the saved train/val
  split; reproduces the Kaggle AUC to within ~0.001 (status.md Phase 3).

## Conventions

- Session handoffs go in `docs/status.md`, rewritten in full (not
  incrementally patched) whenever drift between it and reality is found —
  keep doing that rather than appending an ever-growing changelog.
- Claims in these docs are generally backed by an actual verification step
  (a real test run, a real API check, reading actual source instead of a
  README) rather than assumption — match that bar when adding new claims.
