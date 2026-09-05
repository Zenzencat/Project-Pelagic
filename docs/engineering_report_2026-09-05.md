# Project Pelagic engineering report

Date: 2026-09-05

This report records the work completed during the supplementary-evidence and
verification iterations, what was validated locally, and what remains blocked
or intentionally skipped. `docs/status.md` remains the authoritative project
handoff; this report is a summary for review and decision-making.

## Completed implementation work

### Supplementary live evidence

- Added optional ERA5 10 m wind evidence for live Sentinel-1 observations.
  The path validates the UTC hour, location, units, finite values and grid
  distance before computing `hypot(u10, v10)`. Missing configuration and failed
  retrievals produce explicit statuses without a fabricated measurement.
- Added optional different-date Sentinel-1 comparison. It reuses the existing
  catalog, authentication, Process API, preprocessing and inference path.
  Candidate products must be valid, genuinely different dates, compatible IW
  GRD VV/VH observations and full-footprint matches. Duplicate payloads are
  rejected.
- Added optional Sentinel-2 L2A true-color evidence with cloud filtering,
  acquisition-date preservation, source validation, full-coverage checks and
  explicit `no_match`/`unavailable` states.
- Added API persistence and React evidence panels for wind, SAR comparison and
  optical previews. No automatic oil/lookalike or persistence verdict was added.

### Data-integrity fixes prepared as pull requests

- [PR #2](https://github.com/Zenzencat/Project-Pelagic/pull/2) removes the fake
  square polygon returned by `/api/predict` when no valid contour exists. It
  returns empty GeoJSON coordinates, repairs legacy cached rows when reanalyzed,
  clears stale vessel rows, and guards the map against invalid empty bounds.
- [PR #3](https://github.com/Zenzencat/Project-Pelagic/pull/3) fixes saturated
  synthetic preprocessing by routing `run_full_preprocessing()` through the
  existing scale-aware preprocessing helper. It preserves dB behavior and
  patch/sampling contracts.
- Both pull requests are open against `main`; neither has been merged as of
  this report.

## Local verification completed

- Full supplementary suite: **30 passed**. It covers ERA5 NetCDF selection,
  HTTP serialization/persistence, optional-source failures, temporal and
  optical filtering, source and pixel validation, and synthetic inference
  plumbing.
- ERA5-specific and live-evidence checks passed in a disposable environment.
  The NetCDF fixture uses synthetic components; it is not a real weather record.
- Browser fixture checks passed in Chromium for available, skipped,
  unavailable, no-match and positive/zero-confidence empty-geometry states.
- Frontend Vite build passed. Frontend lint exited successfully with one
  pre-existing unused `loading` variable warning.
- Python compilation and `git diff --check` passed.
- PR #2 regression suite: **5 passed**. It covers empty masks, one-pixel
  geometry, legacy cache repair, land/filter suppression and real contours.
- PR #3 regression suite: **6 passed**. It covers independent linear/dB
  formulas, HWC/CHW input, patch/mask alignment, balanced sampling and a
  temporary synthetic DataLoader integration.
- Public CDSE OData catalog verification found one real primary Sentinel-1
  catalog product and eight eligible different-date candidate *rows* for the
  tested Singapore Strait query. Those eight rows are four distinct
  acquisitions listed twice each (COG and non-COG); see `docs/status.md`
  Task 2. A nearby Sentinel-2 query returned an honest
  `no_match` under the configured coverage/cloud criteria.

## External verification status

No real satellite pixels, authenticated Process API response, ERA5 record,
GFW vessel response or external U-Net inference was verified in this session.
Public catalog metadata is not equivalent to authenticated image retrieval.

The following configuration is absent from the current environment:

- `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`
- `CDSAPI_URL` and `CDSAPI_KEY`
- `GFW_TOKEN`
- `.env` and `~/.cdsapirc`

The first remaining end-to-end acceptance check is a real live Sentinel-1
fetch/inference with ERA5 enabled. It requires CDSE credentials, CDS
credentials and ERA5 dataset-license acceptance. Real temporal pixels and a
real SAR/optical visual pair require CDSE credentials as well.

## Blocked or skipped work

- Real ERA5 retrieval is blocked by missing CDS credentials and license
  acceptance.
- Real Sentinel-1 Process image retrieval and live inference are blocked by
  missing CDSE credentials.
- Real different-date Sentinel-1 pixel comparison is therefore unverified.
- Real Sentinel-2 RGB rendering and geographic visual alignment are therefore
  unverified.
- GFW AIS attribution was not externally rechecked because `GFW_TOKEN` is
  absent.
- DSen2-CR remains stopped at its isolated viability checkpoint. The required
  legacy environment, official source/weights and runnable import path were
  not established; no dependency was added to Pelagic.
- No model retraining, GPU oversampling ablation or holdout metric rerun was
  performed.
- No automated persistence classification, oil confirmation, cloud removal or
  SAR-optical fusion was introduced.
- Historical Zenodo scenes still have no acquisition timestamps, so they cannot
  be matched retrospectively to ERA5 or optical observations.

## Repository state and next decisions

- The original checkout contains the existing supplementary-evidence changes as
  uncommitted work. They were preserved and not mixed into the two focused PRs.
- PR #2 and PR #3 are ready for review and merge.
- After merging, the highest-value next step is to configure CDSE/CDS
  credentials and run one real all-supplements live request, retaining the
  returned source, timestamp, footprint and preview metadata as evidence.
- If credentials will remain unavailable, the next credential-free options are
  explicit demo-database seeding, GitHub CI for the regression suites, or a
  separate audit of the remaining synthetic calibration convention.
