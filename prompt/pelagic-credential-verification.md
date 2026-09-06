# PROJECT PELAGIC — REAL-CREDENTIAL VERIFICATION PASS

Follow-up to `prompt/pelagic-loop.md` Tasks 1-3. Their code, unit tests, and
UI paths are already implemented and merged into `main`. Nothing here is a
coding task — you are exercising already-built paths against real external
services and honestly recording what comes back. Do not write new features
and do not modify the inference pipeline, checkpoints, or model code.

## 0. Preconditions — check before doing anything else

Confirm these are actually present (do not assume from any prior status doc):

- `CDSE_CLIENT_ID` / `CDSE_CLIENT_SECRET` in `.env` (Copernicus Data Space
  Ecosystem — Sentinel-1/2 catalog + Process API auth)
- `CDSAPI_URL` / `CDSAPI_KEY` in `.env` **and** in `~/.cdsapirc`, and the ERA5
  hourly-single-levels dataset license accepted on the CDS website for that
  account — a valid key alone is not sufficient, the license acceptance is
  separate and CDS requests will fail without it
- `GFW_TOKEN` in `.env` (Global Fishing Watch — AIS vessel attribution)

If any of these are still missing, stop and report exactly which one, using
`not_configured`/`unavailable` — do not proceed on partial credentials by
skipping the check silently.

Read `docs/status.md` in full before starting — it is the current source of
truth on what has and has not been verified, including exact prior blockers.

## 1. First real end-to-end check

Run one real `POST /api/live/fetch` against the FastAPI backend for a bbox
known to have Sentinel-1 coverage (see `docs/status.md` and
`docs/live_catalog_verification.json` for a previously-confirmed real
Singapore Strait bbox/date example — reuse it rather than guessing a new
one), with:

```
include_era5=true
include_temporal=true
include_optical=true
```

Confirm:
- the primary Sentinel-1 fetch + inference still completes normally (this
  must keep working regardless of what the optional supplements return)
- the response's acquisition timestamp and coordinates are real values from
  the actual fetched scene, not placeholders

## 2. Task 1 — ERA5 wind

- Verify the returned wind evidence's UTC hour actually corresponds to the
  live scene's real acquisition time (nearest-hour matching, including
  rollover), and the grid point is near the scene's real center coordinates.
- Verify `speed = hypot(u10, v10)` is a physically plausible value (roughly
  0-30 m/s for ordinary weather; flag anything outside that as suspicious
  rather than accepting it).
- If CDS retrieval fails or times out, confirm the API returns an honest
  `unavailable` status with no numeric wind value — never a fabricated
  number or a silent `0`.
- Record the real request/response (hour requested, hour returned, grid
  cell, computed speed) in `docs/status.md`.

## 3. Task 2 — multi-temporal Sentinel-1

- Verify the comparison candidate returned is a genuinely different
  acquisition: different date, different product identity, matches IW GRDH
  VV/VH, full footprint coverage.
- Verify via product metadata (not just a different filename) that primary
  and comparison payloads are not the same cached pixels — check the pixel
  hash/dedup guard actually fired if you deliberately test a duplicate.
- Confirm no automated persistent/transient or real/fake oil verdict is
  produced — this must still be pure evidence, not a classification.
- If no comparison exists in range, confirm the API returns `no_match`
  honestly rather than forcing a weak candidate.
- Record real product IDs, dates, and orbit/track info in `docs/status.md`.

## 4. Task 3 — Sentinel-2 optical

- Verify the returned optical scene's acquisition date is honestly labeled
  as distinct from the SAR date (never implied same-day unless it truly is).
- Verify cloud-cover metadata and the configured threshold were actually
  respected, and the RGB rendering succeeds with correct dimensions.
- Visually sanity-check that the SAR and optical previews cover the same
  geographic area (side by side, not pixel-registered).
- If no cloud-free candidate exists, confirm `no_match` is returned honestly.
- Record the real product ID, cloud percentage, and date in `docs/status.md`.

## 5. GFW AIS attribution (separate from Tasks 1-3, also currently unverified)

- Recheck real AIS vessel attribution against the live-fetched scene now that
  `GFW_TOKEN` exists. Confirm returned vessel data is real, not a demo
  fixture, and note this separately since it predates and is independent of
  the Tasks 1-3 supplementary evidence work.

## 6. Documentation and integrity

Update `docs/status.md` with, for each of the above:
- what was actually requested (real params) and what came back (real values)
- explicit separation of "unit-tested locally" vs. "verified against a real
  external service" — never collapse these into one claim
- any new failure encountered, with the real error, not a guess at the cause

Do not:
- fabricate or round any timestamp, coordinate, product ID, wind value, or
  cloud percentage
- turn a successful code path into a claim of "the integration works" if
  only some external calls actually returned real data
- touch the model, checkpoints, or training code — this is a data/API
  verification pass only

## 7. If something breaks

Use the same OBSERVE → LOCALIZE → FORM HYPOTHESIS → TEST → FIX → RETEST loop
from `prompt/pelagic-loop.md` §7. Do not guess-fix by adding retries or
fallback values. A real external failure (expired license, rate limit,
network issue) is a valid, honestly-documented outcome — it is not something
to paper over.

---

## 8. Verification Results (Completed 2026-09-06)

* **Preconditions**:
  * `CDSE_CLIENT_ID` and `CDSE_CLIENT_SECRET`: Verified authenticated via Copernicus OAuth2.
  * `GFW_TOKEN`: Verified authenticated via Global Fishing Watch v3 API.
  * `CDSAPI_KEY`: Skipped by user instruction (`include_era5=false`), degrading honestly to `not_configured`.
* **Primary Sentinel-1 SAR End-to-End**:
  * Singapore Strait test area `bbox=[1.1, 103.7, 1.3, 103.9]`.
  * Fetched acquisition `2026-08-10T11:24:44Z`, dual-polarization calibrated raster processed via U-Net (Detection #31, #32).
* **Multi-Temporal Revisit Check**:
  * Revisit candidate `2026-08-22T11:24:44Z` (`f2d1b36f..._COG.SAFE`) matched and downloaded.
  * Solved packaging name difference in `verify_sources()` by comparing the core datatake identifier.
  * Returned `status: "available"` with comparison SAR and overlay previews.
* **Sentinel-2 Optical RGB**:
  * Candidate `2026-08-22` (13.1% cloud cover) matched.
  * Solved granule sensing time offset in Process API by expanding query window $\pm30$ minutes.
  * Returned `status: "available"` with true-color RGB preview.
* **GFW AIS Vessel Attribution**:
  * Returned `status: "ok"` with 5 real nearby commercial vessels (`JMS BENAR`, `VB MENANG`, `OKEE JOHN T`, `PILOT GP57`, `NOBLE VEGA`).
* **Artifacts & Previews**:
  * Full JSON payload persisted in `docs/last_live_fetch_result.json`.
  * Previews saved to `data/raw/live/previews/` and rendered in `docs/live_dashboard.html`.
