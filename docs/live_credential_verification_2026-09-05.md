# Live Credential Verification — Item 1: `_COG` Product-Name Matching

**Date:** 2026-09-05
**Scope:** `cdse_fetch.py` / `sentinel_process.py::verify_sources` only. No changes
to ERA5, multi-temporal, optical, or preview storage.
**Bottom line:** The feared `_COG`/non-COG name mismatch is **not real** — verified
against a real authenticated Process response. A *different* real blocker in the
same code path was found and fixed, and a full live `/api/live/fetch` now
completes end-to-end with a real detection.

---

## 1. Credentials

`scripts/validate_credentials.py` CDSE check: **PASS** — `CDSE_CLIENT_ID` /
`CDSE_CLIENT_SECRET` in `.env` obtained a real OAuth2 access token from
`identity.dataspace.copernicus.eu`. (One transient `Read timed out` on that
endpoint was seen mid-run and cleared on retry — network flake, not a credential
problem.)

`CDSAPI_KEY` (ERA5) remains absent — out of scope for this item.

## 2. Real authenticated Process API request

`POST https://sh.dataspace.copernicus.eu/api/v1/process`, bbox
`[1.1, 103.7, 1.3, 103.9]` (Singapore Strait, from
`docs/live_catalog_verification.json`), `find_best_product` selected:

| field | value |
|---|---|
| catalog id | `d514853f-fc94-4799-b40e-c44e81412d0b` |
| catalog `product['name']` | `S1D_IW_GRDH_1SDV_20260810T112444_20260810T112509_004062_007671_C8DC_COG.SAFE` |
| `product['start']` | `2026-08-10T11:24:44.116503Z` |
| `product['end']` | `2026-08-10T11:25:09.115202Z` |

### Captured VERBATIM from `userdata.json` (`updateOutputMetadata` → `scenes.orbits[].tiles`)

```json
{
  "tiles": [
    {
      "shId": 3943532,
      "date": "2026-08-10T11:24:44Z",
      "dataPath": "creo://eodata/Sentinel-1/SAR/IW_GRDH_1S-COG/2026/08/10/S1D_IW_GRDH_1SDV_20260810T112444_20260810T112509_004062_007671_C8DC_COG.SAFE",
      "sentinel1ProductId": "S1D_IW_GRDH_1SDV_20260810T112444_20260810T112509_004062_007671_C8DC_COG.SAFE"
    }
  ]
}
```

### Comparison performed by `verify_sources`

| side | value | after `.removesuffix('.SAFE')` |
|---|---|---|
| `product['name']` | `S1D_…_C8DC_COG.SAFE` | `S1D_…_C8DC_COG` |
| `tile['sentinel1ProductId']` | `S1D_…_C8DC_COG.SAFE` | `S1D_…_C8DC_COG` |

**→ MATCH.** The Process API reports the product under the **exact same
`_COG.SAFE` identifier** the OData catalog uses. There is no COG/non-COG naming
divergence. `verify_sources`' existing name comparison is correct for a `_COG`
primary. The tile `date` is truncated to whole seconds (`…T11:24:44Z`); the
existing `start.replace(microsecond=0) <= dt <= end` check already tolerates
that.

## 3. The real blocker (found while capturing the above)

With the **exact sub-second catalog interval** that `fetch_scene_geotiff` sends
verbatim — `timeRange.from = "2026-08-10T11:24:44.116503Z"` — the Process API
returned:

```
tiles = []          # -> verify_sources raises "Process response lacks source tiles"
pixels: all zero, validity: all zero
```

Every live fetch would fail at `verify_sources`, silently, for every request —
the exact symptom the task anticipated, via a different mechanism.

**Cause:** Sentinel Hub filters scenes on **whole-second** acquisition
timestamps (the returned tile `date` is `2026-08-10T11:24:44Z`). The catalog's
sub-second `start` (`…44.116503Z`), used as an inclusive `from` bound, sorts
*after* the scene's truncated time, so the scene is excluded from its own
fetch.

**Three real calls, same scene, same bbox:**

| `timeRange` | tiles | validity==1 | id returned |
|---|---|---|---|
| `from=…44.116503Z  to=…09.115202Z` (current code) | **0** | 0.000 | — |
| `from=…44Z  to=…10Z` (floor / ceil to whole second) | **1** | 1.000 | `S1D_…_C8DC_COG.SAFE` |
| `from=…44Z  to=…09Z` | **1** | 1.000 | `S1D_…_C8DC_COG.SAFE` |

## 4. Fix

`src/data/cdse_fetch.py::fetch_scene_geotiff` — floor `from` / ceil `to` to whole
seconds before building the Process body:

```python
start_dt, end_dt = acquisition_time(product['start']), acquisition_time(product['end'])
time_from = start_dt.replace(microsecond=0).strftime('%Y-%m-%dT%H:%M:%SZ')
time_to   = (end_dt.replace(microsecond=0) + timedelta(seconds=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
```

This is **not** widening the check to always pass:

- `verify_sources` still requires every returned tile's `sentinel1ProductId` to
  equal the selected product name, and its `date` to fall in
  `[start_to_second, end]`. A wrong-product response is still rejected.
- Sentinel-1 IW GRDH slices are ~25 s long and same-track revisits are days
  apart, so a ≤1 s widening cannot reach a neighbouring acquisition.

## 5. Regression tests (real captured values)

`tests/test_observations.py`:

- **`test_process_identity_real_cog_product`** (new) — feeds `verify_sources`
  the verbatim captured pair: product name
  `S1D_…_C8DC_COG.SAFE`, `sentinel1ProductId` `S1D_…_C8DC_COG.SAFE`, tile
  `date` `2026-08-10T11:24:44Z`, sub-second product `start`/`end`. Asserts
  `status == "verified"`, and that a genuinely different `_COG` product id
  (the 2026-08-22 F65B_COG scene) is still rejected. This is the `_COG`-name
  case the QC pass flagged as untested.
- **`test_fetch_validates_pixels_and_interval`** (updated) — now uses a
  sub-second acquisition interval and asserts the Process body carries the
  whole-second-widened `timeRange` (`22:00:00Z` … `22:00:26Z`).

## 6. Full test suite

```
CUDA_VISIBLE_DEVICES="" python -m pytest -q
61 passed, 1 skipped in 6.65s
```

(`1 skipped` is pre-existing. `global-land-mask`, listed in `requirements.txt`,
was missing from the environment and had to be installed —
`pip install --user --break-system-packages global-land-mask` — before
`tests/test_scale_y_alignment.py` would collect. No dependency pins changed.)

## 7. End-to-end live fetch (real, not cached/holdout)

`POST /api/live/fetch`, bbox `[1.1, 103.7, 1.3, 103.9]`, `date_from=2026-08-05`,
`date_to=2026-08-15`, supplements off (item 1 exercises the primary path;
ERA5/temporal/optical are separate prompt items).

**Result: `status: "OK"`.**

| field | value |
|---|---|
| detection id | `31` (persisted to `data/pelagic.db`, with `supplementary_json`) |
| scene_id | `live_ff555771a909` |
| product | `S1D_IW_GRDH_1SDV_20260810T112444_20260810T112509_004062_007671_C8DC_COG.SAFE` |
| `provenance.status` | `verified` (tile id + date matched) |
| acquisition (real, from scene) | `2026-08-10T11:24:44.116503Z` … `2026-08-10T11:25:09.115202Z` |
| bbox (real, from GeoTIFF) | `[1.1, 103.7, 1.3, 103.9]` |
| confidence_score | `0.7475` |
| predicted oil px (post land-mask) | `200,682` (raw `621,821`; `421,139` removed as land, `39,937` via OSM island refinement) |
| pixel SHA-256 | `41e98adcdb98ed6cf46220fe8d6c4b3c4549ead2e0eaf172df5e920eb43bd44b` |
| GFW attribution | `empty` — 0 candidate vessels within 10 km on 2026-08-10 (see status.md GFW section) |

`verify_sources` passed → tiled U-Net inference ran → a real detection with a
real mask polygon was returned and stored. Acquisition time and coordinates are
the actual fetched-scene values, not placeholders.

**Environment note:** the default checkpoint loads on the local 4 GB CUDA device,
but `run_tiled_inference` OOMs on a full 2048² live scene there. The run above
forced CPU inference (`CUDA_VISIBLE_DEVICES=""`, ~124 s). Resource limit, not a
code defect; out of scope for this check.

## 8. Files changed

| file | change |
|---|---|
| `src/data/cdse_fetch.py` | whole-second `timeRange` bounds in `fetch_scene_geotiff` (+ `acquisition_time` import) |
| `tests/test_observations.py` | new `test_process_identity_real_cog_product`; `test_fetch_validates_pixels_and_interval` uses sub-second interval + asserts widening |
| `docs/status.md` | "Live credential verification" subsection; environment/scope/next-action updates |
| `data/pelagic.db` | contains detection id 31 from the end-to-end run + `supplementary_json` column (revert with `git checkout data/pelagic.db` if not wanted) |

## Verification summary

| item | result |
|---|---|
| Real `sentinel1ProductId` vs `product['name']` | **MATCH** — both `S1D_…_C8DC_COG.SAFE` → `S1D_…_C8DC_COG` after `.SAFE` strip |
| COG/non-COG mismatch real? | **No** |
| Fix required? | Yes — but for a different bug: sub-second `timeRange.from` → empty Process tiles → `verify_sources` fails every request. Fixed by whole-second bounds. |
| Regression test added w/ real values | Yes — `test_process_identity_real_cog_product` |
| Full suite | **61 passed, 1 skipped** |
| End-to-end live fetch | **SUCCESS** — detection id 31, confidence 0.7475, real mask, `provenance.status = verified` |
