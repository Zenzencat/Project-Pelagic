# Live supplementary evidence: verification guide

The 2026-09-05 continuation verified local NetCDF, HTTP and browser paths and
retrieved real public catalog metadata. Satellite pixels and ERA5 records remain
unverified because credentials are absent. Read [status.md](status.md) for current
blockers and measured results; catalog evidence is in
[live_catalog_verification.json](live_catalog_verification.json).
The original investigation details remain in [status_history.md](status_history.md).

From the repository root, run offline checks:

```bash
rtk proxy python -m pytest tests -q
rtk proxy python -m compileall -q src tests
```

Tests use disposable SQLite databases and explicitly mocked service responses.
One test runs the actual v2 U-Net on a synthetic calibrated raster; that is not
a satellite-data verification. The NetCDF test skips if xarray is unavailable;
it passed in the disposable environment with the optional dependencies installed.
An HTTP test uses FastAPI TestClient and real missing-CDS handling, with satellite
fetch/inference doubles and a disposable database.

To repeat the browser check, start the frontend with `npm run dev`, then run:

```bash
rtk proxy python scripts/verify_supplementary_ui.py --url http://127.0.0.1:5173
```

Requires Playwright and its Chromium browser. The script intercepts API responses
with explicit fixtures and uses test graphics; it never calls the backend or
satellite services. It verifies wind/status displays, acquisition dates, image
loading and optional request flags at 1366×768. This does not establish scientific
validity or the geographic correspondence of real previews.

The current disposable environment can run the complete Python suite with:

```bash
rtk proxy /tmp/pelagic-verification-py314/bin/python -m pytest tests -q
```

It produced 30 passes, no skips, with dependency warnings described in status.md.
This `/tmp` environment is disposable and is not assumed to survive a restart.

On a configured machine, install existing requirements in the project environment,
populate the existing `.env.example` variables and start the backend/frontend per
README. CDS access also requires the ERA5 single-level dataset license acceptance.
Do not install DSen2-CR dependencies in that environment.

In Live Fetch, enter a real area and date window. Expand **Supplementary evidence**
and select the desired sources. Equivalent request fields are:

| Field | Default | Meaning |
|---|---|---|
| `include_era5` | `false` | Scene-center hourly wind evidence |
| `include_temporal` | `false` | One different-date Sentinel-1 comparison |
| `temporal_window_days` | `30` | Symmetric search window, 1–90 days |
| `include_optical` | `false` | Sentinel-2 L2A true-color supplement |
| `optical_window_days` | `10` | Symmetric search window, 1–30 days |
| `optical_max_cloud_pct` | `20` | Maximum tile-wide cloud percentage, 0–100 |

The new fields are additive to the existing bbox/date/radius request. Results
are in `detection.supplementary`; GET detection details returns the same persisted
evidence. `original` contains the original SAR observation. `era5`, `temporal`
and `optical` each carry their own status. Primary `status: OK` alone does **not**
prove that any supplementary source succeeded.

Verify one real returned pair before claiming integration success:

1. Confirm the original catalog product ID/name/start/end agree with Process
   source tiles and GeoTIFF bounds. Retain the actual response as evidence.
2. For available wind, compare requested coordinates to the original bbox center,
   acquisition time to the real SAR record, and ERA5 valid time to its nearest UTC
   hour. Recompute `hypot(u10_ms, v10_ms)`; investigate implausible values instead
   of labeling them as oil or substituting another value. Inspect the returned
   grid location and coarse-resolution disclosure.
3. For temporal evidence, compare actual product names, UTC dates and pixel hashes.
   Inspect both SAR previews; they must cover the same region with genuinely
   different acquisition metadata. A failed or no-match status is an honest result,
   but does not satisfy a successful real second-pass verification.
4. For optical evidence, check L2A identity, cloud metadata, separate optical date,
   source tiles and shared bbox. Open the evidence panel and visually check the
   coastlines/landmarks against SAR. Tile-wide cloud percentage cannot prove a
   cloud-free crop; no fusion or cloud removal is performed.
5. Reload the detection and verify all dates/statuses/previews persist. Toggle
   optional sources off in a new request and confirm primary inference still works.
6. Run `npm ci`, `npm run build`, and `npm run lint` in `frontend/`, then verify
   available, unavailable, no-match and skipped states in the actual browser.

Requests can be slow: ERA5 has a 60-second local budget; Sentinel catalog and
Process requests are bounded but may add several minutes when retries on alternate
candidates are needed. Evidence is currently stored inline (PNG data URIs) for
this proof of concept. Strict full-coverage/source checks can reject partial or
ambiguous mosaics; do not weaken these checks merely to obtain a success banner.
