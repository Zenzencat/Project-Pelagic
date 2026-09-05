# Empty prediction geometry

`POST /api/predict` returns a GeoJSON `Polygon` with an empty `coordinates`
array when contour extraction produces no valid polygon. This covers a fully
empty mask, pixels removed by the land mask, and candidates suppressed by the
opt-in lookalike classifier. The endpoint still returns HTTP 201, the scene
bounds, the saved black-or-partial mask PNG, and the confidence calculated from
the surviving predicted pixels. A positive confidence with empty coordinates
is therefore possible for a prediction too small to form a polygon.

This was originally scoped to `/api/predict` alone, leaving `/api/live/fetch`'s
own placeholder fallback in place. That is no longer accurate: the integration
into `main` (commit `ac8c318`) removed the live path's fallback too, so **both**
endpoints now return empty `coordinates` rather than a fabricated square. The
live path builds its geometry in `_analyze_live_scene()`.

The cache recipe includes `+empty_geometry_v1`, before the optional
`+lookalike_filter` suffix. A POST for an existing scene repairs legacy rows
in place when the recipe differs and removes stale nearby-vessel rows. A GET
alone does not rewrite historical rows.

Run the focused regression suite with:

```bash
rtk proxy python -m pytest -q tests/test_predict_empty_masks.py
```

Install the project's requirements plus `pytest` and `httpx` in the chosen
Python environment first. The focused run completed with **5 passed**. The
frontend build also passed, and a real-browser check confirmed that both
positive-confidence empty geometry and zero-confidence empty geometry fall
back to the scene bounds without invalid Leaflet bounds.

The tests use a temporary SQLite database and scene, replace inference,
geolocation, land-mask, and classifier behavior, and make no external service
calls. They retain
the real contour extraction, database writes, PNG output, and API
serialization. They do not require a checkpoint or network access.
