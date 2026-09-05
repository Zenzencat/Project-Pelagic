"""Offline API tests use a disposable DB and explicit external/inference doubles."""
import importlib
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def api(monkeypatch, tmp_path):
    # The checkout may lack the raster package. This is an explicit test double,
    # never a production fallback or evidence of real land-mask verification.
    if importlib.util.find_spec('global_land_mask') is None:
        monkeypatch.setitem(sys.modules, 'global_land_mask', types.SimpleNamespace(globe=None))
    from src.api import database
    monkeypatch.setattr(database, 'DB_PATH', str(tmp_path / 'test.db'))
    database.init_db()
    from src.api import main
    monkeypatch.setattr(main, 'BASE_DIR', str(tmp_path))
    monkeypatch.setattr(main, 'model_loaded', True)
    monkeypatch.setenv('CDSE_CLIENT_ID', 'test-only')
    monkeypatch.setenv('CDSE_CLIENT_SECRET', 'test-only')
    monkeypatch.setattr(main, 'find_best_product', lambda *a: {'id': 'test-product', 'name': 'test-S1-GRD', 'start': '2026-08-11T22:47:44Z', 'end': '2026-08-11T22:48:09Z'})
    monkeypatch.setattr(main, 'get_cdse_token', lambda *a: 'test-token')
    monkeypatch.setattr(main, 'fetch_scene_geotiff', lambda *a, **kw: None)
    monkeypatch.setattr(main.tifffile, 'imread', lambda *a: np.ones((256, 256, 2), dtype=np.float32))
    monkeypatch.setattr(main, 'preprocess_for_prediction', lambda *a, **kw: np.ones((2, 256, 256)))
    monkeypatch.setattr(main, 'run_tiled_inference', lambda *a: (np.zeros((256, 256)), np.zeros((256, 256), dtype=np.uint8)))
    monkeypatch.setattr(main, 'get_scene_geolocation', lambda *a: {'center_lat': 1.2, 'center_lon': 103.8, 'pixel_scale_deg': 0.2/256, 'pixel_scale_y_deg': 0.2/256, 'min_lat': 1.1, 'min_lon': 103.7, 'max_lat': 1.3, 'max_lon': 103.9})
    monkeypatch.setattr(main, 'strip_land_pixels', lambda mask, *a, **kw: (mask, {'oil_px_total_before': 0}))
    monkeypatch.setattr(main, 'get_nearby_vessels', lambda *a, **kw: {'status': 'skipped_no_credentials', 'detail': 'test double', 'vessels': []})
    return main


@pytest.fixture
def live_payload():
    return {'min_lat': 1.1, 'min_lon': 103.7, 'max_lat': 1.3, 'max_lon': 103.9,
            'date_from': '2026-08-05', 'date_to': '2026-08-15'}
