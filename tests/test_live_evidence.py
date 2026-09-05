from fastapi.encoders import jsonable_encoder
from pathlib import Path


def post(api, payload):
    return jsonable_encoder(api.live_fetch(api.LiveFetchRequest(**payload)))


def test_era5_optional_and_persisted(api, live_payload, monkeypatch):
    calls = []
    def wind(lat, lon, timestamp):
        calls.append((lat, lon, timestamp))
        return {'status': 'available', 'u10_ms': 3, 'v10_ms': 4, 'wind_speed_ms': 5}
    monkeypatch.setattr(api, 'get_wind_evidence', wind)
    body = post(api, live_payload)
    assert body['status'] == 'OK' and not calls
    assert body['detection']['supplementary']['era5']['status'] == 'skipped'
    body = post(api, {**live_payload, 'include_era5': True})
    assert calls == [(1.2, 103.8, '2026-08-11T22:47:44Z')]
    detail = jsonable_encoder(api.get_detection_details(body['detection']['id']))
    assert detail['supplementary']['era5']['wind_speed_ms'] == 5


def test_optional_failure_preserves_primary(api, live_payload, monkeypatch):
    def fail(*a):
        raise RuntimeError('secret')
    monkeypatch.setattr(api, 'get_wind_evidence', fail)
    body = post(api, {**live_payload, 'include_era5': True})
    assert body['status'] == 'OK'
    result = body['detection']['supplementary']['era5']
    assert result['status'] == 'unavailable' and 'secret' not in str(result)


def test_http_wind_status_and_persistence(api, live_payload, monkeypatch, tmp_path):
    """Exercise HTTP routing/validation with the real missing-CDS adapter.

    Satellite fetch/inference use the explicit api fixture doubles; storage is
    disposable. This is not an external satellite or ERA5 integration test.
    """
    from fastapi.testclient import TestClient
    from src.analysis.era5_wind_check import get_wind_evidence

    monkeypatch.delenv('CDSAPI_KEY', raising=False)
    monkeypatch.delenv('CDSAPI_URL', raising=False)
    monkeypatch.setenv('CDSAPI_RC', str(tmp_path / 'missing-cds-config'))
    monkeypatch.setattr(api, 'get_wind_evidence', get_wind_evidence)
    with TestClient(api.app) as client:
        response = client.post('/api/live/fetch', json={**live_payload, 'include_era5': True})
        assert response.status_code == 200
        body = response.json()
        assert body['status'] == 'OK'
        detection = body['detection']
        wind = detection['supplementary']['era5']
        assert wind['status'] == 'not_configured'
        assert 'wind_speed_ms' not in wind
        assert wind['acquisition_time_utc'] == detection['acquisition_start_utc']
        assert [wind['requested_lat'], wind['requested_lon']] == [1.2, 103.8]
        detail = client.get(f"/api/detections/{detection['id']}")
        assert detail.status_code == 200
        assert detail.json()['supplementary'] == detection['supplementary']
        default = client.post('/api/live/fetch', json=live_payload)
        assert default.status_code == 200
        assert default.json()['detection']['supplementary']['era5']['status'] == 'skipped'
        invalid = client.post('/api/live/fetch', json={**live_payload, 'min_lat': 91})
        assert invalid.status_code == 422


def test_unserializable_evidence_degrades_instead_of_failing(api, live_payload, monkeypatch):
    """A non-finite value anywhere in the evidence must not 500 a detection
    that already committed. json.dumps(allow_nan=False) rejects it, and the
    endpoint has to fall back to the statuses rather than propagate."""
    monkeypatch.setattr(api, 'get_wind_evidence',
                        lambda *a: {'status': 'available', 'wind_speed_ms': float('nan')})
    body = post(api, {**live_payload, 'include_era5': True})

    assert body['status'] == 'OK'
    supplementary = body['detection']['supplementary']
    # The status survives; the unserializable payload does not.
    assert supplementary['era5']['status'] == 'available'
    assert 'wind_speed_ms' not in supplementary['era5']
    assert supplementary['original']['status'] == 'unavailable'
    assert supplementary['temporal']['status'] == 'skipped'

    detail = jsonable_encoder(api.get_detection_details(body['detection']['id']))
    assert detail['supplementary'] == supplementary


def test_storage_failure_does_not_fail_the_detection(api, live_payload, monkeypatch):
    """A write error while persisting optional evidence leaves the row's NULL
    in place -- which reads back as {} -- and still returns the detection."""
    import sqlite3
    real_connection = api.get_db_connection
    calls = {'n': 0}

    class LockedOnExecute:
        """sqlite3.Connection.execute is read-only, so wrap rather than patch."""
        def __init__(self, conn):
            self._conn = conn

        def execute(self, *args, **kwargs):
            raise sqlite3.OperationalError('database is locked')

        def __getattr__(self, name):
            return getattr(self._conn, name)

    def flaky():
        conn = real_connection()
        calls['n'] += 1
        # The supplementary UPDATE opens the second connection, after the
        # one the detection INSERT used.
        return LockedOnExecute(conn) if calls['n'] == 2 else conn

    monkeypatch.setattr(api, 'get_db_connection', flaky)
    body = post(api, live_payload)
    assert body['status'] == 'OK'
    assert body['detection']['supplementary'] == {}


def test_empty_primary_has_no_fabricated_polygon(api, live_payload):
    result = post(api, live_payload)
    assert result['status'] == 'OK'
    assert result['detection']['geojson_mask']['coordinates'] == []


def test_comparison_results_and_failure_modes(api, live_payload, monkeypatch):
    from test_observations import product
    import numpy as np
    original, candidate = product(), product('alternate', '05')
    monkeypatch.setattr(api, 'find_best_product', lambda *a: original)
    monkeypatch.setattr('src.analysis.sentinel1_revisit_check.search_same_track_passes', lambda *a, **kw: [candidate])
    # The explicit test rasters differ so duplicate-payload validation really runs.
    monkeypatch.setattr(api.tifffile, 'imread', lambda path: np.full((256, 256, 2), 0.1 if Path(path).name.startswith('comparison_') else 0.2, dtype=np.float32))
    result = post(api, {**live_payload, 'include_temporal': True})
    temporal = result['detection']['supplementary']['temporal']
    assert temporal['status'] == 'available'
    assert temporal['observations'][0]['acquisition_start_utc'] == candidate['start']
    assert temporal['observations'][0]['scene_id'] != result['detection']['scene_id']
    monkeypatch.setattr(api.tifffile, 'imread', lambda path: np.ones((256, 256, 2), dtype=np.float32))
    assert post(api, {**live_payload, 'include_temporal': True})['detection']['supplementary']['temporal']['status'] == 'unavailable'
    def fail(*a, **kw):
        raise RuntimeError('test failure')
    monkeypatch.setattr(api, 'fetch_scene_geotiff', fail)
    # Isolate optional comparison failures without failing the primary fetch.
    observation = {'bbox': [1.1, 103.7, 1.3, 103.9], 'pixel_sha256': 'original'}
    payload = api.LiveFetchRequest(**live_payload)
    assert api._temporal_evidence('token', original, observation, payload)['status'] == 'unavailable'
    monkeypatch.setattr(api, 'fetch_scene_geotiff', lambda *a, **kw: {})
    monkeypatch.setattr(api, '_analyze_live_scene', fail)
    assert api._temporal_evidence('token', original, observation, payload)['status'] == 'unavailable'
    monkeypatch.setattr('src.analysis.sentinel1_revisit_check.search_same_track_passes', lambda *a, **kw: [])
    assert api._temporal_evidence('token', original, observation, payload)['status'] == 'no_match'


def test_optical_opt_in_persistence_and_failure(api, live_payload, monkeypatch):
    calls = []
    def optical(token, bbox, timestamp, **kwargs):
        calls.append((bbox, timestamp, kwargs))
        return {'status': 'available', 'observation': {'acquisition_start_utc': '2026-08-10T03:00:00Z'}}
    monkeypatch.setattr(api, 'get_optical_evidence', optical)
    assert post(api, live_payload)['detection']['supplementary']['optical']['status'] == 'skipped'
    assert not calls
    result = post(api, {**live_payload, 'include_optical': True, 'optical_max_cloud_pct': 5})
    assert calls == [([1.1, 103.7, 1.3, 103.9], '2026-08-11T22:47:44Z', {'max_cloud_pct': 5, 'window_days': 10})]
    detail = api.get_detection_details(result['detection']['id'])
    assert detail['supplementary']['optical']['observation']['acquisition_start_utc'] != detail['acquisition_start_utc']
    def fail(*a, **kw):
        raise ValueError('secret')
    monkeypatch.setattr(api, 'get_optical_evidence', fail)
    result = post(api, {**live_payload, 'include_optical': True})
    assert result['status'] == 'OK'
    assert result['detection']['supplementary']['optical']['status'] == 'unavailable'


def test_actual_checkpoint_in_shared_live_analysis(api, live_payload, monkeypatch):
    """Real trained U-Net on an explicitly synthetic calibrated raster, offline.

    This verifies inference plumbing, not scientific validity or external data.
    Land masking and catalog/fetch remain the fixture's explicit doubles.
    """
    import numpy as np
    import torch
    from src.data.preprocess import preprocess_for_prediction
    from src.inference import run_tiled_inference
    assert api.model is not None, 'Real default checkpoint must load for this test'
    raster = np.geomspace(0.001, 0.3, 256 * 256 * 2).reshape(256, 256, 2).astype(np.float32)
    monkeypatch.setattr(api.tifffile, 'imread', lambda *a: raster)
    monkeypatch.setattr(api, 'preprocess_for_prediction', preprocess_for_prediction)
    monkeypatch.setattr(api, 'run_tiled_inference', run_tiled_inference)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(2)
        result = post(api, live_payload)
    finally:
        torch.set_num_threads(old_threads)
    assert result['status'] == 'OK'
    original = result['detection']['supplementary']['original']
    assert 0 <= original['confidence_score'] <= 1
    assert 0 <= original['predicted_pixel_count'] <= 256 * 256
    assert original['overlay_preview'].startswith('/api/previews/')


def test_live_missing_credentials_and_invalid_request(api, live_payload, monkeypatch):
    import pytest
    from pydantic import ValidationError
    monkeypatch.delenv('CDSE_CLIENT_ID')
    result = post(api, live_payload)
    assert result['status'] == 'NOT_CONFIGURED' and result['detection'] is None
    for changes in ({'min_lat': float('nan')}, {'min_lon': 200}, {'temporal_window_days': 0},
                    {'optical_max_cloud_pct': 101}, {'date_from': '2026-09-01'}):
        with pytest.raises(ValidationError):
            api.LiveFetchRequest(**{**live_payload, **changes})
