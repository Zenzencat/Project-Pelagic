import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from src.api import preview_storage


def png_uri(value=80):
    ok, encoded = cv2.imencode('.png', np.full((3, 4), value, dtype=np.uint8))
    assert ok
    return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii')


def all_preview_payload():
    return {
        'original': {'sar_preview': png_uri(1), 'overlay_preview': png_uri(2)},
        'temporal': {'status': 'available', 'observations': [
            {'sar_preview': png_uri(3), 'overlay_preview': png_uri(4)}]},
        'optical': {'status': 'available', 'observation': {'rgb_preview': png_uri(5)}},
    }


def large_png_uri(value):
    """Deterministic 640x640 test graphic, not satellite data."""
    rng = np.random.default_rng(value)
    image = rng.integers(0, 256, (640, 640), dtype=np.uint8)
    ok, encoded = cv2.imencode('.png', image)
    assert ok
    return 'data:image/png;base64,' + base64.b64encode(encoded).decode('ascii'), encoded.tobytes()


def five_live_observations():
    values = {}
    original = {
        'scene_id': 'live-test-primary', 'product': {'id': 'test-product', 'name': 'test-S1-GRD'},
        'acquisition_start_utc': '2026-08-11T22:47:44Z', 'acquisition_end_utc': '2026-08-11T22:48:09Z',
        'source': 'test', 'provenance': {}, 'bbox': [1.1, 103.7, 1.3, 103.9],
        'pixel_sha256': 'primary', 'confidence_score': 0.5, 'predicted_pixel_count': 10,
        'geojson_mask': {'type': 'Polygon', 'coordinates': []},
        'preview_note': 'Test graphic; not satellite data.',
    }
    for field, key in (('sar_preview', 'original_sar'), ('overlay_preview', 'original_overlay')):
        original[field], values[key] = large_png_uri(len(values) + 1)
    comparison = {**original, 'scene_id': 'live-test-comparison', 'product': {'id': 'comparison', 'name': 'test-comparison'}}
    for field, key in (('sar_preview', 'temporal_0_sar'), ('overlay_preview', 'temporal_0_overlay')):
        comparison[field], values[key] = large_png_uri(len(values) + 1)
    optical_preview, values['optical_rgb'] = large_png_uri(5)
    optical = {
        'product': {'id': 'optical', 'name': 'test-S2-L2A'},
        'acquisition_start_utc': '2026-08-10T03:00:00Z', 'cloud_cover_pct': 4,
        'rgb_preview': optical_preview,
    }
    return original, comparison, optical, values


def test_five_previews_round_trip_to_urls_and_disk(tmp_path):
    payload = all_preview_payload()
    preview_storage.persist_previews(payload, 42, tmp_path)
    root = preview_storage.preview_directory(tmp_path)
    expected = {
        '42_original_sar.png', '42_original_overlay.png',
        '42_temporal_0_sar.png', '42_temporal_0_overlay.png', '42_optical_rgb.png',
    }
    assert {p.name for p in root.iterdir()} == expected
    assert payload['original']['sar_preview'] == '/api/previews/42_original_sar.png'
    assert payload['original']['overlay_preview'] == '/api/previews/42_original_overlay.png'
    assert payload['temporal']['observations'][0]['overlay_preview'] == '/api/previews/42_temporal_0_overlay.png'
    assert payload['optical']['observation']['rgb_preview'] == '/api/previews/42_optical_rgb.png'
    assert len(json.dumps(payload)) < 2_000


def test_invalid_new_inline_previews_are_removed_without_base64_fallback(tmp_path):
    payload = {'original': {
        'sar_preview': 'data:image/png;base64,not-base64',
        'overlay_preview': 'data:image/svg+xml;base64,AAAA',
    }}
    preview_storage.persist_previews(payload, 7, tmp_path)
    assert 'sar_preview' not in payload['original']
    assert 'overlay_preview' not in payload['original']
    assert 'base64' not in json.dumps(payload)
    assert 'preview_error' in payload['original']


def test_disk_failure_degrades_preview_only(tmp_path, monkeypatch):
    monkeypatch.setattr(preview_storage, '_atomic_write', lambda *args: (_ for _ in ()).throw(OSError('full')))
    payload = {'original': {'sar_preview': png_uri()}}
    preview_storage.persist_previews(payload, 8, tmp_path)
    assert 'sar_preview' not in payload['original']
    assert payload['original']['preview_error'] == 'Preview storage failed (OSError).'


def test_oldest_cap_only_evicts_owned_regular_pngs(tmp_path, monkeypatch):
    root = preview_storage.preview_directory(tmp_path)
    root.mkdir(parents=True)
    monkeypatch.setattr(preview_storage, 'MAX_PREVIEW_FILES', 2)
    paths = [root / f'9_original_sar.png', root / '9_original_overlay.png', root / '9_optical_rgb.png']
    for index, path in enumerate(paths):
        path.write_bytes(b'png')
        os.utime(path, (100 + index, 100 + index))
    (root / 'keep.txt').write_text('not owned')
    preview_storage.cleanup_previews(root)
    assert not paths[0].exists()
    assert paths[1].exists() and paths[2].exists() and (root / 'keep.txt').exists()


def test_preview_route_round_trip_and_rejects_traversal_symlink(api, tmp_path):
    from fastapi.testclient import TestClient

    root = preview_storage.preview_directory(tmp_path)
    root.mkdir(parents=True)
    target = root / '42_original_sar.png'
    target_bytes = base64.b64decode(png_uri().split(',', 1)[1])
    target.write_bytes(target_bytes)
    outside = tmp_path / 'outside.png'
    outside.write_bytes(b'outside')
    symlink = root / '42_original_overlay.png'
    symlink.symlink_to(outside)
    with TestClient(api.app) as client:
        response = client.get('/api/previews/42_original_sar.png')
        assert response.status_code == 200
        assert response.headers['content-type'].startswith('image/png')
        assert response.content == target_bytes
        assert cv2.imdecode(np.frombuffer(response.content, dtype=np.uint8), cv2.IMREAD_UNCHANGED) is not None
        assert client.get('/api/previews/../outside.png').status_code == 404
        assert client.get('/api/previews/42_original_overlay.png').status_code == 404
        assert client.get('/api/previews/nope.txt').status_code == 404


def test_preview_route_rejects_symlinked_ancestor(api, tmp_path):
    from fastapi.testclient import TestClient

    outside = tmp_path / 'outside-live'
    outside.mkdir()
    live_parent = tmp_path / 'data' / 'raw' / 'live'
    live_parent.parent.mkdir(parents=True)
    live_parent.symlink_to(outside, target_is_directory=True)
    (outside / 'previews').mkdir()
    (outside / 'previews' / '42_original_sar.png').write_bytes(b'outside')
    with TestClient(api.app) as client:
        assert client.get('/api/previews/42_original_sar.png').status_code == 404


def test_live_payload_stores_preview_reference_and_keeps_legacy_inline_bytes(api, live_payload):
    from fastapi.encoders import jsonable_encoder

    legacy_inline = 'data:image/png;base64,legacy-row-bytes'
    legacy_json = json.dumps({'original': {'overlay_preview': legacy_inline}})
    conn = api.get_db_connection()
    conn.execute("""INSERT INTO detections
        (scene_id, confidence_score, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon,
         geojson_mask, image_path, source, supplementary_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                 ('legacy-inline', 0, 1, 2, 3, 4, '{"type":"Polygon","coordinates":[]}',
                  'data/processed/legacy.png', 'holdout', legacy_json))
    conn.commit()
    legacy_id = conn.execute("SELECT id FROM detections WHERE scene_id = ?", ('legacy-inline',)).fetchone()[0]
    conn.close()

    result = jsonable_encoder(api.live_fetch(api.LiveFetchRequest(**live_payload)))
    assert result['status'] == 'OK'
    detection = result['detection']
    assert detection['supplementary']['original']['overlay_preview'].startswith('/api/previews/')
    assert 'base64' not in json.dumps(detection['supplementary'])
    preview_name = Path(detection['supplementary']['original']['overlay_preview']).name
    assert (preview_storage.preview_directory(api.BASE_DIR) / preview_name).is_file()
    assert api.get_detection_details(legacy_id)['supplementary']['original']['overlay_preview'] == legacy_inline
    conn = api.get_db_connection()
    assert conn.execute('SELECT supplementary_json FROM detections WHERE id = ?', (legacy_id,)).fetchone()[0] == legacy_json
    conn.close()


def test_http_live_post_stores_all_five_previews_and_serves_exact_bytes(api, live_payload, monkeypatch):
    from fastapi.testclient import TestClient

    original, comparison, optical, expected_bytes = five_live_observations()
    geo = {'center_lat': 1.2, 'center_lon': 103.8, 'pixel_scale_deg': 0.2 / 256,
           'min_lat': 1.1, 'min_lon': 103.7, 'max_lat': 1.3, 'max_lon': 103.9}
    monkeypatch.setattr(api, '_analyze_live_scene', lambda *args, **kwargs: (original, geo, {'oil_px_total_before': 0}))
    monkeypatch.setattr(api, '_temporal_evidence', lambda *args, **kwargs: {
        'status': 'available', 'observations': [comparison], 'reason': 'test-only'})
    monkeypatch.setattr(api, 'get_optical_evidence', lambda *args, **kwargs: {
        'status': 'available', 'observation': optical, 'reason': 'test-only'})
    captured = {}
    real_persist = api.persist_previews

    def capture_persist(payload, detection_id, base_dir):
        captured['before_bytes'] = len(json.dumps(payload, allow_nan=False))
        real_persist(payload, detection_id, base_dir)
        captured['after_bytes'] = len(json.dumps(payload, allow_nan=False))

    monkeypatch.setattr(api, 'persist_previews', capture_persist)
    with TestClient(api.app) as client:
        response = client.post('/api/live/fetch', json={**live_payload, 'include_temporal': True, 'include_optical': True})
        assert response.status_code == 200
        body = response.json()
        assert body['status'] == 'OK'
        supplementary = body['detection']['supplementary']
        assert captured['before_bytes'] > captured['after_bytes']
        assert captured['after_bytes'] == len(json.dumps(supplementary, allow_nan=False))
        assert 'data:image' not in json.dumps(supplementary)
        refs = {
            'original_sar': supplementary['original']['sar_preview'],
            'original_overlay': supplementary['original']['overlay_preview'],
            'temporal_0_sar': supplementary['temporal']['observations'][0]['sar_preview'],
            'temporal_0_overlay': supplementary['temporal']['observations'][0]['overlay_preview'],
            'optical_rgb': supplementary['optical']['observation']['rgb_preview'],
        }
        for kind, reference in refs.items():
            assert reference == f"/api/previews/{body['detection']['id']}_{kind}.png"
            fetched = client.get(reference)
            assert fetched.status_code == 200
            assert fetched.content == expected_bytes[kind]
        conn = api.get_db_connection()
        row = conn.execute(
            'SELECT supplementary_json FROM detections WHERE id = ?', (body['detection']['id'],)).fetchone()
        conn.close()
        assert len(row['supplementary_json']) == captured['after_bytes']
        print(f"preview payload bytes: {captured['before_bytes']} -> {captured['after_bytes']}")


@pytest.mark.parametrize('failure_mode', ['disk', 'unexpected'])
def test_http_live_post_storage_failure_preserves_primary_success(api, live_payload, monkeypatch, failure_mode):
    from fastapi.testclient import TestClient

    if failure_mode == 'disk':
        def fail_write(*args, **kwargs):
            raise OSError('test disk full')
        monkeypatch.setattr(preview_storage, '_atomic_write', fail_write)
    else:
        def fail_storage(*args, **kwargs):
            raise RuntimeError('test storage exception')
        monkeypatch.setattr(api, 'persist_previews', fail_storage)
    with TestClient(api.app) as client:
        response = client.post('/api/live/fetch', json=live_payload)
        assert response.status_code == 200
        body = response.json()
        assert body['status'] == 'OK'
        supplementary = body['detection']['supplementary']
        assert 'data:image' not in json.dumps(supplementary)
        assert 'preview_error' in supplementary['original']
