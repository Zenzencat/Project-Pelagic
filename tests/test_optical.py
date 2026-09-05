import cv2
import numpy as np
import pytest

from src.data import sentinel2_optical as optical
from test_observations import product, BBOX


def s2(id='optical', day='10', cloud=10, **kwargs):
    return product(id, day, name=f'S2A_MSIL2A_{id}', attributes={'productType': 'S2MSI2A', 'cloudCover': cloud}, **kwargs)


def test_cloud_type_coverage_and_date_filtering():
    candidates = [s2(), s2('cloudy', cloud=30), s2('missing', cloud=None), s2('invalid', cloud=float('nan')),
                  s2('clearer', '05', 1), s2('outside', '01', 0), s2('nocover', geometry=None, footprint=None)]
    selected = optical.select_optical_products(candidates, BBOX, product()['start'], window_days=7)
    assert [p['id'] for p in selected] == ['clearer', 'optical']
    assert optical.select_optical_products(candidates, BBOX, product()['start'], max_cloud_pct=0, window_days=1) == []


def test_render_request_and_validation(monkeypatch):
    candidate = s2()
    image = np.full((640, 640, 4), 255, dtype=np.uint8)
    tile = {'sentinel2ProductId': candidate['name'], 'date': candidate['start'], 'cloudCoverage': 10}
    def process(token, body):
        assert body['input']['data'][0]['type'] == 'sentinel-2-l2a'
        assert body['input']['bounds']['bbox'] == [103, 1, 104, 2]
        assert body['input']['data'][0]['dataFilter']['maxCloudCoverage'] == 20
        return {'default.png': cv2.imencode('.png', image)[1].tobytes()}, {'tiles': [tile]}
    monkeypatch.setattr(optical, 'process_request', process)
    result = optical.render_optical('test-token', BBOX, candidate, 20)
    assert result['acquisition_start_utc'] == candidate['start']
    assert result['rgb_preview'].startswith('data:image/png;base64,')
    tile['cloudCoverage'] = 90
    with pytest.raises(ValueError):
        optical.render_optical('test-token', BBOX, candidate, 20)
    tile['cloudCoverage'] = 10
    image[0, 0, 3] = 0
    with pytest.raises(ValueError):
        optical.render_optical('test-token', BBOX, candidate, 20)


def test_no_match_fetch_failure_and_success(monkeypatch):
    monkeypatch.setattr(optical, 'query_products', lambda *a: [])
    assert optical.get_optical_evidence('test-token', BBOX, product()['start'])['status'] == 'no_match'
    monkeypatch.setattr(optical, 'query_products', lambda *a: [s2()])
    def fail(*a):
        raise ValueError('service secret')
    monkeypatch.setattr(optical, 'render_optical', fail)
    result = optical.get_optical_evidence('test-token', BBOX, product()['start'])
    assert result['status'] == 'unavailable' and 'secret' not in str(result)
    monkeypatch.setattr(optical, 'render_optical', lambda *a: {'acquisition_start_utc': s2()['start']})
    result = optical.get_optical_evidence('test-token', BBOX, product()['start'])
    assert result['status'] == 'available'
    assert result['sar_acquisition_utc'] != result['observation']['acquisition_start_utc']
