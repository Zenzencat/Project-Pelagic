import pytest

from src.analysis.sentinel1_revisit_check import select_comparison_products
from src.data.cdse_fetch import find_best_product
from src.data.observation_catalog import covers_bbox

BBOX = (1, 103, 2, 104)


def product(id='original', day='11', **kwargs):
    return {'id': id, 'name': f'S1A_IW_GRDH_1SDV_{id}',
            'start': f'2026-08-{day}T22:00:00Z', 'end': f'2026-08-{day}T22:00:25Z',
            'geometry': {'type': 'Polygon', 'coordinates': [[[102, 0], [105, 0], [105, 3], [102, 3], [102, 0]]]},
            'attributes': {'relativeOrbitNumber': 42, 'orbitDirection': 'DESCENDING'}, **kwargs}


def test_date_identity_coverage_and_ranking():
    original = product()
    best = product('other', '05')
    partial = product('partial', '06', geometry={'type': 'Polygon', 'coordinates': [[[103, 1], [103.1, 1], [103.1, 1.1], [103, 1.1], [103, 1]]]})
    other_track = product('closer', '10', attributes={})
    results = select_comparison_products([original, product('same-day'), best, best, partial, other_track,
                                          product('invalid', '07', end=None)], original, BBOX)
    assert [p['id'] for p in results] == ['other', 'closer']
    assert select_comparison_products([best], original, BBOX, window_days=2) == []
    assert not covers_bbox({'footprint': None}, BBOX)


def test_one_acquisition_listed_twice_is_one_candidate():
    """CDSE returns a COG and a non-COG row per sensing time. Those are one
    observation, so the caller's bounded retry must not spend both attempts on
    them -- it has to reach the next real date. Mirrors the real catalog rows
    in docs/live_catalog_verification.json, where 8 rows are 4 acquisitions."""
    original = product()
    rows = []
    for day in ('22', '29', '03', '17'):
        rows.append(product(f'{day}-plain', day))
        rows.append(product(f'{day}-cog', day, name=f'S1A_IW_GRDH_1SDV_{day}_COG.SAFE'))

    results = select_comparison_products(rows, original, BBOX, window_days=90)

    assert len(results) == 4
    assert len({p['start'] for p in results}) == 4
    # The retry bound used by src/api/main.py::_temporal_evidence.
    assert len({p['start'] for p in results[:2]}) == 2
    # Ranked by distance from the original's 2026-08-11, one row per date, and
    # the best-ranked representation of each acquisition is the one kept.
    assert [p['id'] for p in results] == ['17-plain', '03-plain', '22-plain', '29-plain']


def test_primary_date_range_is_exact(monkeypatch):
    monkeypatch.setattr('src.data.cdse_fetch.search_same_track_passes',
                        lambda *a, **kw: [product('inside', '11'), product('outside', '13')])
    assert find_best_product(*BBOX, '2026-08-10', '2026-08-11')['id'] == 'inside'
    assert find_best_product(*BBOX, '2026-08-01', '2026-08-02') is None


def test_process_identity_and_date():
    from src.data.sentinel_process import verify_sources
    selected = product()
    tile = {'sentinel1ProductId': selected['name'], 'date': selected['start']}
    assert verify_sources({'tiles': [tile]}, selected, 'S1')['status'] == 'verified'
    for bad in ({'tiles': []}, {'tiles': [{**tile, 'sentinel1ProductId': 'other'}]},
                {'tiles': [{**tile, 'date': '2026-08-05T22:00:00Z'}]}):
        with pytest.raises(ValueError):
            verify_sources(bad, selected, 'S1')


def test_process_identity_real_cog_product():
    """Regression for the _COG/.SAFE naming case, using the VERBATIM product
    name and Process `sentinel1ProductId` captured from a real authenticated
    Sentinel Hub Process response for the Singapore Strait scene on
    2026-09-05 (docs/status.md). The Process API reports the product under the
    exact same `..._COG.SAFE` identifier the catalog uses, and the tile date
    is truncated to whole seconds -- both must still verify, and a genuinely
    different product id must still be rejected."""
    from src.data.sentinel_process import verify_sources
    selected = {
        'id': 'd514853f-fc94-4799-b40e-c44e81412d0b',
        'name': 'S1D_IW_GRDH_1SDV_20260810T112444_20260810T112509_004062_007671_C8DC_COG.SAFE',
        'start': '2026-08-10T11:24:44.116503Z',
        'end': '2026-08-10T11:25:09.115202Z',
    }
    tile = {'sentinel1ProductId': selected['name'], 'date': '2026-08-10T11:24:44Z'}
    assert verify_sources({'tiles': [tile]}, selected, 'S1')['status'] == 'verified'
    wrong = {'sentinel1ProductId':
             'S1D_IW_GRDH_1SDV_20260822T112444_20260822T112509_004237_007C8B_F65B_COG.SAFE',
             'date': '2026-08-10T11:24:44Z'}
    with pytest.raises(ValueError):
        verify_sources({'tiles': [wrong]}, selected, 'S1')


def test_fetch_validates_pixels_and_interval(monkeypatch, tmp_path):
    import io
    import numpy as np
    import tifffile
    from src.data.cdse_fetch import fetch_scene_geotiff, CdseFetchError
    # Sub-second acquisition interval, like the real catalog -- the fix floors
    # `from` and ceils `to` to whole seconds before calling the Process API.
    selected = product(start='2026-08-11T22:00:00.116503Z', end='2026-08-11T22:00:25.115202Z')
    def tiff(array):
        stream = io.BytesIO()
        tifffile.imwrite(stream, array, photometric='minisblack',
                         planarconfig='contig' if array.ndim == 3 else None, extratags=[
            (33550, 'd', 3, (0.01/256, 0.01/256, 0), False),
            (33922, 'd', 6, (0, 0, 0, 103, 1.01, 0), False)])
        return stream.getvalue()
    files = {'default.tif': tiff(np.ones((256, 256, 2), dtype=np.float32)),
             'validity.tif': tiff(np.ones((256, 256), dtype=np.uint8))}
    def process(token, body):
        # Time bounds are floored/ceiled to whole seconds: Sentinel Hub filters
        # on whole-second scene timestamps and a sub-second `from` bound returns
        # zero tiles (verified against a real Process response, docs/status.md).
        assert body['input']['data'][0]['dataFilter']['timeRange'] == {
            'from': '2026-08-11T22:00:00Z', 'to': '2026-08-11T22:00:26Z'}
        return files, {'tiles': [{'sentinel1ProductId': selected['name'], 'date': selected['start']}]}
    monkeypatch.setattr('src.data.sentinel_process.process_request', process)
    path = tmp_path / 'scene.tif'
    assert fetch_scene_geotiff('test', 1, 103, 1.01, 103.01, selected['start'], path, product=selected)['status'] == 'verified'
    files['validity.tif'] = tiff(np.zeros((256, 256), dtype=np.uint8))
    with pytest.raises(CdseFetchError):
        fetch_scene_geotiff('test', 1, 103, 1.01, 103.01, selected['start'], path, product=selected)
    files['validity.tif'] = tiff(np.ones((256, 256), dtype=np.uint8))
    with pytest.raises(CdseFetchError):
        fetch_scene_geotiff('test', 2, 103, 2.01, 103.01, selected['start'], path, product=selected)
