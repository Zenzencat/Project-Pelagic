"""Browser acceptance checks with explicit test-only API responses.

Start the frontend, then run this script with Playwright/Chromium installed.
No backend, production database, satellite service or real measurement is used.
"""
import argparse
import copy
import json
import re

from playwright.sync_api import expect, sync_playwright


def verify(url):
    # A tiny test graphic, deliberately not a satellite image.
    preview = 'data:image/svg+xml,%3Csvg xmlns="http://www.w3.org/2000/svg" width="64" height="64"%3E%3Ctext y="32"%3ETEST%3C/text%3E%3C/svg%3E'
    original = {
        'scene_id': 'test-only-original', 'acquisition_start_utc': '2026-08-11T22:47:44Z',
        'predicted_pixel_count': 0, 'confidence_score': 0,
        'overlay_preview': preview, 'preview_note': 'Test graphic; not satellite data.',
        'product': {'name': 'TEST-ONLY-S1'},
    }
    detection = {
        'id': 900001, 'scene_id': original['scene_id'], 'source': 'live',
        'acquisition_start_utc': original['acquisition_start_utc'],
        'detected_at': '2026-08-12T00:00:00Z', 'confidence_score': 0,
        'bbox': [1.1, 103.7, 1.3, 103.9],
        'geojson_mask': {'type': 'Polygon', 'coordinates': []},
        'nearby_vessels': [], 'vessel_attribution_status': 'skipped_no_credentials',
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = browser.new_page(viewport={'width': 1366, 'height': 768})
        page_errors, requests = [], []
        page.on('pageerror', lambda error: page_errors.append(str(error)))

        def api_route(route):
            path = route.request.url.split('8000', 1)[1]
            if path == '/api/live/fetch':
                requests.append(route.request.post_data_json)
                body = {'status': 'OK', 'detail': 'TEST ONLY', 'detection': detection}
            elif path == '/api/detections':
                body = [detection]
            elif path == f"/api/detections/{detection['id']}":
                body = detection
            else:
                raise AssertionError(f'Unexpected API request: {path}')
            route.fulfill(json=body, headers={'Access-Control-Allow-Origin': '*'})

        page.route('http://localhost:8000/**', api_route)
        # Basemap networking is outside this UI check.
        page.route(re.compile(r'https://.*(?:tile|basemaps).*'), lambda route: route.abort())
        for status in ('available', 'not_configured', 'unavailable', 'skipped', 'no_match'):
            wind = {'status': status, 'reason': f'Test-only {status} reason.'}
            temporal = {'status': status, 'reason': f'Test-only {status} reason.'}
            optical = {'status': status, 'reason': f'Test-only {status} reason.'}
            if status == 'available':
                wind.update(wind_speed_ms=5, u10_ms=3, v10_ms=-4,
                            valid_time_utc='2026-08-11T23:00:00+00:00', grid_lat=1.25, grid_lon=103.75)
                alternate = {**original, 'scene_id': 'test-only-alternate',
                             'acquisition_start_utc': '2026-08-05T22:47:44Z'}
                temporal['observations'] = [alternate]
                optical['observation'] = {
                    'acquisition_start_utc': '2026-08-10T03:00:00Z', 'cloud_cover_pct': 4,
                    'rgb_preview': preview, 'product': {'name': 'TEST-ONLY-S2-L2A'},
                }
            detection['supplementary'] = copy.deepcopy({
                'original': original, 'era5': wind, 'temporal': temporal, 'optical': optical,
            })
            page.goto(url)
            panel = page.get_by_role('region', name='ERA5 wind evidence')
            panel.scroll_into_view_if_needed()
            expect(panel).to_be_visible()
            expect(panel).to_contain_text('0.25°')
            if status == 'available':
                expect(panel).to_contain_text('5.00 m/s')
                expect(panel).to_contain_text(wind['valid_time_utc'])
            else:
                expect(panel).to_contain_text(wind['reason'])
                expect(panel).not_to_contain_text('m/s')
            page.get_by_text('Observation evidence — SAR and optical', exact=True).click()
            evidence = page.locator('details').filter(has=page.get_by_text('Observation evidence — SAR and optical', exact=True))
            expect(evidence).to_contain_text(f'Comparison: {status}')
            expect(evidence).to_contain_text(f'Optical: {status}')
            expect(evidence.locator('img')).to_have_count(3 if status == 'available' else 1)
            if status == 'available':
                expect(evidence).to_contain_text(alternate['acquisition_start_utc'])
                expect(evidence).to_contain_text(optical['observation']['acquisition_start_utc'])
            assert evidence.locator('img').evaluate_all('(imgs) => imgs.every(i => i.complete && i.naturalWidth > 0)')
            print(f'PASS browser rendering: {status} (test fixtures)')

        page.get_by_text('Supplementary evidence (optional)', exact=True).click()
        checkboxes = [page.get_by_role('checkbox', name=name) for name in (
            'ERA5 wind evidence', 'Compare another Sentinel-1 pass', 'Sentinel-2 optical supplement')]
        for checkbox in checkboxes:
            expect(checkbox).not_to_be_checked()
            checkbox.check()
        page.get_by_role('button', name='ดึงภาพจริง (Live Fetch)', exact=True).click()
        expect(page.get_by_text('TEST ONLY', exact=False)).to_be_visible()
        assert all(requests[-1][key] is True for key in ('include_era5', 'include_temporal', 'include_optical'))
        for checkbox in checkboxes:
            checkbox.uncheck()
        with page.expect_response('http://localhost:8000/api/live/fetch'):
            page.get_by_role('button', name='ดึงภาพจริง (Live Fetch)', exact=True).click()
        assert all(requests[-1][key] is False for key in ('include_era5', 'include_temporal', 'include_optical'))
        assert not page_errors, json.dumps(page_errors)
        print('PASS optional request flags and no browser exceptions')
        browser.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:5173')
    verify(parser.parse_args().url)
