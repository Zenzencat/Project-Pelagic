"""Low-cloud Sentinel-2 L2A RGB evidence; no fusion or cloud removal.

OData uses Attributes/cloudCover and productType S2MSI2A:
https://documentation.dataspace.copernicus.eu/APIs/OData.html
Process tile metadata uses sentinel2ProductId/cloudCoverage:
https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L1C.html
L2A explicitly references the same scene metadata schema.
"""
import base64
import math
from datetime import timedelta

import cv2
import numpy as np

from src.data.observation_catalog import acquisition_time, query_products, covers_bbox, valid_product
from src.data.sentinel_process import process_request, verify_sources


def select_optical_products(products, bbox, timestamp, max_cloud_pct=20, window_days=10):
    ref = acquisition_time(timestamp)
    candidates = []
    seen = set()
    for product in products:
        if not valid_product(product) or product['id'] in seen:
            continue
        attrs = product.get('attributes', {})
        cloud = attrs.get('cloudCover')
        if attrs.get('productType') != 'S2MSI2A' or not isinstance(cloud, (int, float)):
            continue
        if not math.isfinite(cloud) or not 0 <= cloud <= max_cloud_pct:
            continue
        if abs((acquisition_time(product['start']) - ref).total_seconds()) > window_days * 86400:
            continue
        if not covers_bbox(product, bbox):
            continue
        seen.add(product['id'])
        candidates.append(product)
    # Full coverage first (required above), then low cloud, then date proximity.
    return sorted(candidates, key=lambda p: (p['attributes']['cloudCover'],
                  abs((acquisition_time(p['start']) - ref).total_seconds())))


def render_optical(token, bbox, product, max_cloud_pct):
    min_lat, min_lon, max_lat, max_lon = bbox
    # Match the region and aspect ratio, cap display pixels (not scientific fusion).
    ratio = (max_lon - min_lon) / (max_lat - min_lat)
    width, height = (640, max(1, round(640 / ratio))) if ratio >= 1 else (max(1, round(640 * ratio)), 640)
    start, end = acquisition_time(product['start']), acquisition_time(product['end'])
    body = {
        'input': {'bounds': {'bbox': [min_lon, min_lat, max_lon, max_lat]}, 'data': [{
            'type': 'sentinel-2-l2a', 'dataFilter': {
                'timeRange': {'from': (start - timedelta(seconds=1)).isoformat(),
                              'to': (end + timedelta(seconds=1)).isoformat()},
                'maxCloudCoverage': max_cloud_pct}}]},
        'output': {'width': width, 'height': height, 'responses': [
            {'identifier': 'default', 'format': {'type': 'image/png'}},
            {'identifier': 'userdata', 'format': {'type': 'application/json'}}]},
        'evalscript': '''
//VERSION=3
function setup() {
  return {input: ["B04", "B03", "B02", "dataMask"], mosaicking: "ORBIT",
          output: {id: "default", bands: 4}};
}
function evaluatePixel(samples) {
  if (!samples.length) return [0, 0, 0, 0];
  let s = samples[0];
  return [2.5*s.B04, 2.5*s.B03, 2.5*s.B02, s.dataMask];
}
function updateOutputMetadata(scenes, inputMetadata, outputMetadata) {
  outputMetadata.userData = {tiles: [].concat.apply([], scenes.orbits.map(o => o.tiles))};
}
'''}
    files, metadata = process_request(token, body)
    provenance = verify_sources(metadata, product, 'S2')
    for tile in metadata['tiles']:
        cloud = tile.get('cloudCoverage')
        if not isinstance(cloud, (int, float)) or not math.isfinite(cloud) or not 0 <= cloud <= max_cloud_pct:
            raise ValueError('Rendered optical source lacks acceptable cloud metadata')
    png = files['default.png']
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.shape != (height, width, 4) or not (image[:, :, 3] == 255).all():
        raise ValueError('Optical rendering is malformed or lacks full valid coverage')
    return {'product': product, 'acquisition_start_utc': product['start'],
            'acquisition_end_utc': product['end'], 'bbox': bbox,
            'source': 'Sentinel-2 L2A / CDSE Process', 'provenance': provenance,
            'cloud_cover_pct': product['attributes']['cloudCover'],
            'rgb_preview': 'data:image/png;base64,' + base64.b64encode(png).decode('ascii')}


def get_optical_evidence(token, bbox, timestamp, *, max_cloud_pct=20, window_days=10):
    try:
        ref = acquisition_time(timestamp)
        products = query_products(bbox, (ref - timedelta(days=window_days)).isoformat(),
                                  (ref + timedelta(days=window_days)).isoformat(), 'SENTINEL-2',
                                  "Attributes/OData.CSC.StringAttribute/any(a:a/Name eq 'productType' and a/OData.CSC.StringAttribute/Value eq 'S2MSI2A')")
        candidates = select_optical_products(products, bbox, timestamp, max_cloud_pct, window_days)
    except Exception as exc:
        return {'status': 'unavailable', 'reason': f'Optical catalog search failed ({type(exc).__name__}).'}
    if not candidates:
        return {'status': 'no_match', 'reason': 'No low-cloud optical scene available with full coverage within the requested window.',
                'max_cloud_pct': max_cloud_pct, 'window_days': window_days}
    failures = []
    for product in candidates[:2]:
        try:
            observation = render_optical(token, bbox, product, max_cloud_pct)
            return {'status': 'available', 'observation': observation, 'sar_acquisition_utc': timestamp,
                    'max_cloud_pct': max_cloud_pct, 'window_days': window_days, 'attempt_failures': failures,
                    'reason': 'True-color visual supplement. Tile-wide cloud metadata does not guarantee a cloud-free crop; dates may differ.'}
        except Exception as exc:
            failures.append({'product_id': product['id'], 'reason': f'Optical render/validation failed ({type(exc).__name__}).'})
    return {'status': 'unavailable', 'reason': 'Optical candidates found, but rendering/validation failed.', 'attempt_failures': failures}
