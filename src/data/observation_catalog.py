"""Shared public CDSE OData search and conservative footprint validation."""
from datetime import datetime, timezone

import requests

ODATA_URL = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'


def acquisition_time(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('Acquisition metadata must include a timezone')
    return dt.astimezone(timezone.utc)


def query_products(bbox, start, end, collection, extra_filter):
    min_lat, min_lon, max_lat, max_lon = bbox
    wkt = f'POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},{min_lon} {max_lat},{min_lon} {min_lat}))'
    filt = (f"Collection/Name eq '{collection}' and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
            f"ContentDate/Start ge {start} and ContentDate/Start le {end} and {extra_filter}")
    params = {'$filter': filt, '$top': 100, '$orderby': 'ContentDate/Start desc', '$expand': 'Attributes'}
    products = []
    # Bounded pagination. Refuse to turn a truncated search into a no-match claim.
    for page in range(10):
        response = requests.get(ODATA_URL, params={**params, '$skip': page * 100}, timeout=15)
        response.raise_for_status()
        body = response.json()
        if not isinstance(body.get('value'), list):
            raise ValueError('Malformed CDSE catalog response')
        for p in body['value']:
            attrs = {a['Name']: a.get('Value') for a in p.get('Attributes', [])}
            products.append({'id': p['Id'], 'name': p['Name'],
                             'start': p['ContentDate']['Start'], 'end': p['ContentDate'].get('End'),
                             'footprint': p.get('Footprint'), 'geometry': p.get('GeoFootprint'),
                             'attributes': attrs})
        if not body.get('@odata.nextLink') and len(body['value']) < 100:
            return products
    raise ValueError('CDSE catalog search exceeded 1000 products; narrow the search window')


def covers_bbox(product, bbox):
    # An intersection alone is insufficient. Unknown/invalid footprints are not matches.
    from shapely.geometry import box, shape
    from shapely import wkt
    from shapely.errors import ShapelyError
    try:
        geometry = product.get('geometry')
        if geometry:
            footprint = shape(geometry)
        else:
            value = product.get('footprint', '')
            value = value.split(';')[-1].rstrip("'")
            footprint = wkt.loads(value)
        min_lat, min_lon, max_lat, max_lon = bbox
        return footprint.is_valid and footprint.covers(box(min_lon, min_lat, max_lon, max_lat))
    except (ValueError, TypeError, AttributeError, ShapelyError):
        return False


def valid_product(product):
    try:
        return bool(product['id'] and product['name'] and
                    acquisition_time(product['end']) >= acquisition_time(product['start']))
    except (KeyError, ValueError, TypeError, AttributeError):
        return False
