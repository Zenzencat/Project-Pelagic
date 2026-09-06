"""Shared Process transport and returned-source checks; never synthesize imagery."""
import io
import json
import tarfile

import requests

from src.data.observation_catalog import acquisition_time

PROCESS_URL = 'https://sh.dataspace.copernicus.eu/api/v1/process'


def process_request(token, body):
    response = requests.post(PROCESS_URL, headers={'Authorization': f'Bearer {token}',
                             'Content-Type': 'application/json', 'Accept': 'application/x-tar'},
                             json=body, timeout=120)
    if response.status_code != 200:
        # Do not echo service responses or request headers into API errors.
        raise ValueError(f'CDSE Process request rejected: HTTP {response.status_code}')
    # Read only expected members, never extract untrusted archive paths.
    with tarfile.open(fileobj=io.BytesIO(response.content)) as archive:
        files = {}
        for name in ('default.tif', 'default.png', 'validity.tif', 'userdata.json'):
            try:
                member = archive.getmember(name)
            except KeyError:
                continue
            if not member.isfile() or member.size > 100_000_000:
                raise ValueError('Invalid Process response member')
            files[name] = archive.extractfile(member).read()
    metadata = json.loads(files['userdata.json'])
    return files, metadata


def verify_sources(metadata, product, sensor):
    """Require returned product names and times to match selected catalog identity."""
    tiles = metadata.get('tiles')
    if not isinstance(tiles, list) or not tiles:
        raise ValueError('Process response lacks source tiles')
    expected_name = product['name'].removesuffix('.SAFE')
    start, end = acquisition_time(product['start']), acquisition_time(product['end'])
    for tile in tiles:
        name = tile.get('sentinel1ProductId' if sensor == 'S1' else 'sentinel2ProductId')
        if not isinstance(name, str):
            raise ValueError('Process response contains an unidentified product')
        tile_name = name.removesuffix('.SAFE')
        if tile_name != expected_name:
            if sensor == 'S1':
                exp_core = '_'.join(expected_name.split('_')[:8])
                tile_core = '_'.join(tile_name.split('_')[:8])
                if not (exp_core and exp_core == tile_core):
                    raise ValueError('Process response contains a different or unidentified product')
            else:
                raise ValueError('Process response contains a different or unidentified product')
        dt = acquisition_time(tile['date'])
        # S1 slices are within seconds; S2 granule sensing times can be ~20m into the pass on the same date
        if sensor == 'S1':
            if not start.replace(microsecond=0) <= dt <= end:
                raise ValueError('Process source date differs from catalog acquisition')
        else:
            if dt.date() != start.date():
                raise ValueError('Process source date differs from catalog acquisition')
    return {'status': 'verified', 'catalog_product_id': product['id'],
            'catalog_product_name': product['name'], 'tiles': tiles}
