"""Remote rejection/malformed-data tests do not contact any external API."""
import io
import json
import tarfile
from unittest.mock import Mock

import pytest
import requests

from src.data import observation_catalog as catalog, sentinel_process as process
from test_observations import BBOX


def test_catalog_metadata_and_pagination(monkeypatch):
    item = {'Id': 'id', 'Name': 'product', 'ContentDate': {'Start': '2026-08-11T00:00:00Z', 'End': '2026-08-11T00:00:25Z'},
            'Attributes': [{'Name': 'cloudCover', 'Value': 10}]}
    first = Mock()
    first.json.return_value = {'value': [item], '@odata.nextLink': 'next'}
    second = Mock()
    second.json.return_value = {'value': []}
    get = Mock(side_effect=[first, second])
    monkeypatch.setattr(catalog.requests, 'get', get)
    result = catalog.query_products(BBOX, '2026-08-01T00:00:00Z', '2026-08-15T00:00:00Z', 'SENTINEL-2', 'true')
    assert result[0]['attributes']['cloudCover'] == 10
    assert get.call_args.kwargs['params']['$skip'] == 100
    assert '$expand' in get.call_args.kwargs['params']
    first.json.return_value = {'error': 'malformed'}
    get.side_effect = [first]
    with pytest.raises(ValueError):
        catalog.query_products(BBOX, 'start', 'end', 'SENTINEL-2', 'true')
    assert catalog.covers_bbox({'footprint': 'not a geometry'}, BBOX) is False


@pytest.mark.parametrize('status', [401, 403, 429, 500])
def test_process_rejected_without_secret_echo(monkeypatch, status):
    monkeypatch.setattr(process.requests, 'post', lambda *a, **kw: Mock(status_code=status, text='secret'))
    with pytest.raises(ValueError, match=f'HTTP {status}') as error:
        process.process_request('secret', {})
    assert 'secret' not in str(error.value)


def test_process_timeout_and_malformed_archive(monkeypatch):
    post = Mock(side_effect=requests.Timeout)
    monkeypatch.setattr(process.requests, 'post', post)
    with pytest.raises(requests.Timeout):
        process.process_request('test-token', {})
    post.side_effect = None
    post.return_value = Mock(status_code=200, content=b'not a satellite response')
    with pytest.raises(tarfile.ReadError):
        process.process_request('test-token', {})


def test_process_archive_metadata(monkeypatch):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w') as archive:
        data = json.dumps({'tiles': []}).encode()
        info = tarfile.TarInfo('userdata.json')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    monkeypatch.setattr(process.requests, 'post', lambda *a, **kw: Mock(status_code=200, content=stream.getvalue()))
    files, metadata = process.process_request('test-token', {})
    assert files['userdata.json'] and metadata == {'tiles': []}
