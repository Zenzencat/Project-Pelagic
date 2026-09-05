import math
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from src.analysis import era5_wind_check as wind


def test_magnitude():
    assert wind.wind_magnitude(3, -4) == 5
    assert wind.wind_magnitude(0, 0) == 0
    with pytest.raises(ValueError):
        wind.wind_magnitude(math.nan, 1)


def test_missing_configuration(monkeypatch, tmp_path):
    monkeypatch.delenv('CDSAPI_KEY', raising=False)
    monkeypatch.delenv('CDSAPI_URL', raising=False)
    monkeypatch.setenv('CDSAPI_RC', str(tmp_path / 'missing'))
    result = wind.get_wind_evidence(1.2, 103.8, '2026-08-11T22:47:44Z')
    assert result['status'] == 'not_configured'
    assert 'wind_speed_ms' not in result


@pytest.mark.parametrize('timestamp', [None, 'bad', '2026-08-11T12:00:00'])
def test_invalid_timestamp(timestamp):
    assert wind.get_wind_evidence(1, 2, timestamp)['status'] == 'skipped'


def test_worker_success_and_failure(monkeypatch):
    connection = Mock()
    monkeypatch.setattr(wind, '_retrieve_wind', lambda *a: {'wind_speed_ms': 5, 'u10_ms': 3, 'v10_ms': 4})
    wind._wind_worker(connection, 1, 2, '2026-08-11T22:47:44Z')
    assert connection.send.call_args.args[0]['status'] == 'available'
    def fail(*args):
        raise ValueError('secret must not appear')
    monkeypatch.setattr(wind, '_retrieve_wind', fail)
    wind._wind_worker(connection, 1, 2, '2026-08-11T22:47:44Z')
    result = connection.send.call_args.args[0]
    assert result['status'] == 'unavailable'
    assert 'secret' not in str(result)
    assert 'wind_speed_ms' not in result


def test_timeout_stops_worker_without_measurement(monkeypatch):
    import multiprocessing
    monkeypatch.setenv('CDSAPI_URL', 'https://example.invalid')
    monkeypatch.setenv('CDSAPI_KEY', 'test-only')
    monkeypatch.setattr(wind, 'CDSAPI_AVAILABLE', True)
    parent, child, process = Mock(), Mock(), Mock()
    parent.poll.return_value = False
    process.pid = 123
    context = Mock()
    context.Pipe.return_value = (parent, child)
    context.Process.return_value = process
    monkeypatch.setattr(multiprocessing, 'get_context', lambda *a: context)
    result = wind.get_wind_evidence(1, 2, '2026-08-11T22:47:44Z', timeout_seconds=1)
    assert result['status'] == 'unavailable' and 'wind_speed_ms' not in result
    process.terminate.assert_called_once()
    parent.close.assert_called_once()


def test_netcdf_selection_and_request(monkeypatch, tmp_path):
    xr = pytest.importorskip('xarray', reason='Optional NetCDF stack unavailable')
    import numpy as np
    ds = xr.Dataset({name: (('valid_time', 'latitude', 'longitude'), [[[value]]], {'units': 'm s**-1'})
                     for name, value in [('u10', 3.), ('v10', -4.)]},
                    coords={'valid_time': [np.datetime64('2026-01-01T00:00')], 'latitude': [30.], 'longitude': [-10.]})
    client = Mock()
    def retrieve(dataset, request, target):
        assert request['year'] == ['2026'] and request['day'] == ['01']
        assert request['time'] == ['00:00']  # nearest hour crosses UTC year boundary
        ds.to_netcdf(target)
    client.retrieve.side_effect = retrieve
    result = wind._retrieve_wind(30, -10, '2025-12-31T23:45:00Z', client)
    assert result['wind_speed_ms'] == 5 and result['grid_lon'] == -10
    hour = datetime(2026, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(KeyError):
        wind.extract_wind_record(ds, 30, -10, hour.replace(hour=1))
    with pytest.raises(ValueError):
        wind.extract_wind_record(ds, 35, -10, hour)
    ds['u10'].values[:] = np.nan
    with pytest.raises(ValueError):
        wind.extract_wind_record(ds, 30, -10, hour)
