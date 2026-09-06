"""Unit tests for src/analysis/gfw_client.py — the real GFW 4Wings AIS
attribution client.

All GFW responses here are SYNTHETIC fixtures hand-built to match the real
v3 4Wings Report JSON shape (verified against a real
`public-global-presence:latest` response for the Singapore Strait, 2026-08-11
— see docs/status.md "GFW AIS attribution"). They are NOT real API output and
must never be presented as such: the point of these tests is the parsing,
dedup, radius-filter, null-handling and honest-failure logic, which
`tests/conftest.py` stubs out for every API test. A real external GFW
verification is recorded separately in docs/status.md.
"""

import pytest

from src.analysis import gfw_client as gfw

CENTER_LAT, CENTER_LON = 1.2, 103.8  # Singapore Strait, matches the doc example


class _FakeResp:
    """Minimal stand-in for requests.Response."""

    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _report(rows, dataset_key="public-global-presence:v4.0"):
    """Wrap synthetic vessel rows in the real nested 4Wings Report shape:
    {"entries": [ {"<dataset>:<version>": [ {row}, ... ]} ]}."""
    return {"total": 1, "entries": [{dataset_key: rows}]}


def _row(mmsi, name, lat, lon, date="2026-08-11"):
    # Only the fields gfw_client actually reads, plus a couple of the real
    # extras so the fixture looks like the genuine row shape.
    return {
        "mmsi": mmsi,
        "shipName": name,
        "lat": lat,
        "lon": lon,
        "date": date,
        "flag": "SGP",
        "vesselType": "CARGO",
        "hours": 1,
    }


def _patch_post(monkeypatch, response):
    """Route gfw_client's requests.post at a fixed synthetic response (or a
    callable(url, **kwargs) -> _FakeResp for request-assertion tests)."""
    def fake_post(url, **kwargs):
        return response(url, **kwargs) if callable(response) else response
    monkeypatch.setattr(gfw.requests, "post", fake_post)


# --- credentials gate -------------------------------------------------------

@pytest.mark.parametrize("token", ["", None])
def test_missing_token_is_skipped_not_empty(token):
    result = gfw.get_nearby_vessels(token, CENTER_LAT, CENTER_LON, "2026-08-11T11:24:44Z")
    assert result["status"] == "skipped_no_credentials"
    assert result["vessels"] == []


# --- normal parsing -------------------------------------------------------

def test_parses_real_shape_sorted_nearest_first(monkeypatch):
    rows = [
        _row("111", "FAR CARGO", 1.25, 103.80),      # ~5.6 km
        _row("222", "NEAR TUG", 1.21, 103.80),       # ~1.1 km
        _row("333", "MID BUNKER", 1.22, 103.79),     # ~2.5 km
    ]
    _patch_post(monkeypatch, _FakeResp(_report(rows)))

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z", radius_km=10.0)

    assert result["status"] == "ok"
    assert [v["mmsi"] for v in result["vessels"]] == ["222", "333", "111"]
    assert [v["vessel_name"] for v in result["vessels"]] == ["NEAR TUG", "MID BUNKER", "FAR CARGO"]
    # distances are monotonically increasing and each vessel carries the
    # honest grid-cell resolution disclosure
    dists = [v["distance_meters"] for v in result["vessels"]]
    assert dists == sorted(dists)
    assert all(v["position_resolution_m"] == pytest.approx(1572.4, abs=1.0) for v in result["vessels"])
    assert all(v["timestamp"] == "2026-08-11" for v in result["vessels"])


def test_dedupes_by_mmsi_keeping_closest_cell(monkeypatch):
    # Same real vessel reported in two 0.01deg cells on the same day — must
    # not occupy two candidate slots.
    rows = [
        _row("999", "WANDERER", 1.27, 103.80),   # ~7.8 km
        _row("999", "WANDERER", 1.205, 103.80),  # ~0.6 km
        _row("888", "OTHER", 1.24, 103.80),      # ~4.4 km
    ]
    _patch_post(monkeypatch, _FakeResp(_report(rows)))

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z", radius_km=10.0)

    mmsis = [v["mmsi"] for v in result["vessels"]]
    assert mmsis.count("999") == 1
    kept = next(v for v in result["vessels"] if v["mmsi"] == "999")
    assert kept["distance_meters"] < 1000  # the near cell, not the 7.8 km one


def test_radius_filter_excludes_vessels_outside(monkeypatch):
    rows = [
        _row("in", "INSIDE", 1.22, 103.80),    # ~2.2 km
        _row("out", "OUTSIDE", 1.20, 103.95),  # ~16.7 km
    ]
    _patch_post(monkeypatch, _FakeResp(_report(rows)))

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z", radius_km=10.0)

    assert [v["mmsi"] for v in result["vessels"]] == ["in"]


def test_top_n_caps_the_candidate_list(monkeypatch):
    rows = [_row(str(i), f"V{i}", 1.20 + i * 0.001, 103.80) for i in range(1, 9)]
    _patch_post(monkeypatch, _FakeResp(_report(rows)))

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z", radius_km=10.0, top_n=5)

    assert result["status"] == "ok"
    assert len(result["vessels"]) == 5


def test_rows_missing_position_or_mmsi_are_skipped(monkeypatch):
    rows = [
        {"mmsi": "no-pos", "shipName": "NO POS", "date": "2026-08-11"},        # no lat/lon
        {"lat": 1.21, "lon": 103.80, "shipName": "NO MMSI", "date": "x"},      # no mmsi
        _row("good", "GOOD", 1.21, 103.80),
    ]
    _patch_post(monkeypatch, _FakeResp(_report(rows)))

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z", radius_km=10.0)

    assert [v["mmsi"] for v in result["vessels"]] == ["good"]


# --- honest "no vessels" outcomes ---------------------------------------

def test_empty_entry_group_returns_empty_status(monkeypatch):
    # The real shape for a box/date with zero AIS presence: the outer entry
    # dict carries no dataset key at all (observed live for 2026-08-10).
    _patch_post(monkeypatch, _FakeResp({"total": 1, "entries": [{}]}))

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-10T11:24:44Z", radius_km=10.0)

    assert result["status"] == "empty"
    assert result["vessels"] == []
    assert "2026-08-10" in result["detail"]


def test_null_dataset_rows_do_not_crash(monkeypatch):
    # A genuinely vessel-free region returns `null` (not []) for its dataset
    # key — the real Chonos Archipelago case that once raised a 500.
    _patch_post(monkeypatch, _FakeResp({"total": 1, "entries": [{"public-global-presence:v4.0": None}]}))

    result = gfw.get_nearby_vessels("real-token", -45.0, -74.0,
                                    "2026-08-11T11:24:44Z", radius_km=10.0)

    assert result["status"] == "empty"
    assert result["vessels"] == []


def test_all_rows_outside_radius_returns_empty(monkeypatch):
    _patch_post(monkeypatch, _FakeResp(_report([_row("far", "FAR", 1.20, 103.99)])))

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z", radius_km=10.0)

    assert result["status"] == "empty"
    assert result["vessels"] == []


# --- honest failure paths (never fabricate) ---------------------------

@pytest.mark.parametrize("code", [401, 403])
def test_token_rejected_returns_error(monkeypatch, code):
    _patch_post(monkeypatch, _FakeResp({"message": "unauthorized"}, status_code=code))

    result = gfw.get_nearby_vessels("bad-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z")

    assert result["status"] == "error"
    assert result["vessels"] == []
    assert str(code) in result["detail"]


def test_server_error_returns_error(monkeypatch):
    _patch_post(monkeypatch, _FakeResp("upstream boom", status_code=502))

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z")

    assert result["status"] == "error"
    assert result["vessels"] == []


def test_network_exception_returns_error(monkeypatch):
    def boom(url, **kwargs):
        raise gfw.requests.RequestException("connection reset")
    monkeypatch.setattr(gfw.requests, "post", boom)

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z")

    assert result["status"] == "error"
    assert result["vessels"] == []


def test_unparseable_body_returns_error(monkeypatch):
    _patch_post(monkeypatch, _FakeResp(ValueError("not json"), status_code=200))

    result = gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                                    "2026-08-11T11:24:44Z")

    assert result["status"] == "error"
    assert result["vessels"] == []


# --- request construction ---------------------------------------------

def test_request_targets_the_real_presence_report_endpoint(monkeypatch):
    seen = {}

    def capture(url, **kwargs):
        seen["url"] = url
        seen["params"] = kwargs.get("params")
        seen["json"] = kwargs.get("json")
        seen["headers"] = kwargs.get("headers")
        return _FakeResp(_report([_row("1", "V", 1.21, 103.80)]))

    monkeypatch.setattr(gfw.requests, "post", capture)
    gfw.get_nearby_vessels("real-token", CENTER_LAT, CENTER_LON,
                           "2026-08-10T11:24:44Z", radius_km=10.0)

    assert seen["url"] == "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
    assert seen["params"]["datasets[0]"] == "public-global-presence:latest"
    assert seen["params"]["date-range"] == "2026-08-10,2026-08-11"  # acq day .. +1
    assert seen["params"]["spatial-resolution"] == "HIGH"
    assert seen["headers"]["Authorization"] == "Bearer real-token"
    assert seen["json"]["group-by"] == "MMSI"
    assert seen["json"]["geojson"]["type"] == "Polygon"
