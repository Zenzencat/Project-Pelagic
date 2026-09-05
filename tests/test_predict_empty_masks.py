"""Regression coverage for truthful empty / post-filter prediction geometry."""

import json
import os
from pathlib import Path
import cv2
import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient


@pytest.fixture
def api(tmp_path_factory, monkeypatch):
    root = tmp_path_factory.mktemp("pelagic-api")
    from src.api import database

    monkeypatch.setattr(database, "DB_DIR", str(root / "data"))
    monkeypatch.setattr(database, "DB_PATH", str(root / "data" / "test.db"))

    import os
    real_exists = os.path.exists
    checkpoint = str(Path(__file__).parents[1] / "checkpoints" / "model_real_v2_best.pt")
    def exists_without_checkpoint(path):
        return False if os.path.abspath(path) == checkpoint else real_exists(path)
    monkeypatch.setattr(os.path, "exists", exists_without_checkpoint)
    import src.api.main as main

    monkeypatch.setattr(main, "BASE_DIR", str(root))
    image_dir = root / "data" / "holdout" / "images"
    image_dir.mkdir(parents=True)
    tifffile.imwrite(image_dir / "scene.tif", np.zeros((8, 8, 2), dtype=np.float32))

    # The endpoint is exercised with the same request, DB, contour and
    # response serialization code as production, without a trained checkpoint.
    monkeypatch.setattr(main, "model_loaded", True)
    monkeypatch.setattr(main, "model", object())
    monkeypatch.setattr(main, "CHECKPOINT_HASH", "test-checkpoint")
    monkeypatch.setattr(main, "preprocess_for_prediction", lambda image: image)
    monkeypatch.setattr(main, "get_scene_geolocation", lambda path: (_ for _ in ()).throw(ValueError("synthetic")))
    monkeypatch.setattr(main, "strip_land_pixels", lambda mask, *args, **kwargs: (mask, {}))

    database.init_db()
    conn = database.get_db_connection()
    conn.execute("DELETE FROM nearby_vessels")
    conn.execute("DELETE FROM detections")
    conn.commit()
    conn.close()
    return main


@pytest.fixture
def client(api):
    with TestClient(api.app) as client:
        yield client


def set_inference(api, mask, monkeypatch, confidence=0.9):
    mask = np.asarray(mask, dtype=np.uint8)

    def fake_inference(model, norm, device):
        return np.where(mask > 0, confidence, 0.01).astype(np.float32), mask.copy()

    monkeypatch.setattr(api, "run_tiled_inference", fake_inference)


def ensure_scene(api, scene_id):
    path = Path(api.BASE_DIR) / "data" / "holdout" / "images" / f"{scene_id}.tif"
    if not path.exists():
        tifffile.imwrite(path, np.zeros((8, 8, 2), dtype=np.float32))


def test_one_pixel_empty_geometry_keeps_confidence_and_serializes(api, client, monkeypatch):
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[3, 3] = 1
    set_inference(api, mask, monkeypatch)

    response = client.post("/api/predict", json={"scene_id": "scene"})
    assert response.status_code == 201
    body = response.json()
    assert body["confidence_score"] == pytest.approx(0.9)
    assert body["geojson_mask"] == {"type": "Polygon", "coordinates": []}
    assert body["nearby_vessels"] == []
    assert cv2.imread(os.path.join(api.BASE_DIR, body["image_path"]), cv2.IMREAD_GRAYSCALE).max() == 255

    assert client.get("/api/detections").json()[0]["geojson_mask"]["coordinates"] == []
    assert client.get(f"/api/detections/{body['id']}").json()["geojson_mask"]["coordinates"] == []


def test_all_zero_mask_is_empty_and_black(api, client, monkeypatch):
    ensure_scene(api, "zero")
    set_inference(api, np.zeros((8, 8), dtype=np.uint8), monkeypatch)
    response = client.post("/api/predict", json={"scene_id": "zero"})
    assert response.status_code == 201
    body = response.json()
    assert body["confidence_score"] == 0
    assert body["geojson_mask"]["coordinates"] == []
    assert cv2.imread(os.path.join(api.BASE_DIR, body["image_path"]), cv2.IMREAD_GRAYSCALE).max() == 0


def test_legacy_hash_repair_removes_stale_vessels_and_is_stable(api, client, monkeypatch):
    ensure_scene(api, "legacy")
    set_inference(api, np.zeros((8, 8), dtype=np.uint8), monkeypatch, confidence=0.9)
    conn = api.get_db_connection()
    conn.execute(
        """INSERT INTO detections
        (scene_id, confidence_score, bbox_min_lat, bbox_min_lon, bbox_max_lat,
         bbox_max_lon, geojson_mask, image_path, checkpoint_hash, source,
         vessel_attribution_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("legacy", 0.8, 8.9, 100.4, 9.1, 100.6,
         json.dumps({"type": "Polygon", "coordinates": [[[100.49, 8.99], [100.51, 8.99], [100.51, 9.01], [100.49, 9.01], [100.49, 8.99]]]}),
         "old.png", "test-checkpoint", "holdout", "skipped_no_timestamp"),
    )
    det_id = conn.execute("SELECT id FROM detections WHERE scene_id='legacy'").fetchone()[0]
    conn.execute(
        "INSERT INTO nearby_vessels (detection_id, mmsi, latitude, longitude, timestamp, distance_meters) VALUES (?, ?, ?, ?, ?, ?)",
        (det_id, 123, 9, 100.5, "2026-01-01", 10),
    )
    conn.commit()
    conn.close()

    first = client.post("/api/predict", json={"scene_id": "legacy"}).json()
    assert first["id"] == det_id
    assert first["geojson_mask"]["coordinates"] == []
    assert first["nearby_vessels"] == []
    conn = api.get_db_connection()
    assert conn.execute("SELECT checkpoint_hash FROM detections WHERE id = ?", (det_id,)).fetchone()[0] == "test-checkpoint+empty_geometry_v1"
    conn.close()
    second = client.post("/api/predict", json={"scene_id": "legacy"}).json()
    assert second["id"] == det_id
    assert second["geojson_mask"]["coordinates"] == []
    assert len([row for row in client.get("/api/detections").json() if row["scene_id"] == "legacy"]) == 1


def test_landmask_removal_and_filter_suppression_are_empty_but_default_is_unchanged(api, client, monkeypatch):
    ensure_scene(api, "land")
    ensure_scene(api, "filtered")
    full = np.ones((8, 8), dtype=np.uint8)
    set_inference(api, full, monkeypatch)
    monkeypatch.setattr(api, "get_scene_geolocation", lambda path: {
        "center_lat": 9.0, "center_lon": 100.5, "pixel_scale_deg": 0.001,
        "min_lat": 8.9, "min_lon": 100.4, "max_lat": 9.1, "max_lon": 100.6,
    })
    monkeypatch.setattr(api, "strip_land_pixels", lambda mask, *args, **kwargs: (np.zeros_like(mask), {"oil_px_removed_on_land": 64}))
    removed = client.post("/api/predict", json={"scene_id": "land"}).json()
    assert removed["confidence_score"] == 0
    assert removed["geojson_mask"]["coordinates"] == []

    monkeypatch.setattr(api, "strip_land_pixels", lambda mask, *args, **kwargs: (mask, {}))
    monkeypatch.setattr(api, "classify_and_filter", lambda image, probs, mask: (np.zeros_like(mask), {"decision": "lookalike"}))
    filtered = client.post("/api/predict", json={"scene_id": "filtered", "apply_lookalike_filter": True}).json()
    assert filtered["confidence_score"] == 0
    assert filtered["geojson_mask"]["coordinates"] == []
    assert filtered["lookalike_filter"]["decision"] == "lookalike"

    genuine = client.post("/api/predict", json={"scene_id": "filtered"}).json()
    assert genuine["confidence_score"] == pytest.approx(0.9)
    assert genuine["geojson_mask"]["coordinates"]


def test_regular_mask_keeps_real_contour(api, client, monkeypatch):
    ensure_scene(api, "regular")
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    set_inference(api, mask, monkeypatch)
    body = client.post("/api/predict", json={"scene_id": "regular"}).json()
    assert body["confidence_score"] == pytest.approx(0.9)
    assert body["geojson_mask"]["coordinates"]
    assert len(body["geojson_mask"]["coordinates"][0]) >= 4
