"""
Project Pelagic — FastAPI Backend Service
SWU Prasarnmit AI Engineering Final Project

This is the main API backend service. It serves prediction endpoints 
running U-Net inference, manages spatial metadata records in SQLite, and
exposes endpoints for UI rendering.
"""

import os
import sys
import json
import sqlite3
import hashlib
import uuid
import numpy as np
import cv2
import tifffile
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # optional; CDSE_CLIENT_ID/SECRET and GFW_TOKEN can be exported directly instead

# Add root folder to path to enable package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.api.database import init_db, get_db_connection
from src.models.unet import UNet
from src.data.preprocess import preprocess_for_prediction
from src.inference import run_tiled_inference
from src.analysis.geoutils import get_scene_geolocation
from src.analysis.contour import mask_to_polygons
from src.analysis.landmask import strip_land_pixels
from src.analysis.cdse_auth import get_cdse_token, CdseAuthError
from src.data.cdse_fetch import find_best_product, fetch_scene_geotiff, CdseFetchError
from src.analysis.gfw_client import get_nearby_vessels

# Real Sentinel-1 GRD ground sampling distance, confirmed directly from the
# holdout GeoTIFFs' ModelPixelScaleTag (consistent across all 30 scenes checked:
# 8.983152841195218e-05 deg/pixel, ~10m/pixel at these latitudes). Used as the
# fallback scale for scenes with no embedded georeferencing (synthetic
# verification scenes) -- previously this whole module used an arbitrary
# 0.00015 deg/pixel display-only mock constant for every scene.
REAL_PIXEL_SCALE_DEG = 8.983152841195218e-05

# 1. Initialize Database on startup
init_db()

app = FastAPI(title="Project Pelagic - Oil Slick Detection API", version="1.0.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In development, allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup paths and check U-Net checkpoint
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "model_real_v2_best.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
model_loaded = False
CHECKPOINT_HASH = None

print("[*] Checking U-Net model checkpoint...")
if os.path.exists(CHECKPOINT_PATH):
    try:
        with open(CHECKPOINT_PATH, "rb") as f:
            CHECKPOINT_HASH = hashlib.md5(f.read()).hexdigest()[:12]
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
        model = UNet(in_channels=2, out_channels=1)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']
        else:
            model.load_state_dict(checkpoint)
            epoch = "N/A"
        model.to(device)
        model.eval()
        model_loaded = True
        print(f"[+] U-Net model loaded successfully on device: {device} (Epoch: {epoch})")
    except Exception as e:
        print(f"[!] Warning: Failed to load U-Net checkpoint: {e}", file=sys.stderr)
else:
    print(f"[!] WARNING: Checkpoint '{CHECKPOINT_PATH}' not found. Inference endpoints will operate in safe-fallback mode (HTTP 503).")

# Request schemas
class PredictRequest(BaseModel):
    scene_id: str

class LiveFetchRequest(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    date_from: str  # "YYYY-MM-DD"
    date_to: str    # "YYYY-MM-DD"
    radius_km: float = 10.0

# 3. Routes
@app.get("/health")
def health_check():
    """Returns api status and model loading status."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "device": str(device)
    }

@app.get("/api/detections")
def get_detections():
    """Retrieves all historical oil slick detection records."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    SELECT id, scene_id, detected_at, confidence_score, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon, geojson_mask, image_path,
           source, acquisition_start_utc, acquisition_end_utc, vessel_attribution_status
    FROM detections
    ORDER BY detected_at DESC;
    """)
    rows = cursor.fetchall()
    conn.close()

    detections = []
    for r in rows:
        detections.append({
            "id": r["id"],
            "scene_id": r["scene_id"],
            "detected_at": r["detected_at"],
            "confidence_score": r["confidence_score"],
            "bbox": [r["bbox_min_lat"], r["bbox_min_lon"], r["bbox_max_lat"], r["bbox_max_lon"]],
            "geojson_mask": json.loads(r["geojson_mask"]),
            "image_path": r["image_path"],
            "source": r["source"],
            "acquisition_start_utc": r["acquisition_start_utc"],
            "acquisition_end_utc": r["acquisition_end_utc"],
            "vessel_attribution_status": r["vessel_attribution_status"],
        })
    return detections

@app.get("/api/detections/{det_id}")
def get_detection_details(det_id: int):
    """Retrieves metadata and nearby vessels for a specific detection."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Query detection
    cursor.execute("SELECT * FROM detections WHERE id = ?;", (det_id,))
    det_row = cursor.fetchone()
    if not det_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Detection record not found")
        
    # Query nearby vessels
    cursor.execute("SELECT * FROM nearby_vessels WHERE detection_id = ?;", (det_id,))
    vessel_rows = cursor.fetchall()
    conn.close()
    
    vessels = []
    for v in vessel_rows:
        vessels.append({
            "id": v["id"],
            "mmsi": v["mmsi"],
            "vessel_name": v["vessel_name"],
            "latitude": v["latitude"],
            "longitude": v["longitude"],
            "timestamp": v["timestamp"],
            "distance_meters": v["distance_meters"],
            "position_resolution_m": v["position_resolution_m"]
        })
        
    return {
        "id": det_row["id"],
        "scene_id": det_row["scene_id"],
        "detected_at": det_row["detected_at"],
        "confidence_score": det_row["confidence_score"],
        "bbox": [det_row["bbox_min_lat"], det_row["bbox_min_lon"], det_row["bbox_max_lat"], det_row["bbox_max_lon"]],
        "geojson_mask": json.loads(det_row["geojson_mask"]),
        "image_path": det_row["image_path"],
        "source": det_row["source"],
        "acquisition_start_utc": det_row["acquisition_start_utc"],
        "acquisition_end_utc": det_row["acquisition_end_utc"],
        "cdse_product_id": det_row["cdse_product_id"],
        "vessel_attribution_status": det_row["vessel_attribution_status"],
        "vessel_search_radius_km": det_row["vessel_search_radius_km"],
        "nearby_vessels": vessels
    }

@app.post("/api/predict", status_code=status.HTTP_201_CREATED)
def predict(payload: PredictRequest):
    """
    Accepts a Sentinel-1 scene ID, runs U-Net inference,
    traces contours, logs the detection in SQLite, and returns results.
    """
    if not model_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="U-Net model is not loaded. Train the baseline model first."
        )
        
    scene_id = payload.scene_id
    img_path = os.path.join(BASE_DIR, "data", "holdout", "images", f"{scene_id}.tif")
    scene_source = "holdout"
    if not os.path.exists(img_path):
        img_path = os.path.join(BASE_DIR, "data", "synthetic", "images", f"{scene_id}.tif")
        scene_source = "synthetic"

    if not os.path.exists(img_path):
        raise HTTPException(
            status_code=404,
            detail=f"SAR Scene file '{scene_id}.tif' not found in holdout or synthetic datasets."
        )
        
    # 1. Read GeoTIFF image (VV, VH channels)
    try:
        image_raw = tifffile.imread(img_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading GeoTIFF: {e}")
        
    # 2. Run Preprocessing (shared with /api/live/fetch -- see
    # src/data/preprocess.py::preprocess_for_prediction)
    norm = preprocess_for_prediction(image_raw)

    # 3. U-Net Inference
    # Tiled to match the 256x256 patch regime the model was trained/evaluated on
    # (see src/inference.py and src/evaluate_holdout.py) rather than a single
    # whole-image forward pass, which puts every interior pixel in a receptive-field
    # context the model never saw during training.
    probs, preds_bin = run_tiled_inference(model, norm, device)
    preds = preds_bin * 255
        
    # 4. Contour Tracing (Convert binary mask to GeoJSON Polygon, via
    # src/analysis/contour.py's mask_to_polygons -- see that module for the
    # simplification-tolerance and smoothing rationale)

    # Coordinate system mapping: real WGS84 georeferencing embedded in the
    # scene's GeoTIFF (all real Trujillo-Acatitla scenes carry this). Falls
    # back to a fixed placeholder point only for scenes with no embedded
    # georeferencing (synthetic verification scenes have no real-world
    # location to report) -- the pixel scale itself is always the real GSD.
    try:
        geo = get_scene_geolocation(img_path)
        center_lat = geo["center_lat"]
        center_lon = geo["center_lon"]
        scale = geo["pixel_scale_deg"]
        scene_bbox = (geo["min_lat"], geo["min_lon"], geo["max_lat"], geo["max_lon"])
        has_real_geo = True
    except ValueError:
        center_lat = 9.0
        center_lon = 100.5
        scale = REAL_PIXEL_SCALE_DEG
        scene_bbox = None
        has_real_geo = False
    H, W = preds.shape

    # Land-sea masking (src/analysis/landmask.py): strip predicted "oil"
    # pixels that fall on land before contour extraction, so a SAR dark-
    # backscatter false positive over land never reaches the reported
    # polygon/area/confidence. Only meaningful with a real lat/lon center --
    # skipped for the synthetic-scene placeholder-coordinate fallback above,
    # which has no real location to check land/sea against.
    if has_real_geo:
        preds, _land_stats = strip_land_pixels(preds, center_lat, center_lon, scale, bbox=scene_bbox)

    polygons = mask_to_polygons(preds, center_lat, center_lon, scale)

    # Fallback to a tiny mock polygon if no slicks were segmented
    if not polygons:
        polygons = [[
            [center_lon - 0.01, center_lat - 0.01],
            [center_lon + 0.01, center_lat - 0.01],
            [center_lon + 0.01, center_lat + 0.01],
            [center_lon - 0.01, center_lat + 0.01],
            [center_lon - 0.01, center_lat - 0.01]
        ]]
        
    geojson = {
        "type": "Polygon",
        "coordinates": polygons
    }
    
    # Calculate average confidence of predicted pixels
    slick_pixels = probs[preds == 255]
    confidence_score = float(np.mean(slick_pixels)) if len(slick_pixels) > 0 else 0.0
    
    # 5. Save the output mask PNG
    processed_mask_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_mask_dir, exist_ok=True)
    mask_png_name = f"{scene_id}_mask.png"
    mask_png_path = os.path.join(processed_mask_dir, mask_png_name)
    cv2.imwrite(mask_png_path, preds)
    
    # Calculate scene footprint bbox
    bbox_min_lat = center_lat - (H // 2) * scale
    bbox_min_lon = center_lon - (W // 2) * scale
    bbox_max_lat = center_lat + (H // 2) * scale
    bbox_max_lon = center_lon + (W // 2) * scale
    
    # Write to Database
    conn = get_db_connection()
    cursor = conn.cursor()

    # No real acquisition timestamp exists for any holdout/synthetic scene
    # (confirmed extensively -- see docs/status.md), so real GFW attribution
    # cannot honestly run here (it would mean defaulting the timestamp,
    # which the task this was built for explicitly forbids). This replaces
    # the old hardcoded mock_vessels with an honest "why there's nothing"
    # status instead of fake vessel rows -- see src/analysis/gfw_client.py
    # for the real attribution path, used by /api/live/fetch below.
    vessel_status = "skipped_no_timestamp"

    def insert_vessels(det_id, vessels):
        if not vessels:
            return
        cursor.executemany("""
        INSERT INTO nearby_vessels (detection_id, mmsi, vessel_name, latitude, longitude, timestamp, distance_meters)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, [
            (det_id, v["mmsi"], v["vessel_name"], v["latitude"], v["longitude"], v["timestamp"], v["distance_meters"])
            for v in vessels
        ])

    try:
        cursor.execute("""
        INSERT INTO detections (scene_id, confidence_score, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon, geojson_mask, image_path, checkpoint_hash, source, vessel_attribution_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            scene_id,
            confidence_score,
            bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon,
            json.dumps(geojson),
            f"data/processed/{mask_png_name}",
            CHECKPOINT_HASH,
            scene_source,
            vessel_status
        ))
        det_id = cursor.lastrowid
        conn.commit()

    except sqlite3.IntegrityError:
        # scene_id already has a row (UNIQUE constraint). Only treat it as a
        # valid cache hit if it was produced by the checkpoint currently loaded —
        # otherwise this is stale data left by a since-swapped checkpoint, and
        # silently returning it would misreport a re-analysis as unchanged.
        cursor.execute("SELECT id, checkpoint_hash FROM detections WHERE scene_id = ?;", (scene_id,))
        existing = cursor.fetchone()
        det_id = existing["id"]

        if existing["checkpoint_hash"] != CHECKPOINT_HASH:
            cursor.execute("""
            UPDATE detections SET
                confidence_score = ?, bbox_min_lat = ?, bbox_min_lon = ?, bbox_max_lat = ?, bbox_max_lon = ?,
                geojson_mask = ?, image_path = ?, checkpoint_hash = ?, detected_at = CURRENT_TIMESTAMP,
                source = ?, vessel_attribution_status = ?
            WHERE id = ?;
            """, (
                confidence_score,
                bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon,
                json.dumps(geojson),
                f"data/processed/{mask_png_name}",
                CHECKPOINT_HASH,
                scene_source,
                vessel_status,
                det_id
            ))
            cursor.execute("DELETE FROM nearby_vessels WHERE detection_id = ?;", (det_id,))
            conn.commit()
    finally:
        conn.close()

    # Return detail response
    return get_detection_details(det_id)


@app.post("/api/live/fetch")
def live_fetch(payload: LiveFetchRequest):
    """
    Live pipeline (additive, alongside the 4 cached demo scenes): search CDSE
    for a real Sentinel-1 GRD product covering the given bbox/date range,
    fetch real calibrated+orthorectified imagery via the Sentinel Hub Process
    API (src/data/cdse_fetch.py), run it through the same preprocessing +
    tiled U-Net inference path as /api/predict, and attribute the detection
    to real nearby AIS vessels via GFW (src/analysis/gfw_client.py) using the
    scene's real acquisition datetime.

    Returns a status envelope instead of relying on HTTP status codes alone,
    so an expected "not configured" / "no product found" state doesn't read
    as a hard failure the way a 4xx/5xx would in the frontend's existing
    error-banner path:
      {"status": "OK"|"NOT_CONFIGURED"|"SKIPPED"|"ERROR", "detail": str, "detection": {...}|null}
    """
    if not model_loaded:
        return {"status": "ERROR", "detail": "U-Net model is not loaded. Train the baseline model first.", "detection": None}

    cdse_client_id = os.environ.get("CDSE_CLIENT_ID")
    cdse_client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not cdse_client_id or not cdse_client_secret:
        return {
            "status": "NOT_CONFIGURED",
            "detail": "CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set — copy .env.example to .env and fill in real CDSE OAuth2 client credentials.",
            "detection": None,
        }

    # 1. Real product search (public OData catalog, no auth needed)
    try:
        product = find_best_product(
            payload.min_lat, payload.min_lon, payload.max_lat, payload.max_lon,
            payload.date_from, payload.date_to,
        )
    except Exception as e:
        return {"status": "ERROR", "detail": f"CDSE product search failed: {e}", "detection": None}

    if not product:
        return {
            "status": "SKIPPED",
            "detail": "No Sentinel-1 GRD product found for this bounding box and date range.",
            "detection": None,
        }

    # 2. Real OAuth2 token, then real calibrated pixel data
    try:
        token = get_cdse_token(cdse_client_id, cdse_client_secret)
    except CdseAuthError as e:
        return {"status": "NOT_CONFIGURED", "detail": f"CDSE credentials rejected: {e}", "detection": None}

    live_dir = os.path.join(BASE_DIR, "data", "raw", "live")
    os.makedirs(live_dir, exist_ok=True)
    scene_id = f"live_{uuid.uuid4().hex[:12]}"
    img_path = os.path.join(live_dir, f"{scene_id}.tif")

    try:
        fetch_scene_geotiff(
            token, payload.min_lat, payload.min_lon, payload.max_lat, payload.max_lon,
            product["start"], img_path,
        )
    except CdseFetchError as e:
        return {"status": "ERROR", "detail": f"CDSE Process API fetch failed: {e}", "detection": None}

    # 3. Read + preprocess (shared with /api/predict) + tiled inference
    try:
        image_raw = tifffile.imread(img_path)
    except Exception as e:
        return {"status": "ERROR", "detail": f"Error reading fetched GeoTIFF: {e}", "detection": None}

    norm = preprocess_for_prediction(image_raw, already_calibrated=True)
    probs, preds_bin = run_tiled_inference(model, norm, device)
    preds = preds_bin * 255

    try:
        geo = get_scene_geolocation(img_path)
    except ValueError as e:
        return {"status": "ERROR", "detail": f"Fetched scene has no embedded georeferencing: {e}", "detection": None}
    center_lat, center_lon, scale = geo["center_lat"], geo["center_lon"], geo["pixel_scale_deg"]
    scene_bbox = (geo["min_lat"], geo["min_lon"], geo["max_lat"], geo["max_lon"])
    H, W = preds.shape

    # Land-sea masking (src/analysis/landmask.py): live-fetched scenes cover
    # arbitrary real-world bboxes (unlike the curated cached demo scenes,
    # pre-verified as open water), so a SAR dark-backscatter false positive
    # over land is a real, expected risk here -- strip it at the pixel level
    # before contour extraction so the reported polygon/area/confidence are
    # all corrected, not just the map drawing. Always applicable here since
    # geo above is always real (the except branch above already returned).
    # bbox enables the real-OSM island refinement (src/analysis/osm_coastline.py,
    # Round 16) on top of the ~930m raster (Round 15).
    preds, land_stats = strip_land_pixels(preds, center_lat, center_lon, scale, bbox=scene_bbox)

    polygons = mask_to_polygons(preds, center_lat, center_lon, scale)
    if not polygons:
        polygons = [[
            [center_lon - 0.01, center_lat - 0.01],
            [center_lon + 0.01, center_lat - 0.01],
            [center_lon + 0.01, center_lat + 0.01],
            [center_lon - 0.01, center_lat + 0.01],
            [center_lon - 0.01, center_lat - 0.01],
        ]]
    geojson = {"type": "Polygon", "coordinates": polygons}

    slick_pixels = probs[preds == 255]
    confidence_score = float(np.mean(slick_pixels)) if len(slick_pixels) > 0 else 0.0

    processed_mask_dir = os.path.join(BASE_DIR, "data", "processed")
    os.makedirs(processed_mask_dir, exist_ok=True)
    mask_png_name = f"{scene_id}_mask.png"
    cv2.imwrite(os.path.join(processed_mask_dir, mask_png_name), preds)

    bbox_min_lat = center_lat - (H // 2) * scale
    bbox_min_lon = center_lon - (W // 2) * scale
    bbox_max_lat = center_lat + (H // 2) * scale
    bbox_max_lon = center_lon + (W // 2) * scale

    # 4. Real GFW attribution -- possible here (unlike /api/predict's cached
    # holdout/synthetic scenes) because this detection has a real acquisition
    # datetime straight from CDSE's catalog (product["start"]), never defaulted.
    gfw_token = os.environ.get("GFW_TOKEN")
    gfw_result = get_nearby_vessels(
        gfw_token, center_lat, center_lon, product["start"], radius_km=payload.radius_km,
    )

    # 5. Write to DB
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO detections (
        scene_id, confidence_score, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon,
        geojson_mask, image_path, checkpoint_hash, source,
        acquisition_start_utc, acquisition_end_utc, cdse_product_id,
        vessel_attribution_status, vessel_search_radius_km
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """, (
        scene_id, confidence_score,
        bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon,
        json.dumps(geojson),
        f"data/processed/{mask_png_name}",
        CHECKPOINT_HASH,
        "live",
        product["start"], product["end"], product["id"],
        gfw_result["status"], payload.radius_km,
    ))
    det_id = cursor.lastrowid
    if gfw_result["vessels"]:
        cursor.executemany("""
        INSERT INTO nearby_vessels (detection_id, mmsi, vessel_name, latitude, longitude, timestamp, distance_meters, position_resolution_m)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, [
            (det_id, v["mmsi"], v["vessel_name"], v["latitude"], v["longitude"], v["timestamp"], v["distance_meters"], v["position_resolution_m"])
            for v in gfw_result["vessels"]
        ])
    conn.commit()
    conn.close()

    osm_note = (
        f" ({land_stats['oil_px_removed_by_osm_only']:,} px via real OSM island refinement)"
        if land_stats.get("oil_px_removed_by_osm_only") else ""
    )
    land_note = (
        f" Land-sea mask removed {land_stats['oil_px_removed_on_land']:,} of "
        f"{land_stats['oil_px_total_before']:,} raw predicted oil px "
        f"({land_stats['oil_px_removed_on_land'] / land_stats['oil_px_total_before'] * 100:.1f}%) "
        f"that fell on land{osm_note}."
        if land_stats["oil_px_total_before"] > 0 else ""
    )
    return {
        "status": "OK",
        "detail": f"Live detection complete ({product['name']}).{land_note} GFW attribution: {gfw_result['detail']}",
        "detection": get_detection_details(det_id),
    }
