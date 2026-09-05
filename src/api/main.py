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
from pydantic import BaseModel, Field, model_validator

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
from src.analysis.lookalike_filter import classify_and_filter
from src.analysis.cdse_auth import get_cdse_token, CdseAuthError
from src.data.cdse_fetch import find_best_product, fetch_scene_geotiff, CdseFetchError
from src.analysis.gfw_client import get_nearby_vessels
from src.analysis.era5_wind_check import get_wind_evidence
from src.data.sentinel2_optical import get_optical_evidence

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
    # Phase 4 (docs/status.md): opt-in, additive post-hoc lookalike filter
    # (src/analysis/lookalike_filter.py) on top of the existing v2 U-Net.
    # Default False -- every existing caller (the 4 verified cached demo
    # scenes, any client that predates this field) gets byte-identical
    # behavior to before this was added. NOT a silent replacement.
    apply_lookalike_filter: bool = False

class LiveFetchRequest(BaseModel):
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    date_from: str  # "YYYY-MM-DD"
    date_to: str    # "YYYY-MM-DD"
    radius_km: float = 10.0
    include_era5: bool = False
    include_optical: bool = False
    optical_window_days: int = Field(default=10, ge=1, le=30)
    optical_max_cloud_pct: float = Field(default=20, ge=0, le=100, allow_inf_nan=False)
    include_temporal: bool = False
    temporal_window_days: int = Field(default=30, ge=1, le=90)

    @model_validator(mode="after")
    def validate_region(self):
        import math
        from datetime import date
        if not all(math.isfinite(v) for v in (self.min_lat, self.min_lon, self.max_lat, self.max_lon, self.radius_km)):
            raise ValueError("Coordinates and radius must be finite")
        if not (-90 <= self.min_lat < self.max_lat <= 90 and -180 <= self.min_lon < self.max_lon <= 180):
            raise ValueError("Invalid bounding box; dateline-crossing regions must be split")
        if date.fromisoformat(self.date_from) > date.fromisoformat(self.date_to):
            raise ValueError("date_from must not follow date_to")
        if self.radius_km <= 0:
            raise ValueError("radius_km must be positive")
        return self

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
        "supplementary": json.loads(det_row["supplementary_json"] or "{}"),
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

    # 3b. Phase 4, opt-in only (docs/status.md): post-hoc lookalike filter.
    # Untouched (preds_bin passes through exactly as before) unless the
    # caller explicitly sets apply_lookalike_filter=True -- see PredictRequest.
    lookalike_filter_info = None
    if payload.apply_lookalike_filter:
        preds_bin, lookalike_filter_info = classify_and_filter(image_raw, probs, preds_bin)

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

    # get_scene_geolocation() only returns the GeoTIFF's ScaleX as
    # "pixel_scale_deg", so derive the latitude degrees-per-pixel from the
    # real corner bounds instead of assuming square pixels. For every cached
    # holdout scene ScaleY == ScaleX, so this is a no-op there; it matters for
    # any scene whose bbox isn't square in degrees. Both the land mask and the
    # contour tracing below must use the SAME scale_y or they end up on two
    # different latitude grids -- see strip_land_pixels' docstring.
    scale_y = (geo["max_lat"] - geo["min_lat"]) / H if has_real_geo else scale

    # Land-sea masking (src/analysis/landmask.py): strip predicted "oil"
    # pixels that fall on land before contour extraction, so a SAR dark-
    # backscatter false positive over land never reaches the reported
    # polygon/area/confidence. Only meaningful with a real lat/lon center --
    # skipped for the synthetic-scene placeholder-coordinate fallback above,
    # which has no real location to check land/sea against.
    if has_real_geo:
        preds, _land_stats = strip_land_pixels(preds, center_lat, center_lon, scale,
                                               bbox=scene_bbox, scale_y=scale_y)

    polygons = mask_to_polygons(preds, center_lat, center_lon, scale, scale_y=scale_y)

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

    # The versioned empty-geometry suffix also invalidates pre-fix cached rows
    # whose fabricated placeholder polygon must be repaired on POST.
    # Phase 4: fold apply_lookalike_filter into the cache key so toggling it
    # for an already-cached scene_id can't silently return a stale result
    # produced under the other setting -- reuses Resolution #6's existing
    # "hash mismatch -> overwrite in place" cache-invalidation logic (that
    # mechanism was built for checkpoint swaps, but "which detection recipe
    # produced this row" generalizes the same way).
    effective_hash = CHECKPOINT_HASH + "+empty_geometry_v1" + ("+lookalike_filter" if payload.apply_lookalike_filter else "")

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
            effective_hash,
            scene_source,
            vessel_status
        ))
        det_id = cursor.lastrowid
        conn.commit()

    except sqlite3.IntegrityError:
        # scene_id already has a row (UNIQUE constraint). Only treat it as a
        # valid cache hit if it was produced by the same checkpoint AND the
        # same apply_lookalike_filter setting currently in effect (both are
        # folded into effective_hash) -- otherwise this is stale data left by
        # a since-swapped checkpoint or a different filter setting, and
        # silently returning it would misreport a re-analysis as unchanged.
        cursor.execute("SELECT id, checkpoint_hash FROM detections WHERE scene_id = ?;", (scene_id,))
        existing = cursor.fetchone()
        det_id = existing["id"]

        if existing["checkpoint_hash"] != effective_hash:
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
                effective_hash,
                scene_source,
                vessel_status,
                det_id
            ))
            cursor.execute("DELETE FROM nearby_vessels WHERE detection_id = ?;", (det_id,))
            conn.commit()
    finally:
        conn.close()

    # Return detail response
    response = get_detection_details(det_id)
    if lookalike_filter_info is not None:
        response["lookalike_filter"] = lookalike_filter_info
    return response


def _png_data_uri(image):
    import base64
    height, width = image.shape[:2]
    factor = min(1, 640 / max(height, width))
    if factor < 1:
        image = cv2.resize(image, (round(width * factor), round(height * factor)), interpolation=cv2.INTER_AREA)
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('Could not encode observation preview')
    return 'data:image/png;base64,' + base64.b64encode(data).decode('ascii')


def _analyze_live_scene(img_path, scene_id, product, provenance):
    """One primary/comparison pipeline: calibration, U-Net, land mask, contours."""
    image_raw = tifffile.imread(img_path)
    norm = preprocess_for_prediction(image_raw, already_calibrated=True)
    probs, preds_bin = run_tiled_inference(model, norm, device)
    if not np.isfinite(probs).all():
        raise ValueError('Inference returned nonfinite probabilities')
    geo = get_scene_geolocation(img_path)
    bbox = [geo['min_lat'], geo['min_lon'], geo['max_lat'], geo['max_lon']]
    center_lat, center_lon, scale = geo['center_lat'], geo['center_lon'], geo['pixel_scale_deg']
    # Live Process scenes are resampled to a fixed width/height, so ScaleY != ScaleX
    # whenever the requested bbox isn't square in degrees. The land mask and the
    # contours must share this, or they land on two different latitude grids.
    scale_y = (geo['max_lat'] - geo['min_lat']) / preds_bin.shape[0]
    preds, land_stats = strip_land_pixels(preds_bin * 255, center_lat, center_lon, scale,
                                          bbox=bbox, scale_y=scale_y)
    # Separate rings match the existing frontend contract. No placeholder for an empty mask.
    polygons = mask_to_polygons(preds, center_lat, center_lon, scale, scale_y=scale_y)
    slick_pixels = probs[preds == 255]
    confidence = float(np.mean(slick_pixels)) if len(slick_pixels) else 0.0
    mask_dir = os.path.join(BASE_DIR, 'data', 'processed')
    os.makedirs(mask_dir, exist_ok=True)
    if not cv2.imwrite(os.path.join(mask_dir, f'{scene_id}_mask.png'), preds):
        raise ValueError('Could not save segmentation mask')
    # Fixed VV dB display stretch makes different passes visually comparable.
    vv_db = 10 * np.log10(np.maximum(image_raw[:, :, 0], 1e-10))
    gray = (np.clip((vv_db + 25) / 25, 0, 1) * 255).astype(np.uint8)
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay[preds == 255] = (0.45 * overlay[preds == 255] + 0.55 * np.array([0, 215, 255])).astype(np.uint8)
    return ({'scene_id': scene_id, 'product': product, 'acquisition_start_utc': product['start'],
             'acquisition_end_utc': product['end'], 'source': 'Sentinel-1 GRD / CDSE Process',
             'provenance': provenance, 'bbox': bbox,
             'pixel_sha256': hashlib.sha256(image_raw.tobytes()).hexdigest(),
             'confidence_score': confidence, 'predicted_pixel_count': int((preds == 255).sum()),
             'geojson_mask': {'type': 'Polygon', 'coordinates': polygons},
             'sar_preview': _png_data_uri(gray), 'overlay_preview': _png_data_uri(overlay),
             'preview_note': 'VV sigma0, fixed -25 to 0 dB stretch; yellow is the U-Net mask after land masking.'},
            geo, land_stats)


def _temporal_evidence(token, product, original, payload):
    from src.analysis.sentinel1_revisit_check import search_same_track_passes, select_comparison_products
    bbox = original['bbox']
    try:
        candidates = search_same_track_passes(bbox[0], bbox[2], bbox[1], bbox[3], product['start'],
                                            window_days=payload.temporal_window_days)
        candidates = select_comparison_products(candidates, product, bbox, payload.temporal_window_days)
    except Exception as exc:
        return {'status': 'unavailable', 'reason': f'Comparison search failed ({type(exc).__name__}).'}
    if not candidates:
        return {'status': 'no_match', 'reason': 'No comparison available with full footprint and a different acquisition date.'}
    failures = []
    # A bounded attempt on at most two candidates, returning one useful comparison.
    for candidate in candidates[:2]:
        try:
            scene_id = f'comparison_{uuid.uuid4().hex[:12]}'
            path = os.path.join(BASE_DIR, 'data', 'raw', 'live', f'{scene_id}.tif')
            provenance = fetch_scene_geotiff(token, *bbox, candidate['start'], path, product=candidate)
            comparison, _, _ = _analyze_live_scene(path, scene_id, candidate, provenance)
            if comparison['pixel_sha256'] == original['pixel_sha256']:
                raise ValueError('Comparison pixels duplicate the original observation')
            return {'status': 'available', 'observations': [comparison], 'attempt_failures': failures,
                    'window_days': payload.temporal_window_days,
                    'reason': 'Different-date observation. Orbit, sea state and geometry may differ; no change classification is inferred.'}
        except Exception as exc:
            failures.append({'product_id': candidate['id'], 'reason': f'Fetch/inference/validation failed ({type(exc).__name__}).'})
    return {'status': 'unavailable', 'reason': 'Comparison candidates found, but fetch/inference/validation failed.', 'attempt_failures': failures}


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
        provenance = fetch_scene_geotiff(
            token, payload.min_lat, payload.min_lon, payload.max_lat, payload.max_lon,
            product["start"], img_path, product=product,
        )
    except CdseFetchError as e:
        return {"status": "ERROR", "detail": f"CDSE Process API fetch failed: {e}", "detection": None}

    # Same analysis function is used for primary and comparison observations.
    try:
        observation, geo, land_stats = _analyze_live_scene(img_path, scene_id, product, provenance)
    except Exception as exc:
        return {"status": "ERROR", "detail": f"Live scene inference/validation failed ({type(exc).__name__}).", "detection": None}
    center_lat, center_lon = geo['center_lat'], geo['center_lon']
    bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon = observation['bbox']
    confidence_score, geojson = observation['confidence_score'], observation['geojson_mask']
    mask_png_name = f"{scene_id}_mask.png"

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

    supplementary = {"original": observation, "era5": {"status": "skipped", "reason": "ERA5 was not requested."}}
    if payload.include_era5:
        try:
            supplementary["era5"] = get_wind_evidence(center_lat, center_lon, product["start"])
        except Exception as exc:
            supplementary["era5"] = {"status": "unavailable", "reason": f"ERA5 analysis failed ({type(exc).__name__})."}
    supplementary['temporal'] = {'status': 'skipped', 'reason': 'Temporal comparison was not requested.'}
    if payload.include_temporal:
        supplementary['temporal'] = _temporal_evidence(token, product, observation, payload)
    supplementary['optical'] = {'status': 'skipped', 'reason': 'Optical supplement was not requested.'}
    if payload.include_optical:
        try:
            supplementary['optical'] = get_optical_evidence(token, observation['bbox'], product['start'],
                                                          max_cloud_pct=payload.optical_max_cloud_pct,
                                                          window_days=payload.optical_window_days)
        except Exception as exc:
            supplementary['optical'] = {'status': 'unavailable', 'reason': f'Optical analysis failed ({type(exc).__name__}).'}
    # Persisting optional evidence must not take down a detection that has
    # already committed -- same rule as every supplementary call above. Two
    # things can fail here: json.dumps(allow_nan=False) refuses a non-finite
    # value rather than emitting invalid JSON, and the write itself can fail.
    # Neither is a reason to 500 a successful primary inference.
    try:
        evidence_json = json.dumps(supplementary, allow_nan=False)
    except (ValueError, TypeError) as exc:
        # Keep each source's status even when a payload (a preview, a stray
        # non-finite number) can't be serialized. A degraded but honest record
        # beats an empty one; the statuses are our own strings, so this is safe.
        reason = f"Evidence could not be serialized ({type(exc).__name__})."
        degraded = {"original": {"status": "unavailable", "reason": reason}}
        for key in ("era5", "temporal", "optical"):
            value = supplementary.get(key)
            degraded[key] = {"status": value.get("status", "unavailable"),
                             "reason": value.get("reason", reason)} if isinstance(value, dict) else \
                            {"status": "unavailable", "reason": reason}
        evidence_json = json.dumps(degraded)
        print(f"[!] Supplementary evidence degraded for detection {det_id}: {reason}", file=sys.stderr)

    conn = None
    try:
        conn = get_db_connection()
        conn.execute("UPDATE detections SET supplementary_json = ? WHERE id = ?", (evidence_json, det_id))
        conn.commit()
    except sqlite3.Error as exc:
        # The row keeps its NULL, which surfaces as {} -- no evidence claimed.
        print(f"[!] Supplementary evidence not persisted for detection {det_id} "
              f"({type(exc).__name__}); primary detection is unaffected.", file=sys.stderr)
    finally:
        if conn is not None:
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
