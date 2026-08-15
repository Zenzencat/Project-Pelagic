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
import numpy as np
import cv2
import tifffile
import torch
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add root folder to path to enable package imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.api.database import init_db, get_db_connection
from src.models.unet import UNet
from src.data.preprocess import calibrate_to_sigma, speckle_filter, to_decibels, normalize_image
from src.inference import run_tiled_inference

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

print("[*] Checking U-Net model checkpoint...")
if os.path.exists(CHECKPOINT_PATH):
    try:
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
    SELECT id, scene_id, detected_at, confidence_score, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon, geojson_mask, image_path 
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
            "image_path": r["image_path"]
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
            "distance_meters": v["distance_meters"]
        })
        
    return {
        "id": det_row["id"],
        "scene_id": det_row["scene_id"],
        "detected_at": det_row["detected_at"],
        "confidence_score": det_row["confidence_score"],
        "bbox": [det_row["bbox_min_lat"], det_row["bbox_min_lon"], det_row["bbox_max_lat"], det_row["bbox_max_lon"]],
        "geojson_mask": json.loads(det_row["geojson_mask"]),
        "image_path": det_row["image_path"],
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
    if not os.path.exists(img_path):
        img_path = os.path.join(BASE_DIR, "data", "synthetic", "images", f"{scene_id}.tif")
    
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
        
    # 2. Run Preprocessing
    if np.any(image_raw < 0):
        # Already in dB scale (Real Sentinel-1 imagery)
        if len(image_raw.shape) == 3 and image_raw.shape[0] == 2:
            image_raw = image_raw.transpose(1, 2, 0)
        # Convert to linear for speckle filtering
        linear = 10.0 ** (image_raw.astype(np.float32) / 10.0)
        filt = speckle_filter(linear, window_size=5)
        # Convert back to dB
        db = 10.0 * np.log10(np.clip(filt, 1e-5, None))
        norm = normalize_image(db)
    else:
        # Linear scale (Synthetic data)
        sig = calibrate_to_sigma(image_raw)
        filt = speckle_filter(sig, window_size=5)
        db = to_decibels(filt)
        norm = normalize_image(db)
    
    # 3. U-Net Inference
    # Tiled to match the 256x256 patch regime the model was trained/evaluated on
    # (see src/inference.py and src/evaluate_holdout.py) rather than a single
    # whole-image forward pass, which puts every interior pixel in a receptive-field
    # context the model never saw during training.
    probs, preds_bin = run_tiled_inference(model, norm, device)
    preds = preds_bin * 255
        
    # 4. Contour Tracing (Convert binary mask to GeoJSON Polygon)
    contours, _ = cv2.findContours(preds, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Coordinate system mapping (Gulf of Thailand mock offsets)
    center_lat = 9.0
    center_lon = 100.5
    scale = 0.00015 # scale factor degrees per pixel
    H, W = preds.shape
    
    polygons = []
    for cnt in contours:
        # Simplify contour lines to make GeoJSON lighter
        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        poly_pts = []
        for pt in approx:
            x, y = pt[0]
            # Convert pixel coords to Latitude/Longitude
            lat = center_lat + (H // 2 - y) * scale
            lon = center_lon + (x - W // 2) * scale
            poly_pts.append([lon, lat])
            
        if len(poly_pts) >= 3:
            poly_pts.append(poly_pts[0]) # Close polygon
            polygons.append(poly_pts)
            
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

    try:
        cursor.execute("""
        INSERT INTO detections (scene_id, confidence_score, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon, geojson_mask, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            scene_id,
            confidence_score,
            bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon,
            json.dumps(geojson),
            f"data/processed/{mask_png_name}"
        ))
        det_id = cursor.lastrowid

        # Add mock AIS vessels for the newly processed scene
        cursor.executemany("""
        INSERT INTO nearby_vessels (detection_id, mmsi, vessel_name, latitude, longitude, timestamp, distance_meters)
        VALUES (?, ?, ?, ?, ?, datetime('now'), ?);
        """, [
            (det_id, 351980000, "Petro Express", center_lat + 0.02, center_lon - 0.03, 4200.0),
            (det_id, 567409210, "Nakhon Fishery 21", center_lat - 0.03, center_lon + 0.01, 5100.0)
        ])
        conn.commit()

    except sqlite3.IntegrityError:
        # Scene already processed, fetch its ID
        cursor.execute("SELECT id FROM detections WHERE scene_id = ?;", (scene_id,))
        det_id = cursor.fetchone()["id"]
    finally:
        conn.close()
        
    # Return detail response
    return get_detection_details(det_id)
