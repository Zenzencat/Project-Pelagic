"""
Project Pelagic — Database Connection and Seeding
SWU Prasarnmit AI Engineering Final Project

This module manages the SQLite database (data/pelagic.db), creating tables 
for oil slick detections and correlated AIS vessel tracking data. It seeds
mock data on startup if the database is empty.
"""

import os
import sqlite3
import json

DB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
DB_PATH = os.path.join(DB_DIR, "pelagic.db")

def get_db_connection():
    """Returns a connection to the SQLite database with dict-like row access."""
    os.makedirs(DB_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes tables and seeds mock records if database is new/empty."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # 1. Create detections table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scene_id TEXT NOT NULL UNIQUE,
        detected_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        confidence_score REAL NOT NULL,
        bbox_min_lat REAL NOT NULL,
        bbox_min_lon REAL NOT NULL,
        bbox_max_lat REAL NOT NULL,
        bbox_max_lon REAL NOT NULL,
        geojson_mask TEXT NOT NULL,
        image_path TEXT NOT NULL
    );
    """)
    
    # 2. Create nearby_vessels table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS nearby_vessels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        detection_id INTEGER NOT NULL,
        mmsi INTEGER NOT NULL,
        vessel_name TEXT,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        timestamp DATETIME NOT NULL,
        distance_meters REAL NOT NULL,
        position_resolution_m REAL,
        FOREIGN KEY (detection_id) REFERENCES detections (id) ON DELETE CASCADE
    );
    """)
    
    # Create indices
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_detections_scene ON detections(scene_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_vessels_detection ON nearby_vessels(detection_id);")

    conn.commit()

    # Migration: add checkpoint_hash so predict() can tell a genuinely-cached
    # result apart from a stale one left by a since-swapped checkpoint.
    cursor.execute("PRAGMA table_info(detections);")
    existing_cols = [row[1] for row in cursor.fetchall()]
    if "checkpoint_hash" not in existing_cols:
        cursor.execute("ALTER TABLE detections ADD COLUMN checkpoint_hash TEXT;")
        conn.commit()

    # Migration: live-fetch + real GFW attribution columns (replaces the old
    # hardcoded mock_vessels). `source` distinguishes cached holdout/synthetic
    # scenes (no real acquisition timestamp -- see docs/status.md) from
    # live-fetched CDSE scenes (which do). `vessel_attribution_status`
    # records *why* nearby_vessels is empty or populated -- 'ok' (real GFW
    # vessels found), 'empty' (GFW queried, zero within radius),
    # 'skipped_no_credentials' (GFW_TOKEN unset), 'skipped_no_timestamp' (no
    # real acquisition datetime to query GFW with -- every pre-existing
    # holdout/synthetic detection), or 'error' (GFW query failed). The
    # frontend uses this to render an honest reason instead of a silent "0
    # vessels" that looks identical to "never checked."
    if "source" not in existing_cols:
        cursor.execute("ALTER TABLE detections ADD COLUMN source TEXT NOT NULL DEFAULT 'holdout';")
        cursor.execute("ALTER TABLE detections ADD COLUMN acquisition_start_utc TEXT;")
        cursor.execute("ALTER TABLE detections ADD COLUMN acquisition_end_utc TEXT;")
        cursor.execute("ALTER TABLE detections ADD COLUMN cdse_product_id TEXT;")
        cursor.execute("ALTER TABLE detections ADD COLUMN vessel_attribution_status TEXT;")
        cursor.execute("ALTER TABLE detections ADD COLUMN vessel_search_radius_km REAL;")
        conn.commit()

        # One-time cleanup: every nearby_vessels row that exists at this
        # point was inserted by the old mock_vessels logic in main.py's
        # predict() (2 fake vessels per scene) -- there is no real
        # acquisition timestamp for any pre-existing detection to have run
        # real GFW attribution against, so these rows cannot be real data.
        # Clearing them (rather than leaving them to render as if unchanged)
        # is the actual "remove the mock layer" behavior for scenes that
        # were already analyzed before this migration ran.
        cursor.execute("DELETE FROM nearby_vessels;")
        cursor.execute("UPDATE detections SET vessel_attribution_status = 'skipped_no_timestamp';")
        conn.commit()

    # Migration: honest position-precision disclosure for GFW-attributed
    # vessels (Round 15). GFW's 4Wings Report endpoint reports each vessel's
    # position as a ~0.01deg grid-cell center, not a precise ping -- distinct
    # vessels sharing a grid cell get identical lat/lon and therefore
    # identical distance_meters, which rendered as an unexplained "0.0km for
    # 3 vessels" and stacked/hidden map markers. src/analysis/gfw_client.py
    # now reports the real grid-cell size per vessel so the frontend can
    # disclose it instead of implying false precision -- see
    # gfw_client.py's module docstring for the full investigation.
    cursor.execute("PRAGMA table_info(nearby_vessels);")
    vessel_cols = [row[1] for row in cursor.fetchall()]
    if "position_resolution_m" not in vessel_cols:
        cursor.execute("ALTER TABLE nearby_vessels ADD COLUMN position_resolution_m REAL;")
        conn.commit()

    # Optional evidence survives history/detail refreshes; legacy rows stay null.
    if "supplementary_json" not in existing_cols:
        cursor.execute("ALTER TABLE detections ADD COLUMN supplementary_json TEXT;")
        conn.commit()

    # 3. Seed mock data if empty
    cursor.execute("SELECT COUNT(*) FROM detections;")
    if cursor.fetchone()[0] == 0:
        print("[*] Database is empty. Seeding mock oil slick detections and vessels...")
        
        # Mock 1: Gulf of Thailand
        mock1_geojson = {
            "type": "Polygon",
            "coordinates": [[
                [101.42, 9.25],
                [101.45, 9.28],
                [101.55, 9.22],
                [101.62, 9.15],
                [101.60, 9.13],
                [101.52, 9.18],
                [101.45, 9.20],
                [101.42, 9.25]
            ]]
        }
        cursor.execute("""
        INSERT INTO detections (scene_id, detected_at, confidence_score, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon, geojson_mask, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            "S1A_IW_GRDH_1SDV_20260627T101402",
            "2026-06-27 10:14:02",
            0.942,
            8.70, 100.90, 9.70, 102.10,
            json.dumps(mock1_geojson),
            "data/synthetic/masks/S1A_IW_GRDH_1SDV_20260627T101402_mask.png"
        ))
        det1_id = cursor.lastrowid
        
        cursor.executemany("""
        INSERT INTO nearby_vessels (detection_id, mmsi, vessel_name, latitude, longitude, timestamp, distance_meters)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, [
            (det1_id, 235085320, "MV Sea Voyager", 9.11, 101.64, "2026-06-27 10:12:00", 12400.0),
            (det1_id, 477123900, "Global Gas Carrier", 9.32, 101.40, "2026-06-27 10:15:30, ", 18200.0),
            (det1_id, 567010243, "Thai Fishing Vessel 09", 9.19, 101.53, "2026-06-27 10:13:45", 3500.0)
        ])
        
        # Mock 2: Andaman Sea (near Phuket)
        mock2_geojson = {
            "type": "Polygon",
            "coordinates": [[
                [98.12, 7.87],
                [98.16, 7.90],
                [98.22, 7.83],
                [98.19, 7.81],
                [98.12, 7.87]
            ]]
        }
        cursor.execute("""
        INSERT INTO detections (scene_id, detected_at, confidence_score, bbox_min_lat, bbox_min_lon, bbox_max_lat, bbox_max_lon, geojson_mask, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (
            "S1B_IW_GRDH_1SDV_20260625T220815",
            "2026-06-25 22:08:15",
            0.887,
            7.35, 97.65, 8.35, 98.65,
            json.dumps(mock2_geojson),
            "data/synthetic/masks/S1B_IW_GRDH_1SDV_20260625T220815_mask.png"
        ))
        det2_id = cursor.lastrowid
        
        cursor.execute("""
        INSERT INTO nearby_vessels (detection_id, mmsi, vessel_name, latitude, longitude, timestamp, distance_meters)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """, (det2_id, 354921000, "LPG Pioneer", 7.80, 98.26, "2026-06-25 22:05:00", 8700.0))
        
        conn.commit()
        print("[+] Mock database seeding completed successfully.")
        
    conn.close()

if __name__ == "__main__":
    init_db()
