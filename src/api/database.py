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
