"""Test explicit database initialization and demo seeding decoupling."""
import sqlite3
import pytest
from src.api import database


def test_init_db_leaves_clean_database_empty(tmp_path, monkeypatch):
    db_file = tmp_path / "clean.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))

    # Initialize tables without seeding
    database.init_db(seed_demo=False)

    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections;")
    det_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM nearby_vessels;")
    vessel_count = cursor.fetchone()[0]
    conn.close()

    assert det_count == 0
    assert vessel_count == 0


def test_seed_demo_data_explicit(tmp_path, monkeypatch):
    db_file = tmp_path / "seeded.db"
    monkeypatch.setattr(database, "DB_PATH", str(db_file))

    database.init_db(seed_demo=False)
    seeded = database.seed_demo_data()
    assert seeded == 2

    conn = database.get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM detections;")
    det_count = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM nearby_vessels;")
    vessel_count = cursor.fetchone()[0]
    conn.close()

    assert det_count == 2
    assert vessel_count == 4

    # Idempotent second call
    seeded_again = database.seed_demo_data()
    assert seeded_again == 0
