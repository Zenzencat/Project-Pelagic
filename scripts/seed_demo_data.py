"""
Project Pelagic — Explicit Demo Database Seeder
SWU Prasarnmit AI Engineering Final Project

Usage:
    python scripts/seed_demo_data.py
    python scripts/seed_demo_data.py --clean-first
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.database import init_db, seed_demo_data, get_db_connection


def main():
    parser = argparse.ArgumentParser(description="Explicitly seed Project Pelagic demo records.")
    parser.add_argument("--clean-first", action="store_true",
                        help="Drop existing detections and vessels before seeding.")
    args = parser.parse_args()

    init_db(seed_demo=False)

    if args.clean_first:
        print("[*] Cleaning existing detections and nearby_vessels...")
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM nearby_vessels;")
        cursor.execute("DELETE FROM detections;")
        conn.commit()
        conn.close()

    count = seed_demo_data()
    print(f"[+] Done. Seeded {count} demo records.")


if __name__ == "__main__":
    main()
