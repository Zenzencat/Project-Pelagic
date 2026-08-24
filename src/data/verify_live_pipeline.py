"""
Project Pelagic — Live Fetch + Real GFW Attribution Verification Script
SWU Prasarnmit AI Engineering Final Project

Verifies the live pipeline (src/data/cdse_fetch.py, src/analysis/gfw_client.py,
POST /api/live/fetch in src/api/main.py) that replaced the old hardcoded
mock_vessels. Two modes, chosen automatically based on what's in .env:

1. Credentials NOT configured/invalid: verifies every SKIPPED/NOT_CONFIGURED/
   skipped_no_credentials gate fires correctly and nothing gets fabricated.
2. Credentials configured and valid (this repo's current state, confirmed via
   scripts/validate_credentials.py): additionally runs one real end-to-end
   live fetch (real CDSE search + real Sentinel Hub Process API fetch + real
   inference + real GFW attribution) against a known-good bbox/date, and
   asserts the result has a real, non-fabricated acquisition timestamp and a
   real GFW attribution status. This writes one real row to data/pelagic.db
   (source='live') and makes real calls against both external APIs -- run
   deliberately, not in a tight loop.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from src.analysis.cdse_auth import get_cdse_token, CdseAuthError
from src.analysis.gfw_client import get_nearby_vessels
from src.data.cdse_fetch import find_best_product

# A real, verified-covered bbox/window (Singapore Strait -- confirmed via a
# real CDSE search during development to have Sentinel-1 GRD coverage).
KNOWN_GOOD_BBOX = (1.10, 103.70, 1.30, 103.90)  # min_lat, min_lon, max_lat, max_lon
KNOWN_GOOD_WINDOW = ("2026-08-05", "2026-08-15")

# A date range before Sentinel-1 existed -- guaranteed zero real products,
# for exercising the honest "no product found" SKIPPED gate.
IMPOSSIBLE_WINDOW = ("2010-01-01", "2010-01-02")


def check(label, condition):
    status = "[+] PASS" if condition else "[!] FAIL"
    print(f"{status} - {label}")
    return condition


def main():
    print("=== Project Pelagic live pipeline verification ===\n")
    all_ok = True

    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    gfw_token = os.environ.get("GFW_TOKEN")

    # --- 1. Bad-credential gates (never touches the real .env values) ---
    print("[*] Gate check: invalid CDSE credentials must raise CdseAuthError, not fabricate a token.")
    try:
        get_cdse_token("bad-client-id", "bad-client-secret")
        all_ok &= check("bad CDSE credentials rejected", False)
    except CdseAuthError:
        all_ok &= check("bad CDSE credentials rejected", True)

    print("\n[*] Gate check: empty GFW token must return skipped_no_credentials, not an empty-but-'ok' result.")
    res = get_nearby_vessels("", 1.2, 103.8, "2026-08-11T22:47:44Z")
    all_ok &= check(
        "empty GFW token -> skipped_no_credentials",
        res["status"] == "skipped_no_credentials" and res["vessels"] == [],
    )

    # --- 2. Real "no product found" SKIPPED gate (real API call, no auth needed) ---
    print("\n[*] Gate check: a date range before Sentinel-1 existed must find zero real products.")
    product = find_best_product(*KNOWN_GOOD_BBOX, *IMPOSSIBLE_WINDOW)
    all_ok &= check("impossible date range -> no product found", product is None)

    # --- 3. Real credential validity (reuses scripts/validate_credentials.py's checks) ---
    cdse_valid = False
    if client_id and client_secret:
        try:
            get_cdse_token(client_id, client_secret)
            cdse_valid = True
        except CdseAuthError:
            pass
    print(f"\n[*] CDSE credentials configured and valid: {cdse_valid}")
    print(f"[*] GFW token configured: {bool(gfw_token)}")

    if not cdse_valid or not gfw_token:
        print(
            "\n[SKIP] Real end-to-end live fetch not run — CDSE and/or GFW "
            "credentials are not both configured/valid in this environment. "
            "Future round with real credentials must additionally verify:\n"
            "  1. POST /api/live/fetch on a real bbox/date returns status=OK\n"
            "     with a real, non-null acquisition_start_utc/acquisition_end_utc\n"
            "     (never defaulted).\n"
            "  2. The stored detection's vessel_attribution_status is 'ok' or\n"
            "     'empty' (both are honest real-query outcomes), never a\n"
            "     fabricated vessel list.\n"
            "  3. The live scene's inference output is visually sane (open\n"
            "     browser, use the Live Fetch panel, inspect the rendered\n"
            "     contour against the real basemap)."
        )
    else:
        print("\n[*] Both credentials valid — running one real end-to-end live fetch...")
        import requests
        try:
            r = requests.post(
                "http://127.0.0.1:8000/api/live/fetch",
                json={
                    "min_lat": KNOWN_GOOD_BBOX[0], "min_lon": KNOWN_GOOD_BBOX[1],
                    "max_lat": KNOWN_GOOD_BBOX[2], "max_lon": KNOWN_GOOD_BBOX[3],
                    "date_from": KNOWN_GOOD_WINDOW[0], "date_to": KNOWN_GOOD_WINDOW[1],
                    "radius_km": 15.0,
                },
                timeout=180,
            )
            body = r.json()
        except requests.RequestException as e:
            all_ok &= check(f"live API reachable at 127.0.0.1:8000 (start it with uvicorn first): {e}", False)
            body = None

        if body is not None:
            all_ok &= check("live fetch status == OK", body.get("status") == "OK")
            det = body.get("detection") or {}
            all_ok &= check("real (non-null) acquisition_start_utc stored", bool(det.get("acquisition_start_utc")))
            all_ok &= check("real (non-null) acquisition_end_utc stored", bool(det.get("acquisition_end_utc")))
            all_ok &= check("source == 'live'", det.get("source") == "live")
            all_ok &= check(
                "vessel_attribution_status is a real outcome (ok/empty/error), not fabricated",
                det.get("vessel_attribution_status") in ("ok", "empty", "error"),
            )
            print(f"    detail: {body.get('detail')}")

    print("\n=== Summary ===")
    print("[+] ALL CHECKS PASSED" if all_ok else "[!] SOME CHECKS FAILED")
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
