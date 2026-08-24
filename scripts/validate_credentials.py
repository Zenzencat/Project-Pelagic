"""
Project Pelagic — Credential Validation
SWU Prasarnmit AI Engineering Final Project

Run this the moment you have real CDS and/or CDSE credentials to know
instantly whether they work, before spending any more time building on top
of them. Does the minimum possible real API call to each service -- an auth
check, not a real data download.

Usage:
    pip install cdsapi requests python-dotenv   # python-dotenv is optional
    cp .env.example .env   # then fill in real values
    python scripts/validate_credentials.py

Reads credentials from environment variables (loads .env automatically if
python-dotenv is installed; otherwise export the vars yourself, or copy
.env.example to ~/.cdsapirc for the CDS half — see that file's comments).
Does NOT read or accept credentials as command-line arguments (would leak
into shell history) and does NOT fabricate/guess at values -- every check
below is skipped with a clear reason if its env vars aren't set, not run
against placeholders.
"""

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # optional; user can export env vars directly instead

# Add root folder to path so `src.*` imports work when run as a script.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def check_cds():
    """
    Validates CDS credentials with the smallest real ERA5 request possible:
    a single variable, single hour, single ~0.25deg grid cell. There is no
    documented lightweight "auth ping" endpoint for the CDS API as of this
    writing (checked; see docs/status.md) -- a bad key/URL fails immediately
    with an auth error, a good one queues and completes in anywhere from
    seconds to a few minutes depending on CDS's queue load. That wait is
    inherent to their system, not a flaw here.
    """
    url = os.environ.get("CDSAPI_URL")
    key = os.environ.get("CDSAPI_KEY")
    if not url or not key:
        print("[SKIP] CDS: CDSAPI_URL / CDSAPI_KEY not set in environment.")
        return None

    try:
        import cdsapi
    except ImportError:
        print("[FAIL] CDS: cdsapi package not installed. Run: pip install cdsapi")
        return False

    import tempfile
    target = tempfile.NamedTemporaryFile(suffix=".nc", delete=False).name

    print("[*] CDS: submitting a minimal real ERA5 request (1 variable, 1 hour, "
          "1 grid cell)... this can take a few minutes depending on CDS queue load.")
    try:
        client = cdsapi.Client(url=url, key=key)
        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": "reanalysis",
                "variable": "10m_u_component_of_wind",
                "year": "2024",
                "month": "01",
                "day": "01",
                "time": "00:00",
                "area": [35.1, 34.9, 34.9, 35.1],  # tiny box, ~1 grid cell
                "format": "netcdf",
            },
            target,
        )
        print(f"[PASS] CDS: credentials work. Retrieved a real ERA5 file to {target}")
        return True
    except Exception as e:
        print(f"[FAIL] CDS: {e}")
        return False
    finally:
        if os.path.exists(target):
            os.remove(target)


def check_cdse():
    """
    Validates CDSE OAuth2 client-credentials against the real token endpoint.
    This is a genuine lightweight check (a few hundred ms, no data transfer) --
    a bad client id/secret fails the token request immediately and
    unambiguously (400/401), a good one returns a real bearer token.
    """
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        print("[SKIP] CDSE: CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set in environment.")
        return None

    from src.analysis.cdse_auth import get_cdse_token, CdseAuthError

    try:
        get_cdse_token(client_id, client_secret)
        print("[PASS] CDSE: credentials work, obtained a real OAuth2 access token.")
        return True
    except CdseAuthError as e:
        print(f"[FAIL] CDSE: {e}")
        return False


def check_gfw():
    """
    Validates the GFW API token against a real, minimal v3 endpoint (a single
    dataset lookup -- a few hundred ms, no report/download data transfer). A
    bad/expired token fails with 401, a good one returns the dataset's real
    metadata.
    """
    token = os.environ.get("GFW_TOKEN")
    if not token:
        print("[SKIP] GFW: GFW_TOKEN not set in environment.")
        return None

    import requests

    try:
        r = requests.get(
            "https://gateway.api.globalfishingwatch.org/v3/datasets/public-global-presence:latest",
            headers={"Authorization": f"Bearer {token}"},
            timeout=15,
        )
        if r.status_code == 200:
            print("[PASS] GFW: token works, fetched real dataset metadata (public-global-presence:latest).")
            return True
        print(f"[FAIL] GFW: HTTP {r.status_code} — {r.text[:300]}")
        return False
    except Exception as e:
        print(f"[FAIL] GFW: {e}")
        return False


def main():
    print("=== Project Pelagic credential validation ===\n")
    cds_result = check_cds()
    print()
    cdse_result = check_cdse()
    print()
    gfw_result = check_gfw()

    print("\n=== Summary ===")
    for name, result in [
        ("CDS (ERA5)", cds_result),
        ("CDSE (Sentinel-1)", cdse_result),
        ("GFW (AIS attribution)", gfw_result),
    ]:
        status = {True: "PASS", False: "FAIL", None: "SKIPPED (not configured)"}[result]
        print(f"  {name:24s} {status}")

    if cds_result is False or cdse_result is False or gfw_result is False:
        sys.exit(1)


if __name__ == "__main__":
    main()
