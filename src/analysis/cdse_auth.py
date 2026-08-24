"""
Project Pelagic — Shared CDSE OAuth2 Authentication
SWU Prasarnmit AI Engineering Final Project

Single source of truth for obtaining a Copernicus Data Space Ecosystem (CDSE)
OAuth2 access token via the client-credentials grant. Extracted out of
scripts/validate_credentials.py's inline check so both that script and the
live-fetch pipeline (src/data/cdse_fetch.py) share one implementation instead
of two copies that can drift apart.
"""

import requests

TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)


class CdseAuthError(Exception):
    """Raised when the CDSE token endpoint rejects the given credentials."""


def get_cdse_token(client_id, client_secret, timeout=15):
    """
    Exchanges a CDSE OAuth2 client id/secret for a real bearer access token.

    Raises CdseAuthError on any non-200 response or network failure -- never
    returns a fabricated/placeholder token. Callers that only want a
    pass/fail check (no token needed) should catch this and treat it as
    "invalid credentials", matching validate_credentials.py's check_cdse().
    """
    try:
        r = requests.post(
            TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
            },
            timeout=timeout,
        )
    except requests.RequestException as e:
        raise CdseAuthError(f"CDSE token request failed: {e}") from e

    if r.status_code != 200:
        raise CdseAuthError(f"CDSE token request rejected: HTTP {r.status_code} — {r.text[:300]}")

    token = r.json().get("access_token")
    if not token:
        raise CdseAuthError("CDSE token response had no access_token field.")
    return token
