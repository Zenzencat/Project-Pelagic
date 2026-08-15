"""
Project Pelagic — SAR-Optical Cloud-Removal Fusion (design sketch, NOT wired
into the live pipeline, NOT runnable yet)
SWU Prasarnmit AI Engineering Final Project

Goal: reconstruct a cloud-free Sentinel-2 optical view of the same location/time
as each SAR scene, as a second modality specifically to help discriminate real
oil slicks from lookalikes (which can be SAR-identical but visually different in
optical color/texture) -- using DSen2-CR (Meraner et al., ISPRS 2020 Best Paper),
github.com/ameraner/dsen2-cr, pretrained, not retrained.

STATUS: design sketch only. Every function below is either (a) real, verified
logic that already works today (the CDSE search calls, reusing the same
credentials prepped for the multi-temporal Sentinel-1 work), or (b) clearly
marked as a stub that CANNOT run until upstream blockers clear. See
docs/cloud_removal_scoping.md for the full write-up; the short version:

  1. BLOCKED on the same missing acquisition-timestamp problem as ERA5 and
     multi-temporal SAR (src/analysis/era5_wind_check.py,
     sentinel1_revisit_check.py) -- being chased separately, not a repo problem
     to keep digging at.
  2. BLOCKED on CDSE credentials (same .env vars already documented in
     .env.example -- CDSE_CLIENT_ID / CDSE_CLIENT_SECRET). Confirmed Sentinel-2
     needs no additional credential setup beyond what's already prepped: a real,
     unauthenticated catalog search against Collection/Name='SENTINEL-2' for
     this project's actual scene coordinates returned real MSIL1C product names.
  3. BLOCKED architecturally, not just by data: DSen2-CR requires TensorFlow
     1.15.0 + Keras 2.2.4 + Python 3.7 (confirmed still pip-installable, but
     3.7-only -- no wheels exist for Python 3.8+). This CANNOT be installed into
     this project's own venv (torch>=2.0.0, modern numpy) without a dependency
     collision -- it needs its own isolated environment (conda env or the
     repo's provided Docker image). run_dsen2cr_inference() below is therefore
     a subprocess-boundary stub, not an in-process call -- writing it as an
     import would misrepresent something that cannot actually work that way.

Preprocessing constants below (SAR/optical clip ranges, scale factors, band
counts) are taken directly from a fresh clone of github.com/ameraner/dsen2-cr
(Code/dsen2cr_main.py, Code/tools/dataIO.py), not guessed at.
"""

import os

import requests

CDSE_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
CDSE_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

# --- DSen2-CR's exact expected input spec (verified against a fresh clone,
# not the README) ---
DSEN2CR_CROP_SIZE = 128  # native patch size; our own model uses 256x256, so a
                          # full 2048x2048 scene needs 16x16=256 tiles here vs
                          # our 8x8=64 -- ~4x more tiles to cover the same scene.
DSEN2CR_OPTICAL_BANDS = 13  # full Sentinel-2 L1C band set (includes B10 cirrus,
                             # which L2A products drop) -- confirms we need L1C
                             # products, not atmospherically-corrected L2A.
DSEN2CR_SAR_BANDS = 2  # VV, VH -- matches this project's own SAR convention.
DSEN2CR_DATA_FORMAT = "channels_first"  # (C, H, W), not (H, W, C) like this
                                          # project's own preprocess.py -- needs
                                          # a transpose.

# SAR clip range is band-specific and asymmetric, unlike this project's own
# uniform [-25, 0] dB clip in src/data/preprocess.py -- a real adaptation
# point, not a drop-in reuse.
DSEN2CR_SAR_CLIP_MIN = [-25.0, -32.5]  # [VV, VH], dB
DSEN2CR_SAR_CLIP_MAX = [0.0, 0.0]
DSEN2CR_SAR_MAX_VAL = 2.0  # SAR rescaled to [0, 2] after clipping

DSEN2CR_OPTICAL_CLIP_MIN = 0
DSEN2CR_OPTICAL_CLIP_MAX = 10000  # typical L1C TOA reflectance * 10000 scaling
DSEN2CR_OPTICAL_SCALE = 2000.0  # optical rescaled to [0, 5] after clipping

# Verified real, downloadable Keras/.h5 checkpoints (checked directly against
# Google Drive's actual file bytes, not just that the link returns HTTP 200 --
# see docs/cloud_removal_scoping.md). All four are ~72.4MB, consistent with
# the ~19M-parameter architecture (16 resblocks, 256 features).
DSEN2CR_CHECKPOINTS = {
    "sar_carl": {  # SAR input + CARL loss -- the paper's best-performing config
        "gdrive_id": "1L3YUVOnlg67H5VwlgYO9uC9iuNlq7VMg",
        "filename": "model_SARcarl.hdf5",
    },
    "sar_plain": {
        "gdrive_id": "1zv4_91Yr2IYyYDoqhZw8KpnfvfLhkuBB",
        "filename": "model_SARplain.hdf5",
    },
    "no_sar_carl": {
        "gdrive_id": "1VHZa5-lX68mA2FbHeCiQsUq13oECw9DA",
        "filename": "model_noSARcarl.hdf5",
    },
    "no_sar_plain": {
        "gdrive_id": "11Th6UwKMXla7LGxsFJXwj-Jx9bSKlWUH",
        "filename": "model_noSARplain.hdf5",
    },
}


def get_cdse_token():
    """
    Real, working function -- same OAuth2 client-credentials flow already
    validated in scripts/validate_credentials.py. Reused here rather than
    duplicated logic.
    """
    client_id = os.environ.get("CDSE_CLIENT_ID")
    client_secret = os.environ.get("CDSE_CLIENT_SECRET")
    if not client_id or not client_secret:
        raise RuntimeError("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set — see .env.example")

    r = requests.post(
        CDSE_TOKEN_URL,
        data={"client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"},
        timeout=15,
    )
    r.raise_for_status()
    return r.json()["access_token"]


def find_nearest_sentinel2_l1c(lat, lon, reference_date_iso, window_days=5):
    """
    Real, working search (no auth needed) -- confirmed against this project's
    actual scene coordinates during this session: querying
    Collection/Name='SENTINEL-2', contains(Name,'MSIL1C') at a real E.
    Mediterranean point returned real product names (e.g.
    S2A_MSIL1C_20240115T082301_...). Requires reference_date_iso -- the
    blocker, not this function.
    """
    from datetime import datetime, timedelta
    ref = datetime.fromisoformat(reference_date_iso)
    start = (ref - timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    end = (ref + timedelta(days=window_days)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    filt = (
        f"Collection/Name eq 'SENTINEL-2' and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;POINT({lon} {lat})') and "
        f"contains(Name,'MSIL1C') and "
        f"ContentDate/Start gt {start} and ContentDate/Start lt {end}"
    )
    r = requests.get(CDSE_ODATA_URL, params={"$filter": filt, "$top": 5, "$orderby": "ContentDate/Start"}, timeout=30)
    r.raise_for_status()
    return r.json().get("value", [])


def download_cdse_product(product_id, out_path, access_token):
    """
    Real download logic (untested end-to-end this session, per the "don't
    actually pull data" instruction) using CDSE's documented product download
    endpoint. Would need real credentials + a real product_id from
    find_nearest_sentinel2_l1c() to actually run.
    """
    url = f"https://zipper.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    headers = {"Authorization": f"Bearer {access_token}"}
    with requests.get(url, headers=headers, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
    return out_path


def preprocess_for_dsen2cr(sar_vv_vh_db, optical_13band_toa):
    """
    Real, verified preprocessing matching DSen2-CR's exact normalization
    (see module docstring for source). Both inputs are (H, W, C) numpy arrays
    (this project's convention); returns (C, H, W) channels-first arrays
    matching DSen2-CR's expected input.

    NOT the same as this project's own src/data/preprocess.py normalization --
    a real adaptation, not a passthrough:
      - This project clips SAR uniformly to [-25, 0] dB for both VV/VH.
      - DSen2-CR clips VV to [-25, 0] but VH to [-32.5, 0] -- band-specific.
    """
    import numpy as np

    sar = sar_vv_vh_db.transpose(2, 0, 1).astype("float32")  # (2, H, W)
    for c in range(2):
        sar[c] = np.clip(sar[c], DSEN2CR_SAR_CLIP_MIN[c], DSEN2CR_SAR_CLIP_MAX[c])
        sar[c] -= DSEN2CR_SAR_CLIP_MIN[c]
        sar[c] = DSEN2CR_SAR_MAX_VAL * (sar[c] / (DSEN2CR_SAR_CLIP_MAX[c] - DSEN2CR_SAR_CLIP_MIN[c]))

    opt = optical_13band_toa.transpose(2, 0, 1).astype("float32")  # (13, H, W)
    opt = np.clip(opt, DSEN2CR_OPTICAL_CLIP_MIN, DSEN2CR_OPTICAL_CLIP_MAX)
    opt = opt / DSEN2CR_OPTICAL_SCALE

    return sar, opt


def run_dsen2cr_inference(sar_chw, optical_chw, checkpoint_path, dsen2cr_repo_path, dsen2cr_python_env):
    """
    STUB -- deliberately not implemented as an in-process call. DSen2-CR needs
    TensorFlow 1.15.0 + Keras 2.2.4 + Python 3.7, which cannot coexist in this
    project's own venv (torch>=2.0.0, modern numpy) -- it needs a separate
    conda env or the repo's own Docker image. The real implementation is a
    subprocess boundary: write sar_chw/optical_chw to temp .npy files, shell
    out to a small predict script running inside the isolated dsen2cr
    environment (using DSen2CR_model() + model.load_weights(checkpoint_path)
    + model.predict_on_batch(), NOT the repo's --predict CLI path, which
    hardcodes requiring a ground-truth target for eval metrics we don't have
    for a genuinely novel scene), and read the reconstructed image back.

    Raises NotImplementedError -- deliberately, rather than a fake pass-through
    that would look like it works.
    """
    raise NotImplementedError(
        "DSen2-CR inference must run in a separate environment (own conda env "
        "or the repo's Docker image) -- not implemented here. See "
        "docs/cloud_removal_scoping.md for the subprocess design and time estimate."
    )


def run_fusion_for_scene(scene_id, lat, lon, reference_date_iso, checkpoint="sar_carl"):
    """
    Orchestration sketch showing the intended end-to-end flow. Refuses to run
    without a real reference_date_iso rather than fabricating one -- same
    stance as era5_wind_check.py and sentinel1_revisit_check.py.
    """
    if reference_date_iso is None:
        raise ValueError(
            f"No acquisition timestamp available for {scene_id} -- this dataset "
            "doesn't have one (see docs/status.md). Refusing to fabricate one."
        )

    s2_candidates = find_nearest_sentinel2_l1c(lat, lon, reference_date_iso)
    if not s2_candidates:
        raise RuntimeError(f"No Sentinel-2 L1C product found near {scene_id} within the search window.")

    # ... download S2 + paired S1, preprocess_for_dsen2cr(), run_dsen2cr_inference() ...
    raise NotImplementedError("End-to-end orchestration sketch only — see docs/cloud_removal_scoping.md")


if __name__ == "__main__":
    print(
        "This is a design sketch, not a runnable pipeline yet. See "
        "docs/cloud_removal_scoping.md for status and what's needed to make it real."
    )
