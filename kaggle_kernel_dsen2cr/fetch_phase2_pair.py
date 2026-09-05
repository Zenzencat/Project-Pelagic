"""
PROJECT PELAGIC — DSen2-CR Phase 2: fetch ONE real Sentinel-1 + Sentinel-2 pair.

Runs LOCALLY (not on Kaggle). Uses this project's own CDSE credentials from
.env — same OAuth2 client-credentials flow as src/analysis/cdse_auth.py, same
Process API as src/data/sentinel_process.py — to pull one real co-located
S1 GRD (VV/VH) + S2 L1C (13 bands) 128x128 patch. Saves them to a .npz and
publishes a small PRIVATE Kaggle dataset that the feasibility kernel consumes
for its single inference pass.

No repo code is imported (the Kaggle kernel must stay single-file / isolated),
so the token + Process logic is re-implemented minimally here. No synthetic
data, no fabricated timestamps: every product id / acquisition date written to
the .npz comes straight from the CDSE catalog response.

Usage:
    python kaggle_kernel_dsen2cr/fetch_phase2_pair.py            # fetch + save npz
    python kaggle_kernel_dsen2cr/fetch_phase2_pair.py --push     # also create/version the Kaggle dataset
"""
import io
import json
import os
import sys
import tarfile
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
import tifffile

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
OUT_DIR = os.path.join(HERE, "phase2_data")
NPZ = os.path.join(OUT_DIR, "phase2_input.npz")

# AOI presets. 128 px * 10 m = 1.28 km on a side.
#   rotterdam  — Phase 2 land/highway patch (thin cloud) — kept for reference
#   coalpoint  — Phase 2b: open water inside the Coal Oil Point natural seep
#                field, Santa Barbara Channel. Persistent real oil slicks from
#                seabed seeps (well documented), so this is a "known real oil"
#                open-water scene. SoCal marine-layer stratus gives cloudy S2.
AOIS = {
    "rotterdam": (51.90, 4.55),
    "coalpoint": (34.388, -119.886),
}
AOI = os.environ.get("PHASE2_AOI", "coalpoint")
CENTER_LAT, CENTER_LON = AOIS[AOI]
HALF = 0.0064  # ~0.64 km each way -> ~1.28 km box
BBOX = [CENTER_LAT - HALF, CENTER_LON - HALF, CENTER_LAT + HALF, CENTER_LON + HALF]
SIZE = 128
# Phase 2b wants cloud actually over the water patch, not just the wider scene.
MIN_PATCH_CLOUD_FRAC = float(os.environ.get("PHASE2_MIN_CLOUD", "0.15"))

# S2 L1C 13-band order expected by DSen2-CR (full L1C set incl. B10 cirrus).
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
            "B08", "B8A", "B09", "B10", "B11", "B12"]


def load_env():
    path = os.path.join(REPO, ".env")
    if not os.path.exists(path):
        sys.exit(".env not found — cannot fetch without CDSE credentials")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    cid, csec = os.environ.get("CDSE_CLIENT_ID"), os.environ.get("CDSE_CLIENT_SECRET")
    if not cid or not csec or "your-" in cid:
        sys.exit("CDSE_CLIENT_ID / CDSE_CLIENT_SECRET not set in .env")
    return cid, csec


def get_token(cid, csec):
    r = requests.post(TOKEN_URL, data={"client_id": cid, "client_secret": csec,
                                       "grant_type": "client_credentials"}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]


def search(collection, start, end, extra):
    min_lat, min_lon, max_lat, max_lon = BBOX
    wkt = (f"POLYGON(({min_lon} {min_lat},{max_lon} {min_lat},{max_lon} {max_lat},"
           f"{min_lon} {max_lat},{min_lon} {min_lat}))")
    filt = (f"Collection/Name eq '{collection}' and "
            f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}') and "
            f"ContentDate/Start ge {start} and ContentDate/Start le {end}")
    if extra:
        filt += f" and {extra}"
    params = {"$filter": filt, "$top": 50, "$expand": "Attributes",
              "$orderby": "ContentDate/Start desc"}
    for attempt in range(4):
        try:
            r = requests.get(ODATA_URL, params=params, timeout=90)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 3:
                raise
            print(f"     catalogue retry {attempt + 1}/3 ({type(e).__name__})")
            time.sleep(5 * (attempt + 1))
    out = []
    for p in r.json().get("value", []):
        attrs = {a["Name"]: a.get("Value") for a in p.get("Attributes", [])}
        out.append({"id": p["Id"], "name": p["Name"],
                    "start": p["ContentDate"]["Start"], "end": p["ContentDate"].get("End"),
                    "cloud": attrs.get("cloudCover"), "attrs": attrs})
    return out


def process(token, body):
    r = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {token}",
                      "Accept": "application/x-tar"}, json=body, timeout=180)
    if r.status_code != 200:
        raise RuntimeError(f"Process HTTP {r.status_code}: {r.text[:300]}")
    with tarfile.open(fileobj=io.BytesIO(r.content)) as tar:
        files = {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}
    meta = json.loads(files["userdata.json"])
    return files, meta


def fetch_s1(token, product):
    start = datetime.fromisoformat(product["start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(product["end"].replace("Z", "+00:00"))
    body = {
        "input": {"bounds": {"bbox": [BBOX[1], BBOX[0], BBOX[3], BBOX[2]]},
                  "data": [{"type": "sentinel-1-grd",
                            "dataFilter": {"timeRange": {
                                "from": start.replace(microsecond=0).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                "to": (end.replace(microsecond=0) + timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%SZ")},
                                "resolution": "HIGH", "acquisitionMode": "IW", "polarization": "DV"},
                            "processing": {"backCoeff": "SIGMA0_ELLIPSOID", "orthorectify": True}}]},
        "output": {"width": SIZE, "height": SIZE, "responses": [
            {"identifier": "default", "format": {"type": "image/tiff"}},
            {"identifier": "userdata", "format": {"type": "application/json"}}]},
        "evalscript": """//VERSION=3
function setup(){return{input:["VV","VH"],mosaicking:"ORBIT",
  output:[{id:"default",bands:2,sampleType:"FLOAT32"}]};}
function evaluatePixel(s){if(!s.length)return{default:[0,0]};return{default:[s[0].VV,s[0].VH]};}
function updateOutputMetadata(sc,im,om){om.userData={tiles:[].concat.apply([],sc.orbits.map(o=>o.tiles))};}
"""}
    files, meta = process(token, body)
    arr = tifffile.imread(io.BytesIO(files["default.tif"]))  # (H,W,2) linear sigma0
    if arr.shape != (SIZE, SIZE, 2) or not np.isfinite(arr).all():
        raise RuntimeError(f"bad S1 raster {arr.shape}")
    return np.transpose(arr, (2, 0, 1)).astype(np.float32), meta  # (2,H,W)


def fetch_s2(token, product):
    start = datetime.fromisoformat(product["start"].replace("Z", "+00:00"))
    day = start.strftime("%Y-%m-%d")
    body = {
        "input": {"bounds": {"bbox": [BBOX[1], BBOX[0], BBOX[3], BBOX[2]]},
                  "data": [{"type": "sentinel-2-l1c",
                            "dataFilter": {"timeRange": {
                                "from": f"{day}T00:00:00Z", "to": f"{day}T23:59:59Z"}}}]},
        "output": {"width": SIZE, "height": SIZE, "responses": [
            {"identifier": "default", "format": {"type": "image/tiff"}},
            {"identifier": "userdata", "format": {"type": "application/json"}}]},
        "evalscript": """//VERSION=3
function setup(){return{input:[%s],output:{bands:13,sampleType:"FLOAT32"},mosaicking:"SIMPLE"};}
function evaluatePixel(s){return [%s];}
function updateOutputMetadata(sc,im,om){om.userData={tiles:sc.tiles};}
""" % (",".join(f'"{b}"' for b in S2_BANDS),
       ",".join(f"s.{b}" for b in S2_BANDS))}
    files, meta = process(token, body)
    arr = tifffile.imread(io.BytesIO(files["default.tif"]))  # (H,W,13) reflectance 0..~1
    if arr.shape != (SIZE, SIZE, 13) or not np.isfinite(arr).all():
        raise RuntimeError(f"bad S2 raster {arr.shape}")
    # Process returns reflectance in [0,1]; DSen2-CR expects L1C DN (reflectance*10000).
    return (np.transpose(arr, (2, 0, 1)) * 10000.0).astype(np.float32), meta  # (13,H,W)


def main():
    cid, csec = load_env()
    token = get_token(cid, csec)
    now = datetime.now(timezone.utc)
    s2_start = (now - timedelta(days=120)).strftime("%Y-%m-%dT00:00:00.000Z")
    s2_end = now.strftime("%Y-%m-%dT00:00:00.000Z")

    s2c = [p for p in search("SENTINEL-2", s2_start, s2_end, "")
           if "MSIL1C" in p["name"] and p["cloud"] is not None and 15 <= float(p["cloud"]) <= 95]
    if not s2c:
        sys.exit("no partly-cloudy (15-95%) S2 L1C scene over the AOI in the last 120 days")

    # Collect every viable pair with its patch cloud fraction, then prefer one
    # that is PARTLY clouded (~0.2-0.85): a patch with both clouded and clear
    # water is the informative case — you can judge the reconstruction against
    # real water in the same frame. Fall back to any pair that clears the min.
    viable = []  # (partial_pref_key, pc, cand, s1_cand, o, s, om, sm)
    for cand in s2c[:12]:
        s2_dt = datetime.fromisoformat(cand["start"].replace("Z", "+00:00"))
        w0 = (s2_dt - timedelta(days=20)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        w1 = (s2_dt + timedelta(days=20)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        s1c = [p for p in search("SENTINEL-1", w0, w1, "contains(Name,'GRD')")
               if "_IW_" in p["name"] and "GRDH" in p["name"]]
        if not s1c:
            continue
        s1_cand = min(s1c, key=lambda p: abs(
            datetime.fromisoformat(p["start"].replace("Z", "+00:00")) - s2_dt))
        print(f"try  S2 {cand['name']}  cloud={cand['cloud']:.0f}%   "
              f"S1 {s1_cand['name'][:32]}...")
        try:
            o, om = fetch_s2(token, cand)
            s, sm = fetch_s1(token, s1_cand)
        except Exception as e:
            print(f"     skip: {e}")
            continue
        if float(np.count_nonzero(o)) / o.size < 0.5 or o.mean() <= 0:
            print("     skip: S2 patch is mostly nodata")
            continue
        if float(np.count_nonzero(s)) / s.size < 0.5:
            print("     skip: S1 patch is mostly nodata")
            continue
        # cloud proxy: over water, B02 DN jumps from a few hundred to >2000 under cloud.
        pc = float(np.mean(o[1] > 2000.0))
        print(f"     patch cloud frac ~{pc:.2f}")
        if pc < MIN_PATCH_CLOUD_FRAC:
            print("     skip: not enough cloud over the patch itself")
            continue
        is_partial = 0.20 <= pc <= 0.85
        viable.append((0 if is_partial else 1, abs(pc - 0.5), cand, s1_cand, o, s, om, sm))
        if is_partial:
            break
    if not viable:
        sys.exit("no S2+S1 pair with real pixels AND cloud over the patch found")
    viable.sort(key=lambda v: (v[0], v[1]))
    _, _, s2, s1, opt, sar, s2_meta, s1_meta = viable[0]
    patch_cloud = float(np.mean(opt[1] > 2000.0))
    s2_dt = datetime.fromisoformat(s2["start"].replace("Z", "+00:00"))
    print(f"\nS2 L1C: {s2['name']}  {s2['start']}  scene cloud={s2['cloud']}%  "
          f"patch cloud~{patch_cloud:.2f}")
    print(f"S1 GRD: {s1['name']}  {s1['start']}")

    # If the patch is fully clouded, grab the nearest near-clear S2 pass over the
    # same AOI as an eyeball "ground truth" for the reconstruction. Not fed to
    # the model — reference only. Different day, so not a pixel-exact target.
    opt_ref = None
    ref_meta = {}
    if patch_cloud > 0.9:
        clear = []
        for p in search("SENTINEL-2",
                        (s2_dt - timedelta(days=45)).strftime("%Y-%m-%dT00:00:00.000Z"),
                        (s2_dt + timedelta(days=45)).strftime("%Y-%m-%dT00:00:00.000Z"), ""):
            if "MSIL1C" not in p["name"] or p["cloud"] is None or float(p["cloud"]) > 25:
                continue
            clear.append(p)
        clear.sort(key=lambda p: abs(
            datetime.fromisoformat(p["start"].replace("Z", "+00:00")) - s2_dt))
        for p in clear[:6]:
            try:
                o2, m2 = fetch_s2(token, p)
            except Exception:
                continue
            if float(np.count_nonzero(o2)) / o2.size > 0.5 and float(np.mean(o2[1] > 2000.0)) < 0.1:
                opt_ref = o2
                ref_meta = {"ref_s2_product": p["name"], "ref_s2_acquired": p["start"],
                            "ref_s2_cloud_pct": p["cloud"], "ref_s2_tiles": m2.get("tiles")}
                print(f"clear ref S2: {p['name']}  {p['start']}  cloud={p['cloud']}%")
                break

    os.makedirs(OUT_DIR, exist_ok=True)
    meta = {
        "aoi_name": AOI, "aoi_bbox_latlon": BBOX, "size_px": SIZE,
        "s1_product": s1["name"], "s1_acquired": s1["start"], "s1_tiles": s1_meta.get("tiles"),
        "s2_product": s2["name"], "s2_acquired": s2["start"], "s2_cloud_pct": s2["cloud"],
        "s2_patch_cloud_frac": round(patch_cloud, 3), "s2_tiles": s2_meta.get("tiles"),
        "days_apart": abs((datetime.fromisoformat(s1["start"].replace("Z", "+00:00")) - s2_dt).days),
        "sar_layout": "(2,H,W) linear sigma0 ellipsoid VV,VH",
        "opt_layout": "(13,H,W) L1C DN = reflectance*10000, band order " + ",".join(S2_BANDS),
        "has_clear_ref": opt_ref is not None,
        "fetched_utc": datetime.now(timezone.utc).isoformat(),
        **ref_meta,
    }
    arrs = {"sar": sar, "opt": opt, "meta": json.dumps(meta)}
    if opt_ref is not None:
        arrs["opt_ref"] = opt_ref
    np.savez_compressed(NPZ, **arrs)
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nsaved {NPZ}")
    print(f"  sar {sar.shape} {sar.dtype}  range [{sar.min():.4g},{sar.max():.4g}]")
    print(f"  opt {opt.shape} {opt.dtype}  range [{opt.min():.4g},{opt.max():.4g}]")
    print(json.dumps(meta, indent=2))

    if "--push" in sys.argv:
        dsmeta = {
            "title": "Pelagic DSen2-CR Phase 2 input",
            "id": "papawitsaeliw/pelagic-dsen2cr-phase2-input",
            "licenses": [{"name": "CC0-1.0"}],
        }
        with open(os.path.join(OUT_DIR, "dataset-metadata.json"), "w") as f:
            json.dump(dsmeta, f, indent=2)
        import subprocess
        exists = subprocess.run(["kaggle", "datasets", "status",
                                 dsmeta["id"]], capture_output=True, text=True).returncode == 0
        cmd = (["kaggle", "datasets", "version", "-p", OUT_DIR, "-m", "refresh", "--dir-mode", "zip"]
               if exists else
               ["kaggle", "datasets", "create", "-p", OUT_DIR, "--dir-mode", "zip"])
        print("\n$", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
