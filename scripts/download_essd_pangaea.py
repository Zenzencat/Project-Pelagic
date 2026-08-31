"""
Phase 1 data prep: downloads the ESSD/PANGAEA look-alike dataset (Yang & Singha
2025, PANGAEA DOI 10.1594/PANGAEA.980773) referenced in docs/status.md's
post-hoc-lookalike-classifier investigation.

The dataset's own tab-separated metadata export (saved as
data/external/essd_pangaea/pangaea_980773_metadata.tab) is row-per-OBJECT for
the oil set (oc/ow subsets -- a patch with N oil objects has N rows sharing
one jpg_file) and row-per-PATCH for the no-oil set (nc/nw subsets -- no object
annotations exist for look-alikes, by definition, matching this project's own
all-zero Mask_lookalike ground truth).

Individual files are served at https://download.pangaea.de/dataset/980773/files/<name>
with no auth required (verified directly -- bulk zip download does require a
login, individual files don't). No-oil filenames use a longer
"<subset>-NNNN-00-NNNNNN.jpg" form, not "<subset>-NNNN.jpg" -- confirmed from
the metadata table itself, not guessed (an earlier guess at the short form
produced consistent HTTP 500s that looked like a server-side outage on first
glance; re-checking against the real metadata showed it was a wrong filename,
not a real outage).

Downloads full oil (1365 patches, oc+ow) and no-oil (2290 patches, nc+nw)
sets -- ~550MB total, well within a one-time local prep step.
"""
import os
import sys
import time
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT_DIR = os.path.join(REPO, "data", "external", "essd_pangaea")
TAB_PATH = os.path.join(EXT_DIR, "pangaea_980773_metadata.tab")
BASE_URL = "https://download.pangaea.de/dataset/980773/files/"

IMG_DIR = os.path.join(EXT_DIR, "images")
XML_DIR = os.path.join(EXT_DIR, "xml")
os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(XML_DIR, exist_ok=True)


def load_metadata():
    with open(TAB_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    header_idx = next(i for i, l in enumerate(lines) if l.strip() == "*/") + 1
    df = pd.read_csv(TAB_PATH, sep="\t", skiprows=header_idx, encoding="utf-8")
    df.columns = [c.split(" (")[-1].rstrip(")") for c in df.columns]
    return df


def unique_patches(df):
    """One row per patch: subset, jpg_file, xml_file (NaN for no-oil),
    plus the patch's aggregate object bbox (oil only, union of all its
    objects' bboxes -- multiple oil-object rows can share one jpg_file)."""
    df["subset"] = df["jpg_file"].str[:2]
    records = []
    for jpg, g in df.groupby("jpg_file", sort=False):
        row0 = g.iloc[0]
        has_bbox = g["obj_patchloc_xmin"].notna().any()
        if has_bbox:
            xmin = int(g["obj_patchloc_xmin"].min())
            ymin = int(g["obj_patchloc_ymin"].min())
            xmax = int(g["obj_patchloc_xmax"].max())
            ymax = int(g["obj_patchloc_ymax"].max())
            n_objects = g["obj_patchloc_xmin"].notna().sum()
        else:
            xmin = ymin = xmax = ymax = None
            n_objects = 0
        records.append({
            "subset": row0["subset"],
            "label": "oil" if row0["subset"] in ("oc", "ow") else "no_oil",
            "jpg_file": jpg,
            "xml_file": row0["xml_file"] if pd.notna(row0["xml_file"]) else None,
            "patch_width": row0["patch_width"],
            "patch_height": row0["patch_height"],
            "start_time": row0["start_time"],
            "sentinel_id": row0["Sentinel_ID"],
            "patch_ul_lon": row0["patch_ul_lon"],
            "patch_ul_lat": row0["patch_ul_lat"],
            "n_objects": n_objects,
            "bbox_xmin": xmin, "bbox_ymin": ymin, "bbox_xmax": xmax, "bbox_ymax": ymax,
        })
    return pd.DataFrame.from_records(records)


def download_one(session, filename, out_dir, retries=3):
    out_path = os.path.join(out_dir, filename)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return filename, "cached"
    for attempt in range(retries):
        try:
            r = session.get(BASE_URL + filename, timeout=30)
            if r.status_code == 200:
                with open(out_path, "wb") as f:
                    f.write(r.content)
                return filename, "ok"
            else:
                last = f"HTTP {r.status_code}"
        except Exception as e:
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    return filename, f"FAILED ({last})"


def download_all(filenames, out_dir, label, max_workers=10):
    ok, cached, failed = 0, 0, []
    with requests.Session() as session:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(download_one, session, f, out_dir): f for f in filenames}
            for i, fut in enumerate(as_completed(futures), 1):
                fname, status = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "cached":
                    cached += 1
                else:
                    failed.append((fname, status))
                if i % 200 == 0 or i == len(filenames):
                    print(f"  [{label}] {i}/{len(filenames)} (ok={ok}, cached={cached}, failed={len(failed)})")
    return ok, cached, failed


def main():
    print("[*] Loading PANGAEA metadata table...")
    df = load_metadata()
    print(f"[+] {len(df)} object/patch rows loaded")

    patches = unique_patches(df)
    patches.to_csv(os.path.join(EXT_DIR, "patch_index.csv"), index=False)
    print(f"[+] {len(patches)} unique patches indexed -> data/external/essd_pangaea/patch_index.csv")
    print(patches.groupby(["label", "subset"]).size())

    jpg_files = patches["jpg_file"].tolist()
    xml_files = patches["xml_file"].dropna().tolist()

    print(f"\n[*] Downloading {len(jpg_files)} JPG images...")
    ok, cached, failed = download_all(jpg_files, IMG_DIR, "images")
    print(f"[+] Images: {ok} downloaded, {cached} already cached, {len(failed)} failed")
    if failed:
        print("    Failed:", failed[:10], "..." if len(failed) > 10 else "")

    print(f"\n[*] Downloading {len(xml_files)} XML annotations...")
    ok2, cached2, failed2 = download_all(xml_files, XML_DIR, "xml")
    print(f"[+] XML: {ok2} downloaded, {cached2} already cached, {len(failed2)} failed")
    if failed2:
        print("    Failed:", failed2[:10], "..." if len(failed2) > 10 else "")

    with open(os.path.join(EXT_DIR, "download_failures.txt"), "w") as f:
        for fname, status in failed + failed2:
            f.write(f"{fname}\t{status}\n")

    print(f"\n[Summary] Total patches: {len(patches)} "
          f"({(patches['label']=='oil').sum()} oil, {(patches['label']=='no_oil').sum()} no_oil)")
    print(f"[Summary] Image download failures: {len(failed)}, XML download failures: {len(failed2)}")


if __name__ == "__main__":
    main()
