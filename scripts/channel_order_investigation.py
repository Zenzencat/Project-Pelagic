"""
Reproduces the numbers in docs/CHANNEL_ORDER_INVESTIGATION.md.

Read-only with respect to the project: loads holdout/live GeoTIFFs and the
existing checkpoints, never writes to checkpoints/ or data/. Writes one JSON
results file (default docs/channel_order_investigation.json).

  stats     per-channel slick-vs-adjacent-sea contrast on holdout scenes
            (GT eroded 41x41 = inside; GT dilated 81x81 minus GT = ring),
            per-channel fraction of pixels below -25 dB, and the median
            per-pixel ch0 - ch1 difference; plus the same dB statistics on
            live CDSE scenes, whose order is known by construction
            (src/data/cdse_fetch.py's evalscript returns [VV, VH]).
  ablation  inference-only: runs the real checkpoints on each holdout scene
            as stored and with channels swapped, and on each unique live
            scene as fetched and swapped (~11 s per pass on CPU).

Usage (from repo root):
  python scripts/channel_order_investigation.py --data-dir data [--ablation]
"""

import argparse
import glob
import hashlib
import json
import os
import sys

import cv2
import numpy as np
import tifffile

sys.path.insert(0, os.path.abspath("."))

INSIDE_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
RING_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (81, 81))


def holdout_stats(data_dir):
    rows = []
    for cat in ("oil", "lookalike", "no_oil"):
        for img_path in sorted(glob.glob(os.path.join(data_dir, "holdout", "images", f"{cat}_*.tif"))):
            sid = os.path.splitext(os.path.basename(img_path))[0]
            img = tifffile.imread(img_path).astype(np.float32)
            gt = (tifffile.imread(os.path.join(data_dir, "holdout", "masks", f"{sid}.tif")) > 0).astype(np.uint8)
            # No-data (exact 0 dB in both bands, or nonfinite) is excluded everywhere.
            valid = np.isfinite(img).all(-1) & (img != 0).all(-1)
            r = {"scene": sid, "category": cat, "gt_px": int(gt.sum()), "nodata_frac": float(1 - valid.mean())}
            if not valid.any():
                r["all_nodata"] = True
                rows.append(r)
                continue
            inside = (cv2.erode(gt, INSIDE_KERNEL) > 0) & valid
            ring = (cv2.dilate(gt, RING_KERNEL) > 0) & (gt == 0) & valid
            r["inside_px"], r["ring_px"] = int(inside.sum()), int(ring.sum())
            for c in (0, 1):
                ch = img[..., c]
                v = ch[valid]
                r[f"ch{c}_median_db"] = float(np.median(v))
                r[f"ch{c}_p2_db"], r[f"ch{c}_p98_db"] = map(float, np.percentile(v, [2, 98]))
                r[f"ch{c}_frac_below_-25db"] = float((v < -25).mean())
                r[f"ch{c}_contrast_db"] = (float(np.median(ch[inside]) - np.median(ch[ring]))
                                           if inside.any() and ring.any() else None)
            r["ch0_minus_ch1_median_db"] = float(np.median(img[..., 0][valid] - img[..., 1][valid]))
            rows.append(r)
            print(f"{sid:16s} ch0 med {r['ch0_median_db']:6.1f} ctr {r['ch0_contrast_db']} | "
                  f"ch1 med {r['ch1_median_db']:6.1f} ctr {r['ch1_contrast_db']} | "
                  f"ch0-ch1 {r['ch0_minus_ch1_median_db']:6.2f}", flush=True)
    return rows


def unique_live_scenes(data_dir):
    seen = set()
    for path in sorted(glob.glob(os.path.join(data_dir, "raw", "live", "*.tif"))):
        img = tifffile.imread(path)
        digest = hashlib.sha256(img.tobytes()).hexdigest()[:12]
        if digest not in seen:
            seen.add(digest)
            yield os.path.basename(path), digest, img


def live_stats(data_dir):
    rows = []
    for name, digest, img in unique_live_scenes(data_dir):
        a = img.astype(np.float32)
        valid = (a > 0).all(-1) & np.isfinite(a).all(-1)
        db = 10 * np.log10(np.clip(a, 1e-10, None))
        r = {"scene": name, "sha256_12": digest, "valid_frac": float(valid.mean())}
        for c, pol in ((0, "VV"), (1, "VH")):
            v = db[..., c][valid]
            r[f"ch{c}_{pol}_median_db"] = float(np.median(v))
            r[f"ch{c}_{pol}_frac_below_-25db"] = float((v < -25).mean())
        r["ch0_minus_ch1_median_db"] = float(np.median(db[..., 0][valid] - db[..., 1][valid]))
        rows.append(r)
        print(f"LIVE {name} VV med {r['ch0_VV_median_db']:6.1f} | VH med {r['ch1_VH_median_db']:6.1f} | "
              f"VV-VH {r['ch0_minus_ch1_median_db']:5.2f}", flush=True)
    return rows


def ablation(data_dir):
    import torch
    from src.models.unet import UNet
    from src.data.preprocess import preprocess_for_prediction
    from src.inference import run_tiled_inference
    from src.evaluate_holdout import compute_metrics

    device = torch.device("cpu")
    out = {}
    for version, ckpt in (("v2", "model_real_v2_best.pt"), ("v1", "model_real_best.pt")):
        model = UNet(in_channels=2, out_channels=1)
        state = torch.load(os.path.join("checkpoints", ckpt), map_location=device)
        model.load_state_dict(state["model_state_dict"] if isinstance(state, dict) and "model_state_dict" in state else state)
        model.eval()
        holdout = []
        for img_path in sorted(glob.glob(os.path.join(data_dir, "holdout", "images", "*.tif"))):
            sid = os.path.splitext(os.path.basename(img_path))[0]
            if version == "v1" and not sid.startswith("oil"):
                continue
            img = tifffile.imread(img_path)
            gt = tifffile.imread(os.path.join(data_dir, "holdout", "masks", f"{sid}.tif"))
            r = {"scene": sid}
            for tag, x in (("as_stored", img), ("swapped", img[..., ::-1].copy())):
                norm = preprocess_for_prediction(x)
                _, pred = run_tiled_inference(model, norm, device)
                iou, dice, prec, rec = compute_metrics(pred, gt)
                r[tag] = {"iou": iou, "dice": dice, "precision": prec, "recall": rec,
                          "pred_frac": float(pred.mean())}
            holdout.append(r)
            print(f"{version} {sid:16s} as_stored IoU {r['as_stored']['iou']:.4f} pred {r['as_stored']['pred_frac']:.4f} | "
                  f"swapped IoU {r['swapped']['iou']:.4f} pred {r['swapped']['pred_frac']:.4f}", flush=True)
        live = []
        if version == "v2":
            for name, digest, img in unique_live_scenes(data_dir):
                r = {"scene": name, "sha256_12": digest}
                for tag, x in (("as_fetched_VV_VH", img), ("swapped_VH_VV", img[..., ::-1].copy())):
                    norm = preprocess_for_prediction(x, already_calibrated=True)
                    probs, pred = run_tiled_inference(model, norm, device)
                    r[tag] = {"pred_frac": float(pred.mean()), "mean_prob": float(probs.mean())}
                live.append(r)
                print(f"{version} LIVE {name} as_fetched pred {r['as_fetched_VV_VH']['pred_frac']:.4f} | "
                      f"swapped pred {r['swapped_VH_VV']['pred_frac']:.4f}", flush=True)
        out[version] = {"holdout": holdout, "live": live}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--ablation", action="store_true", help="also run the inference-only channel-swap ablation")
    ap.add_argument("--out", default=os.path.join("docs", "channel_order_investigation.json"))
    args = ap.parse_args()
    results = {"holdout_stats": holdout_stats(args.data_dir), "live_stats": live_stats(args.data_dir)}
    if args.ablation:
        results["ablation"] = ablation(args.data_dir)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
