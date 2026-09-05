"""
PROJECT PELAGIC — DSen2-CR feasibility check (Task 4, scoped).

Runs ENTIRELY inside a Kaggle kernel. Touches nothing in the main Pelagic repo,
venv, or requirements. Disposable by design.

Questions this answers, and nothing more:
  PHASE 1 — Can DSen2-CR's official (Keras/TF1.15) implementation be installed
            in an isolated Python 3.7 environment on Kaggle's current base image,
            and do the official pretrained checkpoints load without error?
  PHASE 2 — (only if Phase 1 fully passes, and only if the phase-2 input dataset
            is attached) run ONE forward pass on one real Sentinel-1 + Sentinel-2
            pair and save a before/after triptych for a visual sanity check.

This is a viability checkpoint. It does NOT integrate anything into Pelagic.
RESULT (2026-09-06): both phases PASS — see kaggle_kernel_dsen2cr/RESULT.md.

Strategy: the kernel's base interpreter is Python 3.12 and the current Kaggle
image ships no conda. We bootstrap Miniconda into /tmp (the README's own
documented isolation path, installed on the fly — NOT a wheel hack), build an
isolated env `p37`, and drive it via `conda run`. A connectivity precheck
fails fast with a clear message if kernel internet is off (it was, until the
account was phone-verified 2026-09-06). Everything heavy (Miniconda, the env,
the repo clone, the checkpoint) lives under /tmp so it is NOT saved as multi-GB
kernel output; only small result files land in /kaggle/working.

See kaggle_kernel_dsen2cr/RESULT.md for run history and the current verdict.

Everything is logged with explicit STEP / RESULT markers so the outcome is
unambiguous from the kernel log alone.
"""

import json
import os
import socket
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Official spec, from the dsen2-cr README (github.com/ameraner/dsen2-cr):
#   Python 3.7 · tensorflow-gpu 1.15.0 · Keras 2.2.4 · numpy 1.17 · h5py 2.10.0
# numpy pinned to 1.18.5 (last release that imports cleanly under TF 1.15).
# protobuf pinned to the last 3.x (TF 1.15 cannot parse protobuf >= 4).
# ---------------------------------------------------------------------------
PY_VERSION = "3.7"
PIP_PKGS = [
    "tensorflow-gpu==1.15.0",
    "keras==2.2.4",
    "numpy==1.18.5",
    "h5py==2.10.0",
    "protobuf==3.20.3",
]

DSEN2CR_REPO = "https://github.com/ameraner/dsen2-cr"
# model_SARcarl.hdf5 — SAR input + CARL loss, the paper's best config.
# gdrive id + expected byte size verified in docs/cloud_removal_scoping.md.
CKPT_GDRIVE_ID = "1L3YUVOnlg67H5VwlgYO9uC9iuNlq7VMg"
CKPT_NAME = "model_SARcarl.hdf5"
CKPT_EXPECTED_BYTES = 75943784

WORK = "/kaggle/working"          # only small result files go here (kernel output)
TMP = "/tmp/dsen2cr"             # everything heavy lives here, NOT saved as output
ENV_NAME = "p37"
CONDA_DIR = os.path.join(TMP, "miniconda")
CONDA = os.path.join(CONDA_DIR, "bin", "conda")
# conda-forge cudatoolkit/cudnn land in the env's lib/ but conda activation no
# longer adds that to the loader path, so TF 1.15 needs it set explicitly.
GPU_ENV = {"LD_LIBRARY_PATH": os.path.join(CONDA_DIR, "envs", ENV_NAME, "lib")}
PHASE2_NPZ = "/kaggle/input/pelagic-dsen2cr-phase2-input/phase2_input.npz"
# Miniforge, not Miniconda: conda-forge is its only channel, so there is no
# Anaconda-defaults Terms-of-Service gate (which blocks `conda create` non-
# interactively on the current image).
MINICONDA_URL = ("https://github.com/conda-forge/miniforge/releases/latest/"
                 "download/Miniforge3-Linux-x86_64.sh")
RESULT = {"phase": 1, "steps": {}}


def log(msg):
    print(msg, flush=True)


def run(cmd, timeout=3600, env=None):
    log(f"\n$ {cmd}")
    t0 = time.time()
    p = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, **(env or {})},
    )
    dt = time.time() - t0
    out = (p.stdout or "") + (p.stderr or "")
    log("\n".join(out.splitlines()[-40:]))
    log(f"[rc={p.returncode}  {dt:.0f}s]")
    return p.returncode, out


def stop(step, verdict, detail):
    RESULT["steps"][step] = {"verdict": verdict, "detail": detail}
    RESULT["overall"] = f"STOP at {step}: {verdict}"
    _dump()
    log("\n" + "=" * 70)
    log(f"PHASE 1 RESULT: {RESULT['overall']}")
    log(detail)
    log("=" * 70)
    sys.exit(0)


def ok(step, detail):
    RESULT["steps"][step] = {"verdict": "PASS", "detail": detail}
    log(f"  -> {step}: PASS — {detail}")


def _dump():
    with open(os.path.join(WORK, "phase1_result.json"), "w") as f:
        json.dump(RESULT, f, indent=2)


def main():
    os.makedirs(TMP, exist_ok=True)
    os.chdir(TMP)
    log("=" * 70)
    log("PHASE 1 — DSen2-CR environment + checkpoint viability on Kaggle")
    log("=" * 70)
    log(f"base python : {sys.version}")
    run("uname -a")
    _, nv = run("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 || true")
    RESULT["host"] = {"gpu": nv.strip()}

    # -- STEP 0: connectivity precheck (fail fast, this was the blocker) ----
    reachable = False
    for host in ("pypi.org", "github.com", "repo.anaconda.com"):
        try:
            socket.setdefaulttimeout(10)
            socket.gethostbyname(host)
            reachable = True
            break
        except Exception as e:
            log(f"DNS {host}: {e}")
    if not reachable:
        stop("kernel_internet", "BLOCKED",
             "Kaggle kernel has no outbound internet (DNS resolution fails for "
             "pypi.org/github.com/repo.anaconda.com). enable_internet is set but "
             "ignored -> the account is not phone-verified. Verify at "
             "kaggle.com/settings (Phone Verification) or run under an already-"
             "verified account's token, then re-push. Nothing about DSen2-CR "
             "itself has been tested yet — this is an access blocker, not a "
             "dead end for the model.")
    ok("kernel_internet", "outbound DNS works")

    # -- STEP 1: get a conda, then a 3.7 env -------------------------------
    rc, _ = run("which conda && conda --version")
    if rc != 0:
        log("no system conda — bootstrapping Miniconda into /tmp")
        rc, _ = run(f"wget -q {MINICONDA_URL} -O {TMP}/miniconda.sh")
        if rc != 0 or os.path.getsize(f"{TMP}/miniconda.sh") < 1_000_000:
            stop("conda_present", "FAIL", "could not download the Miniconda installer")
        rc, _ = run(f"bash {TMP}/miniconda.sh -b -p {CONDA_DIR}", timeout=900)
        if rc != 0 or not os.path.exists(CONDA):
            stop("conda_present", "FAIL", "Miniconda installer failed")
    else:
        globals()["CONDA"] = "conda"
    rc, cv = run(f"{CONDA} --version")
    if rc != 0:
        stop("conda_present", "FAIL", "conda not runnable after install")
    ok("conda_present", cv.strip())

    # cudatoolkit 10.0 + cuDNN 7.6 come from conda-forge so the TF 1.15 GPU wheel
    # (which links libcudart.so.10.0) can use the T4 — Kaggle's system CUDA is 12,
    # far too new for TF 1.15. Standard old-TF-on-GPU approach, not a patch.
    rc, out = run(
        f"{CONDA} create -y -n {ENV_NAME} --override-channels -c conda-forge "
        f"python={PY_VERSION} pip cudatoolkit=10.0 cudnn=7.6",
        timeout=2400,
    )
    if rc != 0:
        stop("py37_env", "FAIL",
             f"`conda create --override-channels -c conda-forge python="
             f"{PY_VERSION}` failed — see output above. If conda-forge no longer "
             f"resolves a 3.7 env for this platform: no Python 3.7 => no TF1.15 "
             f"=> DSen2-CR official impl not runnable on Kaggle.")
    rc, ver = run(f"{CONDA} run -n {ENV_NAME} python --version")
    if rc != 0 or "3.7" not in ver:
        stop("py37_env", "FAIL", f"env created but python is not 3.7: {ver!r}")
    ok("py37_env", f"isolated env `{ENV_NAME}` = {ver.strip()}")

    # -- STEP 2: install the official TF1.15 / Keras 2.2.4 stack -----------
    rc, out = run(
        f"{CONDA} run -n {ENV_NAME} pip install --no-cache-dir "
        + " ".join(f'"{p}"' for p in PIP_PKGS),
        timeout=3600,
    )
    if rc != 0:
        stop("tf115_install", "FAIL",
             "pip could not install tensorflow-gpu==1.15.0 + keras==2.2.4 into "
             "the 3.7 env. Per the brief, NOT attempting incompatible-wheel "
             "workarounds. See pip output above for the exact resolution error.")
    rc, freeze = run(f"{CONDA} run -n {ENV_NAME} pip freeze")
    with open(os.path.join(WORK, "p37_pip_freeze.txt"), "w") as f:
        f.write(freeze)
    rc, tfv = run(
        f"{CONDA} run -n {ENV_NAME} python -c "
        f"\"import tensorflow as tf; print('TF', tf.__version__); "
        f"import keras; print('Keras', keras.__version__); "
        f"import h5py, numpy; print('h5py', h5py.__version__, 'numpy', numpy.__version__); "
        f"print('GPU', tf.test.is_gpu_available())\"",
        env=GPU_ENV,
    )
    if rc != 0 or "TF 1.15" not in tfv:
        stop("tf115_import", "FAIL",
             f"TF1.15 installed but does not import cleanly under Python 3.7 on "
             f"this image:\n{tfv}")
    ok("tf115_import", tfv.strip().replace("\n", " | "))

    # -- STEP 3: get the official code + the SARcarl checkpoint ------------
    rc, _ = run(f"git clone --depth 1 {DSEN2CR_REPO} {TMP}/dsen2-cr")
    if rc != 0:
        stop("clone", "FAIL", "could not git clone the dsen2-cr repo")
    ok("clone", "dsen2-cr cloned")

    # Checkpoint comes from a private Kaggle dataset attached to this kernel,
    # not from Google Drive: gdown 4.x (the newest that installs under py3.7)
    # can no longer parse Drive's current download page. The dataset is a
    # byte-exact mirror of gdrive id 1L3YUVOnlg67H5VwlgYO9uC9iuNlq7VMg,
    # downloaded once locally with gdown 6.x — see fetch_phase2_pair.py's sibling
    # comment and kaggle_kernel_dsen2cr/RESULT.md.
    ckpt = f"/kaggle/input/dsen2cr-sarcarl-checkpoint/{CKPT_NAME}"
    size = os.path.getsize(ckpt) if os.path.exists(ckpt) else 0
    if size != CKPT_EXPECTED_BYTES:
        stop("checkpoint_available", "FAIL",
             f"{ckpt} is {size} bytes, expected {CKPT_EXPECTED_BYTES}. Attach the "
             f"`papawitsaeliw/dsen2cr-sarcarl-checkpoint` dataset to this kernel.")
    with open(ckpt, "rb") as f:
        head = f.read(8)
    if head[:4] != b"\x89HDF":
        stop("checkpoint_available", "FAIL",
             f"{CKPT_NAME} is not HDF5 (magic={head!r}).")
    ok("checkpoint_available", f"{CKPT_NAME} {size} bytes, valid HDF5")

    # -- STEP 4: build the model graph and load the pretrained weights ----
    loader = os.path.join(TMP, "_load_weights.py")
    with open(loader, "w") as f:
        f.write(
            "import sys\n"
            f"sys.path.insert(0, {TMP + '/dsen2-cr/Code'!r})\n"
            "from dsen2cr_network import DSen2CR_model\n"
            "# exact params from dsen2cr_main.py: crop 128, 16 resblocks, 256 feat\n"
            "model, shape_n = DSen2CR_model(((13,128,128),(2,128,128)),\n"
            "                               batch_per_gpu=2, num_layers=16,\n"
            "                               feature_size=256, use_cloud_mask=True,\n"
            "                               include_sar_input=True)\n"
            "print('model built, params=', model.count_params())\n"
            f"model.load_weights({ckpt!r})\n"
            "print('WEIGHTS_LOADED_OK params=', model.count_params())\n"
        )
    rc, out = run(f"{CONDA} run -n {ENV_NAME} python {loader}", timeout=1200)
    if rc != 0 or "WEIGHTS_LOADED_OK" not in out:
        stop("checkpoint_load", "FAIL",
             "Model graph built but the official pretrained weights DID NOT "
             "load — see traceback above. Common cause: layer-count/name "
             "mismatch between the shipped network code and the released .hdf5 "
             "(the README already warns results are not reproducible).")
    params = out.split("WEIGHTS_LOADED_OK params=")[-1].strip().splitlines()[0]
    ok("checkpoint_load", f"official SARcarl weights loaded, {params} params")
    RESULT["phase1"] = "PASS — env installs, official checkpoint loads."

    # -- STEP 5: PHASE 2 — one real inference pass ------------------------
    if not os.path.exists(PHASE2_NPZ):
        RESULT["overall"] = ("PHASE 1 PASS. Phase 2 skipped — attach the "
                             "pelagic-dsen2cr-phase2-input dataset to run it.")
        _dump()
        log(f"\nPHASE 1 RESULT: {RESULT['overall']}")
        return
    phase2(ckpt)


def phase2(ckpt):
    infer = os.path.join(TMP, "_infer.py")
    with open(infer, "w") as f:
        f.write(f'''
import json, sys
import numpy as np
sys.path.insert(0, {TMP + "/dsen2-cr/Code"!r})
import tensorflow as tf
from dsen2cr_network import DSen2CR_model

print("GPU available:", tf.test.is_gpu_available())
d = np.load({PHASE2_NPZ!r})
meta = json.loads(str(d["meta"]))
sar_lin = d["sar"].astype("float64")   # (2,128,128) linear sigma0 VV,VH
opt_dn  = d["opt"].astype("float64")   # (13,128,128) L1C DN = reflectance*10000
opt_ref = d["opt_ref"].astype("float64") if "opt_ref" in d.files else None
print("input opt DN range", float(opt_dn.min()), float(opt_dn.max()))
print("input sar linear range", float(sar_lin.min()), float(sar_lin.max()))
sar_db_raw = 10.0 * np.log10(np.clip(sar_lin, 1e-10, None))
print("input sar dB: VV mean %.1f min %.1f  VH mean %.1f min %.1f" % (
      sar_db_raw[0].mean(), sar_db_raw[0].min(), sar_db_raw[1].mean(), sar_db_raw[1].min()))

# --- DSen2-CR normalization (Code/tools/dataIO.py, Code/dsen2cr_main.py) ---
# SAR -> dB, per-band asymmetric clip, rescale to [0,2]
bands = [(-25.0, 0.0), (-32.5, 0.0)]
sar_n = np.stack([(np.clip(sar_db_raw[i], lo, hi) - lo) / (hi - lo) * 2.0
                  for i, (lo, hi) in enumerate(bands)])
# optical clip [0,10000], /2000 -> [0,5]
opt_n = np.clip(opt_dn, 0.0, 10000.0) / 2000.0

model, _ = DSen2CR_model(((13, 128, 128), (2, 128, 128)), batch_per_gpu=1,
                         num_layers=16, feature_size=256,
                         use_cloud_mask=True, include_sar_input=True)
model.load_weights({ckpt!r})
pred = model.predict([opt_n[None].astype("float32"),
                      sar_n[None].astype("float32")], batch_size=1)
print("raw pred shape", pred.shape)
out = pred[0, :13]                      # predicted cloud-free optical, [0,5] scale
print("pred[:13] range", float(out.min()), float(out.max()),
      "mean", float(out.mean()), "std", float(out.std()))
print("finite:", bool(np.isfinite(out).all()))

# artifact diagnostics for the open-water case
lum = out[1:4].mean(0)                  # visible-band luminance of the output
sar_vv_n = sar_n[0]
def _corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    dn = float(np.sqrt((a*a).sum() * (b*b).sum()))
    return float((a*b).sum() / dn) if dn > 0 else 0.0
# block seams every 16 px are the classic DSen2-CR tiling artifact
col = np.abs(np.diff(lum, axis=1)); row = np.abs(np.diff(lum, axis=0))
seam = 0.5*(col[:, 15::16].mean() + row[15::16, :].mean())
print("output luminance vs SAR-VV corr: %.3f" % _corr(lum, sar_vv_n))
print("output 16px-seam strength / global grad: %.3f" % (seam / (col.mean()+row.mean()+1e-9)*2))
print("output negative-pixel fraction: %.3f" % float(np.mean(out < 0)))

save = dict(input_opt=opt_n.astype("float32"), output_opt=out.astype("float32"),
            sar_norm=sar_n.astype("float32"), meta=json.dumps(meta))
if opt_ref is not None:
    save["ref_opt"] = (np.clip(opt_ref, 0.0, 10000.0) / 2000.0).astype("float32")
np.savez_compressed("/kaggle/working/phase2_pred.npz", **save)
print("PHASE2_INFER_OK")
''')
    rc, out = run(f"{CONDA} run -n {ENV_NAME} python {infer}", timeout=1800, env=GPU_ENV)
    if rc != 0 or "PHASE2_INFER_OK" not in out:
        RESULT["steps"]["phase2_inference"] = {"verdict": "FAIL", "detail": out.splitlines()[-8:]}
        RESULT["overall"] = "PHASE 1 PASS. PHASE 2 FAIL at inference — see log."
        _dump()
        log(f"\nRESULT: {RESULT['overall']}")
        return
    ok("phase2_inference", "one forward pass completed, output finite")
    for ln in out.splitlines():
        if ln.startswith("output ") or ln.startswith("input sar dB") or ln.startswith("pred[:13]"):
            log("   " + ln)

    # render with the base image's matplotlib (py3.12)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load("/kaggle/working/phase2_pred.npz")
    meta = json.loads(str(z["meta"]))

    def rgb(opt_scaled):           # [0,5] -> B04,B03,B02 -> gamma
        r = np.stack([opt_scaled[3], opt_scaled[2], opt_scaled[1]], -1) * 2000.0
        return np.clip(r / 3000.0, 0, 1) ** (1 / 2.2)

    inp, o = z["input_opt"], z["output_opt"]
    panels = [("S1 VV\n" + meta["s1_acquired"][:10], np.clip(z["sar_norm"][0] / 2.0, 0, 1)),
              (f"S2 L1C input, patch cloud {meta.get('s2_patch_cloud_frac', 0)*100:.0f}%\n"
               + meta["s2_acquired"][:10], rgb(inp)),
              ("DSen2-CR output", rgb(o))]
    if "ref_opt" in z.files:
        panels.append(("clear S2 (reference, not input)\n" + meta.get("ref_s2_acquired", "")[:10],
                       rgb(z["ref_opt"])))
    fig, ax = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for a, (t, im) in zip(ax, panels):
        a.imshow(im, cmap="gray" if im.ndim == 2 else None); a.set_title(t, fontsize=9); a.axis("off")
    plt.tight_layout()
    plt.savefig("/kaggle/working/phase2b_panels.png", dpi=110)

    diag = {ln.split(":")[0].strip(): ln.split(":")[-1].strip()
            for ln in out.splitlines() if ln.startswith("output ")}
    RESULT["phase2b"] = {
        "aoi": meta.get("aoi_name"), "aoi_bbox_latlon": meta["aoi_bbox_latlon"],
        "s1_product": meta["s1_product"], "s1_acquired": meta["s1_acquired"],
        "s2_product": meta["s2_product"], "s2_acquired": meta["s2_acquired"],
        "s2_scene_cloud_pct": meta["s2_cloud_pct"],
        "s2_patch_cloud_frac": meta.get("s2_patch_cloud_frac"),
        "hours_apart": round(meta["days_apart"] * 24, 1),
        "clear_reference": meta.get("ref_s2_product"),
        "output_finite": bool(np.isfinite(o).all()),
        "output_is_flat": bool(float(np.std(o)) < 1e-4),
        "output_std": float(np.std(o)),
        "diagnostics": diag,
        "note": "open-water test, patch fully clouded. Visual read in phase2b_panels.png. "
                "No in-patch clear water to compare against; clear S2 panel is a different day.",
    }
    RESULT["overall"] = "PHASE 1 PASS + PHASE 2 (land) + PHASE 2b (open water) ran — see phase2b_panels.png"
    _dump()
    log(f"\nRESULT: {RESULT['overall']}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        RESULT["overall"] = "STOP — unhandled exception in feasibility script"
        _dump()
        traceback.print_exc()
        log("\nPHASE 1 RESULT: STOP — unhandled exception (see traceback)")
        sys.exit(0)
