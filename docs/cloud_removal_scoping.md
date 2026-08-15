# SAR-Optical Cloud-Removal Fusion (DSen2-CR): Scoping Notes

**Status: design/feasibility scoping only. Nothing here is wired into the live pipeline. No pretrained weights were downloaded and kept, no inference was run.**

This document covers a third potential lookalike-discrimination signal, alongside ERA5 wind (`docs/status.md`) and multi-temporal SAR shape comparison — reconstructing a cloud-free Sentinel-2 optical view of the same scene using [DSen2-CR](https://github.com/ameraner/dsen2-cr) (Meraner et al., ISPRS 2020 Best Paper), as a second modality to help tell real oil apart from lookalikes.

---

## 1. Feasibility check — what I actually verified, not what the README claims

I cloned the repo fresh and read the actual code, rather than trusting the README at face value. The README itself is a good instinct to distrust here: it opens with the maintainer's own note that **"I'm not able to properly replicate results and test the code"** — a direct admission from the author, not a red flag I'm inferring.

### Compute requirement — genuinely uncertain, here's my honest read
- The README says "strongly advised to use GPU." I could not verify actual runtime without a GPU, because I don't have access to one in this environment to test against. I won't pretend otherwise.
- What I *can* say from reading the code: the network (`Code/dsen2cr_network.py`, `Code/dsen2cr_pytorch_model.py`) is a fairly ordinary residual CNN — 16 residual blocks, 256 features, standard `Conv2D`/`ReLU`/`Add` layers, no CUDA-only ops. I calculated ~19M parameters from the architecture, and independently cross-checked that against the actual downloaded checkpoint file sizes (~72.4MB / 4 bytes per float32 ≈ 19M) — they match, which gives me confidence the parameter count estimate is right.
- For comparison: this project's own U-Net (a smaller 4-level, 64-base-channel network) already runs CPU-only inference in this exact environment at ~10-14s per 2048x2048 scene (64 patches at 256x256). DSen2-CR's ~19M-parameter net operating on smaller 128x128 patches is the same order of magnitude of compute — full-scene coverage needs ~4x more tiles (256 tiles of 128x128 vs. our 64 tiles of 256x256), so a rough scaling guess would put CPU inference at somewhere in the tens-of-seconds-to-low-minutes-per-scene range.
- **I'm flagging this as a genuine estimate, not a verified number.** I have not run this model. If you need a confirmed number before committing time to this track, the fastest way to get one is a Colab free-tier run of the plain inference (no training) once real input data exists — cheap to test, not worth simulating here with placeholder data.

### Exact input spec — confirmed directly from code, not the README
- **Sentinel-2 optical**: 13 bands (the full L1C band set, including B10 cirrus — this rules out L2A products, which drop B10). 128x128 pixel crops.
- **Sentinel-1 SAR**: 2 bands (VV, VH) — matches this project's own SAR convention. Same 128x128 crop size.
- **Data format**: channels-first `(C, H, W)`, not this project's `(H, W, C)` — needs a transpose.
- **Normalization** (from `Code/dsen2cr_main.py` + `Code/tools/dataIO.py`):
  - SAR: clipped **per-band, asymmetrically** — VV to `[-25, 0]` dB, VH to `[-32.5, 0]` dB — then rescaled to `[0, 2]`. This project's own `src/data/preprocess.py` clips both bands uniformly to `[-25, 0]` — **not a drop-in reuse**, a real adaptation is needed.
  - Optical: clipped to `[0, 10000]` per band (typical L1C TOA reflectance ×10000 scaling), then divided by 2000 → `[0, 5]`.
  - I implemented and tested this exact normalization in `src/analysis/cloud_removal_fusion.py::preprocess_for_dsen2cr()` against dummy arrays of the right shape — output ranges match spec exactly (`[0, 2]` SAR, `[0, 5]` optical).
- **Cloud mask**: I initially assumed this would need to be supplied as a separate input (a whole extra thing to source). It doesn't — it's derived automatically from the cloudy optical image itself via a classical (non-learned) detector, and is only used to shape the *training* loss, not as a network input. One less dependency than expected.
- **A real gap in the shipped code**: the repo's own `--predict` CLI path (`predict_dsen2cr()` in `Code/dsen2cr_tools.py`) hardcodes `include_target=True`, meaning it's written to *evaluate against a known ground-truth cloud-free image*, not to run blind inference on a genuinely novel scene. The underlying `DataGenerator` class does support `include_target=False` for true blind inference — so this is a small (~15-20 line) custom wrapper to write, not a redesign, but it's not "just run `--predict`" as the README implies.

### Checkpoint links — actually verified downloadable, not just "returns 200"
A Google Drive "view" URL returns HTTP 200 even for broken/restricted links, so I didn't stop there. I checked the actual download endpoint and confirmed real file transfers for all four checkpoints:

| Variant | Size | Confirmed |
|---|---|---|
| `model_SARcarl.hdf5` (SAR input, CARL loss — the paper's best config) | 75,943,784 bytes | Real HDF5 signature, correct headers |
| `model_SARplain.hdf5` (SAR input, plain L1 loss) | 75,943,784 bytes | Real HDF5 signature, correct headers |
| `model_noSARcarl.hdf5` (no SAR, CARL loss) | 75,923,640 bytes | Real HDF5 signature, correct headers |
| `model_noSARplain.hdf5` (no SAR, plain L1 loss) | 75,923,640 bytes | Real HDF5 signature, correct headers |

(A first check via a page-content fetch tool showed an ambiguous "sign-in required" result — that turned out to be the tool struggling with Google's JS-heavy UI, not a real restriction. Worth noting since it would have been an easy false negative to report if I'd stopped there.) I deleted all four downloaded files immediately after confirming them — did not keep them, per this session's scope.

For our use case, `sar_carl` is the right pick — it's the only variant that both uses SAR input (relevant since we have real co-located Sentinel-1) and optimizes the paper's best-performing loss.

### License — confirmed, not assumed
**GPLv3** (read the actual `LICENSE` file, not the README's silence on it). Usable for a school project — academic evaluation isn't "distribution" in the sense that triggers GPL's copyleft obligations. Worth knowing if this code is ever incorporated into something you'd share/distribute more broadly later: GPLv3 is a strong copyleft license, and a combined work that includes GPLv3 code generally needs to be GPLv3-compatible too if distributed. Not a blocker for a capstone; flagging so it's a known fact rather than a surprise later.

### The "PyTorch implementation" mentioned in the README is incomplete
`Code/dsen2cr_pytorch_model.py` imports `from .base_model import BaseModel` — I searched the entire repo and **`base_model.py` doesn't exist anywhere in it**. This file is a class definition lifted from a different (unpublished-here) PyTorch training framework — it cannot run as-is. The core `ResnetStackedArchitecture` class itself (the actual network, no `BaseModel` dependency) is self-contained and structurally identical to the Keras version, which is a genuinely useful fact — but there are **no published PyTorch-format pretrained weights**, only the four `.hdf5` (Keras) checkpoints above. So pretrained inference is locked to the Keras/TensorFlow path regardless of the PyTorch class's existence, unless someone manually ports the H5 weights into the PyTorch class layer-by-layer (plausible since the shapes should line up, but real unverified work — not attempted here).

### Environment reality check
`tensorflow-gpu==1.15.0` is still on PyPI and pip-installable (checked directly) — but **only for Python 3.5/3.6/3.7** (`cp35`/`cp36`/`cp37` wheels only, nothing for 3.8+). This cannot live in this project's own venv (`torch>=2.0.0`, modern numpy) without a dependency collision — it needs its own isolated environment: a separate conda env (the README's documented path) or the repo's provided `Docker/Dockerfile` (CUDA 9.0 + cuDNN 7 base image — dated, but Docker isolates the ancient CUDA toolkit from whatever the host actually has, so a modern GPU + modern driver + this specific container should still work via NVIDIA's driver backward-compatibility, in principle — again, unverified without actually running it).

---

## 2. Pipeline design sketch

Written as real, structured code at `src/analysis/cloud_removal_fusion.py`, matching the pattern of the ERA5/multi-temporal modules from the prior session:

- `get_cdse_token()`, `find_nearest_sentinel2_l1c()` — **real, tested this session.** Confirmed Sentinel-2 needs no credential setup beyond what's already in `.env.example` for the multi-temporal Sentinel-1 work (same CDSE account, same `CDSE_CLIENT_ID`/`CDSE_CLIENT_SECRET`) — verified with a real, unauthenticated catalog search against `Collection/Name='SENTINEL-2'` for one of this project's actual scene coordinates, which returned real `MSIL1C` product names.
- `download_cdse_product()` — real logic, not executed end-to-end this session (no product actually downloaded, per scope).
- `preprocess_for_dsen2cr()` — **real, tested this session** against dummy-shaped arrays; output matches DSen2-CR's exact normalization spec.
- `run_dsen2cr_inference()` — **deliberately a stub that raises `NotImplementedError`**, not a fake pass-through. This has to be a subprocess call into a separate DSen2-CR environment (own conda env or Docker), not an in-process import — writing it any other way would misrepresent something that can't actually work that way given the dependency conflict above.
- `run_fusion_for_scene()` — orchestration sketch; refuses to run without a real timestamp, same stance as the ERA5/multi-temporal modules (no fabricated dates).

---

## 3. Where this should integrate — my actual recommendation

You asked for an opinion, not a punt. Two options exist:

**Option A — bake it into the U-Net as extra input channels** (concatenate reconstructed optical bands alongside VV/VH, retrain end-to-end).

**Option B — a separate, post-hoc lookalike-discrimination stage**, run only on candidate detections the existing SAR-only U-Net already flagged, using the reconstructed optical patch to corroborate or cast doubt on that specific detection — the existing v2 checkpoint stays exactly as-is.

**I recommend Option B, clearly**, for one dominant reason: **the timestamp scarcity problem scales completely differently between the two.** Option A requires retraining the segmentation network, which means every one of the ~2,570 training scenes needs its own real acquisition timestamp to fetch a matching Sentinel-2 pass — the same blocker that's stopping ERA5 and multi-temporal SAR, but now applied to roughly 85x more scenes than the 30 in the holdout set. Given timestamps are currently being chased one paywalled-paper-request at a time, that's very likely infeasible on any near-term timeline. Option B only needs a timestamp for the (much smaller) set of scenes you actually want to double-check — the 30 holdout scenes for evaluation purposes, or in a deployed setting, only scenes where the SAR model produced a positive detection at all.

Secondary reasons Option B is the better call regardless of the timestamp math:
- No retraining risk to the v2 checkpoint you've already validated and are presenting.
- Keeps the DSen2-CR dependency (ancient, isolated environment) fully out of the live inference path — consistent with how the ERA5/multi-temporal work was scoped as standalone analysis, not pipeline surgery.
- More defensible as a research narrative: "SAR flags a candidate, optical fusion independently corroborates or casts doubt on it" is a cleaner story for a presentation than "we added 13 more input channels and hoped the network learned to use them well."

Concretely, Option B's discrimination step doesn't need to start as a trained classifier — a simple heuristic comparing optical color/texture statistics inside vs. outside the SAR-detected polygon (real oil often shows a distinct sheen/color signature under the right sun-glint conditions; some lookalike causes like biogenic films or current fronts may show a more uniform water-color signature) is a reasonable, fast-to-prototype starting point before investing in a trained model.

---

## 4. Time estimate once timestamps + credentials exist

This is a judgment call based on code-complexity assessment, not a measured number — I have not run any part of this pipeline. Roughly:

| Step | Estimate |
|---|---|
| Validate CDSE credentials | Already done (Track C) |
| Build/debug real S1+S2 fetch-and-pair logic for the scenes needing checks | 1-2 days |
| Set up isolated DSen2-CR environment (conda or Docker) | 0.5-1 day, possibly more if package-version rot causes friction |
| Adapt the repo's inference path (bypass the ground-truth requirement) | 0.5-1 day |
| Run + validate output against a handful of real scenes | 0.5-1 day |
| Build the Option-B discrimination logic (heuristic first, classifier if warranted) | 1 day (heuristic) to 3-5+ days (trained classifier — open-ended, depends on ambition) |
| Wire as an optional post-hoc stage (if you decide to go beyond analysis scripts) | ~1 day |

**Total: roughly 1-2 weeks of focused work**, with most of the uncertainty sitting in the discrimination-logic step rather than the plumbing (which I'd estimate at 3-5 days combined, with moderate confidence since it's mostly code I can already see the shape of).
