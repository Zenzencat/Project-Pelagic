# DSen2-CR feasibility via Kaggle — result

**Task:** `prompt/PROJECT Pelagic — DSen2-CR feasibil.md` (Task 4, scoped).
**Date:** 2026-09-06
**Verdict: GO (feasibility confirmed).** The official DSen2-CR Keras/TF1.15
checkpoint installs, loads, and runs, and one real Sentinel-1 + Sentinel-2 pair
produces a **plausible cloud-removed reconstruction**. This is *not* an
instruction to integrate it — the task ends at "does it work".

Kernel: `papawitsaeliw/project-pelagic-dsen2-cr-feasibility` (private).
Runs fully isolated — nothing in the main venv / `requirements.txt` / app
runtime is touched.

## Phase 1 — environment + checkpoint (PASS)

| Step | Result |
|---|---|
| kernel internet | PASS (after the account was phone-verified) |
| conda | PASS — bootstrapped **Miniforge** into `/tmp` (Kaggle's base image is Python 3.12, no conda; Miniforge avoids the Anaconda-defaults ToS gate that blocked plain Miniconda) |
| Python 3.7 env | PASS — `conda create -c conda-forge python=3.7` → **Python 3.7.12** |
| TF 1.15 stack | PASS — `tensorflow-gpu==1.15.0`, `keras==2.2.4`, `h5py==2.10.0`, `numpy==1.18.5`, `protobuf==3.20.3` all install and import cleanly in ~50 s |
| dsen2-cr code | PASS — `git clone github.com/ameraner/dsen2-cr` |
| checkpoint | PASS — `model_SARcarl.hdf5`, **75,943,784 bytes**, valid HDF5 (mirrored to a private Kaggle dataset because `gdown` 4.x — the newest that runs on py3.7 — can't parse Google Drive's current download page; fetched once locally with `gdown` 6.x, byte size matches the earlier scoping) |
| **load pretrained weights** | **PASS** — `DSen2CR_model(((13,128,128),(2,128,128)), num_layers=16, feature_size=256)` builds (**18,947,341 params**, matches the ~19M estimate in `docs/cloud_removal_scoping.md`) and `model.load_weights()` completes with **no error** |

This contradicts the premise of the earlier close-by-decision. That
investigation was blocked on the **PyTorch port** (imports a nonexistent
`base_model.BaseModel`, no PyTorch weights). The **Keras/TF path** — the one
with published checkpoints — was never actually tried in a correct Python 3.7
environment. It works.

## Phase 2 — one real inference pass (PASS)

Input pair, fetched locally with this project's own CDSE credentials
(`fetch_phase2_pair.py`, no synthetic data, real product IDs / dates):

- **S1**: `S1C_IW_GRDH_1SDV_20260831T054933_..._BCC2` — acquired 2026-08-31
- **S2**: `S2B_MSIL1C_20260901T104619_..._T31UFT` L1C, 13 bands — acquired
  2026-09-01, scene cloud ~28 %
- 2 days apart, 1.28 km patch (128 px) of a highway interchange SE of Rotterdam
- returned S2 granule (per Process provenance) is the neighbouring
  `T31UET` tile of the **same acquisition** — same datetime, real data, noted
  for honesty.

Preprocessing followed DSen2-CR's spec exactly (SAR → dB, per-band asymmetric
clip VV[-25,0]/VH[-32.5,0] → [0,2]; optical clip [0,10000]/2000 → [0,5];
channels-first).

Result:
- `model.predict` returns shape `(1, 27, 128, 128)` (13 predicted bands + 13
  input echo + 1 cloud-mask slot), as the architecture dictates.
- predicted optical `pred[:13]`: range **-0.05 … 2.62**, mean 0.57, std 0.37,
  **all finite**, not flat — a real, well-distributed reconstruction.
- **Visual sanity check (`phase2_triptych.png`): the cloud + shadow covering
  the top-left ~40 % of the S2 patch is substantially removed; the highway
  interchange is reconstructed continuously through the previously-clouded
  area; the already-clear bottom half is faithfully preserved with correct
  colour.** Soft/hazy where the thick cloud was, as expected for this model —
  not noise, not garbage, not an identity pass.

### GPU note

Kaggle's T4 is **not usable by TF 1.15** (`cuInit: UNKNOWN ERROR (303)` — the
in-env CUDA 10 / cuDNN 7.6 can't drive the host's CUDA-12-era stack). It didn't
matter: the forward pass ran **on CPU in ~10 s** for one 128-px patch. A full
scene (hundreds of patches) would be a few minutes on CPU — acceptable for a
post-hoc analysis stage, which is the only place this was ever proposed
(`docs/cloud_removal_scoping.md` Option B).

## Phase 2b — open-water test (2026-09-06)

Same kernel / env / preprocessing, only the input pair changed.

- **AOI**: Coal Oil Point natural seep field, Santa Barbara Channel (~34.388 N,
  −119.886 W) — open water, persistent real oil slicks from seabed seeps.
- **S1**: `S1C_IW_GRDH_1SDV_20260814T015754…` — 2026-08-14 01:58 UTC
- **S2**: `S2B_MSIL1C_20260813T183919…_T11SKU` L1C — 2026-08-13 18:39 UTC
  (~7 h apart), scene cloud 15 %, **patch cloud ~100 %** (only 0 %- or 100 %-
  cloud passes exist over this AOI in 120 days — no partial)
- Reference (not model input): near-clear S2 `S2A_MSIL1C_20260810T184831…`,
  2026-08-10, 7 % cloud

SAR over the patch: VV mean **−29.7 dB** (min at the −100 dB log-floor for
nodata) — almost entirely **below DSen2-CR's −25 dB VV clip floor**, so the
normalized SAR input is effectively blank apart from a thin sprinkle where the
water is rougher. Combined with the fully-clouded optical, the model got
**near-zero real information** and had to fall back on its prior.

Output (`run_evidence/phase2b_panels.png`, `run_evidence/result_after_phase2b.json`):
- shape `(1, 27, 128, 128)`, finite, not flat (std 0.19 across 13 bands)
- **visible-band means 0.65 / 0.48 / 0.32 vs the clear reference's
  0.66 / 0.48 / 0.30 — average water radiometry is essentially correct**
- but visible-band spatial std ~0.013 — **~5× smoother than real water**:
  a near-featureless flat blue field
- 16 % slightly-negative pixels, all in NIR/SWIR bands over dark water — minor,
  invisible in RGB
- 16 px tiling-seam strength ≈ global gradient (no checkerboard artifact)
- output-luminance vs SAR-VV correlation 0.51 — it carries the faint SAR
  structure through, but it is not visually meaningful

**Visual read:** open water under thick cloud reconstructs as **smooth,
continuous, plausibly-coloured water with no artifacts, no noise, no
hallucinated pattern** — the "does it produce garbage" fear is answered **no**.
But it is **uninformative**: the model produced generic average water and
recovered no surface texture (there was no information to recover from a blank
SAR + 100 % cloud). It did **not** reconstruct the slick / dark band that is
clearly visible in the SAR.

**Help or hurt oil-vs-lookalike discrimination?** From this one test, for the
fully-clouded case: it would not *hurt* (no false texture introduced) but does
not *help* — the reconstruction adds no independent optical cue; the only
structure in it is an echo of the SAR the project already has. Whether it helps
in the **partial-cloud** case — the realistic one, where the clear part of a
patch could anchor the reconstruction of the cloudy part — **is not answerable
from this test**: no partial-cloud pass over this AOI was available, so it
wasn't tried.

## Go / No-Go

**GO for feasibility — DSen2-CR works and is well-behaved, on land and open
water.** Phase 2b **slightly weakens the "useful for this project" side**: under
heavy cloud over water the output is radiometrically correct but featureless, so
its discriminative value is low exactly where you'd most want it. It doesn't
break, it just doesn't add much. Whether to spend more time is still the
deferred call. If pursued, per the existing scoping:

- Post-hoc only (Option B) — never in the live inference path.
- The isolated env is reproducible from `dsen2cr_feasibility.py` in ~5–8 min.
- **The decisive next test**: a *partial-cloud* open-water scene with a known
  slick, so the reconstruction of the cloudy part can be checked against real
  water in the same frame. Until that's done, whether cloud removal adds
  oil-vs-lookalike signal is unproven — and Phase 2b leans mildly negative.
- Still untested: thick multi-layer cloud, and any quantitative comparison
  against a same-day clear reference.

## Files

- `kernel-metadata.json` — kernel def (2 private datasets attached: checkpoint
  mirror + the Phase 2/2b input pair)
- `dsen2cr_feasibility.py` — the kernel (Phase 1 + Phase 2 + Phase 2b when the
  input dataset is attached, ~5–8 min end to end)
- `fetch_phase2_pair.py` — local S1+S2 fetcher (`.env` CDSE creds;
  `PHASE2_AOI=rotterdam|coalpoint`, default `coalpoint`)
- `run_evidence/phase2_triptych.png` — Phase 2 (land)
- `run_evidence/phase2b_panels.png`, `run_evidence/result_after_phase2b.json` —
  Phase 2b (open water)
- Kaggle datasets (both private, account `papawitsaeliw`):
  `dsen2cr-sarcarl-checkpoint`, `pelagic-dsen2cr-phase2-input`
