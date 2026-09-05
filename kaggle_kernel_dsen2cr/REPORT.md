# DSen2-CR feasibility via Kaggle — final report

**Task:** `prompt/PROJECT Pelagic — DSen2-CR feasibil.md` (Task 4, scoped) +
Phase 2b continuation (open-water test).
**Date:** 2026-09-06
**Bottom line: GO for feasibility — the model installs, loads, runs, and is
well-behaved on land and open water. Whether it is *useful* for this project's
oil-vs-lookalike problem is unproven and, after Phase 2b, leans mildly negative.
Not integrated; integration remains a separate explicit decision.**

---

## 0. Account / privacy (stated up front, as requested)

| Item | Value |
|---|---|
| Kaggle account | **`papawitsaeliw`** ("Papawit Saeliw") — the account the supplied `KGAT_` token authenticates as |
| Kernel `papawitsaeliw/project-pelagic-dsen2-cr-feasibility` | **private** |
| Dataset `papawitsaeliw/dsen2cr-sarcarl-checkpoint` (checkpoint mirror) | **private** |
| Dataset `papawitsaeliw/pelagic-dsen2cr-phase2-input` (S1+S2 pairs) | **private** |

The checkpoint is GPLv3 code; private hosting means no redistribution concern.
Local `~/.kaggle/kaggle.json` holds a different legacy account (`stickearthv10`)
and was left untouched — every Kaggle call in this task used the env token.

Isolation held throughout: nothing in the main venv, `requirements.txt`, or app
runtime was touched. All work is in `kaggle_kernel_dsen2cr/` plus doc/memory
updates.

---

## 1. Phase 1 — environment + checkpoint

**Did Python 3.7 / TF 1.15 install on Kaggle? — YES.
Did the checkpoint load? — YES.**

Kaggle's current base image is Python 3.12 with no `conda`, so the kernel
bootstraps **Miniforge** into `/tmp` (Miniforge, not Miniconda — plain Miniconda
hit an Anaconda-defaults Terms-of-Service gate that blocks `conda create`
non-interactively), then:

| Step | Result |
|---|---|
| kernel internet | PASS (only after the account was phone-verified — unverified Kaggle accounts silently get no internet and no GPU) |
| `conda create -c conda-forge python=3.7` | PASS → **Python 3.7.12** |
| `pip install tensorflow-gpu==1.15.0 keras==2.2.4 h5py==2.10.0 numpy==1.18.5 protobuf==3.20.3` | PASS — all import cleanly, ~50 s |
| `git clone github.com/ameraner/dsen2-cr` | PASS |
| `model_SARcarl.hdf5` | PASS — **75,943,784 bytes**, valid HDF5, byte-exact to the earlier scoping. Served from a private Kaggle dataset because `gdown` 4.x (newest that runs on py3.7) can no longer parse Google Drive's download page; fetched once locally with `gdown` 6.x. |
| build `DSen2CR_model(((13,128,128),(2,128,128)), num_layers=16, feature_size=256)` | PASS — **18,947,341 params** (matches the ~19M estimate in `docs/cloud_removal_scoping.md`) |
| **`model.load_weights()`** | **PASS — no error** |

**Correction to the prior close-by-decision:** that investigation was blocked on
the **PyTorch port** (`import base_model.BaseModel` — nonexistent; no PyTorch
weights). The **Keras/TF path — the one with published checkpoints — had never
been tried in a correct Python 3.7 environment.** It works.

---

## 2. Phase 2 — one real inference pass (land)

**Does inference produce a plausible cloud-removed image? — YES.**

| | Product | Acquired |
|---|---|---|
| S1 | `S1C_IW_GRDH_1SDV_20260831T054933_..._BCC2` | 2026-08-31 |
| S2 (L1C, 13-band) | `S2B_MSIL1C_20260901T104619_N0512_R051_T31UFT` | 2026-09-01 |

- 2 days apart, 1.28 km / 128 px patch over a highway interchange SE of
  Rotterdam. Scene cloud ~28%; cloud + shadow cover ~40% of the patch.
- Input pair fetched locally with this project's own CDSE credentials
  (`fetch_phase2_pair.py`) — real product IDs and acquisition dates, no
  synthetic data.
- Preprocessing per DSen2-CR's exact spec (SAR → dB, per-band clip
  VV[-25,0]/VH[-32.5,0] → [0,2]; optical clip[0,10000]/2000 → [0,5];
  channels-first).

Output:
- shape `(1, 27, 128, 128)` — 13 predicted bands + 13 input echo + 1 cloud-mask
  slot, exactly as the architecture dictates
- `pred[:13]`: range −0.05 … 2.62, mean 0.57, std 0.37, **all finite, well
  distributed, not flat**
- **Visual (`run_evidence/phase2_triptych.png`): the cloud + shadow over ~40% of
  the patch is substantially removed; the interchange reconstructs continuously
  through the previously-clouded area; the clear half is preserved with correct
  colour.** Soft/hazy where the thick cloud was, as expected — not noise, not an
  identity pass.

GPU note: TF 1.15 cannot use Kaggle's T4 (`cuInit: UNKNOWN ERROR (303)` — in-env
CUDA 10 vs CUDA-12 host). Inference ran **on CPU in ~10 s** for one patch; a full
scene would be a few minutes on CPU. GPU is not a blocker for a post-hoc stage.

---

## 3. Phase 2b — one real inference pass (open water)

Same kernel / env / checkpoint / preprocessing, only the input pair changed.

| | Product | Acquired |
|---|---|---|
| S1 | `S1C_IW_GRDH_1SDV_20260814T015754_..._3AA8` | 2026-08-14 01:58 UTC |
| S2 (L1C, model input) | `S2B_MSIL1C_20260813T183919_N0512_R070_T11SKU` | 2026-08-13 18:39 UTC |
| S2 (clear reference, **not** fed to model) | `S2A_MSIL1C_20260810T184831_..._T10SGD` | 2026-08-10 18:49 UTC |

- **AOI:** Coal Oil Point natural seep field, Santa Barbara Channel
  (34.388 N, −119.886 W) — open water, persistent real oil slicks from seabed
  seeps.
- S1↔S2 **~7 h apart.** S2 scene cloud 15%, but **the 128 px patch itself is
  ~100% clouded.** Only 0%- or 100%-cloud S2 passes exist over this small AOI in
  120 days — no partial-cloud pass was available, so the realistic
  partly-clouded case could not be tested.
- SAR over the patch: VV mean **−29.7 dB**, mostly **below DSen2-CR's −25 dB VV
  clip floor** → after normalization the SAR input is effectively blank apart
  from a thin sprinkle. Combined with the fully-clouded optical, the model got
  **near-zero real information.**

Output stats:
- shape `(1, 27, 128, 128)`, finite, not flat (std 0.19 across 13 bands)
- **visible-band means 0.65 / 0.48 / 0.32 vs the clear reference's
  0.66 / 0.48 / 0.30 — average water radiometry essentially correct**
- visible-band *spatial* std ≈ 0.013 — **~5× smoother than real water**
- 16% slightly-negative pixels, all NIR/SWIR over dark water (minor, invisible
  in RGB)
- 16 px tiling-seam strength ≈ global gradient → **no checkerboard artifact**
- output-luminance vs SAR-VV correlation 0.51 — carries faint SAR structure
  through, not visually meaningful

**Visual read (`run_evidence/phase2b_panels.png`):** open water under thick
cloud reconstructs as **smooth, continuous, plausibly-coloured water with no
artifacts, no noise, no hallucinated pattern** — the "does it produce garbage
over water" fear is answered **no**. But the output is **featureless**: the model
produced generic average water and **recovered no surface texture**; it did not
reconstruct the slick/dark band clearly visible in the SAR. With a blank SAR and
100% cloud it fell back on a learned "calm water" prior, applied competently but
uninformatively.

**Help or hurt oil-vs-lookalike discrimination?**
- **Doesn't hurt** — no false texture that could be mistaken for oil or a
  lookalike.
- **Doesn't help** (fully-clouded case) — adds no independent optical cue; its
  only structure is an echo of the SAR the project already has.
- **The partial-cloud case is not answerable from this test** — the scenario
  that matters (clear part of a patch anchoring the reconstruction of the cloudy
  part, over a known slick) wasn't obtainable over this AOI. Not forcing a
  conclusion the single test can't support.

---

## 4. Go / No-Go

**GO for feasibility, at the low end of "GO but tempered".**

- **It works** — more firmly than after Phase 2: robust and radiometrically
  correct on **both** land and open water, no artifacts, no crash, CPU-fast,
  reproducible from `dsen2cr_feasibility.py` in ~5–8 min.
- **Usefulness for this project is now mildly negative** — exactly where you'd
  want help (heavy cloud over water) the output is bland and carries no
  discriminative signal.
- **Decisive next test (not yet done):** a **partial-cloud open-water scene with
  a known slick**, so the reconstruction of the cloudy part can be checked
  against real water in the same frame. Until that's run, treat "cloud removal
  helps distinguish oil from lookalikes" as unproven and leaning unlikely.
- If pursued: **post-hoc only (Option B in `docs/cloud_removal_scoping.md`),
  never the live inference path.**

---

## 5. Verification — what was actually tested

| Claim | Evidence |
|---|---|
| py3.7 + TF 1.15 + Keras 2.2.4 install & import on Kaggle | kernel run v7–v10, `run_evidence/p37_pip_freeze.txt`, `result_after_phase2b.json` |
| official checkpoint loads | `checkpoint_load: PASS`, 18,947,341 params |
| Phase 2 land inference plausible | `run_evidence/phase2_triptych.png` |
| Phase 2b open-water inference well-behaved but featureless | `run_evidence/phase2b_panels.png`, `result_after_phase2b.json` diagnostics |
| real S1/S2 data (not synthetic) | product IDs + dates from CDSE catalog, recorded in each npz's `meta` |

External services actually exercised: CDSE OAuth2 token endpoint, CDSE OData
catalog, Sentinel Hub Process API (S1 GRD + S2 L1C), Kaggle kernels + datasets
API, GitHub, PyPI, conda-forge, Google Drive (locally). No real ERA5, GFW, or
oil-detection scoring involved — out of scope.

---

## 6. Files

- `kaggle_kernel_dsen2cr/dsen2cr_feasibility.py` — the kernel (Phase 1 + 2 + 2b)
- `kaggle_kernel_dsen2cr/fetch_phase2_pair.py` — local S1+S2 fetcher
  (`.env` CDSE creds; `PHASE2_AOI=rotterdam|coalpoint`, default `coalpoint`)
- `kaggle_kernel_dsen2cr/kernel-metadata.json` — kernel def, 2 private datasets attached
- `kaggle_kernel_dsen2cr/RESULT.md` — running log of all 10 kernel versions
- `kaggle_kernel_dsen2cr/run_evidence/` — `phase2_triptych.png`,
  `phase2b_panels.png`, `result_after_phase2b.json`, `p37_pip_freeze.txt`
- `docs/status.md` — DSen2-CR section updated
- Kaggle: kernel + `dsen2cr-sarcarl-checkpoint` + `pelagic-dsen2cr-phase2-input`
  (all private, account `papawitsaeliw`)

## 7. Open items (your call)

1. **Commit** — changes are uncommitted on `fix/live-fetch-subsecond-timerange`
   (unrelated branch); wants its own branch.
2. **Disposability** — kernel + 2 datasets still live on Kaggle. Keeping them
   makes re-runs instant; can be deleted on request.
3. **Next test** — partial-cloud open-water slick scene, if this is pursued.
4. **Integration** — out of scope; separate decision.
