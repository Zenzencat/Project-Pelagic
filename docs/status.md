# Project Pelagic: Status & Handoff Document

This document captures the current status of the project, including recent resolutions, verified assumptions, training runs, evaluation metrics, and remaining open issues. Rewritten in full during a health-check pass (session 5) after drift was found across four prior sessions' worth of edits — prior versions of this file had several confirmed fixes missing entirely (the caching fix, inference-path consolidation, coordinate/GSD fix, DSen2-CR scoping) and one section whose "Resolved" framing was no longer accurate. If you're reading this cold, this file should now be trustworthy end to end.

---

## Recent Resolutions

### 1. Part II File Count Discrepancy
* **Status**: Resolved (documentation error, not a data bug).
* Part II (Zenodo Record ID `8253899`) consists of `Mask_no_oil` (685 files) and `Mask_lookalike` (685 files), totaling **1,370 scenes**. The combined Part I + Part II training/validation dataset contains **2,570 scenes** (1,200 from Part I + 1,370 from Part II). README.md, ai_technique.md, and download_sample.py reflect the corrected totals.

### 2. Hard-Fail Guard in `get_real_dataloaders()`
* **Status**: Resolved.
* Added `allow_verification_fallback: bool = False` to `get_real_dataloaders()` in `src/data/dataset.py`. If matching GeoTIFF images for masks are not found and the parameter is `False`, the loader raises a `RuntimeError` at construction time instead of silently falling back to synthetic mock data. `verify_real_pipeline.py` passes `allow_verification_fallback=True` since its purpose is local pipeline-shape verification, not real training.

### 3. Smart dB/Linear-Scale Preprocessing — `main.py` only, NOT the shared preprocessing module
* **Status**: Resolved in `src/api/main.py`'s `/api/predict`. **Not** propagated to `src/data/preprocess.py::run_full_preprocessing()` — see Open Issue #1 below, found during this session's health check.
* `/api/predict` checks `np.any(image_raw < 0)` to distinguish real dB-scale Sentinel-1 imagery from linear-scale synthetic data and runs the correct calibration path for each. This fixed a real zero-confidence segmentation failure when real imagery was run through the synthetic-only preprocessor.
* This exact fix was never applied to the shared `run_full_preprocessing()` used by `get_dataloaders()` (the synthetic-only training path) and by `get_real_dataloaders()`'s verification-fallback branch — that function still unconditionally assumes dB-scale input. See Open Issue #1.

### 4. U-Net v2 Retraining with Scheduler and Oversampling
* **Status**: Completed.
* `kaggle_kernel/train_kaggle.py` v13, three changes vs. the v1 script that produced `model_real_best.pt`:
  1. **Cosine Annealing Scheduler**: `1e-3` decaying to `1e-6` over 15 epochs — fixed v1's wild epoch-to-epoch Val IoU oscillation (0.27–0.70).
  2. **Epoch-Level Global Metric Pooling**: validation IoU/Dice now pool intersection/union across the whole epoch instead of averaging noisy per-batch ratios.
  3. **Lookalike Hard-Negative Oversampling**: background patch sampling prob 0.005 for lookalike vs. 0.002 for no_oil (2.5x), ~71.4% of the negative background patch pool.
* Completed on a Tesla T4 GPU. Checkpoints: `model_real_v2_best.pt`, `model_real_v2_final.pt`. Log: `kaggle_output/training_v2_log.csv`.

### 5. Dataloader Duplication & Drift Between `dataset.py` and `train_kaggle.py`
* **Status**: Resolved.
* `dataset.py`'s local loader now uses the same category-balanced patch sampling as `train_kaggle.py` (via a shared `category` parameter threading through `src/data/preprocess.py::extract_patches_balanced()`). Both files carry an explicit cross-reference banner since `train_kaggle.py` can't import `src/data/` directly (Kaggle kernels must be single-file) — if you change sampling rates or hard-fail behavior in one, mirror it in the other.

### 6. `scene_id` Cache Bug in `/api/predict`
* **Status**: Resolved.
* `scene_id` has a `UNIQUE` constraint on the `detections` table. Re-predicting an already-seen scene under a *different* checkpoint used to silently return the stale cached row instead of re-running inference — the exact reason the demo DB needed manual resets between checkpoint swaps. Fixed via a `checkpoint_hash` column (MD5 of the loaded checkpoint file, migration-safe `ALTER TABLE`): `predict()` now compares the cached row's hash to the currently-loaded checkpoint and overwrites in place on mismatch, while still short-circuiting on a genuine same-checkpoint repeat call (no wasted recompute). Verified via an injected-staleness test.

### 7. Whole-Image vs. Tiled Inference Consolidated
* **Status**: Resolved.
* `/api/predict` used to run the full 2048x2048 scene through the U-Net in one forward pass; `evaluate_holdout.py` tiled into 256x256 patches matching the training regime. Diagnosed a `no_oil_00001` false positive that looked like a live-vs-eval discrepancy (70% confidence live vs. a "minor" 41px offline finding) — the two methods actually agree closely (36 vs. 41 active px, same location); the real story is a genuine **v2-only model regression** (v1 max prob 0.04 at that pixel, v2 max prob 0.83), not a code-path bug. Still consolidated both call sites into `src/inference.py::run_tiled_inference()` so they can't silently drift apart again.

### 8. Fake-Coordinate and Mock-GSD Bugs Fixed
* **Status**: Resolved.
* `/api/predict` used to display every detection at a fixed fake Gulf-of-Thailand coordinate (9.0N, 100.5E) and convert pixel distances using an arbitrary `0.00015` deg/pixel display constant, regardless of which scene was analyzed. Every real holdout GeoTIFF carries real embedded WGS84 georeferencing (`ModelTiepointTag`/`ModelPixelScaleTag`, confirmed ~10 m/pixel) that was simply never read. `/api/predict` now uses each scene's real georeferencing (`src/analysis/geoutils.py`), falling back to the old fixed point only for synthetic scenes with no real-world location. Verified end-to-end in the browser (map actually re-centers and re-renders per scene), and the 5-scene demo lineup re-confirmed with zero regression (see Demo Lineup section below).
* This also surfaced that **the 30 holdout scenes are not all Eastern Mediterranean** — real coordinates span 6 distinct regions (Eastern Mediterranean, Red Sea, Gulf of Mexico, Baltic Sea, Western Mediterranean, Sunda Strait). See Region Findings below.
* `checkpoint_comparison_summary`/`holdout_per_scene_results` now use each scene's real, **latitude-corrected** pixel area (a flat km/degree constant would be wrong by up to ~2x between the Baltic and Sunda Strait scenes — longitude-degree distance shrinks by `cos(latitude)`) instead of the old mock-scale-derived, `_UNVERIFIED`-flagged figures from an earlier pass.
* **Known remaining gap, not fixed**: the frontend's "ขนาดคราบ" (slick size) badge in `App.jsx` is still a hardcoded fake value (`8.5`/`14.5` km²) for every scene — never wired to real geojson-derived area. Flagged, not in scope for any session so far.

---

## Demo Lineup — 4 Oil-Detection Scenes Only (Updated in the presentation-prep sessions after Session 5)

The live demo dropped from 5 scenes to 4, oil-detection only. `no_oil_00004` was
in the lineup as the "clean water, no oil" example, but a later round fixing
the map's fixed-zoom framing (it made every detected slick icon-sized
regardless of real area) also exposed that `no_oil_00004`'s real coordinates
are inland, not water. Investigating every `no_oil` holdout scene as a
replacement found a harder problem: across all 10 `no_oil` scenes,
**genuinely open water and the live v2 checkpoint correctly outputting ~0%
confidence never co-occur** — the two confirmed-water scenes (`no_oil_00000`,
`no_oil_00001`) both false-positive live (~67.8%, ~66.6%), every scene the
model correctly suppresses is on land, and one file (`no_oil_00005`) is a
broken all-zero array. Full investigation, including how each scene was
checked (raw SAR content + reverse geocoding): `docs/deck_assets/MANIFEST.md`.
Rather than show a land scene mislabeled as water, or a live false positive,
the demo is now oil-detection scenes only:

| Scene | Confidence | Polygons | Real coordinates |
|---|---|---|---|
| oil_00000 | 97.5969% | 29 | 35.5307N, 34.7851E |
| oil_00001 | 90.9043% | 27 | 35.4087N, 34.9938E |
| oil_00003 | 96.3351% | 9 | 20.1616N, 38.2182E |
| oil_00004 | 95.2886% | 228 | 35.8224N, 35.0308E |

Values match the last-confirmed numbers exactly (zero regression), and
coordinates are real and vary correctly per scene (confirmed independently
against `docs/holdout_scene_coordinates.json`). Confirmed in-browser:
switching scenes re-centers/re-zooms the Leaflet map to fit the real detected
polygon's bounding box (not a fixed zoom) and re-renders overlay geometry,
not just the API response.

**Also fixed since this table was first written**: the "ขนาดคราบ" (slick
size) badge described as a hardcoded fake `8.5`/`14.5` km² value in Recent
Resolution #8 and Open Issue #2 below is no longer accurate — it's since been
wired to a real shoelace-formula area computed from each detection's own
GeoJSON polygon (same lat-corrected km/deg method used elsewhere in this
repo). Left the original entries below as-is rather than rewriting session
history; treat this note as the current source of truth on that specific
item.

---

## Training Logs Comparison (Val IoU Stability)

* **Val IoU Peaks**: v2 achieved a higher peak Validation IoU (**`0.7424`** at epoch 11) vs. v1 (**`0.6965`** at epoch 12).
* **Metric Stability**: v1's Validation IoU fluctuated wildly epoch-to-epoch (e.g. `0.5010` → `0.6965` → `0.6439` in epochs 11-13); v2's stabilized above `0.70` from epoch 6 onward.

| Epoch | v1 Val Loss | v1 Val IoU | v2 Val Loss | v2 Val IoU | v2 Val Dice |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.725177 | 0.422358 | 0.506773 | 0.641398 | 0.781526 |
| 2 | 1.124958 | 0.274703 | 0.926747 | 0.267861 | 0.422540 |
| 3 | 0.497706 | 0.585735 | 0.428006 | 0.690064 | 0.816613 |
| 4 | 1.122395 | 0.606193 | 0.393626 | 0.711528 | 0.831454 |
| 5 | 0.396908 | 0.673934 | 0.509283 | 0.686476 | 0.814095 |
| 6 | 0.556104 | 0.496347 | 0.367070 | **`0.727504`** | 0.842260 |
| 7 | 0.529022 | 0.534189 | 0.405825 | **`0.706110`** | 0.827743 |
| 8 | 0.479961 | 0.638172 | 0.398910 | **`0.712574`** | 0.832167 |
| 9 | 0.646530 | 0.515526 | 0.351514 | **`0.721042`** | 0.837913 |
| 10 | 0.466078 | 0.674395 | 0.350150 | **`0.733229`** | 0.846084 |
| 11 | 0.697020 | 0.501025 | 0.329764 | **`0.742467`** | 0.852202 |
| 12 | 0.372412 | 0.696502 | 0.347648 | **`0.739894`** | 0.850505 |
| 13 | 0.478718 | 0.643939 | 0.352208 | **`0.739612`** | 0.850318 |
| 14 | 0.403184 | 0.669000 | 0.351726 | **`0.740577`** | 0.850956 |
| 15 | 0.550864 | 0.670618 | 0.346164 | **`0.741993`** | 0.851890 |

---

## Held-Out Test Set Comparison (v1 vs v2, 30 scenes)

| Category | Version | IoU | Dice | Avg FP pixels/scene | Avg FP % of scene | Avg FP area (km², real GSD) |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Oil | v1 | 0.7263 | 0.8044 | 99,575.4 | 2.374% | 8.948 |
| Oil | v2 | 0.7239 | 0.8039 | 89,587.6 | 2.136% | 8.113 |
| No Oil | v1 | 0.7000 | 0.7000 | 300.4 | 0.007% | 0.024 |
| No Oil | v2 | 0.7000 | 0.7000 | 64.8 | 0.002% | 0.005 |
| Lookalike | v1 | 0.0000 | 0.0000 | 1,733,578.4 | 41.332% | 148.354 |
| Lookalike | v2 | 0.0000 | 0.0000 | 1,571,115.3 | 37.458% | 134.461 |

Full source data: `docs/checkpoint_comparison_summary.{json,csv}`, `docs/holdout_per_scene_results.{json,csv}`.

### Confusion matrix (outcome buckets — identical between v1 and v2)

| Category | Correct | Partial | False Positive | False Negative |
|---|:---:|:---:|:---:|:---:|
| Oil (n=10) | 8 | 2 | 0 | 0 |
| No Oil (n=10) | 7 | – | 3 | – |
| Lookalike (n=10) | 0 | – | 10 | – |

Full source: `docs/confusion_matrix_breakdown.{json,csv}`.

### Lookalike Analysis — corrected framing
* **Result**: Lookalike IoU/Dice are exactly `0.0000` across **all 10/10 scenes in both v1 and v2** — this is universal, not "some scenes," and every one of the 10 has at least one false-positive pixel.
* **The real per-scene picture is more nuanced than "still fails everywhere"**: false-positive area ranges from 0.02% to **81.4%** of scene area across the 10 scenes (worse at the tail than an earlier ~70-80% estimate), highly bimodal — 4/10 scenes are catastrophic (45-81%), 2/10 are near-clean (<3%).
* **v1→v2 did produce a real, if modest, improvement**: average false-positive area dropped from 41.3% to 37.5% of scene area (148.4 → 134.5 km² using real, latitude-corrected GSD) — genuine, but far more modest than an earlier "10-45% per-scene" framing implied (that range described the spread across scenes, not the typical improvement).
* **Region matters as much as category** (new finding this round, previously not visible anywhere in the results): pass rate under v2 varies sharply by region, not just by oil/no_oil/lookalike:

  | Region | v2 pass rate |
  |---|:---:|
  | Eastern Mediterranean (bulk of the set) | 11/15 |
  | Sunda Strait | 3/3 |
  | Red Sea | 1/7 |
  | Gulf of Mexico | 0/2 |
  | Western Mediterranean | 0/2 |
  | Baltic Sea | 0/1 |

  Full source: `docs/region_coverage.{json,csv}`. Small per-region sample sizes (n=1-2 for three of the six) mean the 0/2, 0/2, 0/1 results should be read as suggestive, not conclusive on their own — but Eastern Med vs. Red Sea (15 and 7 scenes respectively, both reasonably sized) is a real, sizeable gap worth investigating.
* **Implication unchanged**: simple background-patch oversampling alone is insufficient to separate lookalikes from real oil given identical SAR backscatter-reduction signatures. Distinguishing them needs either structural context (geometry, e.g. length-width ratio) or external metadata (wind speed, multi-temporal shape stability, optical imagery) — see the ERA5/multi-temporal/DSen2-CR investigations below.
* **Isolating the oversampling change's specific effect** (with vs. without, same checkpoint epoch, same eval path) requires a new ablation training run — **blocked pending explicit user go-ahead**, not started.

---

## New Capability Investigation: ERA5 Wind Cross-Check & Multi-Temporal SAR Comparison

* **Status**: Blocked on missing data, code written and ready pending real inputs.
* **Goal**: reduce the lookalike false-positive problem by post-processing detections against (1) ERA5 10m wind speed at acquisition time — oil slicks only show a dark SAR signature within ~1.5-6 m/s — and (2) whether a same-track Sentinel-1 pass exists nearby in time, to check if the detected shape drifted (real oil) or stayed static (lookalike-likely).
* **No acquisition timestamp exists anywhere for this dataset**, confirmed by: (1) no `DateTime` tag or any date-bearing tag in the GeoTIFFs themselves; (2) querying the Zenodo API directly for all three record IDs (8346860 / 8253899 / 13761290) — each deposit is exhaustively just the image `.7z` + mask `.7z` archives, no metadata/sidecar file. This is a property of the original Zenodo deposit, not something a Kaggle mirror stripped out. Without a timestamp there's no way to pick an ERA5 timestep or anchor a "nearby pass" search for any of these 30 scenes as currently downloaded.
* **Sentinel-1 revisit cadence verified current, not assumed**: the 2026 constellation reconfiguration (Sentinel-1C + Sentinel-1D) completed 2026-06-24, restoring a ~6-day nominal same-track revisit — not the ~12-day figure from the prior single-satellite gap.
* **Real, credential-free finding**: Copernicus Data Space Ecosystem's catalog *search* API needs no auth (only downloads do). A real 12-month Sentinel-1 GRD coverage-density check against all 30 scenes' actual coordinates (`docs/sentinel1_coverage_density.json`) confirms real regional variance (57-724 passes/year, mean 274) — every scene has substantial nearby archive density, so the multi-temporal feature has real potential once timestamps exist.
* **Built, not yet run**: `src/analysis/era5_wind_check.py` and `src/analysis/sentinel1_revisit_check.py` — real fetch/classify/compare logic against the documented CDS and CDSE APIs. Standalone scripts only, not wired into `/api/predict` or the DB schema.
* **Credential prep done**: `.env.example` documents the exact env vars each service needs and where to get them; `scripts/validate_credentials.py` does a real minimal auth check against both CDS and CDSE (no data download) — verified its skip-path against this repo's current unconfigured environment.
* **What's still needed**: a free CDS API key, free CDSE OAuth2 client credentials, and — the real blocker — per-scene acquisition timestamps from a source outside this repo. Being chased separately (contacted the companion paper's corresponding author, DOI `10.1016/j.marpolbul.2024.116549`, awaiting reply as of this writing). Not a repo problem to keep digging at further.

---

## New Capability Investigation: SAR-Optical Cloud-Removal Fusion (DSen2-CR)

* **Status**: Scoped, not built. Nothing downloaded/kept, nothing run.
* **Goal**: a third lookalike-discrimination signal — reconstruct a cloud-free Sentinel-2 optical view of each scene (DSen2-CR, Meraner et al. ISPRS 2020 Best Paper, github.com/ameraner/dsen2-cr, pretrained) as a second modality, since lookalikes and real oil can be SAR-identical but visually different in optical color/texture.
* **Feasibility findings from reading the actual code** (not the README's word for it):
  * The maintainer's own README admits they can't currently replicate/test the code themselves.
  * Confirmed exact input spec: 13-band Sentinel-2 L1C + 2-band Sentinel-1, 128x128 crops, channels-first. SAR clipping is band-specific/asymmetric (`[-25,0]` VV, `[-32.5,0]` VH dB) — differs from this project's own uniform `[-25,0]` clip, a real adaptation point. Cloud mask is derived automatically, not a required separate input.
  * All 4 Google Drive pretrained checkpoints verified genuinely downloadable (real HDF5 signatures, correct headers, ~72.4MB each — matches a ~19M-parameter architecture) — not just that the link returns HTTP 200, which alone is not reliable (Drive returns 200 for restricted links too). Downloaded then immediately deleted, not kept.
  * License confirmed GPLv3 by reading the actual file — usable for a school project.
  * The README's "PyTorch implementation" is incomplete: imports `base_model.BaseModel`, which doesn't exist anywhere in the repo. No PyTorch-format pretrained weights are published either — pretrained inference is locked to the old TensorFlow 1.15/Keras 2.2.4/Python 3.7 stack (confirmed still pip-installable but Python-3.7-only), which needs its own isolated environment (can't coexist with this project's own venv).
  * GPU requirement for inference genuinely unverified — flagged as an honest estimate (grounded in parameter count vs. this project's own U-Net's known CPU runtime), not a guess dressed up as confirmed.
  * Confirmed Sentinel-2 access needs no credentials beyond what's already prepped for Sentinel-1 (same CDSE account) — validated with a real search against this project's actual scene coordinates.
* **My recommendation** (see `docs/cloud_removal_scoping.md` for full reasoning): integrate as a **post-hoc lookalike-discrimination stage on top of the existing v2 U-Net**, not retrained input channels — because the timestamp problem scales completely differently between the two (30 holdout scenes vs. ~2,570 training scenes if baked into retraining).
* **Time estimate once timestamps + credentials exist**: roughly 1-2 weeks, with most uncertainty in the discrimination-logic step, not the plumbing (~3-5 days, moderate confidence).
* Design sketch: `src/analysis/cloud_removal_fusion.py`. Real, tested fetch/preprocessing logic where verifiable; the actual DSen2-CR inference call is a deliberate `NotImplementedError` stub (subprocess boundary into a separate environment), not a fake pass-through.

---

## Open Issues

### 1. `run_full_preprocessing()` has no dB/linear-scale branching — found this session
* **Status**: Open, not fixed. Found during this session's health-check test run, not previously documented anywhere.
* `src/data/preprocess.py::run_full_preprocessing()` (used by `get_dataloaders()` — the synthetic-only training path — and by `get_real_dataloaders()`'s verification-fallback branch) unconditionally assumes dB-scale input, unlike `main.py`'s `/api/predict` (Recent Resolution #3), which correctly branches on `np.any(image_raw < 0)`.
* **Confirmed with a direct test**: raw synthetic data from `generate_synthetic.py` is small positive linear values (0.0001-0.30). Running it through `run_full_preprocessing()` produces a preprocessed patch value range of **exactly `[1.0, 1.0]`** — completely saturated, zero information content.
* **Does not affect the real v1/v2 checkpoints** — those were trained exclusively via `train_kaggle.py` on real dB-scale Kaggle data, where the "always assume dB" logic is correct by design (Kaggle never has synthetic fallback data, per the hard-fail guard).
* **Does undermine three verification scripts' actual purpose**, even though all three currently report "success":
  * `verify_pipeline.py` — passes its own shape/range assertions (`[0,1]` bound-check) while actually feeding degenerate flat-`1.0` patches through the pipeline. The assertions check bounds, not information content, so this was never caught.
  * `src/verify_training.py` — the resulting 2-epoch training run collapses to all-zero predictions. The script's own comment attributes this to "common for short runs on small synthetic sets due to random initialization" — given the confirmed flat-`1.0` input, the more likely real cause is that the training data has no signal to learn from at all, not the epoch count.
  * `verify_real_pipeline.py` — same saturated `[1.0, 1.0]` image range observed when run in verification-fallback mode (its normal mode locally, since real images aren't downloaded here).
* **Not fixed as part of this session** — this is a report-only health check per this round's scope. Worth a small, well-scoped fix in a future round (mirror `main.py`'s branching into `run_full_preprocessing()`).

### 2. Frontend slick-size badge is hardcoded fake data
* **Status**: Open, flagged not fixed (see Recent Resolution #8).
* `frontend/src/App.jsx`'s "ขนาดคราบ" (slick size) badge shows a hardcoded `8.5`/`14.5` km² for every single detection regardless of the real geojson-derived area. Not wired to any real data at any point. Low priority relative to other open items, but worth knowing before presenting the demo — a sharp-eyed viewer could notice every detection shows the same area.

### 3. Issue 5 — Lookalike oversampling ablation
* **Status**: Blocked, pending explicit user go-ahead. Not started.
* Isolating the lookalike-oversampling change's real effect (with vs. without, same checkpoint epoch, same eval path) requires a new training run — v1 vs. v2 differ in three things simultaneously (LR scheduler, pooled-IoU, oversampling rate), not a clean ablation. Budget the same order of GPU time as the original v2 training run.

### 4. Acquisition-timestamp gap (ERA5 / multi-temporal SAR / DSen2-CR)
* **Status**: Blocked on a human path outside this repo, being chased separately. See the two "New Capability Investigation" sections above for full detail — not repeated here to avoid the file drifting out of sync with itself again.
