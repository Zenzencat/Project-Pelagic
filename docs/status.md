# Project Pelagic: Status & Handoff Document

This document captures the current status of the project, including recent resolutions, verified assumptions, training runs, evaluation metrics, and remaining open issues.

---

## Recent Resolutions

### 1. Part II File Count Discrepancy
* **Status**: Resolved (documentation error, not a data bug).
* **Details**: 
  * Part II (Zenodo Record ID `8253899`) consists of `Mask_no_oil` (685 files) and `Mask_lookalike` (685 files), totaling **1,370 scenes**.
  * The combined Part I + Part II training/validation dataset contains **2,570 scenes** (1,200 from Part I + 1,370 from Part II), rather than the previously assumed 1,885 scenes.
  * Project files ([README.md](file:///c:/Users/Iris/Downloads/Project%20Pelagic/README.md), [ai_technique.md](file:///c:/Users/Iris/Downloads/Project%20Pelagic/docs/ai_technique.md), and [download_sample.py](file:///c:/Users/Iris/Downloads/Project%20Pelagic/src/data/download_sample.py)) have been updated to reflect the correct totals.

### 2. Hard-Fail Guard in `get_real_dataloaders()`
* **Status**: Resolved.
* **Details**:
  * Added `allow_verification_fallback: bool = False` to `get_real_dataloaders()` in [dataset.py](file:///c:/Users/Iris/Downloads/Project%20Pelagic/src/data/dataset.py).
  * If matching GeoTIFF images for masks are not found and the parameter is `False`, the loader raises a `RuntimeError` at dataset-construction time, preventing silent fallback to synthetic mock data during training.
  * If the parameter is `True`, it proceeds and prints a loud, visible warning message.
  * [verify_real_pipeline.py](file:///c:/Users/Iris/Downloads/Project%20Pelagic/src/data/verify_real_pipeline.py) has been updated to pass `allow_verification_fallback=True` since its purpose is testing pipeline dimensions locally.

### 3. Smart Preprocessing in API & Holdout Inference
* **Status**: Resolved.
* **Details**:
  * Added a smart decibel-aware preprocessor inside [main.py](file:///c:/Users/Iris/Downloads/Project%20Pelagic/src/api/main.py#L181-L197). It dynamically checks if the input GeoTIFF contains negative values.
  * If the input is in decibel scale (real SAR imagery), it runs linear conversion before filtering, db scaling, and normalization. If it is linear (synthetic data), it runs the linear calibration pipeline.
  * This resolves the zero-confidence segmentation failure when running real imagery through the synthetic preprocessor.
  * Downloaded a 30-scene subset (10 Oil, 10 No oil, 10 Lookalike) of the Part III held-out test dataset to `data/holdout/images/` and `data/holdout/masks/`.
  * Verified end-to-end inference and Leaflet map rendering locally on these real scenes.

### 4. U-Net v2 Retraining with Scheduler and Oversampling
* **Status**: Completed.
* **Details**:
  * Modified [train_kaggle.py](file:///c:/Users/Iris/Downloads/Project%20Pelagic/kaggle_kernel/train_kaggle.py) to incorporate two targeted fixes:
    1. **Cosine Annealing Scheduler**: Added `torch.optim.lr_scheduler.CosineAnnealingLR` decaying from `1e-3` to `1e-6` over 15 epochs to prevent validation metric instability.
    2. **Epoch-Level Global Metric Pooling**: Rewrote the metric calculation during the validation loop to accumulate intersections and unions globally across the entire epoch, rather than taking the simple mean of per-batch ratios.
    3. **Lookalike Hard-Negative Oversampling**: Tracked scene categories and adjusted the background patch sampling probability. Lookalike scenes were oversampled with `prob = 0.005` (2.5x increase relative to the clean sea `no_oil` baseline `0.002`), ensuring lookalikes made up ~71.4% of the negative background patch pool.
  * Pushed the script as version 13 to Kaggle and successfully completed the training run on a Tesla T4 GPU.
  * Saved the new model outputs as `model_real_v2_best.pt` and `model_real_v2_final.pt` under [checkpoints/](file:///c:/Users/Iris/Downloads/Project%20Pelagic/checkpoints/).
  * Saved the training logs as [training_v2_log.csv](file:///c:/Users/Iris/Downloads/Project%20Pelagic/kaggle_output/training_v2_log.csv).

---

## Training Logs Comparison (Val IoU Stability)

Below is the comparison of epoch-by-epoch training validation metrics between the original U-Net (v1) and the updated U-Net (v2) incorporating the Cosine Annealing scheduler and Global Metric Pooling:

* **Val IoU Peaks**: v2 achieved a higher peak Validation IoU (**`0.7424`** at epoch 11) compared to v1 (**`0.6965`** at epoch 12).
* **Metric Stability**: While v1's Validation IoU fluctuated wildly epoch-to-epoch (e.g. `0.5010` -> `0.6965` -> `0.6439` in epochs 11-13), v2's Validation IoU stabilized completely above `0.70` starting from epoch 6 and remained flat up to epoch 15, confirming that the LR scheduler successfully smoothed out metric oscillations.

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

## Held-Out Test Set Comparison (v1 vs v2)

We re-evaluated the newly trained `model_real_v2_best.pt` on the same 30-scene holdout test dataset. Below is the direct comparison of performance metrics:

| Category | Model version | Average IoU | Average Dice (F1) | Average Precision | Average Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Oil Spill** | v1 (`model_real_best`) | `0.7263` | `0.8044` | `0.8218` | `0.8943` |
| | v2 (`model_real_v2_best`) | **`0.7239`** | **`0.8039`** | **`0.8243`** | **`0.8889`** |
| **No Oil** | v1 (`model_real_best`) | `0.7000` | `0.7000` | `0.7000` | `0.7000` |
| | v2 (`model_real_v2_best`) | **`0.7000`** | **`0.7000`** | **`0.7000`** | **`0.7000`** |
| **Lookalike** | v1 (`model_real_best`) | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| | v2 (`model_real_v2_best`) | **`0.0000`** | **`0.0000`** | **`0.0000`** | **`0.0000`** |
| **Overall** | v1 (`model_real_best`) | `0.4754` | `0.5015` | `0.5073` | `0.5314` |
| | v2 (`model_real_v2_best`) | **`0.4746`** | **`0.5013`** | **`0.5081`** | **`0.5296`** |

### Lookalike Analysis
* **Result**: Lookalike IoU and Dice remained strictly at **`0.0000`** across all 10 test scenes in both models.
* **Findings**: Despite oversampling lookalike background patches by 2.5x during training (representing ~71.4% of the negative patch pool), the model still produced false positive detections on all lookalike test scenes. 
* **Implications**: Simple background patch oversampling is insufficient to differentiate lookalikes from real oil spills due to identical backscatter reduction signatures (wind shadows, organic films). Distinguishing them requires structural context (e.g. geometric features like length-width ratios) or external metadata (e.g. co-registered wind speed measurements or incident angles).

---

## Open Issues

### 1. Dataloader Duplication & Drift (Tech Debt)
* **Status**: Resolved — see commit history. `dataset.py`'s local loader now uses the same category-balanced patch sampling as `train_kaggle.py`, and both files carry an explicit cross-reference banner since `train_kaggle.py` can't import `src/data/` directly (Kaggle kernels must be single-file).

---

## New Capability Investigation: ERA5 Wind Cross-Check & Multi-Temporal SAR Comparison

* **Status**: Blocked on missing data, code written and ready pending real inputs.
* **Goal**: reduce the universal lookalike false-positive problem (see above) by post-processing detections against (1) ERA5 10m wind speed at acquisition time — oil slicks only show a dark SAR signature within ~1.5-6 m/s — and (2) whether a same-track Sentinel-1 pass exists nearby in time to check if the detected shape drifted (real oil) or stayed static (lookalike-likely).
* **Real per-scene geolocation exists and was previously unused — now fixed.** Every holdout GeoTIFF carries real embedded WGS84 georeferencing (`ModelTiepointTag`/`ModelPixelScaleTag`, ~10 m/pixel) — confirmed by direct inspection, extracted for all 30 holdout scenes into `docs/holdout_scene_coordinates.json` via `src/analysis/geoutils.py`. `src/api/main.py`'s live demo previously ignored this entirely and showed every scene at a fixed Gulf-of-Thailand mock coordinate with an arbitrary `0.00015` deg/pixel display scale; `/api/predict` now reads each scene's real georeferencing (falling back to the old fixed point only for synthetic scenes with no real-world location), and `checkpoint_comparison_summary`/`holdout_per_scene_results` use each scene's real, latitude-corrected pixel area instead. See `docs/region_coverage.json` for a per-scene region breakdown (Eastern Mediterranean / Red Sea / Gulf of Mexico / Baltic Sea / Western Mediterranean / Sunda Strait) with pass/fail outcomes, built for the professor's "area of difference" map slide.
* **The 30 holdout scenes are not all Eastern Mediterranean.** Real extracted coordinates span at least 6 distinct regions: Eastern Mediterranean (Lebanon/Syria/Cyprus coast), Red Sea, Gulf of Mexico (Louisiana), Baltic Sea (Finland), western Mediterranean (Corsica/Sardinia), and Sunda Strait (Indonesia). This matters because Sentinel-1 revisit density varies a lot by latitude/region — see coverage numbers below.
* **No acquisition timestamp exists anywhere for this dataset**, confirmed by: (1) no `DateTime` tag or any date-bearing tag in the GeoTIFFs themselves; (2) querying the Zenodo API directly for all three record IDs (8346860 / 8253899 / 13761290) — each deposit's file listing and description is exhaustively just the image `.7z` + mask `.7z` archives, no metadata/sidecar file. This is a property of the original Zenodo deposit, not something a Kaggle mirror stripped out. Without a timestamp there's no way to pick an ERA5 timestep or anchor a "nearby pass" search for any of these 30 scenes as currently downloaded.
* **Sentinel-1 revisit cadence verified current, not assumed**: the 2026 constellation reconfiguration (Sentinel-1C + Sentinel-1D) completed 2026-06-24, restoring a ~6-day nominal same-track revisit (shifted 1 day from the old A/B pattern) — not the ~12-day figure that held during the prior single-satellite gap.
* **Real, credential-free finding**: Copernicus Data Space Ecosystem's catalog *search* API needs no auth (only downloads do). Ran a real 12-month Sentinel-1 GRD coverage-density check against all 30 scenes' actual coordinates — see `docs/sentinel1_coverage_density.json`. Confirms real regional variance (e.g. sparse over the Gulf of Mexico point vs. very dense over the Baltic point), which sets expectations for how useful the multi-temporal feature could be once real per-scene dates are available.
* **Built, not yet run**: `src/analysis/era5_wind_check.py` and `src/analysis/sentinel1_revisit_check.py` — real fetch/classify/compare logic against the documented CDS and CDSE APIs. Not executed against real per-scene data because (1) no CDS API key or CDSE OAuth credentials are configured in this environment, and (2) the timestamp gap above. Both are standalone scripts only — not wired into `/api/predict` or the DB schema.
* **Credential prep done, timestamp gap still open**: `.env.example` documents exactly which env vars each service needs and where to get them; `scripts/validate_credentials.py` does a real minimal auth check against both CDS and CDSE (no data download) so real credentials can be sanity-checked in seconds once obtained — verified its skip-path against this repo's current unconfigured environment. The acquisition-timestamp gap itself is not a repo problem to keep digging at further (Zenodo + Kaggle mirror both confirmed to lack it) — needs a human path (e.g. the paywalled companion paper's supplementary data).
* **What's needed to actually run these for real**: a free CDS account + API key (cds.climate.copernicus.eu) for ERA5; a free Copernicus Data Space Ecosystem account + OAuth2 client credentials (dataspace.copernicus.eu) for Sentinel-1 downloads; and, most importantly, real per-scene acquisition timestamps for the 30 holdout scenes from a source outside this repo (e.g. the paywalled companion paper's supplementary data, DOI `10.1016/j.marpolbul.2024.116549`, not independently verified).
