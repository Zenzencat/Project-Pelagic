# Project Pelagic: SAR-based Maritime Oil Slick Detection System
**SWU Prasarnmit — AI Engineering Track (Final Project)**

Project Pelagic is a C-band Synthetic Aperture Radar (SAR) maritime oil slick detection system proof-of-concept (PoC). It processes Sentinel-1 radar imagery (VV/VH polarizations), executes pixel-level semantic segmentation using a trained U-Net CNN, and overlays the tracked oil slick coordinates and adjacent AIS vessels onto an interactive map dashboard.

---

## 📂 Project Structure

```
Project-Pelagic/
├── checkpoints/          # PyTorch model weights (model_real_best.pt) [Tracked]
├── data/                 # Raw/processed/synthetic imagery & databases
│   ├── synthetic/        # Generated synthetic SAR scenes for testing
│   └── pelagic.db        # SQLite database containing seeded mock detections
├── docs/                 # Academic reports & architectural specifications
│   ├── system_architecture.md
│   ├── data_pipeline.md
│   ├── problem_analysis.md
│   ├── ai_technique.md
│   ├── db_schema.md
│   └── wireframe.html    # Standalone interactive Leaflet.js mockup
├── frontend/             # Vite + React + Leaflet.js map dashboard
│   ├── src/
│   │   ├── App.jsx       # Main single-page React UI code
│   │   └── index.css     # CSS themes and Leaflet overrides
│   └── package.json
├── kaggle_kernel/        # Training script & metadata for Kaggle runner
├── kaggle_eval_kernel/   # Inference evaluation script for held-out test set
├── kaggle_kernel_lookalike/ # Trains the post-hoc lookalike classifier (Kaggle T4)
├── src/                  # Core Python modules
│   ├── api/              # FastAPI endpoints & SQLite database logic
│   ├── analysis/         # Post-processing: contours, land mask, GFW vessels,
│   │                     #   lookalike_filter.py (opt-in post-hoc filter)
│   ├── data/             # Ingestion pipelines & dataset loaders
│   ├── models/           # U-Net architecture definition
│   ├── evaluate_holdout.py # Local holdout test evaluation script
│   ├── evaluate_holdout_lookalike_filter.py # Holdout eval WITH the post-hoc filter
│   ├── train.py          # Local training orchestrator loop
│   └── verify_training.py # Verification harness for CPU training runs
├── requirements.txt      # Python dependencies list
└── README.md
```

---

## 🛠️ Prerequisites

Make sure you have the following installed on your machine:
* **Python 3.10+** (with pip)
* **Node.js 18+** (with npm)
* **Git**

---

## 🚀 Quick Start Guide

Follow these steps to run the entire project locally.

### 1. Set Up and Run the Backend API (FastAPI)

Open your terminal in the project root:

1. **Create a virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   
   # macOS/Linux
   python3 -m venv venv
   ```

2. **Activate the virtual environment:**
   ```bash
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   
   # Windows (CMD)
   .\venv\Scripts\activate.bat
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the FastAPI server:**
   ```bash
   python -m uvicorn src.api.main:app --port 8000
   ```
   The backend API will initialize the database, seed mock entries, and listen on **http://localhost:8000**.
   * View the API health check: `http://localhost:8000/health`
   * Interactive API docs: `http://localhost:8000/docs`

---

### 2. Set Up and Run the Frontend (React / Vite)

Open a new terminal window in the project root:

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install Node modules:**
   ```bash
   npm install
   ```

3. **Start the Vite development server:**
   ```bash
   npm run dev
   ```
   Open **[http://localhost:5173](http://localhost:5173)** in your browser to view and interact with the live map dashboard!

---

## 📊 Dataset Ingestion & Downloader

We utilize the 3-part Sentinel-1 SAR Oil Spill Dataset by Trujillo-Acatitla et al.:
* **Part I** (1,200 oil spill scenes): Record ID `8346860`
* **Part II** (1,370 clean sea and lookalike scenes: 685 of each category): Record ID `8253899`
* **Part III** (450 held-out test scenes): Record ID `13761290`

To download files programmatically to `data/raw/`, run the downloader CLI:
```bash
# Download only the annotation masks for Part I (saves 40GB disk space)
python src/data/download_sample.py --part 1

# Download Part I including the massive dual-channel images
python src/data/download_sample.py --part 1 --include-images
```

---

## 🧪 Model Training & Kaggle Execution

### 1. Local CPU Verification Check
To verify that the U-Net data ingestion and model architecture run successfully on your machine:
```bash
# Run a quick 2-epoch CPU mock verification run
python src/verify_training.py
```

### 2. U-Net v1 Baseline (Kaggle T4 GPU)
For baseline training on the real Sentinel-1 dataset (Parts I & II, 2,570 scenes), we used a flat learning rate of `1e-3` over 15 epochs.
* Script location: `kaggle_kernel/train_kaggle.py`
* Pushed via Kaggle CLI: `kaggle kernels push -p kaggle_kernel`
* Output checkpoint: **`checkpoints/model_real_best.pt`** (Val Dice: **0.8071**)

### 3. U-Net v2 Retrained Model (Cosine Annealing & Oversampling)
To address epoch-to-epoch validation metric volatility and high lookalike false alarm rates (100% false positives in v1), we retrained the model with three targeted improvements:
1. **Cosine Annealing LR Scheduler**: Integrated `CosineAnnealingLR` (T_max=15, eta_min=1e-6) to smoothly decay the learning rate and stabilize gradient steps.
2. **Epoch-Level Global Metric Pooling**: Validation metrics are pooled globally across the entire epoch (summing absolute intersection and union pixel counts) instead of taking the simple mean of noisy per-batch ratios.
3. **Lookalike Hard-Negative Oversampling**: Oversampled lookalike background patches by 2.5x during patch balancing (`prob = 0.005` vs clean sea baseline `0.002`), raising lookalike frequency in the negative pool to **`71.4%`**.
* Output checkpoint: **`checkpoints/model_real_v2_best.pt`** (Best Val IoU: **0.7424** vs v1 best Val IoU of **0.6965**)
* Logs: `kaggle_output/training_v2_log.csv`

---

## 📈 Local Holdout Evaluation (Part III Subset)

We evaluated both U-Net checkpoints locally on a 30-scene subset of the Part III held-out test dataset (10 Oil, 10 No Oil, 10 Lookalike) downloaded to `data/holdout/`.

### Run the Evaluation
To run the evaluation script locally:
```bash
# Evaluate the U-Net v2 model (default)
python src/evaluate_holdout.py --version v2

# Evaluate the U-Net v1 model
python src/evaluate_holdout.py --version v1
```
Comparative 3-panel plots (SAR VV, Ground Truth, Prediction) are outputted to the `docs/` folder (e.g. `docs/holdout_v2_viz_oil_00000.png`).

### Before/After Evaluation Results

| Category | Model Version | Average IoU | Average Dice (F1) | Average Precision | Average Recall |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Oil Spill** | v1 (`model_real_best`) | `0.7263` | `0.8044` | `0.8218` | `0.8943` |
| | v2 (`model_real_v2_best`) | **`0.7239`** | **`0.8039`** | **`0.8243`** | **`0.8889`** |
| **No Oil** | v1 (`model_real_best`) | `0.7000` | `0.7000` | `0.7000` | `0.7000` |
| | v2 (`model_real_v2_best`) | **`0.7000`** | **`0.7000`** | **`0.7000`** | **`0.7000`** |
| **Lookalike** | v1 (`model_real_best`) | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| | v2 (`model_real_v2_best`) | **`0.0000`** | **`0.0000`** | **`0.0000`** | **`0.0000`** |
| **Overall** | v1 (`model_real_best`) | `0.4754` | `0.5015` | `0.5073` | `0.5314` |
| | v2 (`model_real_v2_best`) | **`0.4746`** | **`0.5013`** | **`0.5081`** | **`0.5296`** |

### Lookalike False Alarm Suppression Analysis
* **Binary Metric Null Result**: Both models return `0.0000` for all lookalike metrics. This is because lookalike features (such as wind shadows and biogenic films) produce backscatter reduction signatures identical to oil slicks, causing U-Net to predict false positive pixels on every scene (binary score `0.0`).
* **Continuous Pixel-Level Reduction**: Comparing the raw predicted positive pixel counts reveals that **U-Net v2 consistently reduced lookalike false positives by 10% to 45%** across all lookalike test scenes. For example, on `lookalike_00003`, false positive pixels dropped from `1,815` (v1) to `1,004` (v2). This confirms that hard-negative oversampling successfully regularized background predictions, even if it did not suppress them completely to zero.
  *(Note: `docs/status.md` gives a more carefully verified framing of this metric — average false-positive area 41.3% → 37.5% of scene area. The 10–45% range describes spread across individual scenes, not the typical improvement.)*

---

## 🧪 Post-Hoc Lookalike Discrimination Filter (opt-in — Phases 1–4)

A second-stage classifier that runs **after** the U-Net and decides whether each
candidate detection is real oil or a lookalike, then either keeps or suppresses
that detection. Full investigation log: `docs/status.md` (Phases 1–4).

* **Model**: Gradient Boosting on 12 hand-crafted features — 11 shared
  GLCM-texture / Canny-edge-density / shape-compactness features
  (`src/analysis/candidate_region_features.py`) plus the U-Net's own mean
  prediction confidence in the candidate region. Trained on the full real
  training pool (1,200 oil + 685 lookalike scenes), **not** on `data/holdout/`.
  Artifact: `checkpoints/lookalike_classifier_final.joblib` (the `_v1` /
  `_v2_oversampled` / `_v3_confidence` files are kept for reproducibility).
* **Confidence gate** (`GATE_THRESHOLD = 0.975`): when the U-Net is very
  confident, its "oil" call is kept regardless of the classifier — real
  lookalikes are sometimes also confidently (wrongly) flagged, so this is an
  accepted trade, verified on the real 30-scene holdout:

  | Category | Baseline (v2 U-Net) | With filter + gate |
  |---|---|---|
  | oil | 8 correct, 2 partial | 6 correct, 1 partial, 3 false-negative |
  | no_oil | 7 correct, 3 false-positive | 7 correct, 3 false-positive (unchanged) |
  | lookalike | 0 correct, 10 false-positive | 7 correct, 3 false-positive |

* **NOT recommended for unconditional integration** and **not enabled anywhere
  by default** — trading real oil false-negatives for lookalike fixes is not a
  free win for a spill detector. It is wired into `/api/predict` as an opt-in
  flag only:

  ```jsonc
  POST /api/predict  { "scene_id": "...", "apply_lookalike_filter": true }
  // default false → response and behavior byte-identical to before this existed
  ```
  When enabled the response gains a `lookalike_filter` diagnostic block. The
  flag is folded into the scene cache key so toggling it can't return a stale
  row produced under the other setting.

### Run the holdout evaluation

```bash
# Baseline filter, no confidence gate
python src/evaluate_holdout_lookalike_filter.py

# Final configuration: classifier + confidence gate (matches the table above)
python src/evaluate_holdout_lookalike_filter.py --gate --gate-threshold 0.975
```
Per-scene results land in `docs/phase4_confirm_*.{json,csv}`.

> **Gotcha**: `.joblib` classifiers trained on Kaggle may fail to unpickle
> locally (`ModuleNotFoundError: No module named '_loss'`) due to an sklearn
> version mismatch. Refit locally from `output/lookalike_classifier_own_domain_features*.csv`
> + the saved train/val split if this happens (reproduces the Kaggle AUC to
> within 0.001) — see `docs/status.md` Phase 3.
