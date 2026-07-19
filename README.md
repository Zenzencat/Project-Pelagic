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
├── src/                  # Core Python modules
│   ├── api/              # FastAPI endpoints & SQLite database logic
│   ├── data/             # Ingestion pipelines & dataset loaders
│   ├── models/           # U-Net architecture definition
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
* **Part II** (685 clean sea and lookalike scenes): Record ID `8253899`
* **Part III** (450 held-out test scenes): Record ID `13761290`

To download files programmatically to `data/raw/`, run the downloader CLI:
```bash
# Download only the annotation masks for Part I (saves 40GB disk space)
python src/data/download_sample.py --part 1

# Download Part I including the massive dual-channel images
python src/data/download_sample.py --part 1 --include-images
```

---

## 🧪 Model Training, Verification & Kaggle Execution

### 1. Local CPU Verification Check
To verify that the U-Net data ingestion and model architecture run successfully on your machine:
```bash
# Run a quick 2-epoch CPU mock verification run
python src/verify_training.py
```

### 2. Real Dataset Training (Kaggle T4 GPU)
For full-scale training on the real Sentinel-1 dataset (Parts I & II, 1,885 scenes), we use Kaggle's T4 GPU accelerator to handle the heavy computations.
* The script is located in `kaggle_kernel/train_kaggle.py`.
* To push the training job to Kaggle via their CLI:
  ```bash
  kaggle kernels push -p kaggle_kernel
  ```
The trained best checkpoint is saved locally as **`checkpoints/model_real_best.pt`** (Val Dice: **0.8071**).

### 3. Held-Out Part III Test Set Evaluation
We evaluated the best checkpoint on the completely unseen held-out Part III test set (450 scenes). The evaluation script runs batch GPU inference:
* The script is located in `kaggle_eval_kernel/eval_kaggle.py`.
* Push the evaluation job:
  ```bash
  kaggle kernels push -p kaggle_eval_kernel
  ```

#### Evaluation Metrics Results:
* **Oil Scenes (Positive Class, Pixel-Level Segmentation)**:
  * **Intersection over Union (IoU)**: `0.7246`
  * **Dice Coefficient (F1-score)**: `0.8172`
  * **Precision**: `0.8717`
  * **Recall**: `0.8399`
* **No Oil background scenes perfect suppression rate**: `61.33%`
* **Lookalike feature scenes perfect suppression rate**: `2.67%`
