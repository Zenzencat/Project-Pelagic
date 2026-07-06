# Project Pelagic: SAR-based Maritime Oil Slick Detection System
**SWU Prasarnmit — AI Engineering Track (Final Project)**

Project Pelagic is a C-band Synthetic Aperture Radar (SAR) maritime oil slick detection system proof-of-concept (PoC). It processes Sentinel-1 radar imagery (VV/VH polarizations), executes pixel-level semantic segmentation using a trained U-Net CNN, and overlays the tracked oil slick coordinates and adjacent AIS vessels onto an interactive map dashboard.

---

## 📂 Project Structure

```
Project-Pelagic/
├── checkpoints/          # PyTorch model weights (best_model.pth) [Tracked]
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
├── src/                  # Core Python modules
│   ├── api/              # FastAPI endpoints & SQLite database logic
│   ├── data/             # Ingestion pipelines & dataset loaders
│   ├── models/           # U-Net, loss functions, & validation metrics
│   ├── train.py          # Main training orchestrator loop
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

## 🧪 Model Training & Verification

To verify that the U-Net model trains successfully on your local machine:
```bash
# Run a quick 2-epoch CPU test verify script
python src/verify_training.py
```
This saves weight checkpoints in `checkpoints/` and logs history in `logs/training_history.csv`.
