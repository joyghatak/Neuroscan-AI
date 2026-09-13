# NeuroScan AI v3.0

**Explainable Brain MRI Tumour Analysis — Research & Demo Application**

> ⚠️ **Research & Demonstration Use Only.** This is an experimental AI research system.
> Its outputs are NOT a medical diagnosis and must not be used as a substitute for
> evaluation by a qualified radiologist or clinician.

---

## Overview

NeuroScan AI v3.0 is a Streamlit web application that performs:

1. **Brain extraction** — Otsu thresholding + largest external contour crop
2. **Attention U-Net segmentation** — tumour probability map and ROI extraction
3. **EfficientNet-B0 classification** — deployed on the full-image domain
4. **MC Dropout uncertainty estimation** — 30 stochastic passes, BatchNorm frozen
5. **Deterministic TTA** — 10 geometric transforms
6. **Fitted ensemble** — 0.75 × MC Dropout + 0.25 × TTA
7. **Grad-CAM explainability** — last backbone Conv2d
8. **Research PDF report** — ReportLab, with provenance and disclaimers

### Performance (BRISC2025 Test Set)

| Configuration | Accuracy | Macro F1 | Cohen κ |
|---|---|---|---|
| Full Image (single pass) | 98.59% | 98.62% | 0.9808 |
| ROI-Guided (single pass) | 97.49% | 97.56% | 0.9657 |
| **Ensemble (deployed)** | **98.79%** | **98.84%** | **0.9835** |

Segmentation: Test Dice = 0.8400 | Test IoU = 0.7756

---

## Project Structure

```
project/
│
├── app.py                          ← Main Streamlit application
├── requirements.txt
├── README.md
├── run_demo.bat                    ← Windows demo launcher
├── run_demo.ps1                    ← PowerShell demo launcher
│
├── neuroscan/                      ← Inference package
│   ├── __init__.py
│   ├── models.py                   ← AttentionUNet, NeuroScanClassifier
│   ├── preprocessing.py            ← brain_crop, to_tensor_imagenet
│   ├── inference.py                ← NeuroScanInference, MC Dropout, TTA, ensemble
│   ├── explainability.py           ← GradCAM, overlay_heatmap, panel builders
│   ├── reporting.py                ← ReportLab PDF generation
│   └── utils.py                    ← Artifact discovery, checkpoint loading
│
└── neuroscan_v3_artifacts/         ← Place extracted artifacts here
    ├── checkpoints/
    │   ├── segmenter.pt            ← REQUIRED
    │   ├── classifier_full_image.pt ← REQUIRED
    │   └── ...
    ├── metrics/
    │   ├── run_summary.json
    │   ├── comparison.json
    │   └── segmentation.json
    ├── figures/
    ├── xai/
    └── reports/
```

---

## Installation

### 1. Create a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install PyTorch

If you don't already have PyTorch installed, visit https://pytorch.org/get-started/locally/
and install the appropriate version for your CUDA version.

Example (CUDA 12.1):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Example (CPU only):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install remaining dependencies

```bash
pip install -r requirements.txt
```

---

## Artifact Setup

Copy the extracted `neuroscan_v3_artifacts` directory next to `app.py`:

```
project/
├── app.py
├── requirements.txt
└── neuroscan_v3_artifacts/
    ├── checkpoints/
    │   ├── segmenter.pt
    │   ├── classifier_full_image.pt
    │   └── ...
    ├── metrics/
    ├── figures/
    ├── xai/
    └── reports/
```

Alternatively, set the environment variable:

```bash
# Windows (PowerShell)
$env:NEUROSCAN_ARTIFACT_DIR = "C:\path\to\neuroscan_v3_artifacts"

# Linux / macOS
export NEUROSCAN_ARTIFACT_DIR=/path/to/neuroscan_v3_artifacts
```

---

## Running the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

---

## Demo on Mobile via ngrok

ngrok creates a temporary public HTTPS tunnel to your local Streamlit server,
allowing you to open the app on any phone or tablet without needing the same Wi-Fi.

### Step 1 — Start the Streamlit app

Open **Terminal 1**:
```bash
streamlit run app.py
```

### Step 2 — Install and authenticate ngrok

Download ngrok from https://ngrok.com/download and add it to PATH.

If required, add your authtoken (free account):
```bash
ngrok config add-authtoken YOUR_TOKEN_HERE
```

### Step 3 — Create the public tunnel

Open **Terminal 2**:
```bash
ngrok http 8501
```

You will see output like:
```
Forwarding   https://xxxx-xxxx.ngrok-free.app -> http://localhost:8501
```

Open that HTTPS URL on your phone browser.

### Security Notes

- The tunnel is **temporary** — it stops when you close the ngrok terminal.
- Do **not** expose real patient data through a public ngrok tunnel.
- Use synthetic or de-identified MRI images for demonstrations.
- The app does **not** contain any ngrok credentials.

---

## Fast Demo Mode

Enable **Fast Demo Mode** in the sidebar to reduce MC Dropout to 10 passes
and TTA to 5 transforms. The result will be faster but uncertainty estimates
will be less stable. The UI will clearly indicate when fast mode is active.

Default (Full Analysis Mode):
- MC Dropout: **30 passes**
- TTA: **10 transforms**
- Grad-CAM: **enabled**
- Segmentation: **enabled**

---

## Environment Variables

| Variable | Description |
|---|---|
| `NEUROSCAN_ARTIFACT_DIR` | Override artifact directory path |

---

## Citation & Dataset

This project uses the **BRISC2025** dataset:
- Kaggle: `briscdataset/brisc2025`

---

## Disclaimer

NeuroScan AI is an **experimental research system**. It:

- Has not been reviewed or validated as a medical device
- Analyses only a single 2D MRI slice
- May produce incorrect or uncertain predictions
- Must not be used for clinical decision-making
- Is intended for research, education, and demonstration purposes only

The authors accept no liability for decisions based on outputs of this system.
