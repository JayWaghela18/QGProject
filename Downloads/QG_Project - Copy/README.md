# QG Project — Camera-Based rPPG Heart Rate Estimator

> A real-time, contactless heart rate estimation system using remote photoplethysmography (rPPG) with classical algorithms (POS, CHROM) and a trained CNN model — powered by OpenCV, MediaPipe, and TensorFlow.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Assets](#models--assets)
- [Testing](#testing)
- [Limitations](#limitations)
- [License](#license)

---

## Overview

This project estimates a person's **heart rate (BPM)** in real time using only a standard webcam — no physical sensor required. It works by detecting subtle color changes in facial skin caused by blood circulation (rPPG), then applying signal processing and deep learning to extract a reliable pulse estimate.

**Key capabilities:**
- Real-time face detection via MediaPipe Face Landmarker (primary) and Haar Cascade (fallback)
- Forehead-region ROI extraction for optimal rPPG signal quality
- Two classical rPPG algorithms: **POS** and **CHROM**
- A trained **CNN (Time-Frequency)** model for heart-rate prediction
- Signal quality scoring and method comparison (MAE, RMSE)

> ⚠️ This is an **educational/research project**. Results must not be used for medical diagnosis.

---

## How It Works

```
Webcam → Video Frames → Face Detection (MediaPipe / Haar Cascade)
  → Forehead ROI → Per-frame RGB averages → RGB time-series signal
    → POS algorithm ──┐
    → CHROM algorithm─┤→ Band-pass filter → Welch PSD → Estimated BPM
    → CNN model ──────┘
```

1. **Capture** — OpenCV reads frames from the webcam continuously
2. **Detect** — MediaPipe Face Landmarker locates the face; Haar Cascade is the fallback
3. **ROI** — An upper-forehead region (~25–75% width, 15–40% height of the face) is selected
4. **Signal** — Mean RGB values of the ROI are collected per frame
5. **Process** — POS/CHROM chrominance projections isolate the blood-flow pulse signal
6. **Filter** — A Butterworth band-pass filter (0.7–4.0 Hz → 42–240 BPM) removes noise
7. **Estimate** — Welch's PSD finds the dominant frequency → `BPM = peak_freq × 60`
8. **CNN** — Time-domain and frequency-domain inputs are fed to the trained Keras model

---

## Project Structure

```
QG_Project/
├── src/
│   ├── main.py               # Full end-to-end pipeline (CNN + POS)
│   ├── face_detection.py     # Basic real-time face detection test
│   ├── roi.py                # Displays face + ROI regions live
│   ├── rgb_signal.py         # Captures & saves RGB signal to .npy files
│   ├── heart_rate.py         # Heart rate estimation using POS algorithm
│   ├── chrom.py              # Heart rate estimation using CHROM algorithm
│   └── evaluate_methods.py   # MAE/RMSE metric comparison of methods
├── tests/
│   ├── test_classical_rppg.py
│   ├── test_evaluate_methods.py
│   ├── test_face_tracking.py
│   ├── test_mediapipe_webcam.py
│   ├── test_multi_roi_rgb.py
│   ├── test_multi_roi_webcam.py
│   ├── webcam_hr_demo.py
│   ├── compare_classical_methods.py
│   ├── estimate_hr_from_rgb.py
│   └── plot_multi_roi_rgb.py
├── models/
│   ├── HR_CNN_Time_Frequency_v3.keras   # Trained CNN model
│   └── face_landmarker.task             # MediaPipe Face Landmarker asset
├── experiments/
│   └── hr_cnn_v1/                       # Experimental model runs
├── data/                                # Raw data (empty by default)
├── outputs/                             # Generated outputs
├── validation/                          # Validation datasets/results
├── requirements.txt                     # Core dependencies
├── requirements-mediapipe.txt           # MediaPipe-specific dependencies
├── MODEL_EVALUATION.md                  # CNN architecture & evaluation notes
├── PROJECT_DOCUMENTATION.md            # Detailed project documentation
├── rgb_signal.npy                       # Saved RGB signal (generated at runtime)
├── timestamps.npy                       # Timestamps for RGB signal
├── camera_test.avi                      # Sample/recorded webcam video
└── README.md
```

---

## Requirements

- Python **3.9 – 3.11** (TensorFlow 2.15 compatibility)
- A working **webcam** connected as camera index `0`
- Windows (tested), Linux/macOS may require minor path adjustments

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd QG_Project
```

### 2. Create and activate the virtual environment

```powershell
# Create venv
python -m venv .venv_mediapipe

# Activate on Windows
.\.venv_mediapipe\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

> For the isolated MediaPipe environment (recommended for `main.py`):
> ```powershell
> pip install -r requirements-mediapipe.txt
> ```

### 4. Download the MediaPipe Face Landmarker model

Place the model at `models/face_landmarker.task`. The official model can be downloaded from:

```
https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**SHA-256 checksum:**
```
64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff
```

> If this file is unavailable, the system automatically falls back to Haar Cascade detection.

---

## Usage

All commands should be run from the project root with the virtual environment active.

### Run the full CNN + POS pipeline (recommended)

```powershell
.\.venv_mediapipe\Scripts\python.exe .\src\main.py
```

### Check face detection and ROI placement

```powershell
python .\src\roi.py
```

### Capture RGB signal to file

```powershell
python .\src\rgb_signal.py
```

### Estimate heart rate with POS

```powershell
python .\src\heart_rate.py
```

### Estimate heart rate with CHROM

```powershell
python .\src\chrom.py
```

### Run method evaluation / comparison

```powershell
python .\src\evaluate_methods.py
```

> Press **Q** in any OpenCV camera window to stop live capture early.

---

## Models & Assets

| Asset | Description |
|---|---|
| `models/HR_CNN_Time_Frequency_v3.keras` | Trained CNN model using time + frequency domain inputs |
| `models/face_landmarker.task` | MediaPipe Face Landmarker (Apache License 2.0, by Google) |
| `rgb_signal.npy` | Saved per-frame RGB values (generated by `rgb_signal.py`) |
| `timestamps.npy` | Capture timestamps corresponding to `rgb_signal.npy` |
| `camera_test.avi` | Recorded webcam video for offline testing |

---

## Testing

```powershell
# Run all unit tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_evaluate_methods.py -v
```

---

## Limitations

- Haar Cascade can fail with large head rotations, poor lighting, or occlusion
- Motion artifacts and changing illumination degrade rPPG signal quality significantly
- The CNN model is only reliable for inputs similar to its training distribution
- The signal quality score is a heuristic, not a clinical-grade confidence measure
- Default camera index is `0`; change `CAMERA_INDEX` in the scripts for other cameras
- The evaluation script uses demo values by default — replace with real measurements for benchmarking

---

## License

This project is intended for **educational and experimental use only**.  
The MediaPipe Face Landmarker model is distributed under the **Apache License 2.0** by Google.
