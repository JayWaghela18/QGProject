# QG Project Documentation

## 1. What this project does

This project is a camera-based remote photoplethysmography (rPPG) system. It estimates a person's heart rate without using a physical sensor. The webcam records small changes in facial skin color caused by blood circulation. Those changes are converted into a pulse signal and then into an estimated heart rate in beats per minute (BPM).

The project contains:

- Webcam and face-detection utilities
- Facial region-of-interest (ROI) selection
- RGB signal capture from the ROI
- Two classical rPPG algorithms: POS and CHROM
- Band-pass filtering and frequency analysis
- Heart-rate and signal-quality estimation
- A TensorFlow/Keras CNN model path
- An evaluation framework for comparing methods

This is an educational/research project. Its heart-rate result should not be treated as a medical diagnosis.

## 2. How the project works

The main processing flow is:

```text
Webcam
  |
  v
Video frames
  |
  v
Haar-cascade face detection
  |
  v
Upper-face rPPG ROI
  |
  v
Average red, green, and blue values for every frame
  |
  v
RGB time-series signal
  |
  +--> POS algorithm
  |
  +--> CHROM algorithm
  |
  +--> CNN time/frequency model
          |
          v
Band-pass filtering and spectral analysis
          |
          v
Estimated heart rate and signal-quality score
```

### Step 1: Capture video

OpenCV opens the default camera using `cv2.VideoCapture(0)`. Frames are read continuously until the capture duration is reached or the user presses **Q**.

### Step 2: Detect the face

OpenCV's built-in Haar cascade file, `haarcascade_frontalface_default.xml`, detects frontal faces. The largest detected face is normally selected when the RGB signal is being captured.

### Step 3: Select the ROI

The project uses an upper-face region as the rPPG ROI. In the ROI script, the region is approximately:

- 25% to 75% of the face width
- 15% to 40% of the face height

The mean BGR value of the pixels inside this region is calculated for each frame and stored as RGB data in the order `R, G, B`.

### Step 4: Generate the pulse signal

The RGB values contain lighting changes, movement noise, and the small color variations related to blood flow. POS and CHROM use chrominance projections to reduce common lighting effects and isolate a pulse-like signal.

### Step 5: Estimate heart rate

The pulse signal is filtered between approximately 0.7 Hz and 4.0 Hz, which corresponds to about 42-240 BPM. Welch's power spectral density method finds the strongest frequency in this physiological range:

```text
heart rate (BPM) = peak frequency (Hz) * 60
```

The project also calculates a signal-quality score based on how dominant the detected frequency is.

## 3. Project files

### `src/roi.py`

Displays the detected face and the upper-face rPPG ROI in real time. It is useful for checking whether the camera and ROI placement work correctly. It does not save a signal or estimate heart rate.

### `src/face_detection.py`

Runs basic real-time face detection and draws a rectangle around each detected face. It is a simpler test than `roi.py`.

### `src/read_video.py`

Records approximately 30 seconds of webcam video and saves it as:

```text
camera_test.avi
```

The recording can be stopped early with **Q**.

### `src/rgb_signal.py`

Captures approximately 30 seconds of facial ROI data. It saves:

```text
rgb_signal.npy
timestamps.npy
```

`rgb_signal.npy` contains rows of `[red, green, blue]` values. `timestamps.npy` contains the capture time for each row.

### `src/heart_rate.py`

Loads `rgb_signal.npy` and `timestamps.npy`, then:

1. Estimates the actual frames per second from the timestamps.
2. Runs the POS algorithm.
3. Filters the resulting pulse signal.
4. Estimates heart rate with Welch frequency analysis.
5. Calculates signal quality.
6. Displays the filtered waveform and frequency spectrum.

### `src/chrom.py`

Provides the CHROM rPPG algorithm and a processing flow similar to `heart_rate.py`. It also includes filtering, heart-rate estimation, signal-quality calculation, and plots.

### `src/main.py`

This is the most complete end-to-end program. It includes:

- Webcam capture
- Face and ROI processing
- POS pulse extraction
- Signal-quality checks
- CNN input creation
- Loading `models/HR_CNN_Time_Frequency_v3.keras`
- CNN heart-rate prediction
- Comparison of CNN and POS results

The script uses a 32-second capture, a 10-second processing window, and both time-domain and frequency-domain CNN inputs.

### `src/evaluate_methods.py`

Contains metrics for comparing predicted heart rates with reference values:

- MAE: mean absolute error
- RMSE: root mean square error
- Mean error percentage

The current `main()` function uses example values for POS and CHROM. Those example values should be replaced with real predictions and ground-truth measurements for a real evaluation.

## 4. Technologies and libraries used

### Python

The project is written in Python.

### OpenCV (`cv2`)

Used for:

- Webcam access
- Reading video frames
- Converting images to grayscale
- Haar-cascade face detection
- Drawing face and ROI rectangles
- Displaying camera windows
- Saving video

### NumPy

Used for:

- Storing RGB signals and timestamps
- Calculating average channel values
- Array and numerical operations
- Saving and loading `.npy` files

### SciPy

Used for:

- Butterworth band-pass filtering through `scipy.signal.butter` and `filtfilt`
- Welch power spectral density through `scipy.signal.welch`
- Signal and frequency analysis

### Matplotlib

Used to plot:

- The filtered rPPG waveform
- The frequency spectrum

### TensorFlow/Keras

Used by `src/main.py` to load and run the trained CNN model:

```text
models/HR_CNN_Time_Frequency_v3.keras
```

### Python virtual environment

The project includes a `venv` directory containing the Python environment and installed packages. It is preferable to run the project using the Python executable inside this environment.

## 5. Important project assets

| Asset | Purpose |
|---|---|
| `models/HR_CNN_Time_Frequency_v3.keras` | Trained CNN model used by `src/main.py` |
| `camera_test.avi` | Example or previously recorded camera video |
| `validation.zip` | Validation data/archive for project experiments |
| `rgb_signal.npy` | Saved RGB signal generated by `src/rgb_signal.py` |
| `timestamps.npy` | Time values corresponding to the saved RGB signal |
| `venv/` | Local Python virtual environment |

The `.npy` files may be generated after running `rgb_signal.py`; they are input files for `heart_rate.py` and `chrom.py`.

## 6. How to run the project

Open PowerShell and run:

```powershell
cd "C:\Users\dhruv\Downloads\QG_Project"
.\venv\Scripts\Activate.ps1
```

To check the ROI and face detection:

```powershell
python .\src\roi.py
```

To capture RGB data:

```powershell
python .\src\rgb_signal.py
```

After the capture finishes, estimate heart rate with POS:

```powershell
python .\src\heart_rate.py
```

To run the CHROM pipeline:

```powershell
python .\src\chrom.py
```

To run the complete CNN/POS program:

```powershell
python .\src\main.py
```

To run the example evaluation:

```powershell
python .\src\evaluate_methods.py
```

Press **Q** in an OpenCV camera window to stop a live capture early.

## 7. Requirements for a useful result

- A working webcam must be connected and available as camera index `0`.
- The face should be visible and reasonably well lit.
- The user should keep their face still during capture.
- The ROI should remain on exposed skin rather than hair, eyes, or background.
- At least enough samples for the selected processing window must be available.
- The required model file must exist when running `src/main.py`.

If the camera cannot be opened, check Windows camera permissions and close other applications that may already be using the webcam.

## 8. Limitations

- Haar-cascade detection can fail with large head rotations, poor lighting, or occlusion.
- Motion and changing illumination can significantly affect rPPG quality.
- The default camera index is `0`; another camera may require changing `CAMERA_INDEX`.
- The quality score is a signal heuristic, not a clinical confidence measurement.
- The CNN model can only be expected to work correctly with inputs similar to the data on which it was trained.
- The evaluation script currently contains demonstration data rather than a complete automated benchmark.
