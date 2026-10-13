# Model Creation, Evaluation, and Improvement Guide

## 1. Project purpose

This project estimates heart rate from a face recorded by a webcam. It uses
remote photoplethysmography (rPPG): small changes in facial skin color caused
by blood circulation are extracted from video and converted into a heart-rate
estimate in beats per minute (BPM).

The main runtime pipeline is:

```text
Webcam video
    -> Haar-cascade face detection
    -> upper-face region of interest (ROI)
    -> average RGB values over time
    -> POS rPPG signal
    -> band-pass filtering and signal-quality analysis
    -> time-domain and frequency-domain inputs
    -> trained CNN
    -> heart rate in BPM
```

The main implementation is in [`src/main.py`](./src/main.py).

## 2. How the current model is made

The repository contains the trained model file:

```text
models/HR_CNN_Time_Frequency_v3.keras
```

The repository does **not** currently contain the original training script,
training dataset loader, training history, loss function, optimizer settings,
or train/validation/test split. Therefore, the exact training procedure
cannot be reconstructed from this project alone. The following details are
verified from the saved Keras model and the inference code.

### Model inputs

For every 10-second window, [`create_cnn_inputs()`](./src/main.py) creates two
normalized inputs:

| Input | Shape | Meaning |
|---|---:|---|
| Time input | `(300, 1)` | Filtered POS pulse signal resampled to 300 points |
| Frequency input | `(128, 1)` | Welch power spectrum between 0.7 and 3.0 Hz, log-scaled and normalized |

The input batch dimension is added at runtime, so the model receives:

```text
(batch, 300, 1)
(batch, 128, 1)
```

The frequency range corresponds approximately to 42-180 BPM:

```text
0.7 Hz * 60 = 42 BPM
3.0 Hz * 60 = 180 BPM
```

### Verified architecture

The saved model is named `HR_CNN_Time_Frequency_v3` and contains:

1. A time branch with 1D convolution layers, batch normalization,
   activations, max pooling, and global average pooling.
2. A frequency branch with 1D convolution layers, batch normalization,
   activations, max pooling, and global average pooling.
3. Concatenation of the time and frequency features.
4. Dense layers with dropout.
5. A single dense output named `hr_output`.

The saved model has:

| Property | Value |
|---|---:|
| Outputs | One scalar heart-rate value |
| Total parameters | 77,953 |
| Trainable parameters | 77,313 |
| Non-trainable parameters | 640 |
| Model size shown by Keras | Approximately 304.5 KB |

The model performs regression, not classification. Its output should be
compared with a reference heart rate in BPM.

## 3. How to run the model

From PowerShell:

```powershell
cd "C:\Users\dhruv\Downloads\QG_Project"
.\venv\Scripts\python.exe .\src\main.py
```

The program captures webcam data, processes model-compatible 10-second windows,
and prints the predicted BPM and signal-quality status. The webcam should be
available as camera index `0`.

The live experiment now captures 60 seconds and processes six non-overlapping
10-second windows. The original model still receives the same two inputs for
each window; the model file and its input shapes are unchanged.

The live run writes a per-window report to:

```text
outputs/live_60s_results.json
```

The report contains window timestamps, CNN/POS/CHROM/GREEN estimates when
available, ROI method, valid sample count, estimated FPS, quality indicators,
processing latency, acceptance status, and rejection reasons. Missing
measurements are recorded as `null`, not as invented values.

By default, windows with a quality score below 35 are rejected. At least two
accepted windows are required. The final CNN estimate is the median of accepted
window predictions rather than the arithmetic mean. These settings are
configurable at the top of [`src/main.py`](./src/main.py):

```python
CAPTURE_DURATION = 60
WINDOW_SECONDS = 10
WINDOW_STRIDE_SECONDS = 10
MIN_ACCEPTED_WINDOWS = 2
MIN_ACCEPTABLE_QUALITY_SCORE = 35.0
```

This aggregation improves resistance to outlier windows but does not prove
that a 60-second recording is more accurate. Accuracy still requires
synchronized reference measurements.

## Stage E face detection and ROI handling

The live pipeline uses the OpenCV Haar Cascade detector and preserves it as the
only detector dependency currently installed in the project environment.
MediaPipe is not required or installed.

Stage E adds:

- Face-box clamping against the current frame boundaries.
- Selection of the face nearest the previous valid face location when multiple
  faces are detected; the largest face is selected when no previous track
  exists.
- Optional face-box smoothing controlled by
  `FACE_BOX_SMOOTHING_ALPHA`.
- Validation of ROI dimensions, bounds, emptiness, and finite pixel values.
- A configurable `FACE_LOSS_GRACE_SECONDS` display grace period.
- No RGB sample or timestamp is created during face loss or invalid ROI
  frames. A reused face box is not treated as proof that a face is currently
  valid.

The existing upper-face ROI geometry and CNN input preprocessing remain
unchanged. The JSON report includes additive capture metadata and detector/ROI
fields while retaining the existing window fields. These changes improve
input validation and tracking stability, but they do not establish improved
heart-rate accuracy without synchronized reference measurements.

## 4. Current evaluation code

The evaluation helper is [`src/evaluate_methods.py`](./src/evaluate_methods.py).
It accepts synchronized, per-window prediction rows and calculates:

- **MAE (Mean Absolute Error):** average absolute difference in BPM.
- **RMSE (Root Mean Squared Error):** penalizes larger errors more strongly.
- **Signed bias:** mean prediction minus reference BPM.
- **Correlation:** reported only with at least three matched windows and
  non-constant prediction and reference values.
- **Within ±5 BPM:** percentage of matched windows whose absolute error is at
  most 5 BPM.
- **Valid coverage:** matched windows divided by windows with a valid reference.

The formulas are:

```text
MAE  = mean(abs(predicted_BPM - reference_BPM))
RMSE = sqrt(mean((predicted_BPM - reference_BPM)^2))
```

### Stage G1 CSV format

The evaluator uses one wide row per uniquely identified window. These columns
are required:

```text
session_id,subject_id,window_number,start_seconds,end_seconds,
reference_hr_bpm,cnn_hr,pos_hr,chrom_hr,green_hr,accepted
```

`session_id` and `window_number` identify a window and must be unique together.
`start_seconds` and `end_seconds` must be finite with end greater than start.
`reference_hr_bpm` is the synchronized reference measurement. Prediction cells
may be blank when a method did not produce a value. `accepted` must be a
boolean (`true`/`false`, `1`/`0`, or `yes`/`no`); rejected windows are excluded
from metric calculations. BPM values outside 20–240 BPM, missing references,
and missing predictions are excluded and counted in the result. No value is
imputed.

Example row:

```text
s01,p01,1,0,10,72.0,73.1,71.8,72.4,70.9,true
```

Run the evaluator with a genuine synchronized file:

```powershell
cd "C:\Users\dhruv\Downloads\QG_Project"
.\venv\Scripts\python.exe .\src\evaluate_methods.py .\data\stage_g1_predictions.csv
```

An optional JSON output can be written with:

```powershell
.\venv\Scripts\python.exe .\src\evaluate_methods.py `
    .\data\stage_g1_predictions.csv `
    --json .\outputs\stage_g1_evaluation.json
```

CNN, POS, CHROM, and GREEN are evaluated on the same rows and window keys.
The output reports matched-window counts, coverage, rejection counts, and
unavailable reasons. The existing
[`validation/real_world_validation.csv`](./validation/real_world_validation.csv)
does not follow this format and is intentionally reported as unavailable; its
single unverified value is not used as a label.

The synthetic fixtures in `tests/test_evaluate_methods.py` verify evaluator
calculations and validation behavior only. They are not model-performance
results.

## 5. Recommended real performance evaluation

### Step 1: Collect a labelled dataset

For each video sample, record:

```text
video_id
subject_id
video_path
reference_heart_rate_bpm
lighting condition
motion condition
camera/device
```

The reference heart rate should come from a reliable synchronized device such
as a pulse oximeter, ECG, or validated chest strap.

The reference and video timestamps must correspond to the same time window.
Do not use a reference value measured at a different time.

### Step 2: Use subject-independent splits

Split by person, not by randomly splitting windows from the same person:

```text
Training subjects:   70%
Validation subjects: 15%
Test subjects:       15%
```

All windows from one subject must stay in only one split. Otherwise, the
evaluation can be overly optimistic because the model may see very similar
signals from the same person in both training and test data.

### Step 3: Evaluate every test window

For every test video:

1. Detect the face and extract the same ROI used during inference.
2. Generate the POS pulse signal.
3. Generate the 300-point time input and 128-point frequency input.
4. Run the saved CNN model.
5. Store the prediction and the synchronized reference BPM.
6. Record quality score, lighting, motion, subject, and device metadata.

### Step 4: Report more than one metric

At minimum, report:

| Metric | Why it matters |
|---|---|
| MAE in BPM | Easy to interpret average error |
| RMSE in BPM | Shows sensitivity to large failures |
| Mean percentage error | Normalizes error by heart rate |
| Median absolute error | Less affected by outliers |
| Standard deviation of error | Shows consistency |
| Pearson/Spearman correlation | Measures agreement in trends |
| Bland-Altman bias and limits of agreement | Measures systematic clinical bias |
| Valid prediction rate | Measures how often the system produces a usable result |

Report each metric overall and separately for different conditions:
stationary versus moving subjects, good versus poor lighting, and each subject
or camera.

## 6. How to interpret results

There is no universal pass/fail threshold for this project until the intended
use case is defined. As an engineering starting point:

- Lower MAE and RMSE are better.
- Mean error close to zero indicates low systematic bias.
- High valid prediction rate indicates reliable operation.
- A large gap between training/validation and test performance suggests
  overfitting or data leakage.
- A good average score with poor results for one subject or lighting condition
  indicates weak generalization.

Do not describe the system as medically accurate or clinically validated based
only on the current example evaluator.

## 7. Current limitations and required improvements

### Highest priority

1. **Add the original training pipeline or recreate a documented one.**
   Include dataset preparation, target labels, normalization, split strategy,
   loss, optimizer, learning rate, batch size, epochs, and random seeds.
2. **Replace demonstration evaluation data with real labelled predictions.**
   The current evaluator does not measure CNN performance.
3. **Add CNN evaluation to `evaluate_methods.py`.**
   Compare CNN, POS, and CHROM predictions against the same reference labels.
4. **Prevent subject leakage.**
   Use subject-independent train, validation, and test sets.
5. **Save reproducible experiment outputs.**
   Store predictions, references, metadata, model version, and evaluation
   metrics in CSV or JSON format.

### Signal and data improvements

- Use a more stable ROI, such as skin regions on the cheeks and forehead,
  instead of relying only on a fixed upper-face crop.
- Track the face across frames so the ROI does not jump when detection changes.
- Reject frames with weak face detection, excessive movement, blur, or
  changing exposure.
- Calibrate or normalize for camera white balance and illumination changes.
- Collect more subjects, skin tones, lighting conditions, cameras, and motion
  levels.
- Balance the dataset across heart-rate ranges so normal resting examples do
  not dominate the model.
- Validate the actual frame rate; the signal-processing code estimates FPS from
  timestamps, and unstable frame timing can affect the spectrum.

### Model and training improvements

- Train with MAE or Huber loss in addition to MSE so extreme noisy labels have
  less influence.
- Use early stopping and save the best validation checkpoint.
- Tune convolution sizes, dropout, learning rate, and input window length
  using only the validation subjects.
- Add data augmentation for realistic noise, illumination changes, motion, and
  small timing distortions.
- Compare the CNN with a simple spectral baseline. A more complex model should
  outperform the baseline on unseen subjects, not only on training data.
- Consider predicting uncertainty or a confidence interval and reject
  low-confidence windows.

## 8. Recommended next experiment

The most useful next step is to create a real CSV with one row per test
window:

```text
subject_id,video_id,window_start_s,reference_bpm,cnn_bpm,pos_bpm,chrom_bpm,quality,lighting,motion
```

Then calculate metrics separately for CNN, POS, and CHROM using exactly the
same rows. This will show whether the CNN improves on the existing signal
processing methods and which conditions need more data or better preprocessing.
