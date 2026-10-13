import cv2
import json
import numpy as np
from pathlib import Path
import time
import tensorflow as tf
from scipy.signal import butter, filtfilt, welch, find_peaks
from chrom import chrom_algorithm, estimate_heart_rate

try:
    import mediapipe as mp
except ImportError:
    mp = None


# ============================================================
# SETTINGS
# ============================================================

CAPTURE_DURATION = 60

CAMERA_INDEX = 0

MODEL_PATH = "models/HR_CNN_Time_Frequency_v3.keras"
REPORT_PATH = "outputs/live_60s_results.json"
FACE_LANDMARKER_MODEL_PATH = "models/face_landmarker.task"

TIME_LENGTH = 300
FREQ_LENGTH = 128

WINDOW_SECONDS = 10
WINDOW_STRIDE_SECONDS = 10
MIN_ACCEPTED_WINDOWS = 2
MIN_ACCEPTABLE_QUALITY_SCORE = 35.0

POS_WINDOW_SECONDS = 1.6

LOW_HZ = 0.7
HIGH_HZ = 3.0

# Physiological range used only for quality assessment
MIN_HR = 40
MAX_HR = 180

FACE_LOSS_GRACE_SECONDS = 1.0
FACE_BOX_SMOOTHING_ALPHA = 0.35
MIN_ROI_WIDTH = 20
MIN_ROI_HEIGHT = 10
CAPTURE_METADATA = {}
MIN_WINDOW_COVERAGE_RATIO = 0.80
# Gaps above this limit are never interpolated.
MAX_TIMESTAMP_GAP_SECONDS = 0.25
MIN_SIGNAL_SECONDS = 1.6
FILTER_PAD_MULTIPLIER = 3


def estimate_green_hr(rgb, fps):

    rgb = validate_rgb_signal(rgb)
    validate_fps(
        fps,
        LOW_HZ,
        HIGH_HZ
    )

    green_signal = rgb[:, 1]

    filtered = bandpass_filter(
        green_signal,
        fps,
        LOW_HZ,
        HIGH_HZ
    )

    frequencies, power = welch(
        filtered,
        fs=fps,
        nperseg=min(
            len(filtered),
            max(8, int(8 * fps))
        ),
        nfft=4096
    )

    valid = (
        (frequencies >= LOW_HZ)
        &
        (frequencies <= HIGH_HZ)
    )

    if not np.any(valid):
        raise ValueError(
            "No valid GREEN heart-rate frequency found."
        )

    valid_frequencies = frequencies[valid]
    valid_power = power[valid]

    return float(
        valid_frequencies[np.argmax(valid_power)]
        * 60.0
    )


def validate_rgb_signal(rgb):

    rgb = np.asarray(
        rgb,
        dtype=np.float64
    )

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(
            "RGB must have shape (samples, 3)."
        )

    if len(rgb) == 0:
        raise ValueError("RGB signal is empty.")

    if not np.all(np.isfinite(rgb)):
        raise ValueError(
            "RGB signal contains non-finite values."
        )

    return rgb


def validate_fps(fps, low=LOW_HZ, high=HIGH_HZ):

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            f"FPS must be positive and finite: {fps}"
        )

    nyquist = fps / 2.0

    if low <= 0 or high <= low or nyquist <= high:
        raise ValueError(
            f"Invalid frequency bounds or FPS: "
            f"fps={fps:.3f}, low={low:.3f}, high={high:.3f}"
        )


def validate_timestamps(rgb, timestamps):

    rgb = validate_rgb_signal(rgb)
    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )

    if timestamps.ndim != 1:
        raise ValueError(
            "Timestamps must be one-dimensional."
        )

    if len(rgb) != len(timestamps):
        raise ValueError(
            "RGB and timestamp lengths do not match."
        )

    if len(timestamps) < 2:
        raise ValueError(
            "At least two timestamps are required."
        )

    if not np.all(np.isfinite(timestamps)):
        raise ValueError(
            "Timestamps contain non-finite values."
        )

    timestamp_diffs = np.diff(timestamps)

    if not np.all(timestamp_diffs > 0):
        raise ValueError(
            "Timestamps must be strictly increasing."
        )

    return rgb, timestamps


def prepare_uniform_window(rgb, timestamps, window_seconds):

    rgb, timestamps = validate_timestamps(
        rgb,
        timestamps
    )

    timestamp_diffs = np.diff(timestamps)
    effective_duration = (
        timestamps[-1] - timestamps[0]
    )
    median_interval = float(
        np.median(timestamp_diffs)
    )
    max_gap = float(
        np.max(timestamp_diffs)
    )
    coverage_ratio = min(
        effective_duration / window_seconds,
        1.0
    )
    effective_fps = 1.0 / median_interval

    diagnostics = {
        "timestamp_gap_count": int(
            np.sum(
                timestamp_diffs
                > MAX_TIMESTAMP_GAP_SECONDS
            )
        ),
        "max_gap_seconds": max_gap,
        "coverage_ratio": float(coverage_ratio),
        "effective_sampling_rate": float(effective_fps),
        "resampled": False,
        "timing_status": "valid"
    }

    if coverage_ratio < MIN_WINDOW_COVERAGE_RATIO:
        diagnostics["timing_status"] = "insufficient_coverage"
        raise ValueError(
            "insufficient timestamp coverage "
            f"({coverage_ratio:.2%})"
        )

    if max_gap > MAX_TIMESTAMP_GAP_SECONDS:
        diagnostics["timing_status"] = "excessive_timestamp_gap"
        raise ValueError(
            "timestamp gap exceeds "
            f"{MAX_TIMESTAMP_GAP_SECONDS:.2f}s"
        )

    if not np.isfinite(effective_fps) or effective_fps <= 0:
        diagnostics["timing_status"] = "invalid_sampling_rate"
        raise ValueError("invalid effective sampling rate")

    target_count = max(
        int(round(effective_duration * effective_fps)) + 1,
        2
    )
    target_timestamps = np.linspace(
        timestamps[0],
        timestamps[-1],
        target_count
    )
    resampled_rgb = np.column_stack([
        np.interp(
            target_timestamps,
            timestamps,
            rgb[:, channel]
        )
        for channel in range(3)
    ])

    if not np.all(np.isfinite(resampled_rgb)):
        diagnostics["timing_status"] = "invalid_resampled_signal"
        raise ValueError(
            "resampled RGB signal contains non-finite values"
        )

    diagnostics["resampled"] = True
    return resampled_rgb, target_timestamps, diagnostics


# ============================================================
# FACE DETECTOR
# ============================================================

FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades
    + "haarcascade_frontalface_default.xml"
)


def create_mediapipe_detector():
    if mp is None:
        print("MediaPipe unavailable; using Haar Cascade fallback.")
        return None

    model_path = Path(FACE_LANDMARKER_MODEL_PATH)
    if not model_path.is_file():
        print(
            f"MediaPipe model asset unavailable at {model_path}; "
            "using Haar Cascade fallback."
        )
        return None

    try:
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path)
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return mp.tasks.vision.FaceLandmarker.create_from_options(options)
    except (OSError, RuntimeError, ValueError) as error:
        print(
            f"MediaPipe initialization failed ({error}); "
            "using Haar Cascade fallback."
        )
        return None


def _landmark_coordinates(landmark):
    try:
        x = float(landmark.x)
        y = float(landmark.y)
    except (AttributeError, TypeError, ValueError):
        return None

    if not np.isfinite(x) or not np.isfinite(y):
        return None
    return x, y


def landmarks_to_face_box(landmarks, frame_shape):
    if landmarks is None:
        return None

    coordinates = [
        coordinate
        for coordinate in (
            _landmark_coordinates(landmark)
            for landmark in landmarks
        )
        if coordinate is not None
    ]
    if len(coordinates) < 3:
        return None

    frame_height, frame_width = frame_shape[:2]
    x_values = [coordinate[0] * frame_width for coordinate in coordinates]
    y_values = [coordinate[1] * frame_height for coordinate in coordinates]
    return clamp_face_box(
        (
            min(x_values),
            min(y_values),
            max(x_values) - min(x_values),
            max(y_values) - min(y_values),
        ),
        frame_shape,
    )


def get_landmark_forehead_roi(frame, landmarks):
    face_box = landmarks_to_face_box(landmarks, frame.shape)
    if face_box is None:
        return None, None, "invalid landmarks"

    x, y, width, height = face_box
    frame_height, frame_width = frame.shape[:2]
    roi_x1 = max(0, min(frame_width, int(round(x + 0.20 * width))))
    roi_y1 = max(0, min(frame_height, int(round(y + 0.08 * height))))
    roi_x2 = max(0, min(frame_width, int(round(x + 0.80 * width))))
    roi_y2 = max(0, min(frame_height, int(round(y + 0.28 * height))))

    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1
    if roi_width < MIN_ROI_WIDTH:
        return None, None, "ROI is too narrow"
    if roi_height < MIN_ROI_HEIGHT:
        return None, None, "ROI is too short"

    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    if roi.size == 0:
        return None, None, "ROI is empty"
    if not np.all(np.isfinite(roi)):
        return None, None, "ROI contains non-finite pixels"

    return (
        roi,
        (roi_x1, roi_y1, roi_width, roi_height),
        None,
    )


def extract_mean_rgb(roi):
    if roi is None or roi.ndim != 3 or roi.shape[2] != 3 or roi.size == 0:
        return None
    mean_bgr = np.mean(roi, axis=(0, 1))
    if not np.all(np.isfinite(mean_bgr)):
        return None
    return mean_bgr[::-1]


def append_valid_sample(rgb_values, timestamps, mean_rgb, timestamp):
    if mean_rgb is None:
        return False

    sample = np.asarray(mean_rgb, dtype=np.float64)
    timestamp = float(timestamp)
    if (
        sample.shape != (3,)
        or not np.all(np.isfinite(sample))
        or not np.isfinite(timestamp)
    ):
        return False

    rgb_values.append(sample)
    timestamps.append(timestamp)
    return True


def capture_window_starts():
    return np.arange(
        0,
        CAPTURE_DURATION - WINDOW_SECONDS + 1,
        WINDOW_STRIDE_SECONDS
    )


def clamp_face_box(face_box, frame_shape):

    if face_box is None or len(face_box) != 4:
        return None

    frame_height, frame_width = frame_shape[:2]

    try:
        x, y, width, height = (
            float(value)
            for value in face_box
        )
    except (TypeError, ValueError):
        return None

    values = np.asarray(
        [x, y, width, height],
        dtype=np.float64
    )

    if not np.all(np.isfinite(values)) or width <= 0 or height <= 0:
        return None

    x1 = max(0, min(frame_width, int(round(x))))
    y1 = max(0, min(frame_height, int(round(y))))
    x2 = max(0, min(frame_width, int(round(x + width))))
    y2 = max(0, min(frame_height, int(round(y + height))))

    if x2 <= x1 or y2 <= y1:
        return None

    return (
        x1,
        y1,
        x2 - x1,
        y2 - y1
    )


def face_box_center(face_box):

    x, y, width, height = face_box

    return (
        x + width / 2.0,
        y + height / 2.0
    )


def select_face_box(faces, previous_face_box, frame_shape):

    valid_faces = []

    for face in faces:
        clamped = clamp_face_box(
            face,
            frame_shape
        )

        if clamped is not None:
            valid_faces.append(clamped)

    if len(valid_faces) == 0:
        return None

    if previous_face_box is None:
        return max(
            valid_faces,
            key=lambda box: box[2] * box[3]
        )

    previous_center = face_box_center(
        previous_face_box
    )

    return min(
        valid_faces,
        key=lambda box: (
            (face_box_center(box)[0] - previous_center[0]) ** 2
            + (face_box_center(box)[1] - previous_center[1]) ** 2
        )
    )


def smooth_face_box(
    previous_face_box,
    current_face_box,
    alpha=FACE_BOX_SMOOTHING_ALPHA
):

    if current_face_box is None:
        return None

    if previous_face_box is None:
        return current_face_box

    alpha = float(
        np.clip(alpha, 0.0, 1.0)
    )

    smoothed = (
        (1.0 - alpha) * np.asarray(
            previous_face_box,
            dtype=np.float64
        )
        + alpha * np.asarray(
            current_face_box,
            dtype=np.float64
        )
    )

    return tuple(
        int(round(value))
        for value in smoothed
    )


def get_upper_face_roi(frame, face_box):

    if frame is None or frame.ndim < 2:
        return None, None, "invalid frame"

    clamped_face = clamp_face_box(
        face_box,
        frame.shape
    )

    if clamped_face is None:
        return None, None, "invalid face box"

    x, y, width, height = clamped_face

    roi_x1 = int(round(x + 0.25 * width))
    roi_y1 = int(round(y + 0.15 * height))
    roi_x2 = int(round(roi_x1 + 0.50 * width))
    roi_y2 = int(round(roi_y1 + 0.25 * height))

    frame_height, frame_width = frame.shape[:2]

    roi_x1 = max(0, min(frame_width, roi_x1))
    roi_y1 = max(0, min(frame_height, roi_y1))
    roi_x2 = max(0, min(frame_width, roi_x2))
    roi_y2 = max(0, min(frame_height, roi_y2))

    roi_width = roi_x2 - roi_x1
    roi_height = roi_y2 - roi_y1

    if roi_width < MIN_ROI_WIDTH:
        return None, None, "ROI is too narrow"

    if roi_height < MIN_ROI_HEIGHT:
        return None, None, "ROI is too short"

    roi = frame[
        roi_y1:roi_y2,
        roi_x1:roi_x2
    ]

    if roi.size == 0:
        return None, None, "ROI is empty"

    if not np.all(np.isfinite(roi)):
        return None, None, "ROI contains non-finite pixels"

    return (
        roi,
        (roi_x1, roi_y1, roi_width, roi_height),
        None
    )


def extract_roi_rgb(frame, face_box):

    roi, roi_box, rejection_reason = get_upper_face_roi(
        frame,
        face_box
    )

    if rejection_reason is not None:
        return None, roi_box, rejection_reason

    mean_bgr = np.mean(
        roi,
        axis=(0, 1)
    )

    if mean_bgr.shape != (3,) or not np.all(
        np.isfinite(mean_bgr)
    ):
        return None, roi_box, "RGB mean is invalid"

    return extract_mean_rgb(roi), roi_box, None


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(
    signal,
    fps,
    low=LOW_HZ,
    high=HIGH_HZ,
    order=3
):

    signal = np.asarray(
        signal,
        dtype=np.float64
    )

    if signal.ndim != 1 or len(signal) == 0:
        raise ValueError(
            "Signal must be a non-empty one-dimensional array."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "Signal contains non-finite values."
        )

    validate_fps(
        fps,
        low,
        high
    )

    nyquist = fps / 2.0
    low_norm = low / nyquist
    high_norm = high / nyquist

    b, a = butter(
        order,
        [low_norm, high_norm],
        btype="band"
    )

    pad_length = FILTER_PAD_MULTIPLIER * max(
        len(a),
        len(b)
    )

    if len(signal) <= pad_length:
        raise ValueError(
            f"Signal is too short for filter padding: "
            f"{len(signal)} samples, need more than {pad_length}."
        )

    return filtfilt(
        b,
        a,
        signal
    )


# ============================================================
# POS rPPG
# ============================================================

def pos_algorithm(
    rgb,
    fps,
    window_seconds=POS_WINDOW_SECONDS
):

    rgb = validate_rgb_signal(rgb)

    validate_fps(
        fps,
        LOW_HZ,
        HIGH_HZ
    )

    window = int(
        window_seconds * fps
    )

    if window < 2 or len(rgb) < window:
        raise ValueError(
            f"Not enough samples for POS. "
            f"Samples={len(rgb)}, required={window}"
        )

    rgb_normalized = rgb / (
        np.mean(rgb, axis=0) + 1e-8
    )

    pulse = np.zeros(
        len(rgb)
    )

    for start in range(
        0,
        len(rgb) - window + 1
    ):

        end = start + window

        C = rgb_normalized[
            start:end
        ].T

        S1 = C[0] - C[1]

        S2 = (
            C[0]
            + C[1]
            - 2 * C[2]
        )

        alpha = (
            np.std(S1)
            / (np.std(S2) + 1e-8)
        )

        h = S1 + alpha * S2

        h = h - np.mean(h)

        pulse[start:end] += h

    return pulse


# ============================================================
# IMPROVED SIGNAL QUALITY
# ============================================================

def calculate_signal_quality(
    pulse,
    fps
):

    # --------------------------------------------------------
    # Filter POS signal
    # --------------------------------------------------------

    filtered = bandpass_filter(
        pulse,
        fps,
        LOW_HZ,
        HIGH_HZ
    )

    # --------------------------------------------------------
    # Welch spectrum
    # --------------------------------------------------------

    nperseg = min(
        len(filtered),
        max(8, int(8 * fps))
    )

    frequencies, power = welch(
        filtered,
        fs=fps,
        nperseg=nperseg,
        nfft=4096
    )

    valid = (
        (frequencies >= LOW_HZ)
        &
        (frequencies <= HIGH_HZ)
    )

    frequencies = frequencies[valid]
    power = power[valid]

    if len(power) < 5:

        return {
            "score": 0.0,
            "quality": "Poor",
            "spectral_hr": None,
            "peak_prominence": 0.0,
            "band_concentration": 0.0,
            "filtered": filtered
        }

    # --------------------------------------------------------
    # Dominant spectral peak
    # --------------------------------------------------------

    peak_index = np.argmax(power)

    peak_frequency = frequencies[
        peak_index
    ]

    spectral_hr = (
        peak_frequency * 60.0
    )

    # --------------------------------------------------------
    # Physiological range score
    # --------------------------------------------------------

    if MIN_HR <= spectral_hr <= MAX_HR:

        range_score = 1.0

    else:

        range_score = 0.0

    # --------------------------------------------------------
    # Peak prominence
    #
    # Compare the largest peak against the median
    # spectral power instead of total power.
    # --------------------------------------------------------

    median_power = np.median(
        power
    )

    if median_power <= 0:

        peak_prominence = 0.0

    else:

        peak_prominence = (
            power[peak_index]
            / (median_power + 1e-12)
        )

    # Normalize prominence roughly to 0-1
    prominence_score = np.clip(
        (peak_prominence - 1.0) / 5.0,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Spectral concentration
    #
    # Measure power around dominant frequency.
    # --------------------------------------------------------

    frequency_resolution = (
        frequencies[1]
        - frequencies[0]
    )

    bandwidth = max(
        0.15,
        3 * frequency_resolution
    )

    local_band = (
        np.abs(
            frequencies
            - peak_frequency
        )
        <= bandwidth
    )

    local_power = np.sum(
        power[local_band]
    )

    total_power = np.sum(
        power
    )

    if total_power <= 0:

        concentration = 0.0

    else:

        concentration = (
            local_power
            / total_power
        )

    concentration_score = np.clip(
        concentration / 0.40,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Composite quality score
    #
    # Range: approximately 0-100
    # --------------------------------------------------------

    score = (
        0.30 * range_score
        + 0.40 * prominence_score
        + 0.30 * concentration_score
    ) * 100.0

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if score >= 60:

        quality = "Good"

    elif score >= 35:

        quality = "Moderate"

    else:

        quality = "Poor"

    return {
        "score": float(score),
        "quality": quality,
        "spectral_hr": float(spectral_hr),
        "peak_prominence": float(
            peak_prominence
        ),
        "band_concentration": float(
            concentration
        ),
        "filtered": filtered
    }


# ============================================================
# CREATE CNN INPUTS
# ============================================================

def create_cnn_inputs(
    pulse,
    fps
):

    if len(pulse) < 2:
        raise ValueError(
            "Pulse signal is too short for CNN preprocessing."
        )

    if not np.all(np.isfinite(pulse)):
        raise ValueError(
            "Pulse signal contains non-finite values."
        )

    validate_fps(
        fps,
        LOW_HZ,
        HIGH_HZ
    )

    filtered = bandpass_filter(
        pulse,
        fps,
        LOW_HZ,
        HIGH_HZ
    )

    # --------------------------------------------------------
    # TIME DOMAIN
    # --------------------------------------------------------

    original_x = np.linspace(
        0,
        1,
        len(filtered)
    )

    target_x = np.linspace(
        0,
        1,
        TIME_LENGTH
    )

    time_signal = np.interp(
        target_x,
        original_x,
        filtered
    )

    time_signal = (
        time_signal
        - np.mean(time_signal)
    ) / (
        np.std(time_signal)
        + 1e-8
    )

    time_input = time_signal.reshape(
        1,
        TIME_LENGTH,
        1
    ).astype(np.float32)

    # --------------------------------------------------------
    # FREQUENCY DOMAIN
    # --------------------------------------------------------

    nperseg = min(
        len(filtered),
        max(8, int(8 * fps))
    )

    frequencies, power = welch(
        filtered,
        fs=fps,
        nperseg=nperseg,
        nfft=4096
    )

    valid = (
        (frequencies >= LOW_HZ)
        &
        (frequencies <= HIGH_HZ)
    )

    valid_frequencies = frequencies[
        valid
    ]

    valid_power = power[
        valid
    ]

    if len(valid_frequencies) < 2:

        raise ValueError(
            "Not enough frequency information."
        )

    target_frequencies = np.linspace(
        LOW_HZ,
        HIGH_HZ,
        FREQ_LENGTH
    )

    frequency_signal = np.interp(
        target_frequencies,
        valid_frequencies,
        valid_power
    )

    frequency_signal = np.log1p(
        frequency_signal
    )

    frequency_signal = (
        frequency_signal
        - np.mean(frequency_signal)
    ) / (
        np.std(frequency_signal)
        + 1e-8
    )

    freq_input = frequency_signal.reshape(
        1,
        FREQ_LENGTH,
        1
    ).astype(np.float32)

    return (
        time_input,
        freq_input
    )


# ============================================================
# CAMERA CAPTURE
# ============================================================

def capture_rgb():

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open camera."
        )

    rgb_values = []
    timestamps = []
    previous_face_box = None
    last_detected_time = None
    face_loss_frames = 0
    invalid_roi_frames = 0
    mediapipe_detector = create_mediapipe_detector()
    detector_samples = {
        "mediapipe": 0,
        "haar_cascade": 0,
    }

    start_time = time.monotonic()

    print()
    print("======================================")
    print("          CAMERA rPPG CAPTURE")
    print("======================================")

    print(
        f"Recording for "
        f"{CAPTURE_DURATION} seconds..."
    )

    print()
    print("Please:")
    print("- Keep your face inside the camera")
    print("- Stay relatively still")
    print("- Use stable lighting")
    print()

    while True:

        ret, frame = cap.read()

        if not ret:
            continue

        current_time = time.monotonic()

        elapsed = (
            current_time
            - start_time
        )

        if elapsed >= CAPTURE_DURATION:
            break

        selected_face_box = None
        roi = None
        roi_box = None
        detector_label = None

        if mediapipe_detector is not None:
            try:
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                result = mediapipe_detector.detect_for_video(
                    image,
                    int(round(elapsed * 1000)),
                )
                if result.face_landmarks:
                    roi, roi_box, roi_reason = get_landmark_forehead_roi(
                        frame,
                        result.face_landmarks[0],
                    )
                    if roi_reason is None:
                        selected_face_box = landmarks_to_face_box(
                            result.face_landmarks[0],
                            frame.shape,
                        )
                        detector_label = "mediapipe"
            except (RuntimeError, ValueError, TypeError) as error:
                print(f"MediaPipe frame detection failed: {error}")

        if selected_face_box is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80)
            )
            selected_face_box = select_face_box(
                faces,
                previous_face_box,
                frame.shape
            )
            if selected_face_box is not None:
                roi, roi_box, roi_reason = get_upper_face_roi(
                    frame,
                    selected_face_box,
                )
                detector_label = "haar_cascade"

        if selected_face_box is None:
            face_loss_frames += 1

            if (
                last_detected_time is not None
                and elapsed - last_detected_time
                <= FACE_LOSS_GRACE_SECONDS
            ):
                face_status = "Face temporarily lost"
            else:
                face_status = "Face not detected"
        else:
            previous_face_box = smooth_face_box(
                previous_face_box,
                selected_face_box
            )
            last_detected_time = elapsed
            detector_samples[detector_label] += 1
            face_status = f"Face detected ({detector_label})"

            if roi_reason is not None:
                invalid_roi_frames += 1
                face_status = f"ROI rejected: {roi_reason}"
            else:
                mean_rgb = extract_mean_rgb(roi)
                if mean_rgb is None:
                    invalid_roi_frames += 1
                    face_status = "ROI rejected: RGB mean is invalid"
                else:
                    if append_valid_sample(
                        rgb_values,
                        timestamps,
                        mean_rgb,
                        elapsed,
                    ):
                        x, y, w, h = previous_face_box

                        cv2.rectangle(
                            frame,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            2
                        )

                        roi_x, roi_y, roi_w, roi_h = roi_box
                        cv2.rectangle(
                            frame,
                            (roi_x, roi_y),
                            (roi_x + roi_w, roi_y + roi_h),
                            (255, 0, 0),
                            2
                        )

        remaining = max(
            0,
            CAPTURE_DURATION - elapsed
        )

        cv2.putText(
            frame,
            f"Recording: {remaining:.1f}s",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Valid samples: {len(rgb_values)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            face_status,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        cv2.imshow(
            "Camera rPPG",
            frame
        )

        if (
            cv2.waitKey(1) & 0xFF
            == ord("q")
        ):

            print(
                "Capture stopped by user."
            )

            break

    cap.release()

    cv2.destroyAllWindows()
    if mediapipe_detector is not None:
        mediapipe_detector.close()

    rgb_values = np.asarray(
        rgb_values,
        dtype=np.float64
    )

    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )

    if len(rgb_values) < 100:

        raise RuntimeError(
            "Too few valid face samples captured."
        )

    print()
    print("======================================")
    print("          CAPTURE COMPLETE")
    print("======================================")

    print(
        f"RGB samples : "
        f"{len(rgb_values)}"
    )
    print(
        f"Face-loss frames: "
        f"{face_loss_frames}"
    )
    print(
        f"Invalid ROI frames: "
        f"{invalid_roi_frames}"
    )
    print(
        f"Detector samples: "
        f"{detector_samples}"
    )

    CAPTURE_METADATA.clear()
    CAPTURE_METADATA.update({
        "detector": (
            "mediapipe_preferred"
            if mediapipe_detector is not None
            else "haar_cascade"
        ),
        "roi_method": (
            "mediapipe_forehead_with_haar_fallback"
            if mediapipe_detector is not None
            else "haar_upper_face"
        ),
        "face_loss_grace_seconds": FACE_LOSS_GRACE_SECONDS,
        "face_box_smoothing_alpha": FACE_BOX_SMOOTHING_ALPHA,
        "face_loss_frames": face_loss_frames,
        "invalid_roi_frames": invalid_roi_frames,
        "detector_samples": detector_samples,
        "valid_rgb_samples": int(len(rgb_values))
    })

    if len(timestamps) > 1:

        duration = (
            timestamps[-1]
            - timestamps[0]
        )

        fps = (
            len(rgb_values)
            / duration
        )

        print(
            f"Duration    : "
            f"{duration:.2f} sec"
        )

        print(
            f"Estimated FPS: "
            f"{fps:.2f}"
        )

    return (
        rgb_values,
        timestamps
    )


# ============================================================
# PROCESS ONE WINDOW
# ============================================================

def process_window(
    rgb_window,
    timestamp_window,
    model,
    window_number
):

    processing_started = time.perf_counter()

    print()
    print("--------------------------------------")

    print(
        f"PROCESSING WINDOW "
        f"{window_number}"
    )

    print("--------------------------------------")

    print(
        f"Window samples: "
        f"{len(rgb_window)}"
    )

    report = {
        "window_number": window_number,
        "start_seconds": float(timestamp_window[0])
        if len(timestamp_window) > 0 else None,
        "end_seconds": float(timestamp_window[-1])
        if len(timestamp_window) > 0 else None,
        "cnn_hr": None,
        "pos_hr": None,
        "chrom_hr": None,
        "green_hr": None,
        "roi_method": CAPTURE_METADATA.get(
            "roi_method",
            "haar_upper_face"
        ),
        "detector": CAPTURE_METADATA.get(
            "detector",
            "haar_cascade"
        ),
        "roi_status": "not_processed",
        "valid_sample_count": int(len(rgb_window)),
        "fps": None,
        "effective_sampling_rate": None,
        "timestamp_gap_count": None,
        "max_gap_seconds": None,
        "coverage_ratio": None,
        "resampled": False,
        "timing_status": "not_processed",
        "quality_score": None,
        "quality": None,
        "processing_latency_seconds": None,
        "accepted": False,
        "rejection_reason": None
    }

    def reject(reason):
        report["rejection_reason"] = reason
        report["processing_latency_seconds"] = (
            time.perf_counter() - processing_started
        )
        print(f"Window rejected: {reason}")
        return report

    if len(timestamp_window) < 2:

        print(
            "Not enough timestamps."
        )

        return reject("fewer than two timestamps")

    try:
        raw_timestamps = np.asarray(
            timestamp_window,
            dtype=np.float64
        )
        if (
            len(raw_timestamps) == len(rgb_window)
            and np.all(np.isfinite(raw_timestamps))
            and np.all(np.diff(raw_timestamps) > 0)
        ):
            raw_gaps = np.diff(raw_timestamps)
            report["timestamp_gap_count"] = int(
                np.sum(
                    raw_gaps > MAX_TIMESTAMP_GAP_SECONDS
                )
            )
            report["max_gap_seconds"] = float(
                np.max(raw_gaps)
            )
            report["coverage_ratio"] = float(
                min(
                    (
                        raw_timestamps[-1]
                        - raw_timestamps[0]
                    ) / WINDOW_SECONDS,
                    1.0
                )
            )
            report["effective_sampling_rate"] = float(
                1.0 / np.median(raw_gaps)
            )
            report["fps"] = report[
                "effective_sampling_rate"
            ]
            if report["coverage_ratio"] < (
                MIN_WINDOW_COVERAGE_RATIO
            ):
                report["timing_status"] = (
                    "insufficient_coverage"
                )
            elif report["max_gap_seconds"] > (
                MAX_TIMESTAMP_GAP_SECONDS
            ):
                report["timing_status"] = (
                    "excessive_timestamp_gap"
                )
            else:
                report["timing_status"] = "valid"
    except (TypeError, ValueError, FloatingPointError):
        pass

    try:
        rgb_window, timestamp_window, timing = (
            prepare_uniform_window(
                rgb_window,
                timestamp_window,
                WINDOW_SECONDS
            )
        )
    except (TypeError, ValueError) as e:
        return reject(f"timing validation failed: {e}")

    report["valid_sample_count"] = int(
        len(rgb_window)
    )
    report["effective_sampling_rate"] = timing[
        "effective_sampling_rate"
    ]
    report["fps"] = timing[
        "effective_sampling_rate"
    ]
    report["timestamp_gap_count"] = timing[
        "timestamp_gap_count"
    ]
    report["max_gap_seconds"] = timing[
        "max_gap_seconds"
    ]
    report["coverage_ratio"] = timing[
        "coverage_ratio"
    ]
    report["resampled"] = timing["resampled"]
    report["timing_status"] = timing["timing_status"]

    window_duration = (
        timestamp_window[-1]
        - timestamp_window[0]
    )

    window_fps = timing[
        "effective_sampling_rate"
    ]

    print(
        f"Window duration: "
        f"{window_duration:.2f} sec"
    )

    print(
        f"Window FPS: "
        f"{window_fps:.2f}"
    )

    minimum_samples = max(
        int(MIN_SIGNAL_SECONDS * window_fps) + 5,
        FILTER_PAD_MULTIPLIER * 4 + 2
    )

    if len(rgb_window) < minimum_samples:

        print(
            "Not enough samples for POS."
        )

        return reject("not enough samples for POS")

    # --------------------------------------------------------
    # POS
    # --------------------------------------------------------

    print(
        "Running POS..."
    )

    try:

        pulse = pos_algorithm(
            rgb_window,
            window_fps
        )

    except Exception as e:

        print(
            f"POS failed: {e}"
        )

        return reject(f"POS failed: {e}")

    print(
        "POS complete."
    )

    # --------------------------------------------------------
    # Improved quality
    # --------------------------------------------------------

    print(
        "Checking signal quality..."
    )

    try:

        quality_info = (
            calculate_signal_quality(
                pulse,
                window_fps
            )
        )

    except Exception as e:

        print(
            f"Signal quality failed: "
            f"{e}"
        )

        return reject(f"signal quality failed: {e}")

    print(
        f"Signal Quality: "
        f"{quality_info['quality']}"
    )

    print(
        f"Quality Score: "
        f"{quality_info['score']:.2f}%"
    )
    report["quality_score"] = float(quality_info["score"])
    report["quality"] = quality_info["quality"]
    report["pos_hr"] = quality_info["spectral_hr"]
    report["roi_status"] = "valid"

    if (
        quality_info["score"]
        < MIN_ACCEPTABLE_QUALITY_SCORE
    ):
        return reject(
            "signal quality below "
            f"{MIN_ACCEPTABLE_QUALITY_SCORE:.1f}%"
        )

    if quality_info[
        "spectral_hr"
    ] is not None:

        print(
            f"POS spectral HR: "
            f"{quality_info['spectral_hr']:.2f} BPM"
        )

    print(
        f"Peak/median power: "
        f"{quality_info['peak_prominence']:.2f}"
    )

    print(
        f"Local spectral concentration: "
        f"{quality_info['band_concentration'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # CNN input
    # --------------------------------------------------------

    print(
        "Creating CNN inputs..."
    )

    try:

        time_input, freq_input = (
            create_cnn_inputs(
                pulse,
                window_fps
            )
        )

    except Exception as e:

        print(
            f"CNN input creation failed: "
            f"{e}"
        )

        return reject(f"CNN input creation failed: {e}")

    print(
        f"Time input : "
        f"{time_input.shape}"
    )

    print(
        f"Freq input : "
        f"{freq_input.shape}"
    )

    # --------------------------------------------------------
    # CNN prediction
    # --------------------------------------------------------

    print(
        "Running CNN v3..."
    )

    try:

        prediction = model.predict(
            {
                "input_layer": time_input,
                "input_layer_1": freq_input
            },
            verbose=0
        )

    except Exception:
        try:
            prediction = model.predict(
                [
                    time_input,
                    freq_input
                ],
                verbose=0
            )
        except Exception as e:
            return reject(f"CNN prediction failed: {e}")

    hr = float(
        np.asarray(
            prediction
        ).flatten()[0]
    )
    report["cnn_hr"] = hr

    print(
        f"CNN HR: "
        f"{hr:.2f} BPM"
    )

    # --------------------------------------------------------
    # CNN vs spectral agreement
    # --------------------------------------------------------

    spectral_hr = (
        quality_info["spectral_hr"]
    )

    if spectral_hr is not None:

        agreement_error = abs(
            hr - spectral_hr
        )

        print(
            f"CNN vs POS difference: "
            f"{agreement_error:.2f} BPM"
        )

    else:

        agreement_error = None

    try:
        chrom_pulse = chrom_algorithm(
            rgb_window,
            window_fps
        )
        report["chrom_hr"] = float(
            estimate_heart_rate(
                chrom_pulse,
                window_fps
            )[0]
        )
    except Exception as e:
        print(f"CHROM failed: {e}")

    try:
        report["green_hr"] = estimate_green_hr(
            rgb_window,
            window_fps
        )
    except Exception as e:
        print(f"GREEN failed: {e}")

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    if quality_info["quality"] == "Good":

        print(
            "STATUS: Good signal."
        )

    elif quality_info["quality"] == "Moderate":

        print(
            "STATUS: Moderate signal."
        )

    else:

        print(
            "STATUS: Poor signal."
        )

    report.update({
        "hr": hr,
        "quality_score": quality_info["score"],
        "quality": quality_info["quality"],
        "spectral_hr": spectral_hr,
        "agreement_error": agreement_error,
        "processing_latency_seconds": (
            time.perf_counter() - processing_started
        ),
        "accepted": True
    })

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("======================================")
    print("        LOADING CNN v3 MODEL")
    print("======================================")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    expected_shapes = [
        (None, TIME_LENGTH, 1),
        (None, FREQ_LENGTH, 1),
    ]
    actual_shapes = [
        tuple(shape)
        for shape in model.input_shape
    ]
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            "CNN input shapes changed unexpectedly: "
            f"expected {expected_shapes}, got {actual_shapes}"
        )

    print(
        "CNN v3 model loaded successfully."
    )

    print(
        "Input shapes:",
        model.input_shape
    )

    # --------------------------------------------------------
    # Capture
    # --------------------------------------------------------

    rgb, timestamps = capture_rgb()

    # --------------------------------------------------------
    # Six 10-second windows from the 60-second recording
    # --------------------------------------------------------

    print()
    print("======================================")
    print("       SIGNAL QUALITY + HR")
    print("======================================")

    reports = []

    window_starts = capture_window_starts()

    for window_index, start_time in enumerate(
        window_starts
    ):

        end_time = (
            start_time
            + WINDOW_SECONDS
        )

        mask = (
            (timestamps >= start_time)
            &
            (timestamps < end_time)
        )

        rgb_window = rgb[mask]

        timestamp_window = (
            timestamps[mask]
        )

        if len(rgb_window) < 50:

            print()
            print(
                f"Window {window_index + 1} "
                f"({start_time}-{end_time}s)"
            )

            print(
                "Too few valid samples. "
                "Skipping."
            )

            reports.append({
                "window_number": window_index + 1,
                "start_seconds": float(start_time),
                "end_seconds": float(end_time),
                "cnn_hr": None,
                "pos_hr": None,
                "chrom_hr": None,
                "green_hr": None,
                "roi_method": CAPTURE_METADATA.get(
                    "roi_method",
                    "haar_upper_face"
                ),
                "detector": CAPTURE_METADATA.get(
                    "detector",
                    "haar_cascade"
                ),
                "roi_status": "not_processed",
                "valid_sample_count": int(len(rgb_window)),
                "fps": None,
                "effective_sampling_rate": None,
                "timestamp_gap_count": None,
                "max_gap_seconds": None,
                "coverage_ratio": None,
                "resampled": False,
                "timing_status": "not_processed",
                "quality_score": None,
                "quality": None,
                "processing_latency_seconds": None,
                "accepted": False,
                "rejection_reason": "fewer than 50 valid samples"
            })
            continue

        report = process_window(
            rgb_window,
            timestamp_window,
            model,
            window_index + 1
        )

        reports.append(report)

    accepted_reports = [
        report
        for report in reports
        if report["accepted"]
    ]

    report_path = Path(REPORT_PATH)
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )
    report_path.write_text(
        json.dumps(
            {
                "capture_duration_seconds": CAPTURE_DURATION,
                "window_seconds": WINDOW_SECONDS,
                "window_stride_seconds": WINDOW_STRIDE_SECONDS,
                "minimum_accepted_windows": MIN_ACCEPTED_WINDOWS,
                "minimum_acceptable_quality_score":
                    MIN_ACCEPTABLE_QUALITY_SCORE,
                "capture_metadata": CAPTURE_METADATA,
                "windows": reports
            },
            indent=2
        ),
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    print()
    print()
    print("======================================")
    print("        CNN v3 LIVE HR RESULT")
    print("======================================")

    if len(accepted_reports) < MIN_ACCEPTED_WINDOWS:

        print(
            "Reliable HR estimate unavailable: "
            f"{len(accepted_reports)} accepted window(s), "
            f"{MIN_ACCEPTED_WINDOWS} required."
        )

        return

    for i, result in enumerate(
        reports,
        start=1
    ):

        status = (
            "Accepted"
            if result["accepted"]
            else f"Rejected: {result['rejection_reason']}"
        )
        print(
            f"Window {i}: "
            f"CNN={result['cnn_hr']:.2f} BPM"
            if result["accepted"]
            else f"Window {i}: "
            f"Quality={result['quality']} "
            f"| {status}"
        )
        if result["accepted"]:
            print(
                f"  POS={result['pos_hr']:.2f} BPM"
                if result["pos_hr"] is not None
                else "  POS=unavailable"
            )
            print(
                f"  CHROM={result['chrom_hr']:.2f} BPM"
                if result["chrom_hr"] is not None
                else "  CHROM=unavailable"
            )
            print(
                f"  GREEN={result['green_hr']:.2f} BPM"
                if result["green_hr"] is not None
                else "  GREEN=unavailable"
            )
            print(
                f"  Quality={result['quality']} "
                f"({result['quality_score']:.1f}%)"
            )

    # --------------------------------------------------------
    # Final HR: robust median over accepted windows
    # --------------------------------------------------------

    final_hr = np.median(
        [
            result["cnn_hr"]
            for result in accepted_reports
        ]
    )

    print()
    print(
        f"Final Heart Rate: "
        f"{final_hr:.2f} BPM"
    )

    # --------------------------------------------------------
    # Average quality
    # --------------------------------------------------------

    quality_scores = [
        result["quality_score"]
        for result in accepted_reports
    ]

    average_quality = np.mean(
        quality_scores
    )

    if average_quality >= 60:

        overall_quality = "Good"

    elif average_quality >= 35:

        overall_quality = "Moderate"

    else:

        overall_quality = "Poor"

    print(
        f"Overall Signal Quality: "
        f"{overall_quality}"
    )

    print(
        f"Average Quality Score: "
        f"{average_quality:.2f}%"
    )

    # --------------------------------------------------------
    # Window consistency
    # --------------------------------------------------------

    hr_values = [
        result["cnn_hr"]
        for result in accepted_reports
    ]

    if len(hr_values) >= 2:

        hr_std = np.std(
            hr_values
        )

        hr_range = (
            max(hr_values)
            - min(hr_values)
        )

        print()
        print(
            f"Window HR standard deviation: "
            f"{hr_std:.2f} BPM"
        )

        print(
            f"Window HR range: "
            f"{hr_range:.2f} BPM"
        )

    print("======================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()