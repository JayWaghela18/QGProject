import numpy as np
from scipy.signal import butter, filtfilt, welch


# ============================================================
# SETTINGS
# ============================================================

LOW_HR = 0.7
HIGH_HR = 4.0


# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(signal, fps, low=0.7, high=4.0, order=3):

    nyquist = fps / 2

    low_norm = low / nyquist
    high_norm = high / nyquist

    b, a = butter(
        order,
        [low_norm, high_norm],
        btype="band"
    )

    return filtfilt(b, a, signal)


# ============================================================
# CHROM ALGORITHM
# ============================================================

def chrom_algorithm(rgb, fps, window_seconds=1.6):

    """
    CHROM rPPG algorithm.

    Input:
        rgb = array of shape (frames, 3)

    Columns:
        R, G, B

    Output:
        pulse signal
    """

    rgb = np.asarray(
        rgb,
        dtype=np.float64
    )

    if rgb.ndim != 2 or rgb.shape[1] != 3:

        raise ValueError(
            "RGB must have shape (frames, 3)."
        )

    window = int(
        window_seconds * fps
    )

    if len(rgb) < window:

        raise ValueError(
            "Not enough RGB samples for CHROM."
        )

    pulse = np.zeros(
        len(rgb)
    )

    # --------------------------------------------------------
    # Process overlapping windows
    # --------------------------------------------------------

    for start in range(
        0,
        len(rgb) - window + 1
    ):

        end = start + window

        C = rgb[start:end].copy()

        # ----------------------------------------------------
        # Normalize RGB
        # ----------------------------------------------------

        C = C / (
            np.mean(
                C,
                axis=0,
                keepdims=True
            )
            + 1e-8
        )

        R = C[:, 0]
        G = C[:, 1]
        B = C[:, 2]

        # ----------------------------------------------------
        # CHROM chrominance projection
        # ----------------------------------------------------

        X = (
            3 * R
            - 2 * G
        )

        Y = (
            1.5 * R
            + G
            - 1.5 * B
        )

        # ----------------------------------------------------
        # Adaptive scaling
        # ----------------------------------------------------

        alpha = (
            np.std(X)
            / (np.std(Y) + 1e-8)
        )

        # ----------------------------------------------------
        # CHROM pulse signal
        # ----------------------------------------------------

        S = X - alpha * Y

        # Remove DC component
        S = S - np.mean(S)

        # ----------------------------------------------------
        # Overlap-add
        # ----------------------------------------------------

        pulse[start:end] += S

    return pulse


# ============================================================
# HEART RATE ESTIMATION
# ============================================================

def estimate_heart_rate(
    signal,
    fps
):

    filtered = bandpass_filter(
        signal,
        fps
    )

    frequencies, power = welch(
        filtered,
        fs=fps,
        nperseg=min(
            len(filtered),
            int(fps * 10)
        )
    )

    valid = (
        (frequencies >= LOW_HR)
        &
        (frequencies <= HIGH_HR)
    )

    if not np.any(valid):

        raise ValueError(
            "No valid heart-rate frequency found."
        )

    valid_frequencies = frequencies[
        valid
    ]

    valid_power = power[
        valid
    ]

    peak_index = np.argmax(
        valid_power
    )

    peak_frequency = (
        valid_frequencies[peak_index]
    )

    heart_rate = (
        peak_frequency * 60
    )

    return heart_rate, filtered


# ============================================================
# SIGNAL QUALITY
# ============================================================

def calculate_signal_quality(
    filtered_signal,
    fps
):

    frequencies, power = welch(
        filtered_signal,
        fs=fps,
        nperseg=min(
            len(filtered_signal),
            int(fps * 10)
        )
    )

    valid = (
        (frequencies >= LOW_HR)
        &
        (frequencies <= HIGH_HR)
    )

    power = power[valid]

    if len(power) == 0:

        return 0.0, "Poor"

    total_power = np.sum(power)

    if total_power <= 0:

        return 0.0, "Poor"

    peak_power = np.max(power)

    score = (
        peak_power
        / total_power
        * 100
    )

    if score >= 25:

        quality = "Good"

    elif score >= 15:

        quality = "Moderate"

    else:

        quality = "Poor"

    return score, quality


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("===================================")
    print("       CHROM rPPG ANALYSIS")
    print("===================================")

    # --------------------------------------------------------
    # Load RGB data
    # --------------------------------------------------------

    print()
    print("Loading RGB signal...")

    rgb = np.load(
        "rgb_signal.npy"
    )

    timestamps = np.load(
        "timestamps.npy"
    )

    print(
        "RGB shape:",
        rgb.shape
    )

    # --------------------------------------------------------
    # Calculate FPS
    # --------------------------------------------------------

    duration = (
        timestamps[-1]
        - timestamps[0]
    )

    if duration <= 0:

        raise ValueError(
            "Invalid timestamps."
        )

    fps = (
        len(rgb)
        / duration
    )

    print(
        f"Estimated FPS: {fps:.2f}"
    )

    # --------------------------------------------------------
    # CHROM
    # --------------------------------------------------------

    print()
    print("Running CHROM...")

    pulse = chrom_algorithm(
        rgb,
        fps
    )

    print(
        "CHROM signal generated."
    )

    # --------------------------------------------------------
    # Heart Rate
    # --------------------------------------------------------

    print()
    print(
        "Estimating heart rate..."
    )

    heart_rate, filtered = (
        estimate_heart_rate(
            pulse,
            fps
        )
    )

    # --------------------------------------------------------
    # Quality
    # --------------------------------------------------------

    score, quality = (
        calculate_signal_quality(
            filtered,
            fps
        )
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    print()
    print("===================================")
    print("          CHROM RESULT")
    print("===================================")

    print(
        f"Heart Rate    : "
        f"{heart_rate:.2f} BPM"
    )

    print(
        f"Signal Quality: "
        f"{quality}"
    )

    print(
        f"Quality Score : "
        f"{score:.2f}%"
    )

    print("===================================")


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    main()