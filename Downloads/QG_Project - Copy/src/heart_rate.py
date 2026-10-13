import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, welch


def bandpass_filter(signal, fps, low=0.7, high=4.0, order=3):
    nyquist = fps / 2

    low = low / nyquist
    high = high / nyquist

    b, a = butter(order, [low, high], btype="band")

    return filtfilt(b, a, signal)


def pos_algorithm(rgb, fps, window_seconds=1.6):
    """
    POS (Plane-Orthogonal-to-Skin) algorithm.

    rgb shape:
        (number_of_frames, 3)

    columns:
        R, G, B
    """

    rgb = np.asarray(rgb, dtype=np.float64)

    window = int(window_seconds * fps)

    if len(rgb) < window:
        raise ValueError("Not enough RGB samples for POS.")

    # Normalize RGB channels
    rgb_normalized = rgb / (
        np.mean(rgb, axis=0) + 1e-8
    )

    pulse = np.zeros(len(rgb))

    for start in range(0, len(rgb) - window + 1):
        end = start + window

        C = rgb_normalized[start:end].T

        # POS projection
        S1 = C[0] - C[1]
        S2 = C[0] + C[1] - 2 * C[2]

        alpha = np.std(S1) / (np.std(S2) + 1e-8)

        h = S1 + alpha * S2

        # Normalize the window
        h = h - np.mean(h)

        pulse[start:end] += h

    return pulse


def estimate_heart_rate(signal, fps):
    """
    Estimate HR using Welch frequency analysis.
    """

    filtered = bandpass_filter(signal, fps)

    frequencies, power = welch(
        filtered,
        fs=fps,
        nperseg=min(len(filtered), int(fps * 10))
    )

    # Physiological HR range:
    # 42 - 240 BPM
    valid = (
        (frequencies >= 0.7) &
        (frequencies <= 4.0)
    )

    if not np.any(valid):
        raise ValueError("No valid heart-rate frequency found.")

    valid_frequencies = frequencies[valid]
    valid_power = power[valid]

    peak_frequency = valid_frequencies[
        np.argmax(valid_power)
    ]

    heart_rate = peak_frequency * 60

    return heart_rate, filtered


def calculate_signal_quality(filtered_signal, fps, heart_rate):
    """
    Calculate a simple signal-quality score based on
    how dominant the detected HR frequency is.
    """

    frequencies, power = welch(
        filtered_signal,
        fs=fps,
        nperseg=min(len(filtered_signal), int(fps * 10))
    )

    valid = (
        (frequencies >= 0.7) &
        (frequencies <= 4.0)
    )

    frequencies = frequencies[valid]
    power = power[valid]

    if len(power) == 0 or np.max(power) == 0:
        return 0.0, "Poor"

    # Dominant peak power
    peak_power = np.max(power)

    # Total power in physiological range
    total_power = np.sum(power)

    ratio = peak_power / total_power

    # Convert to percentage
    quality_score = ratio * 100

    if quality_score >= 25:
        quality = "Good"
    elif quality_score >= 15:
        quality = "Moderate"
    else:
        quality = "Poor"

    return quality_score, quality


def main():

    print("Loading RGB signal...")

    rgb = np.load("rgb_signal.npy")
    timestamps = np.load("timestamps.npy")

    print("RGB shape:", rgb.shape)

    # Estimate actual sampling rate
    duration = timestamps[-1] - timestamps[0]
    fps = len(rgb) / duration

    print(f"Estimated FPS: {fps:.2f}")

    print("\nRunning POS...")

    pulse = pos_algorithm(rgb, fps)

    print("POS signal generated.")

    print("\nEstimating heart rate...")

    heart_rate, filtered_signal = estimate_heart_rate(
        pulse,
        fps
    )

    quality_score, quality = calculate_signal_quality(
        filtered_signal,
        fps,
        heart_rate
    )

    # -----------------------------
    # Display results
    # -----------------------------

    print("\n==============================")
    print(f"❤️ Estimated Heart Rate: {heart_rate:.2f} BPM")
    print(f"📊 Signal Quality: {quality}")
    print(f"📈 Quality Score: {quality_score:.2f}%")
    print("==============================")

    # -----------------------------
    # Plot filtered rPPG waveform
    # -----------------------------

    time_axis = np.arange(len(filtered_signal)) / fps

    plt.figure(figsize=(12, 5))
    plt.plot(time_axis, filtered_signal)

    plt.xlabel("Time (seconds)")
    plt.ylabel("rPPG Signal")
    plt.title("Filtered rPPG Signal")

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -----------------------------
    # Plot frequency spectrum
    # -----------------------------

    frequencies, power = welch(
        filtered_signal,
        fs=fps,
        nperseg=min(len(filtered_signal), int(fps * 10))
    )

    valid = (
        (frequencies >= 0.7) &
        (frequencies <= 4.0)
    )

    plt.figure(figsize=(12, 5))

    plt.plot(
        frequencies[valid] * 60,
        power[valid]
    )

    plt.axvline(
        heart_rate,
        linestyle="--",
        label=f"Detected HR: {heart_rate:.2f} BPM"
    )

    plt.xlabel("Heart Rate (BPM)")
    plt.ylabel("Power")
    plt.title("Heart Rate Frequency Spectrum")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()