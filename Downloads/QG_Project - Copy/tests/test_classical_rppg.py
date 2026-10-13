import importlib.util
import unittest

import numpy as np


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


estimate = load_module(
    "estimate_hr_from_rgb",
    "tests/estimate_hr_from_rgb.py",
)
collector = load_module(
    "test_multi_roi_rgb",
    "tests/test_multi_roi_rgb.py",
)


class ClassicalRppgTests(unittest.TestCase):
    def test_synthetic_frequency_is_recovered(self):
        frequency = 1.2
        timestamps = np.arange(0.0, 20.0, 1.0 / 30.0)
        signal = 100.0 + 4.0 * np.sin(2 * np.pi * frequency * timestamps)
        result = estimate.bandpass_and_estimate(signal, 30.0)
        self.assertAlmostEqual(result["hr_bpm"], frequency * 60.0, delta=1.5)

    def test_uniform_resample_rejects_duplicate_and_large_gap(self):
        with self.assertRaises(ValueError):
            estimate.uniform_resample(
                [0.0, 0.1, 0.1],
                [1.0, 2.0, 3.0],
            )
        with self.assertRaises(ValueError):
            estimate.uniform_resample(
                [0.0, 0.1, 0.5],
                [1.0, 2.0, 3.0],
            )

    def test_timestamp_validation_rejects_invalid_values(self):
        with self.assertRaises(ValueError):
            estimate.validate_timestamps(
                [{"time_seconds": "0"}, {"time_seconds": "0"}]
            )
        with self.assertRaises(ValueError):
            estimate.validate_timestamps(
                [{"time_seconds": "0"}, {"time_seconds": "nan"}]
            )

    def test_rgb_roi_extraction_preserves_rgb_order(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:, :, :] = [30, 20, 10]
        landmarks = [
            type("Landmark", (), {"x": 0.1, "y": 0.1})(),
            type("Landmark", (), {"x": 0.9, "y": 0.9})(),
            type("Landmark", (), {"x": 0.5, "y": 0.5})(),
        ]
        face_box = collector.get_face_box(landmarks, frame.shape)
        values, _ = collector.extract_rois(frame, face_box)
        self.assertEqual(values["forehead_R"], 30.0)
        self.assertEqual(values["forehead_G"], 20.0)
        self.assertEqual(values["forehead_B"], 10.0)


if __name__ == "__main__":
    unittest.main()
