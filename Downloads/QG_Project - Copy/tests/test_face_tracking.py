import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, "src")
import main


class FaceTrackingTests(unittest.TestCase):
    def test_landmark_roi_is_clamped_and_has_minimum_size(self):
        frame = np.zeros((100, 120, 3), dtype=np.uint8)
        landmarks = [
            SimpleNamespace(x=-0.1, y=-0.1),
            SimpleNamespace(x=0.9, y=0.9),
            SimpleNamespace(x=0.5, y=0.4),
        ]

        roi, box, reason = main.get_landmark_forehead_roi(
            frame,
            landmarks,
        )

        self.assertIsNone(reason)
        self.assertGreaterEqual(box[0], 0)
        self.assertGreaterEqual(box[1], 0)
        self.assertLessEqual(box[0] + box[2], frame.shape[1])
        self.assertLessEqual(box[1] + box[3], frame.shape[0])
        self.assertEqual(roi.shape[2], 3)

    def test_invalid_landmarks_and_small_roi_are_rejected(self):
        frame = np.zeros((40, 40, 3), dtype=np.uint8)
        invalid = [SimpleNamespace(x=float("nan"), y=0.2)] * 3
        self.assertEqual(
            main.get_landmark_forehead_roi(frame, invalid)[2],
            "invalid landmarks",
        )

        small_frame = np.zeros((20, 20, 3), dtype=np.uint8)
        landmarks = [
            SimpleNamespace(x=0.2, y=0.2),
            SimpleNamespace(x=0.3, y=0.3),
            SimpleNamespace(x=0.25, y=0.25),
        ]
        self.assertIn(
            main.get_landmark_forehead_roi(small_frame, landmarks)[2],
            {"ROI is too narrow", "ROI is too short"},
        )

    def test_rgb_extraction_is_rgb_not_bgr(self):
        roi = np.array([[[10, 20, 30]]], dtype=np.uint8)
        np.testing.assert_array_equal(
            main.extract_mean_rgb(roi),
            np.array([30.0, 20.0, 10.0]),
        )

    def test_invalid_detection_does_not_append_sample(self):
        rgb_values = []
        timestamps = []
        self.assertFalse(
            main.append_valid_sample(rgb_values, timestamps, None, 1.0)
        )
        self.assertEqual(rgb_values, [])
        self.assertEqual(timestamps, [])

        self.assertTrue(
            main.append_valid_sample(
                rgb_values,
                timestamps,
                np.array([30.0, 20.0, 10.0]),
                1.25,
            )
        )
        self.assertEqual(timestamps, [1.25])
        np.testing.assert_array_equal(
            rgb_values[0],
            np.array([30.0, 20.0, 10.0]),
        )

    def test_capture_has_six_ten_second_windows(self):
        self.assertEqual(
            list(main.capture_window_starts()),
            [0, 10, 20, 30, 40, 50],
        )

    def test_missing_mediapipe_asset_uses_fallback(self):
        with patch.object(main, "mp", object()), patch.object(
            main.Path, "is_file", return_value=False
        ):
            self.assertIsNone(main.create_mediapipe_detector())

    def test_cnn_input_shapes_are_unchanged(self):
        pulse = np.sin(np.linspace(0, 20 * np.pi, 300))
        time_input, freq_input = main.create_cnn_inputs(pulse, 30.0)
        self.assertEqual(time_input.shape, (1, 300, 1))
        self.assertEqual(freq_input.shape, (1, 128, 1))


if __name__ == "__main__":
    unittest.main()
