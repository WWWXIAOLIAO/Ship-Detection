"""CPU-only checks for evaluation validity and fusion behavior."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ship_detection.metrics import evaluate, fuse_labels, operating_metrics


class EvaluationTests(unittest.TestCase):
    def test_perfect_oracle_and_empty_detections(self):
        annotations = {
            "info": {},
            "images": [
                {"id": i + 1, "width": 100, "height": 100, "file_name": f"{i}.jpg"}
                for i in range(17)
            ],
            "categories": [{"id": i + 1, "name": f"class{i}"} for i in range(17)],
            "annotations": [
                {
                    "id": i + 1,
                    "image_id": i + 1,
                    "category_id": i + 1,
                    "bbox": [10, 20, 30, 40],
                    "area": 1200,
                    "iscrowd": 0,
                }
                for i in range(17)
            ],
        }
        oracle = {
            "image_id": np.arange(1, 18),
            "classes": np.arange(17),
            "boxes": np.tile([10, 20, 40, 60], (17, 1)),
            "confidence": np.ones(17),
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "annotations.json"
            path.write_text(json.dumps(annotations))
            result = evaluate(oracle, "val", details=True, annotations_path=path)
            self.assertAlmostEqual(result["map"], 1.0)
            self.assertEqual(result["recall_at_conf025"], 1.0)
            empty = {key: value[:0] for key, value in oracle.items()}
            result = evaluate(empty, "val", details=True, annotations_path=path)
            self.assertEqual(result["map"], 0.0)
            self.assertEqual(result["recall_at_conf025"], 0.0)

    def test_confusion_distinguishes_wrong_class_from_missed_object(self):
        annotations = {
            "images": [{"id": 1}],
            "annotations": [
                {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]},
                {"image_id": 1, "category_id": 2, "bbox": [20, 20, 10, 10]},
            ],
        }
        predictions = [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
            {"image_id": 1, "category_id": 3, "bbox": [20, 20, 10, 10], "score": 0.9},
        ]
        result = operating_metrics(annotations, predictions)
        self.assertEqual(result["precision_at_conf025"], 0.5)
        self.assertEqual(result["recall_at_conf025"], 0.5)
        self.assertEqual(result["localization_recall"], 1.0)
        self.assertEqual(result["confusion_matrix"][1][2], 1)

    def test_text_can_change_specific_class_but_preserves_other_ship(self):
        visual = np.full((2, 17), 0.01)
        visual[:, 0] = 0.8
        text = np.zeros((2, 16))
        text[:, 3] = 10
        data = {"classes": np.array([0, 16]), "visual": visual, "descriptions": text}
        np.testing.assert_array_equal(fuse_labels(data, "descriptions", 0), [0, 16])
        np.testing.assert_array_equal(fuse_labels(data, "descriptions", 1), [3, 16])
        for alpha in [-0.1, 1.1]:
            with self.assertRaises(ValueError):
                fuse_labels(data, "descriptions", alpha)


if __name__ == "__main__":
    unittest.main()
