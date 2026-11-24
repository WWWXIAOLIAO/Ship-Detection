"""Cache invalidation must follow model, image, annotation and prompt changes."""

import tempfile
import unittest
from pathlib import Path

from ship_detection.cache import prediction_signature


class CacheTests(unittest.TestCase):
    def test_each_input_change_invalidates_cached_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            files = [Path(tmp) / name for name in ["model.pt", "annotations.json", "image.jpg"]]
            for file in files:
                file.write_bytes(b"initial")

            def signature(prompts=None):
                return prediction_signature(files[0], files[1], [files[2]], prompts)

            initial = signature()
            self.assertEqual(initial, signature())
            for file in files:
                file.write_bytes(b"changed")
                self.assertNotEqual(initial, signature())
                file.write_bytes(b"initial")
            self.assertNotEqual(initial, signature({"prompt": "a ship"}))


if __name__ == "__main__":
    unittest.main()
