import unittest

import numpy as np
from PIL import Image

from btsp_cam.cifar100 import IndexedRemappedSubset, build_pilot_cifar224_transform
from btsp_cam.protocol import equal_increment_tasks, pilot_class_order


class CIFAR100DataTests(unittest.TestCase):
    def test_train_transform_shape_dtype_and_range(self):
        image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        tensor = build_pilot_cifar224_transform(True)(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_test_transform_shape_dtype_and_range(self):
        image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))
        tensor = build_pilot_cifar224_transform(False)(image)
        self.assertEqual(tuple(tensor.shape), (3, 224, 224))
        self.assertEqual(str(tensor.dtype), "torch.float32")
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)
        # PILOT cifar224 applies no normalization after ToTensor.
        self.assertAlmostEqual(float(tensor.mean()), 128 / 255, places=6)

    def test_current_train_and_cumulative_test_class_sets(self):
        class FakeDataset:
            targets = np.repeat(np.arange(100), 2)

            def __getitem__(self, index):
                return index, int(self.targets[index])

        order = pilot_class_order(100, 1993)
        tasks = equal_increment_tasks(100, 5, 1993)
        for task_index in (0, 1, 19):
            train = IndexedRemappedSubset(FakeDataset(), tasks[task_index], order)
            test = IndexedRemappedSubset(
                FakeDataset(), np.concatenate(tasks[: task_index + 1]), order
            )
            self.assertEqual(len(train), 10)
            self.assertEqual(set(train.labels), set(range(5 * task_index, 5 * (task_index + 1))))
            self.assertEqual(len(test), 10 * (task_index + 1))
            self.assertEqual(set(test.labels), set(range(5 * (task_index + 1))))


if __name__ == "__main__":
    unittest.main()
