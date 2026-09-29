import importlib
import sys
import unittest
from pathlib import Path

import numpy as np

from btsp_cam.protocol import equal_increment_tasks, pilot_class_order


ROOT = Path(__file__).resolve().parents[1]
PILOT_ROOT = ROOT / "upstream" / "LAMDA-PILOT"


class PilotProtocolParityTests(unittest.TestCase):
    def test_class_order_matches_actual_pilot_data_manager(self):
        sys.path.insert(0, str(PILOT_ROOT))
        try:
            dm = importlib.import_module("utils.data_manager")

            class FakeData:
                use_path = False
                train_trsf = []
                test_trsf = []
                common_trsf = []

                def download_data(self):
                    self.train_data = np.zeros((100, 1), dtype=np.uint8)
                    self.train_targets = np.arange(100)
                    self.test_data = self.train_data.copy()
                    self.test_targets = self.train_targets.copy()

            original_get_idata = dm._get_idata
            global_state = np.random.get_state()
            try:
                dm._get_idata = lambda name, args=None: FakeData()
                manager = dm.DataManager("fake", True, 1993, 5, 5, {})
            finally:
                dm._get_idata = original_get_idata
                np.random.set_state(global_state)
        finally:
            sys.path.remove(str(PILOT_ROOT))

        np.testing.assert_array_equal(pilot_class_order(100, 1993), manager._class_order)
        self.assertEqual(manager._increments, [5] * 20)

    def test_twenty_disjoint_five_class_tasks(self):
        tasks = equal_increment_tasks(100, 5, 1993)
        self.assertEqual(len(tasks), 20)
        self.assertTrue(all(len(task) == 5 for task in tasks))
        self.assertEqual(np.unique(np.concatenate(tasks)).size, 100)


if __name__ == "__main__":
    unittest.main()
