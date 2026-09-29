import unittest

import torch

from btsp_cam.backbone import PilotFrozenViT


class BackboneTests(unittest.TestCase):
    def test_eval_feature_is_exactly_repeatable(self):
        backbone = PilotFrozenViT(device="cpu", pretrained=False)
        image = torch.zeros(1, 3, 224, 224)
        with torch.inference_mode():
            first = backbone.model(image)
            second = backbone.model(image)
        self.assertEqual(tuple(first.shape), (1, 768))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(any(p.requires_grad for p in backbone.model.parameters()))


if __name__ == "__main__":
    unittest.main()
