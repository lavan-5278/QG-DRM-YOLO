"""Real Ultralytics tests, automatically skipped if wheel not present.

Run on a machine with installed requirements:
 python -m unittest tests.test_ultra_integration -v
"""
import importlib.util
import unittest
import torch


@unittest.skipUnless(importlib.util.find_spec('ultralytics'), 'Ultralytics is not installed')
class UltralyticsSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        from qgdrm.ultra import ABLATIONS, make_detection_model
        cls.flags = ABLATIONS
        cls.factory = staticmethod(make_detection_model)

    def test_each_variant_forward(self):
        for variant in self.flags:
            with self.subTest(variant=variant):
                net = self.factory('yolov8n.yaml', nc=3, flags=self.flags[variant])
                net.eval()
                with torch.no_grad():
                    output = net(torch.randn(1, 3, 64, 64))
                pred = output[0] if isinstance(output, tuple) else output
                self.assertEqual(tuple(pred.shape), (1, 7, 84))

    def test_loss_and_backward(self):
        net = self.factory('yolov8n.yaml', nc=3, flags=self.flags['a4'])
        net.train()
        batch = {
            'img': torch.rand(2, 3, 64, 64),
            'batch_idx': torch.tensor([0, 1]),
            'cls': torch.tensor([[0.0], [1.0]]),
            'bboxes': torch.tensor([[.5,.5,.3,.3],[.5,.5,.3,.3]]),
        }
        loss, items = net(batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(net.model[6].residual_scale.grad)
        self.assertIsNotNone(net.model[-1].qg_pre.p3_detail.residual_scale.grad)


if __name__ == '__main__':
    unittest.main()
