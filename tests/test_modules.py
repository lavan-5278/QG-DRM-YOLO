"""Pure-PyTorch unit tests; no ultralytics wheel/GPU/dataset required."""
import unittest
import torch
from torch import nn

from qgdrm.modules import RGDRC2fV3, QGAFPN, MLKCM, P3DetailRestore, QGFeatureNeck


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_rgdr_starts_exactly_at_native_c2f_and_zero_gate(self):
        baseline = nn.Sequential(nn.Conv2d(128, 128, 1), nn.ReLU())
        model = RGDRC2fV3(baseline, 128, 128)
        x = torch.randn(1, 128, 4, 4, requires_grad=True)
        output = model(x)
        expected = baseline(x)
        self.assertTrue(torch.equal(output, expected))
        self.assertEqual(model.hidden, 32)
        self.assertEqual(model.reliability(baseline(x)).shape[1], 1)
        self.assertAlmostEqual(float(model.residual_scale.detach()), 0.0)
        output.mean().backward()
        self.assertIsNotNone(model.residual_scale.grad)

    def test_qgafpn_output_shapes_and_six_zero_initializations(self):
        pyramid = [
            torch.randn(1, 64, 8, 8, requires_grad=True),
            torch.randn(1, 128, 4, 4, requires_grad=True),
            torch.randn(1, 256, 2, 2, requires_grad=True),
        ]
        fuser = QGAFPN((64, 128, 256))
        out = fuser(pyramid)
        self.assertEqual([x.shape for x in out], [x.shape for x in pyramid])
        strengths = [i.raw_strength for stages in fuser.injections for i in stages]
        self.assertEqual(len(strengths), 6)
        self.assertTrue(all(float(x.detach()) == 0 for x in strengths))
        sum(x.mean() for x in out).backward()
        self.assertTrue(all(t.grad is not None for t in pyramid))

    def test_mlkcm_asymmetric_channel_partition(self):
        for c, expected in [(64,[22,21,21]), (128,[43,43,42]), (256,[86,85,85])]:
            m = MLKCM(c)
            self.assertEqual(m.group_channels, expected)
            x = torch.randn(1, c, 4, 4)
            self.assertEqual(m(x).shape, x.shape)

    def test_p3_identity_at_start_and_trainability(self):
        model = P3DetailRestore(64)
        x = torch.randn(2, 64, 4, 4, requires_grad=True)
        out = model(x)
        self.assertTrue(torch.equal(x, out))
        out.sum().backward()
        self.assertIsNotNone(model.residual_scale.grad)
        self.assertAlmostEqual(float(model.reliability.bias.mean().detach()), 0)

    def test_progressive_neck(self):
        model = QGFeatureNeck((64, 128, 256), True, True, True)
        x = [torch.randn(1, c, d, d) for c, d in [(64,8),(128,4),(256,2)]]
        y = model(x)
        self.assertEqual([tuple(t.shape) for t in y], [tuple(t.shape) for t in x])


if __name__ == '__main__':
    unittest.main()
