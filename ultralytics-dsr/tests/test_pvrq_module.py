"""Unit tests for the PVR-Q latent perspective field and P3 refiner."""

import torch

from ultralytics.nn.modules import LatentPerspectiveField, PVRF
from ultralytics.nn.tasks import parse_model


CHANNELS = (32, 64, 128, 256)
SHAPES = ((32, 160, 160), (64, 80, 80), (128, 40, 40), (256, 20, 20))


def _features(*, requires_grad=False):
    return [torch.randn(2, c, h, w, requires_grad=requires_grad) for c, h, w in SHAPES]


def test_latent_perspective_field_returns_p3_resolution_outputs():
    module = LatentPerspectiveField(c_p5=256, c_ctx=24, c_field=16)
    field, log_scale = module(torch.randn(2, 256, 20, 20), (80, 80))

    assert field.shape == (2, 16, 80, 80)
    assert log_scale.shape == (2, 1, 80, 80)
    assert torch.isfinite(field).all()
    assert torch.isfinite(log_scale).all()


def test_pvrf_returns_p3_shape_and_training_aux_only():
    module = PVRF(CHANNELS, {"field_channels": 16, "fusion_alpha_init": 0.25})
    module.train()
    output = module(_features())

    assert output.shape == (2, 64, 80, 80)
    assert set(module.last_aux) == {"log_scale"}
    assert module.last_aux["log_scale"].shape == (2, 1, 80, 80)

    module.eval()
    with torch.no_grad():
        module(_features())
    assert module.last_aux is None


def test_pvrf_has_finite_input_and_parameter_gradients():
    module = PVRF(CHANNELS, {"field_channels": 16, "fusion_alpha_init": 0.25})
    module.train()
    features = _features(requires_grad=True)

    output = module(features)
    loss = output.square().mean() + module.last_aux["log_scale"].square().mean()
    loss.backward()

    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in features)
    gradients = [parameter.grad for parameter in module.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_zero_initialized_fusion_is_exact_p3_identity():
    module = PVRF(CHANNELS, {"field_channels": 16, "fusion_alpha_init": 0.0}).eval()
    features = _features()

    with torch.no_grad():
        output = module(features)

    assert torch.equal(output, features[1])


def test_parse_model_resolves_pvrq_config_and_preserves_p3_channels():
    config = {
        "nc": 3,
        "pvrq": {"field_channels": 12, "fusion_alpha_init": 0.0},
        "backbone": [
            [-1, 1, "Conv", [32, 3, 2]],
            [-1, 1, "Conv", [64, 3, 2]],
            [-1, 1, "Conv", [128, 3, 2]],
            [-1, 1, "Conv", [256, 3, 2]],
        ],
        "head": [
            [[0, 1, 2, 3], 1, "PVRF", ["pvrq"]],
            [-1, 1, "Conv", [32, 1, 1]],
        ],
    }

    model, save = parse_model(config, ch=3, verbose=False)
    pvrf = model[-2]
    assert isinstance(pvrf, PVRF)
    assert pvrf.field_channels == 12
    assert save == [0, 1, 2, 3]
    assert model[-1].conv.in_channels == 64
