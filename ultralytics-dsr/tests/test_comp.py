"""CompositionHead / CompositionDetect unit tests (CAAL)."""

from pathlib import Path

import torch

from ultralytics.models.yolo.comp.model import CompositionDetectionModel
from ultralytics.nn.tasks import DetectionModel

ROOT = Path(__file__).resolve().parents[1]
COMP_YAML = ROOT / "ultralytics/cfg/models/11/yolo11n-comp.yaml"
YOLO_YAML = ROOT / "ultralytics/cfg/models/11/yolo11.yaml"


def _batch(images, objs):
    batch_idx, cls, bboxes = [], [], []
    for bi, objs_i in enumerate(objs):
        for o in objs_i:
            batch_idx.append(bi)
            cls.append(o[0])
            bboxes.append(o[1:])
    return {
        "img": images,
        "batch_idx": torch.tensor(batch_idx, dtype=torch.long),
        "cls": torch.tensor(cls, dtype=torch.long),
        "bboxes": torch.tensor(bboxes, dtype=torch.float32).reshape(-1, 4),
    }


def test_comp_disabled_output_identical_to_official():
    """comp.enabled=false (or comp_active=False) => byte-identical to official Detect."""
    m_plain = DetectionModel(str(YOLO_YAML), nc=3, verbose=False).eval()
    m_comp = DetectionModel(str(COMP_YAML), nc=3, verbose=False).eval()
    m_comp.model[-1].comp_active = False
    m_comp.load_state_dict(m_plain.state_dict(), strict=False)
    x = torch.randn(1, 3, 64, 64)
    y1 = m_plain(x)
    y2 = m_comp(x)
    y1 = y1[0] if isinstance(y1, tuple) else y1
    y2 = y2[0] if isinstance(y2, tuple) else y2
    assert torch.equal(y1, y2), "comp disabled must produce byte-identical output"
    # active with zero-initialised comp_bias is also identical at init
    m_comp2 = DetectionModel(str(COMP_YAML), nc=3, verbose=False).eval()
    m_comp2.load_state_dict(m_plain.state_dict(), strict=False)
    y3 = m_comp2(x)
    y3 = y3[0] if isinstance(y3, tuple) else y3
    assert torch.equal(y1, y3), "zero-initialised composition bias must not change output"


def test_comp_loss_shape_and_grad():
    model = CompositionDetectionModel(str(COMP_YAML), nc=3, verbose=False)
    model.train()
    B = 2
    img = torch.rand(B, 3, 64, 64)
    batch = _batch(img, [[(0, 0.5, 0.5, 0.3, 0.3), (2, 0.3, 0.3, 0.2, 0.2)], [(1, 0.7, 0.7, 0.2, 0.2)]])
    loss, items = model.loss(batch)
    assert loss.shape == (4,)  # box, cls, dfl, comp
    assert items.shape == (4,)
    assert torch.isfinite(loss).all()
    assert float(loss[3]) > 0.0, "comp loss must be positive when GT classes are present"
    loss.sum().backward()
    comp_grads = [p.grad for p in model.model[-1].comp_head.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(comp_grads) > 0, "CompositionHead must receive gradients"


def test_comp_train_step_smoke():
    model = CompositionDetectionModel(str(COMP_YAML), nc=3, verbose=False)
    model.train()
    img = torch.rand(1, 3, 64, 64)
    batch = _batch(img, [[(1, 0.5, 0.5, 0.3, 0.3), (0, 0.3, 0.3, 0.2, 0.2)]])
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(2):
        opt.zero_grad()
        loss, items = model.loss(batch)
        loss.sum().backward()
        opt.step()
        assert torch.isfinite(loss).all()
    head = model.model[-1]
    assert head.comp_logits is not None and head.comp_logits.shape == (1, 3)
