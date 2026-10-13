"""Phase 1 training-path smoke tests (v1.1 22.3)."""

from pathlib import Path

import pytest
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.dsr.model import DSRDetectionModel
from ultralytics.nn.modules.dsr import DSRDetect

ROOT = Path(__file__).resolve().parents[1]
DSR_YAML = ROOT / "ultralytics/cfg/models/11/yolo11n-dsr.yaml"


def _build_smoke_model():
    model = DSRDetectionModel(str(DSR_YAML), nc=3, verbose=False)
    head = model.model[-1]
    # simulate a trained base detector so low-threshold candidates exist at init
    for cv3 in head.cv3:
        cv3[-1].bias.data.zero_()
    head.candidate_conf = 0.05
    head.pre_topk = 100
    head.relation_topk = 40
    head.uncertainty_threshold = 0.0  # v2: select all candidates in the smoke test
    return model


def test_forward_backward_step():
    """One full forward/backward step through DSRDetectionModel.loss."""
    model = _build_smoke_model()
    model.train()
    B, H, W = 2, 64, 64
    img = torch.rand(B, 3, H, W)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0, 0, 1], dtype=torch.long),
        "cls": torch.tensor([0, 2, 1], dtype=torch.long),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.3, 0.3], [0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    }
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        loss, items = model.loss(batch)
        assert loss.shape == (6,)  # box, cls, dfl, relation, preserve, gicr (v3)
        assert items.shape == (6,)
        loss.sum().backward()
        opt.step()
    head_grads = [p.grad for p in model.model[-1].refine_head.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(head_grads) > 0, "refine head must receive gradients"
    mdrr_grads = [p.grad for p in model.model[-1].mdrr.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
    assert len(mdrr_grads) > 0, "MDRR parameters must receive gradients after head updates"


def test_candidate_source_ema_step():
    """v1.1 14.1: EMA head is created, updated, and used for candidate generation."""
    model = _build_smoke_model()
    head = model.model[-1]
    head.candidate_source = "ema"
    head.ema_decay = 0.9
    model.train()
    B, H, W = 2, 64, 64
    img = torch.rand(B, 3, H, W)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0, 0, 1], dtype=torch.long),
        "cls": torch.tensor([0, 2, 1], dtype=torch.long),
        "bboxes": torch.tensor(
            [[0.5, 0.5, 0.3, 0.3], [0.3, 0.3, 0.2, 0.2], [0.7, 0.7, 0.2, 0.2]],
            dtype=torch.float32,
        ),
    }
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        loss, items = model.loss(batch)
        assert torch.isfinite(loss).all()
        loss.sum().backward()
        opt.step()
        head.ema_update()
    assert head.ema_head is not None
    assert head.ema_step == 3
    ema_w = head.ema_head.cv2[0][0].conv.weight
    cur_w = head.cv2[0][0].conv.weight
    assert not torch.equal(ema_w.data, cur_w.data), "EMA weights must track the base head"


def test_inference_uses_ema_candidate_source():
    """EMA candidate selection must be used at inference, not only on the training path."""
    model = _build_smoke_model()
    head = model.model[-1]
    head.candidate_source = "ema"
    head.candidate_conf = 0.5
    ema = head._ensure_ema_head()
    for current, ema_level in zip(head.cv3, ema.cv3):
        current[-1].bias.data.fill_(-10.0)  # current head yields no candidates
        ema_level[-1].bias.data.fill_(10.0)  # EMA head yields candidates

    model.eval()
    with torch.no_grad():
        model(torch.rand(1, 3, 64, 64))

    assert head.last_cand_mask is not None and head.last_cand_mask.any()


def test_frozen_bn_stays_eval():
    """v3: frozen base BN layers must be kept in eval so running stats don't drift."""
    import torch.nn as nn

    from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer

    model = _build_smoke_model()
    model.train()
    base_bns = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    assert base_bns and all(m.training for m in base_bns), "base BN should be trainable initially"
    DSRDetectionTrainer.apply_freeze(model, ["backbone", "neck", "base_detect"])
    DSRDetectionTrainer.keep_frozen_bn_eval(model)
    frozen_bns = [
        m for m in model.modules() if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters())
    ]
    assert frozen_bns, "expected frozen BN layers after apply_freeze"
    assert all(not m.training for m in frozen_bns), "frozen BN must be in eval mode"


def test_stability_compare():
    """Mean max-IoU helper: identical sets -> 1.0; disjoint sets -> < 1.0."""
    a = torch.tensor([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]])
    assert DSRDetect.stability_compare(a, a.clone()) == 1.0
    c = torch.tensor([[10.0, 10.0, 20.0, 20.0], [60.0, 60.0, 70.0, 70.0]])
    assert DSRDetect.stability_compare(a, c) < 1.0
    assert DSRDetect.stability_compare(a, torch.empty(0, 4)) == 0.0


@pytest.mark.slow
def test_trainer_smoke_one_epoch():
    """Real model.train() smoke run on the tiny dataset (entry-constraint check)."""
    from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer

    smoke_yaml = ROOT / "tests/data/smoke.yaml"
    out_dir = ROOT / "runs" / "smoke"
    model = YOLO(str(DSR_YAML))
    # The trainer rebuilds the model from ``self.model.yaml``, so the DSR candidate
    # settings must be lowered there: a random-init base head (bias_init) produces
    # confidences around 1e-3, far below the default candidate_conf=0.03.
    dsr_cfg = model.model.yaml["dsr"]
    dsr_cfg["candidate_conf"] = 0.0001
    dsr_cfg["pre_topk"] = 100
    dsr_cfg["relation_topk"] = 40
    dsr_cfg["validator"] = True  # exercise DSRDetectionValidator during trainer val
    for cv3 in model.model.model[-1].cv3:
        cv3[-1].bias.data.zero_()
    model.train(
        trainer=DSRDetectionTrainer,
        data=str(smoke_yaml),
        epochs=1,
        imgsz=128,
        batch=2,
        device="cpu",
        workers=0,
        project=str(out_dir),
        name="step",
        plots=False,
        verbose=True,
        deterministic=True,
    )
    metrics = model.val(data=str(smoke_yaml), imgsz=128, device="cpu", split="val")
    assert metrics is not None
    save_dir = Path(model.trainer.save_dir)
    stability_report = save_dir / "candidate_stability.json"
    assert stability_report.exists(), "candidate stability probe must write its report"
    audit_report = save_dir / "uncertainty_audit.json"
    assert audit_report.exists(), "uncertainty audit must write its report"
    scenario_report = save_dir / "dsr_scenario.json"
    assert scenario_report.exists(), "DSRDetectionValidator must write dsr_scenario.json"


@pytest.mark.slow
def test_official_resume_link():
    """Official resume: kill a background run mid-training, then model.train(resume=True)."""
    import subprocess
    import sys
    import time

    work = ROOT / "runs/resume_it"
    if work.exists():
        import shutil

        shutil.rmtree(work)
    worker = ROOT / "tests/resume_worker.py"
    proc = subprocess.Popen(
        [sys.executable, str(worker), str(work)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 300
        last_epoch = -1
        while time.time() < deadline and proc.poll() is None:
            ck_path = work / "a/weights/last.pt"
            if ck_path.exists():
                try:
                    ck = torch.load(str(ck_path), map_location="cpu", weights_only=False)
                    last_epoch = int(ck.get("epoch", -1))
                    if last_epoch >= 1:
                        break
                except Exception:
                    pass
            time.sleep(2)
        assert last_epoch >= 1, f"worker did not reach a resumable epoch (epoch={last_epoch})"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()

    m = YOLO(str(work / "a/weights/last.pt"))
    m.train(resume=True, epochs=6, device="cpu", plots=False, verbose=False)
    assert m.trainer.epoch >= 5, "resume must continue training past the interrupted epoch"
