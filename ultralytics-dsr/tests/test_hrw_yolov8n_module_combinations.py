"""Contract tests for the 11 unseen unique LGC/DR-SATA/QG-AFPN/DSR combinations."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ultralytics.models.yolo.dsr.model import DSRDetectionModel
from ultralytics.models.yolo.dsr.train import DSRDetectionTrainer
from ultralytics.nn.modules import DRSATA, DSRDetect, LGC, QGAFPN
from ultralytics.nn.tasks import DetectionModel

from experiments.train_hrw_yolov8n_module_combinations_queue import (
    JOBS,
    job_uses_dsr,
    trainer_for,
    write_combo_yamls,
)


EXPECTED_JOB_IDS = {
    "lgc_drsata",
    "lgc_qgafpn",
    "lgc_dsr",
    "drsata_qgafpn",
    "drsata_dsr",
    "qgafpn_dsr",
    "lgc_drsata_qgafpn",
    "lgc_drsata_dsr",
    "lgc_qgafpn_dsr",
    "drsata_qgafpn_dsr",
    "lgc_drsata_qgafpn_dsr",
}


def test_declares_exactly_the_eleven_unseen_unique_subsets():
    """Four existing singleton results plus these 11 jobs form all 15 nonempty subsets."""
    assert {job_id for job_id, *_ in JOBS} == EXPECTED_JOB_IDS
    assert len(JOBS) == 11


@pytest.mark.parametrize("job_id", sorted(EXPECTED_JOB_IDS))
def test_generated_combo_yaml_registers_every_requested_component(tmp_path: Path, job_id: str):
    """Every configuration must parse before the 300-epoch queue can claim a GPU."""
    paths = write_combo_yamls(tmp_path)
    active = set(job_id.split("_"))
    if "dsr" in active:
        model = DSRDetectionModel(str(paths[job_id]), nc=3, verbose=False)
    else:
        model = DetectionModel(str(paths[job_id]), nc=3, verbose=False)

    modules = tuple(model.modules())
    assert any(isinstance(module, LGC) for module in modules) == ("lgc" in active)
    assert any(isinstance(module, DRSATA) for module in modules) == ("drsata" in active)
    assert any(isinstance(module, QGAFPN) for module in modules) == ("qgafpn" in active)
    assert isinstance(model.model[-1], DSRDetect) == ("dsr" in active)
    model.eval()
    with torch.inference_mode():
        prediction = model(torch.zeros(1, 3, 64, 64))
    tensor = prediction[0] if isinstance(prediction, (tuple, list)) else prediction
    assert torch.isfinite(tensor).all()


def test_lgc_then_drsata_is_the_fixed_nonduplicated_backbone_fusion(tmp_path: Path):
    """The joint backbone design must not silently reverse the approved C4/C5 operator order."""
    path = write_combo_yamls(tmp_path)["lgc_drsata"]
    model = DetectionModel(str(path), nc=3, verbose=False)
    layer_types = [type(layer).__name__ for layer in model.model]
    assert layer_types.index("LGC") < layer_types.index("DRSATA")
    assert layer_types.count("LGC") == 2
    assert layer_types.count("DRSATA") == 2


def test_only_dsr_subsets_select_the_dsr_trainer():
    """DSR combinations need DSRDetectionModel/loss instead of the ordinary trainer."""
    for job_id, *_ in JOBS:
        assert job_uses_dsr(job_id) == ("dsr" in job_id.split("_"))
        if job_uses_dsr(job_id):
            assert trainer_for(job_id) is DSRDetectionTrainer
        else:
            assert trainer_for(job_id) is None
