"""YOLOv8n + DSR registration contract."""

from pathlib import Path

from ultralytics.nn.modules.dsr import DSRDetect
from ultralytics.nn.tasks import DetectionModel


ROOT = Path(__file__).resolve().parents[1]
MODEL_YAML = ROOT / "ultralytics/cfg/models/v8/yolov8n-dsr.yaml"


def test_yolov8n_dsr_builds_a_registered_dsr_detect_head():
    """The YOLOv8n YAML must route P3/P4/P5 features into DSRDetect."""
    model = DetectionModel(str(MODEL_YAML), nc=3, verbose=False)

    assert isinstance(model.model[-1], DSRDetect)
    assert model.model[-1].nc == 3
    assert model.model[-1].nl == 3
