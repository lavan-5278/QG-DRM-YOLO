"""Fixed-protocol launcher contract for YOLOv8n + DSR on 5k_HRW."""

from pathlib import Path

from experiments.train_hrw_yolov8n_dsr import MODEL_YAML, build_train_kwargs, format_report_section


def test_yolov8n_dsr_launcher_uses_the_scratch_baseline_protocol():
    kwargs = build_train_kwargs(Path("runs") / "yolov8n_dsr")

    assert MODEL_YAML.name == "yolov8n-dsr.yaml"
    assert kwargs["pretrained"] is False
    assert kwargs["epochs"] == 300
    assert kwargs["batch"] == 8
    assert kwargs["lr0"] == 0.01
    assert kwargs["imgsz"] == 640
    assert kwargs["seed"] == 42
    assert kwargs["optimizer"] == "SGD"


def test_dsr_final_section_contains_metrics_and_efficiency_values():
    text = format_report_section(
        {
            "epoch": "176", "precision": "0.6000", "recall": "0.6100", "map50": "0.6200", "map5095": "0.5000",
            "train_box": "1.0", "train_cls": "2.0", "train_dfl": "3.0",
            "val_box": "4.0", "val_cls": "5.0", "val_dfl": "6.0",
        },
        {"parameters": 123456, "weight_bytes": 1048576, "flops_g": 8.5, "inference_ms_b1_640": 4.25},
    )

    assert "YOLOv8n + DSR" in text
    assert "0.5000" in text
    assert "123,456" in text
    assert "1.00 MB" in text
    assert "8.500" in text
    assert "4.250" in text
