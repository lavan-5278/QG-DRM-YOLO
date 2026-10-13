"""Ultralytics 8.3.241 adapter without editing site-packages or parse_model.

Strategy: build the ORIGINAL yolov8n DetectionModel (native head, native losses,
PAN, task aligner, checkpoints), replace stride-16 C2f by a wrapped native C2f,
and attach the paper's custom neck as a preprocessor inside the native Detect.
This ensures Ultralytics training/inference handles exactly the same output type.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any

from torch import nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.head import Detect
from ultralytics.models.yolo.detect import DetectionTrainer

from .modules import RGDRC2fV3, QGFeatureNeck


@dataclass(frozen=True)
class AblationFlags:
    rgdr: bool = False
    qg_afpn: bool = False
    mlkcm: bool = False
    p3dr: bool = False


ABLATIONS = {
    "a0": AblationFlags(),
    "a1": AblationFlags(rgdr=True),
    "a2": AblationFlags(rgdr=True, qg_afpn=True),
    "a3": AblationFlags(rgdr=True, qg_afpn=True, mlkcm=True),
    "a4": AblationFlags(rgdr=True, qg_afpn=True, mlkcm=True, p3dr=True),
}


class QGDRMDetect(Detect):
    """Detect head with custom pre-head feature fusion. Native Detect logic unchanged.

    Instantiated by casting a *just-built*, native YOLOv8 Detect to this subclass;
    hence ``cv2``, ``cv3``, stride, biases, DFL, state are those of original head.
    """

    def forward(self, x):
        return super().forward(self.qg_pre(list(x)))


def attach_modules(model: DetectionModel, flags: AblationFlags) -> DetectionModel:
    """Add modules to a fresh *YOLOv8n* DetectionModel. Not an arbitrary YOLO model.

    Validates layer indices and feature sizes so Ultralytics upstream changes
    fail explicitly instead of silently creating a wrong architecture.
    """
    layers = model.model
    if len(layers) != 23:
        raise RuntimeError(f"Expected 23 YOLOv8n layers; got {len(layers)}. Check Ultralytics version.")
    c4 = layers[6]
    if getattr(c4, "__class__").__name__ != "C2f":
        raise RuntimeError(f"Expected C2f at backbone layer 6, got {type(c4).__name__}")
    old_head = layers[22]
    if type(old_head) is not Detect:
        raise RuntimeError(f"Expected standard Detect at layer 22, got {type(old_head).__name__}")
    if list(old_head.f) != [15, 18, 21]:
        raise RuntimeError(f"Unexpected head input layers: {old_head.f}")
    channels = tuple(branch[0].conv.in_channels for branch in old_head.cv2)
    if channels != (64, 128, 256):
        raise RuntimeError(f"Expected YOLOv8n PAN channels (64,128,256), got {channels}")

    if flags.rgdr:
        c1 = c4.cv1.conv.in_channels
        c2 = c4.cv2.conv.out_channels
        if (c1, c2) != (128, 128):
            raise RuntimeError(f"Unexpected C4 channels: {(c1,c2)}")
        replacement = RGDRC2fV3(base=c4, in_channels=c1, out_channels=c2)
        for meta in ("i", "f", "type", "np"):
            if hasattr(c4, meta):
                setattr(replacement, meta, getattr(c4, meta))
        replacement.type = "RGDRC2fV3"
        replacement.np = sum(p.numel() for p in replacement.parameters())
        layers[6] = replacement

    if flags.qg_afpn or flags.mlkcm or flags.p3dr:
        if flags.p3dr and not flags.mlkcm:
            raise ValueError("P3DR requires MLKCM for the paper's progressive ablations")
        # Both types are Python subclasses of nn.Module with the same instance
        # layout. No new Detect parameters or losses are created here.
        old_head.__class__ = QGDRMDetect
        old_head.qg_pre = QGFeatureNeck(
            channels=channels, qg_afpn=flags.qg_afpn,
            mlkcm=flags.mlkcm, p3dr=flags.p3dr,
        )
        old_head.type = "QGDRMDetect (native Detect + pre-head modules)"
        old_head.np = sum(p.numel() for p in old_head.parameters())

    model.qgdrm_ablation = flags
    # State dict, pretrained=False, train loop, native Detect and DFL retained.
    return model


def make_detection_model(cfg: Any, nc: int, flags: AblationFlags,
                         ch: int = 3, verbose: bool = False) -> DetectionModel:
    model = DetectionModel(cfg=cfg, nc=nc, ch=ch, verbose=verbose)
    return attach_modules(model, flags)


def trainer_for(variant: str):
    """Return a trainer class compatible with ``YOLO(...).train(trainer=...)``."""
    if variant not in ABLATIONS:
        raise ValueError(f"Unknown variant {variant}; valid: {', '.join(ABLATIONS)}")
    flags = ABLATIONS[variant]

    class PaperDetectionTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            model = make_detection_model(
                cfg=cfg, nc=self.data["nc"], ch=self.data.get("channels", 3),
                flags=flags, verbose=verbose,
            )
            if weights is not None:
                # Manuscript uses no pretrained weights; this branch is useful
                # only for explicitly requested continuation / fine-tuning.
                model.load(weights)
            return model

    PaperDetectionTrainer.__name__ = f"PaperDetectionTrainer_{variant}"
    return PaperDetectionTrainer
