# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
DSR-Det efficiency profiling (v1.1 20.2).

Library function (no standalone train/val script): measures, for a fixed candidate
top-K and input size:

- per-component latency: base Detect decode, candidate selection, RoI Align, relation
  module (graphs / aggregation / gate), refinement head, write-back, NMS;
- total inference latency / FPS, FP32 vs FP16;
- Params / GFLOPs (thop, fixed ``relation_topk``; dynamic ops caveat reported);
- peak inference memory and peak training memory (one forward+backward).

Usage:
    from ultralytics.models.yolo.dsr.bench import profile_dsr
    report = profile_dsr(model, imgsz=640, device="0", repeats=50)

Measurement protocol (20.2): fixed input size, warm-up, deterministic settings,
batch size and whether NMS is included are all reported in the output dict.
"""

from __future__ import annotations

import time
from collections import defaultdict

import torch

from ultralytics.nn.modules.dsr import DSRDetect
from ultralytics.utils import LOGGER
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.torch_utils import select_device, unwrap_model


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _stat(vals: list[float], batch: int) -> dict[str, float]:
    """Mean / median milliseconds per image for a component."""
    if not vals:
        return {"mean_ms": 0.0, "median_ms": 0.0}
    per = [v / batch for v in vals]
    return {
        "mean_ms": round(float(sum(per) / len(per)), 4),
        "median_ms": round(float(sorted(per)[len(per) // 2]), 4),
    }


def _component_report(head: DSRDetect, batch: int) -> dict[str, dict[str, float]]:
    """Aggregate per-component timings from DSRDetect + MDRR timing sinks."""
    out: dict[str, dict[str, float]] = {}
    sinks = [head.timings, getattr(head.mdrr, "timings", None)]
    for sink in sinks:
        if not sink:
            continue
        for key, vals in sink.items():
            out[key] = _stat(vals, batch)
    return out


def _gflops_params(model, x: torch.Tensor) -> tuple[float, int]:
    """thop-based GFLOPs (FP32, fixed top-K) and trainable parameter count."""
    try:
        import thop
    except ImportError:
        LOGGER.warning("thop not installed; GFLOPs unavailable")
        return 0.0, 0
    macs, _ = thop.profile(model, inputs=(x,), verbose=False)
    gflops = float(macs) * 2 / 1e9
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return gflops, params


def profile_dsr(
    model,
    imgsz: int = 640,
    batch: int = 1,
    device: str | None = None,
    warmup: int = 10,
    repeats: int = 50,
    half: bool = False,
    fixed_k: int = 100,
    include_nms: bool = True,
    compare_half: bool = False,
) -> dict:
    """Profile DSR-Det inference (latency components, FLOPs, memory)."""
    device = select_device(device or ("0" if torch.cuda.is_available() else "cpu"))
    model = unwrap_model(model).to(device).eval()
    head = model.model[-1]
    assert isinstance(head, DSRDetect), "profile_dsr requires a DSR-Det model"
    head.relation_topk = int(fixed_k)
    head.pre_topk = max(100, int(fixed_k * 3))

    def run_once(dtype_half: bool) -> dict:
        m = model.half() if dtype_half else model.float()
        x = torch.randn(batch, 3, imgsz, imgsz, device=device)
        x = x.half() if dtype_half else x.float()
        head.timings = defaultdict(list)
        if head.mdrr is not None:
            head.mdrr.timings = defaultdict(list)
        with torch.no_grad():
            for _ in range(warmup):
                m(x)
            _sync(device)
            totals = []
            for _ in range(repeats):
                _sync(device)
                t0 = time.perf_counter()
                y = m(x)
                _sync(device)
                totals.append((time.perf_counter() - t0) * 1000.0)
        total = _stat(totals, batch)
        components = _component_report(head, batch)
        if "mdrr" not in components:
            LOGGER.warning(
                "no candidates selected during profiling; lower candidate_conf "
                "or profile a trained detector to measure the relation branch"
            )
        nms_ms = 0.0
        if include_nms:
            with torch.no_grad():
                nms_times = []
                for _ in range(max(1, min(repeats, 20))):
                    _sync(device)
                    t0 = time.perf_counter()
                    non_max_suppression(y, 0.001, 0.7, max_det=300)
                    _sync(device)
                    nms_times.append((time.perf_counter() - t0) * 1000.0)
            nms_ms = round(float(sum(nms_times) / len(nms_times)) / batch, 4)
        return {
            "precision": "fp16" if dtype_half else "fp32",
            "total_mean_ms": total["mean_ms"],
            "fps": round(1000.0 / max(total["mean_ms"], 1e-9), 1),
            "components": components,
            "nms_mean_ms": nms_ms,
        }

    fp32 = run_once(False)
    runs = [fp32]
    if compare_half or half:
        runs.append(run_once(True))

    gflops, params = _gflops_params(model.float(), torch.randn(batch, 3, imgsz, imgsz, device=device))
    # Peak inference memory
    peak_infer_mb = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            model(torch.randn(batch, 3, imgsz, imgsz, device=device))
        peak_infer_mb = round(torch.cuda.max_memory_allocated(device) / 1e6, 2)

    report = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "imgsz": imgsz,
        "batch": batch,
        "warmup": warmup,
        "repeats": repeats,
        "relation_topk": int(fixed_k),
        "include_nms": include_nms,
        "params": params,
        "gflops_fp32": round(gflops, 2),
        "gflops_note": "thop with fixed relation_topk; RoIAlign/scatter dynamic ops are not counted",
        "peak_inference_memory_mb": peak_infer_mb,
        "runs": runs,
    }
    LOGGER.info(_format_report(report))
    return report


def peak_memory_train(model, imgsz: int = 640, batch: int = 1, device: str | None = None) -> dict:
    """Peak training memory for one forward+backward step (v1.1 20.2)."""
    device = select_device(device or ("0" if torch.cuda.is_available() else "cpu"))
    model = unwrap_model(model).to(device).train()
    img = torch.randn(batch, 3, imgsz, imgsz, device=device)
    n = max(2, batch * 2)
    batch_dict = {
        "img": img,
        "batch_idx": torch.zeros(n, dtype=torch.long, device=device),
        "cls": torch.zeros(n, dtype=torch.long, device=device),
        "bboxes": torch.rand(n, 4, device=device) * 0.5 + 0.25,
    }
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    opt.zero_grad()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    loss, _ = model.loss(batch_dict)
    loss.sum().backward()
    opt.step()
    peak = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else 0.0
    report = {
        "device": str(device),
        "imgsz": imgsz,
        "batch": batch,
        "peak_training_memory_mb": round(peak, 2),
        "note": "single forward+backward step on synthetic batch",
    }
    LOGGER.info(f"peak training memory: {report['peak_training_memory_mb']} MB ({device})")
    return report


def _format_report(report: dict) -> str:
    """Human-readable summary of the profiling report."""
    lines = [
        f"DSR-Det profile: device={report['device']} ({report['device_name']}) imgsz={report['imgsz']} "
        f"batch={report['batch']} topk={report['relation_topk']} warmup={report['warmup']} repeats={report['repeats']}",
        f"params={report['params']}  gflops(fp32)={report['gflops_fp32']}  peak_infer_mem={report['peak_inference_memory_mb']} MB",
    ]
    for run in report["runs"]:
        lines.append(f"[{run['precision']}] total={run['total_mean_ms']} ms/img  fps={run['fps']}  nms={run['nms_mean_ms']} ms/img")
        parts = ", ".join(f"{k}={v['mean_ms']}" for k, v in sorted(run["components"].items()))
        lines.append(f"    components(ms/img): {parts}")
    return "\n".join(lines)
