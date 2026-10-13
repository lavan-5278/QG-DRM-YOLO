# PVR-Q Phase 0 Diagnosis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible, instance-level diagnosis of the existing YOLO11n baseline that decides whether perspective-conditioned routing and localization-quality calibration merit implementation.

**Architecture:** A standalone `pvrq_diagnosis` package will run an existing baseline weight over the SCB-Dataset3-S validation images, pair predictions with labelled instances, and persist one JSON record per instance. Pure NumPy metric functions then build vertical-position × scale maps, instance-competition analyses, confidence–IoU calibration curves, image-bootstrap confidence intervals, and a machine-readable G0 decision. This phase is read-only with respect to model weights and does not import or add PVR-Q modules.

**Tech Stack:** Python 3.11, PyTorch 2.3, vendored Ultralytics 8.3.247, NumPy, SciPy, Matplotlib, pytest.

## Global Constraints

- Primary data is `ultralytics/cfg/datasets/SCB-Dataset3-S.yaml`; never use SCB-Dataset3-U as evidence.
- Primary weight is an already trained YOLO11n baseline; do not train or modify any checkpoint in Phase 0.
- Use the same 640-pixel image size, confidence threshold `0.001`, IoU NMS threshold `0.7`, and maximum detections `300` as the baseline evaluation protocol.
- Match only predictions of the same class to a GT instance and match each prediction at most once, descending by confidence.
- Treat per-GT pairwise box overlap plus scale-normalized neighbour count as **instance competition**, never ground-truth visibility.
- Use image-level bootstrap with `B=2000`, RNG seed `3407`, and report 95% percentile intervals.
- No Git commands, commits, model YAML changes, or neural-network code in this phase.

---

## File Structure

| File | Responsibility |
|---|---|
| `pvrq_diagnosis/__init__.py` | Marks the diagnostic package and re-exports public metric helpers. |
| `pvrq_diagnosis/phase0_metrics.py` | Deterministic geometry, matching, AP, calibration, bootstrap, binning and G0 decision functions; no model loading. |
| `pvrq_diagnosis/phase0_runner.py` | Loads data/weights, obtains detections, writes JSONL records, invokes metrics, writes JSON/Markdown/PNG outputs. |
| `pvrq_diagnosis/run_phase0.py` | Stable command-line entry point with explicit baseline/data/output arguments. |
| `tests/test_pvrq_phase0.py` | Unit tests for competition, one-to-one matching, AP/calibration, bootstrap and G0 decisions. |
| `runs/pvrq_phase0/baseline_seed42/` | Generated run directory only: manifest, `instances.jsonl`, `phase0_summary.json`, report and figures. |

### Task 1: Deterministic per-instance diagnostic metrics

**Files:**
- Create: `pvrq_diagnosis/__init__.py`
- Create: `pvrq_diagnosis/phase0_metrics.py`
- Create: `tests/test_pvrq_phase0.py`

**Interfaces:**
- Consumes: `gt_xyxy: np.ndarray[N,4]`, `gt_cls: np.ndarray[N]`, `pred_xyxy: np.ndarray[M,4]`, `pred_cls: np.ndarray[M]`, `pred_conf: np.ndarray[M]`, and `(image_h, image_w)`.
- Produces: `compute_competition(...) -> np.ndarray[N]`, `match_instances(...) -> list[dict]`, `compute_group_ap(...) -> dict[str,float]`, `calibration_summary(...) -> dict[str,float]`, `bootstrap_mean_diff(...) -> dict[str,float]`, and `gate_g0(...) -> dict[str, object]`.

- [ ] **Step 1: Write the failing unit tests**

Create `tests/test_pvrq_phase0.py` with these exact tests:

```python
import numpy as np

from pvrq_diagnosis.phase0_metrics import (
    bootstrap_mean_diff,
    compute_competition,
    gate_g0,
    match_instances,
)


def test_competition_combines_overlap_and_nearby_centres():
    boxes = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [40, 0, 50, 10]], dtype=np.float32)
    values = compute_competition(boxes, radius=2.0, beta=1.0, n_ref=4)
    assert values.shape == (3,)
    assert values[0] > values[2]
    assert values[1] > values[2]
    assert np.all((values >= 0.0) & (values < 1.0))


def test_matching_is_class_aware_confidence_ordered_and_one_to_one():
    gt_boxes = np.array([[0, 0, 10, 10], [20, 0, 30, 10]], dtype=np.float32)
    gt_cls = np.array([0, 0], dtype=np.int64)
    pred_boxes = np.array([[0, 0, 10, 10], [0, 0, 9, 9], [20, 0, 30, 10], [20, 0, 30, 10]], dtype=np.float32)
    pred_cls = np.array([0, 0, 1, 0], dtype=np.int64)
    pred_conf = np.array([0.90, 0.80, 0.99, 0.70], dtype=np.float32)
    rows = match_instances(gt_boxes, gt_cls, pred_boxes, pred_cls, pred_conf)
    assert [row["matched_pred_index"] for row in rows] == [0, 3]
    assert [row["matched_iou"] for row in rows] == [1.0, 1.0]
    assert rows[0]["matched_conf"] == 0.90


def test_bootstrap_and_gate_proceed_for_localization_and_calibration_failures():
    focus = np.full(16, 0.35, dtype=np.float64)
    reference = np.full(16, 0.70, dtype=np.float64)
    diff = bootstrap_mean_diff(focus, reference, n_boot=2000, seed=3407)
    assert diff["ci95_high"] < 0.0
    decision = gate_g0(
        iou_diff=diff,
        calibration_gap={"mean": 0.20, "ci95_low": 0.12, "ci95_high": 0.28},
        vertical_area_spearman=-0.45,
    )
    assert decision["decision"] == "PROCEED_PVRQ"
    assert decision["lpfield"] is True
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_pvrq_phase0.py -v`

Expected: collection failure because `pvrq_diagnosis.phase0_metrics` does not exist.

- [ ] **Step 3: Implement the metric module**

Create `pvrq_diagnosis/__init__.py`:

```python
"""Read-only diagnostics for the PVR-Q research gate."""

from .phase0_metrics import compute_competition, gate_g0, match_instances

__all__ = ("compute_competition", "gate_g0", "match_instances")
```

Implement `phase0_metrics.py` with the following exact behavioral rules:

```python
def box_iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return pairwise IoU of [N,4] and [M,4] xyxy boxes; empty inputs return [N,M]."""


def compute_competition(
    boxes: np.ndarray, radius: float = 2.0, beta: float = 1.0, n_ref: int = 4
) -> np.ndarray:
    """Return 1-exp(-(max IoA + beta*nearby/n_ref)) for every GT box."""


def match_instances(
    gt_xyxy: np.ndarray, gt_cls: np.ndarray, pred_xyxy: np.ndarray,
    pred_cls: np.ndarray, pred_conf: np.ndarray,
) -> list[dict]:
    """Greedily assign each prediction at most once by descending confidence within class."""


def bootstrap_mean_diff(
    focus: np.ndarray, reference: np.ndarray, n_boot: int = 2000, seed: int = 3407
) -> dict[str, float]:
    """Bootstrap mean(focus)-mean(reference) with independent draws and percentile CI."""


def gate_g0(iou_diff: dict, calibration_gap: dict, vertical_area_spearman: float) -> dict[str, object]:
    """Return PROCEED_PVRQ, PROCEED_CQ_ONLY, or STOP from predeclared G0 rules."""
```

`match_instances` must sort prediction indices by `(-confidence, index)`, ignore class-mismatched boxes, allocate the available GT of the matching class with maximum IoU, and record zero IoU/confidence for unassigned GT. Each returned row must have `gt_index`, `matched_pred_index` (`None` if unassigned), `matched_iou`, and `matched_conf`.

`compute_competition` must use `intersection_area / area_i` for max IoA and centre distance `< radius * sqrt(area_i)` for neighbour count. Self-pairs are excluded. Use `np.clip` before exponentiation so every return value is finite in `[0, 1)`.

`gate_g0` returns `PROCEED_PVRQ` only when `iou_diff["ci95_high"] < 0.0` and `calibration_gap["ci95_low"] > 0.0`; it sets `lpfield = abs(vertical_area_spearman) >= 0.20`. If both statistical conditions hold but `lpfield` is false, return `PROCEED_CQ_ONLY`; otherwise return `STOP`.

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `pytest tests/test_pvrq_phase0.py -v`

Expected: all three tests pass.

- [ ] **Step 5: Add AP and calibration coverage**

Extend the same test file with:

```python
from pvrq_diagnosis.phase0_metrics import calibration_summary, compute_group_ap


def test_ap_and_calibration_are_bounded_and_penalize_overconfidence():
    records = [
        {"image_id": "a", "gt_index": 0, "matched_iou": 1.0, "matched_conf": 0.95},
        {"image_id": "b", "gt_index": 0, "matched_iou": 0.20, "matched_conf": 0.95},
    ]
    ap = compute_group_ap(records)
    cal = calibration_summary(records, n_bins=5)
    assert 0.0 <= ap["AP50"] <= 1.0
    assert 0.0 <= ap["AP50_95"] <= 1.0
    assert cal["mean_conf_minus_iou"] > 0.0
    assert cal["brier"] >= 0.0
```

Implement `compute_group_ap(records)` as an 11-point precision-envelope AP at thresholds 0.50, 0.55, ..., 0.95 based on per-GT matched confidence/IoU; return `AP50`, `AP75`, and their ten-threshold mean `AP50_95`. Implement `calibration_summary(records, n_bins)` using all records, target `matched_iou`, prediction `matched_conf`, Brier mean squared error, equal-width-bin ECE, and a list of `{lo, hi, n, mean_conf, mean_iou}` rows.

- [ ] **Step 6: Run the complete metric test file**

Run: `pytest tests/test_pvrq_phase0.py -v`

Expected: all four tests pass.

### Task 2: Baseline inference and immutable instance-record generation

**Files:**
- Create: `pvrq_diagnosis/phase0_runner.py`
- Create: `pvrq_diagnosis/run_phase0.py`
- Modify: `tests/test_pvrq_phase0.py`

**Interfaces:**
- Consumes: `--weights PATH`, `--data PATH`, `--output PATH`, `--imgsz INT`, `--device STR`, and `--reuse-records`.
- Produces: `instances.jsonl`, each row containing `image_id`, `gt_index`, `gt_cls`, normalized geometry, `competition`, matched prediction values, plus run manifest metadata.

- [ ] **Step 1: Write the failing record-schema test**

Append:

```python
import json
from pathlib import Path

from pvrq_diagnosis.phase0_runner import write_instance_records


def test_record_writer_preserves_required_instance_fields(tmp_path: Path):
    item = {
        "image_id": "img.jpg", "gt_index": 0, "gt_cls": 1, "x_center_norm": 0.4,
        "y_center_norm": 0.6, "area_norm": 0.01, "log_area_norm": -4.60517,
        "competition": 0.3, "matched_pred_index": 2, "matched_iou": 0.7, "matched_conf": 0.8,
    }
    path = tmp_path / "instances.jsonl"
    write_instance_records([item], path)
    loaded = json.loads(path.read_text(encoding="utf-8").strip())
    assert loaded == item
```

- [ ] **Step 2: Run the schema test to verify it fails**

Run: `pytest tests/test_pvrq_phase0.py::test_record_writer_preserves_required_instance_fields -v`

Expected: collection failure because `phase0_runner` does not exist.

- [ ] **Step 3: Implement record generation and CLI**

Implement these public functions in `phase0_runner.py`:

```python
def write_instance_records(records: list[dict], path: Path) -> None:
    """Write UTF-8 JSONL, one stable-key-sorted object per line."""


def collect_instance_records(
    weights: Path, data_yaml: Path, imgsz: int, device: str, conf: float, iou: float, max_det: int,
) -> tuple[list[dict], dict]:
    """Run Ultralytics YOLO over the validation split and return records and manifest."""


def run_phase0(args: argparse.Namespace) -> Path:
    """Create/reuse records, generate diagnostic report, and return output directory."""
```

Use `ultralytics.data.utils.check_det_dataset` to resolve the validation image directory and `ultralytics.data.utils.img2label_paths` to resolve each label path. Load labels with `np.loadtxt(..., ndmin=2)`; an empty/missing label file becomes `np.empty((0, 5), dtype=np.float32)`. Convert YOLO normalized `class, cx, cy, w, h` labels to original-image `xyxy` using the image dimensions read by `cv2.imread`.

Load `YOLO(str(weights))`, invoke `model.predict(source=str(image_path), imgsz=imgsz, conf=conf, iou=iou, max_det=max_det, device=device, verbose=False)`, and convert `result.boxes.xyxy`, `.cls`, `.conf` to NumPy. Pass them to Task 1 functions. Every GT row must include these exact keys:

```python
{
    "image_id": image_path.name, "gt_index": int(index), "gt_cls": int(cls),
    "x_center_norm": float(cx), "y_center_norm": float(cy),
    "area_norm": float(w * h), "log_area_norm": float(np.log(max(w * h, 1e-12))),
    "competition": float(competition[index]), "matched_pred_index": int_or_none,
    "matched_iou": float(iou), "matched_conf": float(conf),
}
```

The manifest must record absolute `weights`, absolute `data_yaml`, image count, GT count, prediction arguments, UTC timestamp, and SHA-256 of `instances.jsonl`. With `--reuse-records`, refuse reuse if the manifest's weights/data/imgsz/conf/iou/max_det differ from arguments.

Create `run_phase0.py` with these arguments and defaults:

```python
parser.add_argument("--weights", type=Path, required=True)
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--imgsz", type=int, default=640)
parser.add_argument("--device", default="0")
parser.add_argument("--conf", type=float, default=0.001)
parser.add_argument("--iou", type=float, default=0.7)
parser.add_argument("--max-det", type=int, default=300)
parser.add_argument("--reuse-records", action="store_true")
```

- [ ] **Step 4: Run the record-schema test to verify it passes**

Run: `pytest tests/test_pvrq_phase0.py::test_record_writer_preserves_required_instance_fields -v`

Expected: PASS.

- [ ] **Step 5: Run static and import checks**

Run: `python -m compileall pvrq_diagnosis; pytest tests/test_pvrq_phase0.py -v`

Expected: compilation succeeds and all five tests pass.

### Task 3: Report, figures, bootstrap G0 decision, and actual baseline run

**Files:**
- Modify: `pvrq_diagnosis/phase0_metrics.py`
- Modify: `pvrq_diagnosis/phase0_runner.py`
- Modify: `tests/test_pvrq_phase0.py`
- Create at runtime: `runs/pvrq_phase0/baseline_seed42/phase0_summary.json`
- Create at runtime: `runs/pvrq_phase0/baseline_seed42/phase0_report.md`
- Create at runtime: `runs/pvrq_phase0/baseline_seed42/figures/vertical_area_heatmap.png`
- Create at runtime: `runs/pvrq_phase0/baseline_seed42/figures/competition_metrics.png`
- Create at runtime: `runs/pvrq_phase0/baseline_seed42/figures/calibration.png`

**Interfaces:**
- Consumes: JSONL records emitted by Task 2.
- Produces: `build_phase0_summary(records, n_boot=2000, seed=3407) -> dict` with keys `vertical_area`, `competition`, `calibration`, `correlations`, `g0`.

- [ ] **Step 1: Write the failing report test**

Append:

```python
from pvrq_diagnosis.phase0_metrics import build_phase0_summary


def test_phase0_summary_has_predeclared_sections_and_decision():
    records = []
    for i in range(24):
        small_competitive = i < 12
        records.append({
            "image_id": f"{i // 2}", "gt_index": i, "y_center_norm": i / 24,
            "area_norm": 0.002 if small_competitive else 0.04,
            "log_area_norm": -6.2 if small_competitive else -3.2,
            "competition": 0.8 if small_competitive else 0.1,
            "matched_iou": 0.30 if small_competitive else 0.80,
            "matched_conf": 0.75 if small_competitive else 0.80,
        })
    summary = build_phase0_summary(records, n_boot=200, seed=3407)
    assert set(summary) == {"vertical_area", "competition", "calibration", "correlations", "g0"}
    assert summary["g0"]["decision"] == "PROCEED_PVRQ"
```

- [ ] **Step 2: Run the report test to verify it fails**

Run: `pytest tests/test_pvrq_phase0.py::test_phase0_summary_has_predeclared_sections_and_decision -v`

Expected: import failure because `build_phase0_summary` does not exist.

- [ ] **Step 3: Implement summary, figures, and report**

Implement `build_phase0_summary` with these fixed rules:

1. Use vertical bins `[0.00, 0.25, 0.50, 0.75, 1.00]` and three area bins defined by the 1/3 and 2/3 quantiles of `log_area_norm`; each heat-map cell reports `n`, mean matched IoU, AP50, AP75 and AP50-95.
2. Use competition tertiles; every bin reports the same metrics plus `mean_conf_minus_iou` and Brier score.
3. Define the focus population as `area_norm <= q33(area_norm)` **and** `competition >= q50(competition)`, and reference as all remaining records. Bootstrap the mean-IoU difference and the focus calibration gap `mean(conf - IoU)` by resampling image IDs, not individual rows.
4. Compute Spearman `y_center_norm`–`log_area_norm`, competition–`(1-IoU)`, and confidence–IoU with `scipy.stats.spearmanr`; convert NaN to `0.0` for a constant synthetic test only.
5. Call `gate_g0` with the focus/reference IoU bootstrap, focus calibration-gap bootstrap, and vertical/area correlation.

Add `write_phase0_outputs(summary, output_dir)` in `phase0_runner.py`. It must write UTF-8 pretty JSON, Markdown that states the predeclared G0 rule and its result, and three PNGs using Matplotlib `Agg` backend. The heat map uses rows `y-bin` and columns `area-bin`; competition figure has mean IoU and AP75 bars; calibration figure plots confidence vs mean IoU with a dashed `y=x` reference. Do not overwrite `instances.jsonl` during report generation.

- [ ] **Step 4: Run all Phase 0 tests**

Run: `pytest tests/test_pvrq_phase0.py -v`

Expected: all six tests pass.

- [ ] **Step 5: Execute the actual Phase 0 baseline diagnosis**

Run:

```powershell
python .\pvrq_diagnosis\run_phase0.py `
  --weights .\runs\comp_full\baseline\seed42\weights\best.pt `
  --data .\ultralytics\cfg\datasets\SCB-Dataset3-S.yaml `
  --output .\runs\pvrq_phase0\baseline_seed42 `
  --imgsz 640 --device 0
```

Expected: exit code 0; output contains `manifest.json`, `instances.jsonl`, `phase0_summary.json`, `phase0_report.md`, and all three figures.

- [ ] **Step 6: Verify artifacts and report the G0 result**

Run:

```powershell
Get-ChildItem .\runs\pvrq_phase0\baseline_seed42 -Recurse -File |
  Select-Object FullName,Length
Get-Content -Raw -Encoding UTF8 .\runs\pvrq_phase0\baseline_seed42\phase0_summary.json
```

Expected: summary has five top-level sections; report whether the outcome is `PROCEED_PVRQ`, `PROCEED_CQ_ONLY`, or `STOP`, with its confidence intervals and no claim of final SOTA.

## Self-review

Coverage: Task 1 implements the predeclared definitions and G0 decision; Task 2 supplies immutable baseline evidence; Task 3 provides every Phase 0 heat map, competition/calibration analysis, bootstrap and report required by the approved design. No model module, YAML, loss, trainer or checkpoint mutation is included. All public types and names used in Tasks 2–3 are defined in the preceding tasks. The plan contains no Git operation by user instruction.
