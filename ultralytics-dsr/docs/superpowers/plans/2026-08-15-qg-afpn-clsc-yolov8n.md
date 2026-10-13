# QG-AFPN and CLSC YOLOv8n Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add QG-AFPN and CLSC Head as independently trainable YOLOv8n modules, then run two fixed-protocol 5k_HRW seed-42 experiments and append their efficiency and accuracy results to the existing Chinese baseline report.

**Architecture:** QGAFPN retains the existing AFPN P3→P4→P5 fusion and modulates each source injection using a zero-initialized scale-agreement quality term. CLSC emits separate class-context and localization-detail feature tuples, while CLSCDetect preserves Detect decoding and loss contracts by routing those tuples to `cv3` and `cv2` respectively.

**Tech Stack:** Python 3.11, PyTorch 2.3, Ultralytics 8.3.247, YAML model definitions, pytest, RTX 4080 SUPER.

## Global Constraints

- Dataset: 5k_HRW; seed=42; pretrained=False; batch=8; SGD; lr0=0.01; epochs=300; imgsz=640.
- QG-AFPN and CLSC are standalone experiments; do not stack either with the other.
- All new residual/modulation scales are initialized to exact identity behavior.
- Preserve ordinary YOLOv8 detection loss, DFL, decoding, NMS, and output contracts.
- Use atomic JSON state writes; do not overwrite previous module-experiment report sections.
- Calculate checkpoint efficiency with RTX 4080 SUPER, CUDA FP32, batch=1, 640×640, 50 warmups and 200 synchronized forwards.
- The workspace is not a Git repository; skip commit commands while preserving the validation steps.

---

### Task 1: Add and verify QGAFPN

**Files:**
- Modify: `ultralytics/nn/modules/block.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Create: `tests/test_qg_afpn_clsc.py`

**Interfaces:**
- Consumes: three P3/P4/P5 tensors, `ch: tuple[int, int, int]`.
- Produces: `QGAFPN(ch).forward(features: list[Tensor]) -> list[Tensor]` with the same shape and channel contracts as AFPN.

- [ ] **Step 1: Write the failing QGAFPN contract test**

```python
def test_qgafpn_identity_and_shapes():
    module = QGAFPN((64, 128, 256)).eval()
    features = [torch.randn(1, 64, 80, 80), torch.randn(1, 128, 40, 40), torch.randn(1, 256, 20, 20)]
    outputs = module(features)
    assert [x.shape for x in outputs] == [x.shape for x in features]
    assert all(torch.isfinite(x).all() for x in outputs)
    assert all(torch.equal(scale, torch.zeros_like(scale)) for scale in module.modulation_scales)
```

- [ ] **Step 2: Run the test and confirm the missing-symbol failure**

Run: `python -m pytest tests/test_qg_afpn_clsc.py::test_qgafpn_identity_and_shapes -v`

Expected: FAIL because `QGAFPN` is not importable.

- [ ] **Step 3: Implement `QGAFPN` beside `AFPN`**

```python
class QGAFPN(AFPN):
    def __init__(self, ch):
        super().__init__(ch)
        self.quality = nn.ModuleList(nn.ModuleList(nn.Conv2d(c, 1, 1) for _ in range(2)) for c in self.ch)
        self.modulation_scales = nn.ParameterList(nn.Parameter(torch.zeros(())) for _ in range(6))

    def forward(self, features):
        outputs, scale_index = [], 0
        for target_index, target_channels in enumerate(self.ch):
            target_size = features[target_index].shape[-2:]
            fused = self.projections[target_index][0](features[0])
            if fused.shape[-2:] != target_size:
                fused = F.interpolate(fused, size=target_size, mode="nearest")
            for source_index in range(1, 3):
                source = self.projections[target_index][source_index](features[source_index])
                if source.shape[-2:] != target_size:
                    source = F.interpolate(source, size=target_size, mode="nearest")
                base_gate = self.gates[target_index][source_index - 1](torch.cat((fused, source), 1))
                agreement = torch.exp(-(F.layer_norm(fused, fused.shape[1:]) - F.layer_norm(source, source.shape[1:])).abs().mean(1, keepdim=True))
                quality = torch.sigmoid(self.quality[target_index][source_index - 1](source))
                modulation = 1 + torch.tanh(self.modulation_scales[scale_index]) * (quality * agreement - 1)
                fused = self.fusions[target_index][source_index - 1](fused + base_gate * modulation * source)
                scale_index += 1
            outputs.append(fused)
        return outputs
```

Export `QGAFPN` from `block.py` and `ultralytics.nn.modules.__init__`, import it in `tasks.py`, and add the parser branch used by AFPN so the YAML receives `(P3_channels, P4_channels, P5_channels)`.

- [ ] **Step 4: Run focused and parser validation**

Run: `python -m pytest tests/test_qg_afpn_clsc.py::test_qgafpn_identity_and_shapes -v`

Expected: PASS. Then run an inline Python model build from `yolov8n-qg-afpn.yaml`; assert a 640×640 forward returns finite output with detection channel count `4 + nc`.

- [ ] **Step 5: Record the non-Git validation checkpoint**

Run: `python -m py_compile ultralytics/nn/modules/block.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py`

Expected: exits 0.

### Task 2: Add and verify CLSC and CLSCDetect

**Files:**
- Modify: `ultralytics/nn/modules/head.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Modify: `tests/test_qg_afpn_clsc.py`

**Interfaces:**
- Consumes: `CLSC(ch).forward(features: list[Tensor])`.
- Produces: a six-tensor list ordered `[P3_cls, P4_cls, P5_cls, P3_loc, P4_loc, P5_loc]`; `CLSCDetect(nc, ch)` consumes the six tensors and returns the normal `Detect` contract.

- [ ] **Step 1: Write failing CLSC tests**

```python
def test_clsc_identity_routes_and_detect_contract():
    adapter = CLSC((64, 128, 256)).eval()
    features = [torch.randn(1, 64, 80, 80), torch.randn(1, 128, 40, 40), torch.randn(1, 256, 20, 20)]
    routed = adapter(features)
    assert len(routed) == 6
    assert [x.shape for x in routed[:3]] == [x.shape for x in features]
    assert [x.shape for x in routed[3:]] == [x.shape for x in features]
    head = CLSCDetect(nc=3, ch=(64, 128, 256)).eval()
    y, _ = head(routed)
    assert y.shape[1] == 7
    assert torch.isfinite(y).all()
```

- [ ] **Step 2: Run tests and confirm the missing-symbol failure**

Run: `python -m pytest tests/test_qg_afpn_clsc.py::test_clsc_identity_routes_and_detect_contract -v`

Expected: FAIL because `CLSC` and `CLSCDetect` are not importable.

- [ ] **Step 3: Implement feature adapters and routed detect head**

```python
class _CLSCResidual(nn.Module):
    def __init__(self, c, kernel):
        super().__init__()
        self.branch = nn.Sequential(DWConv(c, c, kernel), Conv(c, c, 1))
        self.scale = nn.Parameter(torch.zeros(()))
    def forward(self, x):
        return x + self.scale * self.branch(x)

class CLSC(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.cls_adapters = nn.ModuleList(_CLSCResidual(c, 5) for c in ch)
        self.loc_adapters = nn.ModuleList(_CLSCResidual(c, 3) for c in ch)
    def forward(self, features):
        cls = [adapter(x) for adapter, x in zip(self.cls_adapters, features)]
        loc = [adapter(x) for adapter, x in zip(self.loc_adapters, features)]
        return [*cls, *loc]

class CLSCDetect(Detect):
    def forward(self, x):
        cls_features, loc_features = x[:self.nl], x[self.nl:]
        pred = [torch.cat((self.cv2[i](loc_features[i]), self.cv3[i](cls_features[i])), 1) for i in range(self.nl)]
        if self.training:
            return pred
        y = self._inference(pred)
        return y if self.export else (y, pred)
```

Register the modules and add parser branches: CLSC receives three input channels; CLSCDetect receives six indices but inserts only the first three channel values for its `ch` constructor and validates paired channel equality.

- [ ] **Step 4: Run focused, training-mode and YAML contract tests**

Run: `python -m pytest tests/test_qg_afpn_clsc.py -v`

Expected: PASS. Build `yolov8n-clsc.yaml`, run eval and train mode 640×640 forwards, assert eval shape `[1, 7, N]` and training returns three feature-level prediction tensors.

- [ ] **Step 5: Record the non-Git validation checkpoint**

Run: `python -m py_compile ultralytics/nn/modules/head.py ultralytics/nn/modules/__init__.py ultralytics/nn/tasks.py`

Expected: exits 0.

### Task 3: Add standalone model YAMLs and a resumable two-job queue

**Files:**
- Create: `ultralytics/cfg/models/v8/yolov8n-qg-afpn.yaml`
- Create: `ultralytics/cfg/models/v8/yolov8n-clsc.yaml`
- Create: `experiments/train_hrw_yolov8n_qg_clsc_queue.py`
- Modify: `tests/test_qg_afpn_clsc.py`

**Interfaces:**
- Consumes: model YAML paths and the fixed 5k_HRW protocol.
- Produces: `runs/hrw_yolov8n_qg_clsc_seed42_scratch/queue_state.json`, per-job native runs, logs, summaries and report section.

- [ ] **Step 1: Write failing YAML build tests**

```python
@pytest.mark.parametrize("name", ["yolov8n-qg-afpn.yaml", "yolov8n-clsc.yaml"])
def test_custom_model_yaml_builds(name):
    model = YOLO(ROOT / "ultralytics/cfg/models/v8" / name).model.eval()
    with torch.inference_mode():
        prediction, _ = model(torch.zeros(1, 3, 640, 640))
    assert prediction.shape[1] == 7
    assert torch.isfinite(prediction).all()
```

- [ ] **Step 2: Run tests and confirm the missing-YAML failure**

Run: `python -m pytest tests/test_qg_afpn_clsc.py::test_custom_model_yaml_builds -v`

Expected: FAIL because the model YAMLs do not exist.

- [ ] **Step 3: Create YAML graphs and queue runner**

The QG YAML copies `yolov8n-afpn.yaml` and replaces `AFPN` with `QGAFPN`. The CLSC YAML copies the normal YOLOv8n P3/P4/P5 path, applies `CLSC`, selects six outputs with `Index`, and passes them to `CLSCDetect`.

The runner must:

```python
JOBS = (
    ("yolov8n_qg_afpn", "YOLOv8n + QG-AFPN", ROOT / "ultralytics/cfg/models/v8/yolov8n-qg-afpn.yaml"),
    ("yolov8n_clsc", "YOLOv8n + CLSC Head", ROOT / "ultralytics/cfg/models/v8/yolov8n-clsc.yaml"),
)
OUTPUT_ROOT = ROOT / "runs/hrw_yolov8n_qg_clsc_seed42_scratch"
```

Reuse the proven queue behavior: atomic `queue_state.json`; completed jobs skipped on resume; results best-selected by mAP50-95; checkpoint benchmark; native summary; explicit `failed` status on exceptions. Use a distinct report marker `YOLOV8N_QG_CLSC_FINAL_START/END` and append the section without replacing `YOLOV8N_MODULES_FINAL_START/END`.

- [ ] **Step 4: Run build, queue initialization and syntax verification**

Run: `python -m pytest tests/test_qg_afpn_clsc.py -v` and `python -m py_compile experiments/train_hrw_yolov8n_qg_clsc_queue.py`.

Expected: all tests pass; compilation exits 0. Run the runner with a `--dry-run` flag that creates/validates state and report section but never calls `YOLO.train`; assert both jobs are pending and all report columns exist.

- [ ] **Step 5: Record the non-Git validation checkpoint**

Run: `python -c "import json; json.load(open('runs/hrw_yolov8n_qg_clsc_seed42_scratch/queue_state.json', encoding='utf-8')); print('valid')"`

Expected: prints `valid`.

### Task 4: Launch, monitor, and hand off results

**Files:**
- Modify at runtime: `runs/hrw_yolov8n_qg_clsc_seed42_scratch/queue_state.json`
- Modify at runtime: `runs/hrw_standard_baselines_seed42_scratch/5k_HRW_seed42_标准基线结果.md`

**Interfaces:**
- Consumes: verified queue runner and model YAMLs.
- Produces: two completed job summaries, report rows and an hourly monitor automation.

- [ ] **Step 1: Start the verified queue as a hidden background process**

Run the fixed environment interpreter with `experiments/train_hrw_yolov8n_qg_clsc_queue.py --resume`, working directory at the project root; redirect stdout and stderr to `runs/hrw_yolov8n_qg_clsc_seed42_scratch/logs/qg_clsc_queue.out` and `.err`.

- [ ] **Step 2: Create an hourly monitoring automation**

Create a recurring monitor that reads `queue_state.json`, current `results.csv`, queue logs, process state and `nvidia-smi`; report epoch, train/val losses, P/R/mAP50/mAP50-95, deltas, GPU and exceptions. The monitor must not modify or stop training. On each completion it must verify that the report row includes metrics, parameter count, best weight size, FLOPs and inference time.

- [ ] **Step 3: Completion verification**

When both jobs are completed, parse the state file, inspect the two report rows and confirm no `failed` job. State the single-seed/single-split limitation, then delete the hourly monitoring automation.

