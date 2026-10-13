# VBR Visible-Body Region Aligner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one VBR module to YOLO11n, verify its identity-safe local upper-body alignment behavior, and screen it against the completed matched baseline on SCB-Dataset3-S.

**Architecture:** `VBR` receives P2 and P3, projects P2 into the P3 channel space, samples P2 at five upper-body behavior-template points with learned bounded offsets, and fuses the weighted samples into P3 through a zero-initialized residual gate. The existing Detect head and `v8DetectionLoss` are unchanged; YAML routes only Detect's P3 input through VBR.

**Tech Stack:** Python 3.11, PyTorch 2.3.0+cu118, vendored Ultralytics 8.3.247, pytest, RTX 4080 SUPER.

## Global Constraints

- Do not use Git or create commits.
- Do not alter `Detect`, `v8DetectionLoss`, data labels, or the existing PVRF module.
- VBR output must have exactly P3's shape and be bitwise identical to P3 at `alpha=0` in eval mode.
- Train only the VBR candidate against `runs/pvrf_matched_baseline_300/seed42`; baseline best mAP50-95 is `0.47010`.
- Use `POLARS_SKIP_CPU_CHECK=1` only in the training process environment because this workstation's installed Polars CPU check otherwise fails.

---

### Task 1: Implement and unit-test VBR

**Files:**
- Create: `ultralytics/nn/modules/vbr.py`
- Modify: `ultralytics/nn/modules/__init__.py`
- Test: `tests/test_vbr_module.py`

**Interfaces:**
- Consumes: `p2: Tensor[B,C2,2H,2W]`, `p3: Tensor[B,C3,H,W]`, and `cfg: dict` containing `samples`, `offset_radius`, and `alpha_init`.
- Produces: `VBR(ch: tuple[int,int], cfg: dict)` and `forward(xs: list[Tensor]) -> Tensor[B,C3,H,W]`.

- [ ] **Step 1: Write failing shape and identity tests**

```python
@pytest.mark.parametrize("p3_hw", [(20, 20), (40, 64)])
def test_vbr_preserves_p3_shape(p3_hw):
    h, w = p3_hw
    module = VBR((16, 32), {"samples": 5, "offset_radius": 2.0, "alpha_init": 0.0}).eval()
    p2, p3 = torch.randn(2, 16, h * 2, w * 2), torch.randn(2, 32, h, w)
    assert module([p2, p3]).shape == p3.shape

def test_vbr_zero_gate_is_exact_identity():
    module = VBR((16, 32), {"alpha_init": 0.0}).eval()
    p2, p3 = torch.randn(1, 16, 40, 40), torch.randn(1, 32, 20, 20)
    assert torch.equal(module([p2, p3]), p3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -m pytest tests/test_vbr_module.py -v`

Expected: FAIL because `ultralytics.nn.modules.vbr` does not exist.

- [ ] **Step 3: Implement the minimal VBR module**

```python
class VBR(nn.Module):
    def __init__(self, ch, cfg):
        c2, c3 = ch
        self.p2_proj = Conv(c2, c3, k=1, s=1)
        self.offset = nn.Conv2d(c3, 2 * samples, 3, padding=1)
        self.weight = nn.Conv2d(c3, samples, 3, padding=1)
        self.fuse = Conv(2 * c3, c3, k=1, s=1)
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

    def forward(self, xs):
        p2, p3 = xs
        detail = self._sample_upper_body_template(self.p2_proj(p2), p3)
        residual = self.fuse(torch.cat((p3, detail), 1))
        return p3 + self.alpha.tanh() * residual
```

Implement `_sample_upper_body_template` with five declared template offsets, `tanh`-bounded learned offsets, normalized `grid_sample(..., align_corners=False)`, and softmax aggregation.

- [ ] **Step 4: Add failing gradient and nonidentity tests**

```python
def test_vbr_nonzero_gate_backpropagates_to_both_scales():
    module = VBR((16, 32), {"alpha_init": 0.25}).train()
    p2 = torch.randn(2, 16, 40, 40, requires_grad=True)
    p3 = torch.randn(2, 32, 20, 20, requires_grad=True)
    out = module([p2, p3])
    out.square().mean().backward()
    assert not torch.equal(out.detach(), p3.detach())
    assert torch.isfinite(out).all() and p2.grad is not None and p3.grad is not None
    assert module.offset.weight.grad is not None and module.weight.weight.grad is not None
```

- [ ] **Step 5: Run module tests**

Run: `D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -m pytest tests/test_vbr_module.py -v`

Expected: PASS.

### Task 2: Register VBR and validate the YAML graph

**Files:**
- Modify: `ultralytics/nn/modules/__init__.py`
- Modify: `ultralytics/nn/tasks.py`
- Create: `ultralytics/cfg/models/11/yolo11n-vbr.yaml`
- Test: `tests/test_vbr_module.py`

**Interfaces:**
- Consumes: YAML top-level `vbr` dict and feature layer indices `[1,16]`.
- Produces: a P3 output at layer 23 and `Detect` input indices `[23,19,22]`.

- [ ] **Step 1: Write the failing parser assertion**

```python
def test_vbr_yaml_builds_and_routes_refined_p3():
    model = YOLO("ultralytics/cfg/models/11/yolo11n-vbr.yaml").model
    assert isinstance(model.model[-2], VBR)
    assert model.model[-1].f == [23, 19, 22]
```

- [ ] **Step 2: Run the parser test to verify it fails**

Run: `D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -m pytest tests/test_vbr_module.py::test_vbr_yaml_builds_and_routes_refined_p3 -v`

Expected: FAIL because `VBR` is not exported or parse_model has no VBR branch.

- [ ] **Step 3: Register and configure the module**

```python
# tasks.py parse_model branch
elif m is VBR:
    for j, a in enumerate(args):
        if isinstance(a, str) and a == "vbr":
            args[j] = d.get("vbr", {})
    args.insert(0, tuple(ch[x] for x in f))
    c2 = ch[f[1]]
```

Export `VBR` from `ultralytics.nn.modules`. Copy `yolo11n-pvrf.yaml`, remove PVRF and `pvrq`, add the declared `vbr` config, append `[[1,16],1,VBR,[vbr]]`, and route Detect as `[[23,19,22],1,Detect,[nc]]`.

- [ ] **Step 4: Run parser and CUDA forward tests**

Run: `D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -m pytest tests/test_vbr_module.py -v`

Expected: PASS, including CUDA `YOLO(vbr_yaml).model(torch.zeros(1,3,640,640,device="cuda"))`.

### Task 3: Regression checks and training launch

**Files:**
- Test: `tests/test_vbr_module.py`
- Create: `runs/vbr_full_300/seed42/` through Ultralytics training output
- Create: `runs/vbr_full_300/launcher.out`
- Create: `runs/vbr_full_300/launcher.err`

**Interfaces:**
- Consumes: `yolo11n-vbr.yaml`, `yolo11n.pt`, and `SCB-Dataset3-S.yaml`.
- Produces: standard Ultralytics `results.csv`, `weights/best.pt`, and `weights/last.pt`.

- [ ] **Step 1: Run focused and detection regression tests**

Run: `D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -m pytest tests/test_vbr_module.py tests/test_pvrq_module.py -v`

Expected: PASS with no PVRF regression.

- [ ] **Step 2: Verify GPU and refuse overwrite of the output directory**

Run: `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader`

Expected: GPU 0 has sufficient available memory; abort before launch if `runs/vbr_full_300/seed42` already exists.

- [ ] **Step 3: Launch the 300-epoch candidate in the CUDA environment**

```powershell
$env:POLARS_SKIP_CPU_CHECK='1'
D:\Users\Admin\anaconda3\envs\lfm-yolo-mamba-stage14\python.exe -c "from ultralytics import YOLO; model=YOLO('ultralytics/cfg/models/11/yolo11n-vbr.yaml').load('yolo11n.pt'); model.train(data='ultralytics/cfg/datasets/SCB-Dataset3-S.yaml', epochs=300, imgsz=640, batch=16, device=0, workers=8, seed=42, deterministic=True, patience=100, project='runs/vbr_full_300', name='seed42', exist_ok=False, pretrained=False)"
```

Run it as a hidden background process with stdout/stderr redirected to the declared launcher files.

- [ ] **Step 4: Verify launch and monitor every 10 minutes**

Read process state, `results.csv`, `launcher.out`, `launcher.err`, and `nvidia-smi`. Report latest losses, precision, recall, mAP50, mAP50-95, and errors. Stop monitoring on normal completion.

### Task 4: Make the screening decision

**Files:**
- Read: `runs/vbr_full_300/seed42/results.csv`
- Read: `runs/pvrf_matched_baseline_300/seed42/results.csv`

**Interfaces:**
- Consumes: best row selected by maximum `metrics/mAP50-95(B)` from each CSV.
- Produces: raw comparison and `PROMOTE` or `REJECT` decision.

- [ ] **Step 1: Extract best rows exactly**

```powershell
$rows = Import-Csv runs/vbr_full_300/seed42/results.csv
$rows | Sort-Object {[double]$_.'metrics/mAP50-95(B)'} -Descending | Select-Object -First 1
```

- [ ] **Step 2: Compute delta against the fixed baseline**

```powershell
$delta = [double]$vbr.'metrics/mAP50-95(B)' - 0.47010
```

- [ ] **Step 3: Report decision**

Report `PROMOTE` only if `delta >= 0.0020`; otherwise report `REJECT`. Include both mAP50-95 values, mAP50, Precision, Recall, best epochs, the exact delta, and the one-seed limitation.
