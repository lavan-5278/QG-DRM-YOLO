"""Generate Grad-CAM, XGrad-CAM or P3 feature-energy overlays (paper Fig. 10).

Examples:
 python -m scripts.cam --weights runs/scb_s/a4/weights/best.pt --image example.jpg \
       --class-id 0 --method gradcam --box 100 100 180 220

This is a *reimplementation* of attribution visualization, not the authors'
original visualization code. Scores and heatmaps depend on checkpoint,
selected layer, target location and input preprocessing.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics import YOLO
import qgdrm.ultra  # noqa: F401


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--class-id', type=int, default=0)
    parser.add_argument('--method', choices=['gradcam','xgradcam','p3energy'], default='gradcam')
    parser.add_argument('--box', type=int, nargs=4, default=None,
                        metavar=('x1','y1','x2','y2'),
                        help='Selected class target bounding box in 640x640 image pixels')
    parser.add_argument('--output', default='cam_output.png')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch_device = torch.device('cuda:0' if args.device.isdigit() else args.device)
    yolo = YOLO(args.weights)
    net = yolo.model.to(torch_device).eval().requires_grad_(True)
    layers = net.model
    head = layers[-1]
    target_layer = (
        head.qg_pre.mlkcm[0]
        if hasattr(head, 'qg_pre') and head.qg_pre.mlkcm is not None
        else layers[15]
    )
    raw = cv2.imread(args.image)
    if raw is None:
        raise FileNotFoundError(args.image)
    image = cv2.resize(raw, (640, 640))
    x = torch.from_numpy(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).permute(2,0,1)
    x = x.unsqueeze(0).float().to(torch_device) / 255.0
    feats = {}

    def forward_hook(_module, _inputs, output):
        f = output if isinstance(output, torch.Tensor) else output[0]
        feats['activation'] = f
        if f.requires_grad:
            f.register_hook(lambda grad: feats.__setitem__('gradient', grad))

    handle = target_layer.register_forward_hook(forward_hook)
    try:
        net.zero_grad(set_to_none=True)
        result = net(x)
        pred = result[0] if isinstance(result, (tuple,list)) else result
        if args.method == 'p3energy':
            heat = feats['activation'][0].abs().mean(0)
            score = float('nan')
        else:
            if not 0 <= args.class_id < pred.shape[1]-4:
                raise ValueError('Class id outside model head outputs')
            s = pred[0, 4+args.class_id]
            if args.box is not None:
                x1,y1,x2,y2 = args.box
                centers = pred[0,:2]
                mask = (centers[0]>=x1)&(centers[0]<=x2)&(centers[1]>=y1)&(centers[1]<=y2)
                if bool(mask.any()):
                    s = s[mask]
            score_t = s.max()
            score = float(score_t.detach().cpu())
            score_t.backward()
            feature = feats['activation'][0]
            gradient = feats['gradient'][0]
            if args.method == 'gradcam':
                weights = gradient.mean(dim=(1,2))
            else:
                denominator = feature.sum(dim=(1,2))
                safe_denominator = denominator + torch.where(
                    denominator >= 0, torch.full_like(denominator, 1e-8),
                    torch.full_like(denominator, -1e-8))
                weights = (gradient * feature).sum(dim=(1,2)) / safe_denominator
            heat = torch.relu((weights[:,None,None]*feature).sum(0))
        heat = heat.detach().cpu().float().numpy()
    finally:
        handle.remove()
    heat -= heat.min()
    heat /= heat.max()+1e-8
    heat = cv2.resize(heat,(640,640))
    cmap = cv2.applyColorMap(np.uint8(255*heat),cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.5, cmap, 0.5, 0)
    if args.box:
        x1,y1,x2,y2 = args.box
        cv2.rectangle(overlay,(x1,y1),(x2,y2),(255,255,255),2)
    cv2.imwrite(str(Path(args.output)),overlay)
    print(f'Saved {args.output}; class score={score:.4f}; layer={target_layer.__class__.__name__}')


if __name__ == '__main__':
    main()
