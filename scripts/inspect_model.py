"""Print architecture, forward shape and approximate FLOPs (requires Ultralytics)."""
import argparse
import torch
from qgdrm.ultra import make_detection_model, ABLATIONS


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--variant',choices=list(ABLATIONS),default='a4')
    p.add_argument('--nc',type=int,default=3)
    p.add_argument('--imgsz',type=int,default=640)
    a=p.parse_args()
    model=make_detection_model('yolov8n.yaml',a.nc,ABLATIONS[a.variant],verbose=True)
    print('parameters:',sum(x.numel() for x in model.parameters()))
    model.eval()
    with torch.no_grad():
        preds=model(torch.zeros(1,3,a.imgsz,a.imgsz))
    pred = preds[0] if isinstance(preds,tuple) else preds
    if isinstance(pred,torch.Tensor):
        print('predictions shape:',tuple(pred.shape))
    elif isinstance(pred,list):
        print('output list shapes:',[tuple(x.shape) for x in pred])
    model.info(imgsz=a.imgsz)


if __name__=='__main__':main()
