"""Ultralytics-supported paper comparator baselines; fresh random initialization."""
import argparse
from pathlib import Path
from ultralytics import YOLO, RTDETR


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',choices=['yolov8n','yolov8s','yolo11n','rtdetr-l'],required=True)
    p.add_argument('--data',required=True)
    p.add_argument('--device',default='0')
    p.add_argument('--project',default='runs/baselines_s')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--epochs',type=int,default=300)
    a=p.parse_args()
    cls = RTDETR if a.model=='rtdetr-l' else YOLO
    model = cls(a.model+'.yaml')
    model.train(data=str(Path(a.data).resolve()), epochs=a.epochs,
                imgsz=640,batch=a.batch,device=a.device,project=a.project,
                name=a.model,pretrained=False,optimizer='SGD',lr0=0.01,lrf=0.01,
                momentum=.937,weight_decay=.0005,warmup_epochs=3,
                warmup_momentum=.8,patience=100,
                mosaic=1.0,close_mosaic=10,seed=42,
                deterministic=True,amp=False,plots=True)


if __name__=='__main__':main()
