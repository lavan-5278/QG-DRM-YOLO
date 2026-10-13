"""Side-by-side qualitative boxes with unified conf / NMS (paper Fig. 9)."""
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import qgdrm.ultra  # noqa: F401


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--image',required=True)
    p.add_argument('--weights',nargs='+',required=True,help='Comparable checkpoints, e.g. n a4 s')
    p.add_argument('--labels',nargs='+',default=None)
    p.add_argument('--conf',type=float,default=0.25)
    p.add_argument('--iou',type=float,default=0.7)
    p.add_argument('--output',default='comparison.jpg')
    p.add_argument('--device',default='0')
    args=p.parse_args()
    labels=args.labels or [Path(w).parent.parent.name for w in args.weights]
    if len(labels)!=len(args.weights): raise ValueError('labels count must match weights count')
    tiles=[]
    for weights,title in zip(args.weights,labels):
        r=YOLO(weights).predict(args.image, imgsz=640, conf=args.conf,iou=args.iou,
                               device=args.device,verbose=False)[0]
        tile=r.plot() # BGR image with boxes, scores and class IDs
        cv2.rectangle(tile,(0,0),(min(tile.shape[1],24+len(title)*15),36),(0,0,0),-1)
        cv2.putText(tile,title,(8,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        tiles.append(tile)
    height=max(x.shape[0] for x in tiles)
    tiles=[cv2.copyMakeBorder(t,0,height-t.shape[0],0,0,cv2.BORDER_CONSTANT) for t in tiles]
    cv2.imwrite(args.output,np.concatenate(tiles,axis=1))
    print('Saved',args.output,'with conf=',args.conf,'iou=',args.iou)


if __name__=='__main__':main()
