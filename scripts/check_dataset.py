"""Inspect YOLO-format SCB split sizes, annotations, label indices, normalized boxes."""
import argparse
from collections import Counter
from pathlib import Path
import yaml

IMAGE_EXTS={'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}
EXPECTED={
  's': {'train':4012,'val':1003,'instances':25810,'classes':[11207,10841,3762]},
  'u': {'train':797,'val':209,'instances':29656,'classes':[22,11733,4472,10464,1422,1543]},
}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--expected',choices=['s','u'],default=None)
    args=p.parse_args()
    yaml_path=Path(args.data).expanduser().resolve()
    data=yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    root=Path(data['path'])
    if not root.is_absolute():root=yaml_path.parent/root
    root=root.resolve()
    nc=int(data['nc'])
    counts=Counter(); images_total={}; invalid=[]
    for split in ('train','val'):
        img_path=Path(data[split]);img_path=img_path if img_path.is_absolute() else root/img_path
        image_paths=sorted(p for p in img_path.rglob('*') if p.suffix.lower() in IMAGE_EXTS) if img_path.exists() else []
        images_total[split]=len(image_paths)
        print(f'{split}: {len(image_paths)} images; image root: {img_path}')
        for image in image_paths:
            try:
                relative=image.relative_to(root/'images')
                label=root/'labels'/relative.with_suffix('.txt')
            except ValueError:
                invalid.append(f'Cannot infer label path from image path: {image}')
                continue
            if not label.exists():
                continue # empty-label images are allowed
            for line_number,line in enumerate(label.read_text(encoding='utf-8').splitlines(),1):
                if not line.strip():continue
                fields=line.split()
                try:
                    if len(fields)!=5:raise ValueError('Expected class x_center y_center w h')
                    cat=int(fields[0]); coordinates=list(map(float,fields[1:]))
                    if not 0<=cat<nc:raise ValueError('Class out of range')
                    x,y,w,h=coordinates
                    if not(0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):
                        raise ValueError('Invalid normalized box')
                    if not(0<=x-w/2<=1 and 0<=x+w/2<=1 and 0<=y-h/2<=1 and 0<=y+h/2<=1):
                        raise ValueError('Box extends outside normalized image')
                    counts[cat]+=1
                except Exception as exc:
                    invalid.append(f'{label}:{line_number}: {exc}')
    print(f'Class counts: {[counts[i] for i in range(nc)]}; instances total: {sum(counts.values())}')
    if args.expected:
        expected=EXPECTED[args.expected]
        print('Reference sizes from manuscript:',expected)
        for key in ('train','val'):
            print(key, 'MATCH' if images_total[key]==expected[key] else 'DIFF')
        print('Instance count:', 'MATCH' if sum(counts.values())==expected['instances'] else 'DIFF')
        print('Class counts:', 'MATCH' if [counts[i] for i in range(nc)]==expected['classes'] else 'DIFF')
    print(f'Invalid annotations: {len(invalid)}')
    for issue in invalid[:30]:print('INVALID',issue)
    if invalid:raise SystemExit(1)


if __name__=='__main__':main()
