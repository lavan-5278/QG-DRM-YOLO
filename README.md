# QG-DRM-YOLO

QG-DRM-YOLO is a YOLOv8n-based classroom behavior detection model designed to address common challenges in intelligent teaching analysis, including large scale variation, heavy occlusion, and subtle differences between behavior categories.

## Abstract

With the development of smart education, computer vision-based classroom behavior detection has become an important approach for intelligent instructional analysis. Classroom behavior detection remains challenging because of pronounced scale variation, severe occlusion, and subtle visual differences among behaviors. We propose QG-DRM-YOLO, a YOLOv8n-based network that integrates four complementary feature-processing modules.

- RGDRC2fV3 combines a stable C2f base with reliability-guided differential enhancement.
- QG-AFPN progressively incorporates source features using quality estimates and source-target consistency.
- MLKCM captures complementary receptive fields.
- P3DetailRestore further refines high-resolution local details for small targets and fine-grained behaviors.

On SCB-Dataset3-S, QG-DRM-YOLO improves Precision, Recall, mAP@0.5, and mAP@0.5:0.95 over YOLOv8n by 3.79, 4.22, 4.09, and 6.08 percentage points, respectively. Its mAP@0.5:0.95 also exceeds YOLOv8s, although YOLOv8s achieves slightly higher Recall. Independently trained on the six-class SCB-Dataset3-U, the proposed model improves mAP@0.5:0.95 by 4.27 points over YOLOv8n. These results indicate that the proposed feature-enhancement strategy improves classroom behavior detection across different class distributions.

## Project Overview

This repository implements QG-DRM-YOLO for classroom behavior detection. The model is built on YOLOv8n and enhanced with four feature-processing modules to improve multi-scale feature representation, detail restoration, and robust perception in crowded classroom scenes.

The main goals are:

- improve detection of small and occluded subjects
- strengthen fine-grained behavior discrimination
- maintain lightweight deployment efficiency
- support classroom behavior understanding in educational scenarios

## Key Contributions

1. Reliability-guided feature enhancement with RGDRC2fV3
2. Quality-aware feature fusion with QG-AFPN
3. Multi-scale receptive field modeling with MLKCM
4. High-resolution detail restoration with P3DetailRestore
5. Strong performance on SCB-Dataset3-S and SCB-Dataset3-U

## Model Architecture

QG-DRM-YOLO adopts YOLOv8n as the base detector and incorporates additional feature enhancement modules into the feature extraction and fusion pipeline.

### Main Modules

- RGDRC2fV3: stabilizes the backbone representation and improves feature reliability through differential enhancement.
- QG-AFPN: uses quality-aware source feature integration and source-target consistency for adaptive multi-scale fusion.
- MLKCM: captures complementary receptive fields to enhance contextual modeling.
- P3DetailRestore: restores local high-resolution details, improving performance on small targets and subtle action patterns.

## Datasets

This project is developed for classroom behavior detection using the SCB-Dataset3 benchmark.

- SCB-Dataset3-S: used for evaluation under the reported settings
- SCB-Dataset3-U: six-class dataset used for independent training and evaluation

## Experimental Results

On SCB-Dataset3-S, compared with YOLOv8n, QG-DRM-YOLO achieves:

- Precision: +3.79%
- Recall: +4.22%
- mAP@0.5: +4.09%
- mAP@0.5:0.95: +6.08%

On SCB-Dataset3-U, the model improves mAP@0.5:0.95 by 4.27 percentage points over YOLOv8n.

These results show that integrating reliability-guided enhancement and multi-scale detail restoration significantly improves classroom behavior detection performance.

## Repository Structure

```text
QG-DRM-YOLO/
├── configs/
│   └── ...
├── datasets/
│   └── ...
├── models/
│   └── ...
├── scripts/
│   └── ...
├── train.py
├── val.py
├── detect.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## Installation

```bash
git clone https://github.com/lavan-5278/QG-DRM-YOLO.git
cd QG-DRM-YOLO
pip install -r requirements.txt
```

## Training

```bash
python train.py --data your_dataset.yaml --epochs 300 --img 640
```

## Validation

```bash
python val.py --weights runs/train/exp/weights/best.pt --data your_dataset.yaml
```

## Inference

```bash
python detect.py --weights runs/train/exp/weights/best.pt --source your_image_or_video
```

## Notes

- This project is intended for research and educational applications in classroom behavior analysis.
- The implementation can be adapted to other YOLOv8-based detection tasks.
- Fine-tuning on custom classroom datasets may require changes to dataset configuration files and class names.

## License

This project is released under the MIT License unless otherwise specified.

## Contact

- GitHub: https://github.com/lavan-5278
- Repository: https://github.com/lavan-5278/QG-DRM-YOLO
