# RefCOCO-Degraded Benchmark Dataset

## Overview
A benchmark for evaluating visual grounding robustness under adverse imaging conditions.

## Dataset Statistics
| Property | Value |
|----------|-------|
| Source | RefCOCO (val split) |
| Samples | 3811 |
| Conditions | 13 |
| Total Images | 49543 |
| Severities | [0.25, 0.5, 0.75, 1.0] |
| Degradations | ['fog', 'smoke', 'thermal'] |

## Directory Structure
```
/
├── images/
│   ├── clean/                 # Original images
│   ├── fog_0.25/              # Fog (severity 0.25)
│   ├── fog_0.5/               # Fog (severity 0.5)
│   ├── fog_0.75/              # Fog (severity 0.75)    
│   ├── fog_1.0/               # Fog (severity 1.0)
│   ├── smoke_0.25/            # Smoke (severity 0.25)
│   ├── smoke_0.5/             # Smoke (severity 0.5)
│   ├── smoke_0.75/            # Smoke (severity 0.75)
│   ├── smoke_1.0/             # Smoke (severity 1.0)
│   ├── thermal_0.25/          # Thermal (severity 0.25)
│   ├── thermal_0.5/           # Thermal (severity 0.5)
│   ├── thermal_0.75/          # Thermal (severity 0.75)
│   └── thermal_1.0/           # Thermal (severity 1.0)
├── annotations/
│   ├── annotations.json       # Main annotations
│   ├── evaluation_prompts.json # Prompts for VLM evaluation
│   └── val_split.json     # Split information
└── README.md
```

## Annotation Format
```json
{
  "id": 0,
  "filename": "000000.jpg",
  "expression": "the lady with the blue shirt",
  "bbox": [x, y, width, height],
  "bbox_xyxy": [x1, y1, x2, y2],
  "image_width": 640,
  "image_height": 480
}
```

## Usage

### Loading the Dataset
```python
import json
from PIL import Image

# Load annotations
with open('annotations/annotations.json', 'r') as f:
    data = json.load(f)

annotations = data['annotations']
conditions = data['info']['conditions']

# Load an image
sample = annotations[0]
condition = 'fog_0.5'
image_path = f'images/{condition}/{sample["filename"]}'
image = Image.open(image_path)
```

### Evaluating a Model
```python
for sample in annotations:
    for condition in conditions:
        image_path = f'images/{condition}/{sample["filename"]}'
        expression = sample['expression']
        gt_bbox = sample['bbox']
        
        # Your model prediction
        pred_bbox = model.predict(image_path, expression)
        
        # Calculate IoU
        iou = calculate_iou(pred_bbox, gt_bbox)
```

## Degradation Methods
- **Fog**: Atmospheric scattering model (Koschmieder's law)
- **Smoke**: Procedural noise blending with varying opacity
- **Thermal**: Grayscale + INFERNO colormap + sensor noise

## Citation
If you use this dataset, please cite:
1. RefCOCO: Yu et al., "Modeling Context in Referring Expressions", ECCV 2016
2. MS COCO: Lin et al., "Microsoft COCO", ECCV 2014

## Created
2026-02-21 17:40:36
