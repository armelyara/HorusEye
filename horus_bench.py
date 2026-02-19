"""
Create Complete RefCOCO-Degraded Benchmark Dataset
Includes: All conditions × Multiple severities × Full annotations
"""

import os
import json
import cv2
import numpy as np
from tqdm import tqdm
from datetime import datetime

from degradation_pipeline import RefCOCOLoader, DegradationPipeline


def create_full_dataset(
    refcoco_path,
    coco_images_path,
    output_path,
    split='val',
    severities=[0.25, 0.5, 0.75, 1.0],
    degradation_types=['fog', 'smoke', 'thermal'],
    max_samples=None
):
    """
    Create complete benchmark dataset with all degradations and severities
    
    Args:
        refcoco_path: Path to RefCOCO annotations
        coco_images_path: Path to COCO images
        output_path: Where to save the dataset
        split: 'train', 'val', 'testA', or 'testB'
        severities: List of severity levels [0.0 - 1.0]
        degradation_types: List of degradation types
        max_samples: None for all samples, or integer to limit
    """
    
    print("=" * 70)
    print("CREATING REFCOCO-DEGRADED BENCHMARK DATASET")
    print("=" * 70)
    
    # Load RefCOCO
    print(f"\nLoading RefCOCO ({split} split)...")
    loader = RefCOCOLoader(refcoco_path, coco_images_path, split=split)
    
    n_samples = len(loader) if max_samples is None else min(max_samples, len(loader))
    
    print(f"\nDataset Configuration:")
    print(f"  - Source: RefCOCO {split} split")
    print(f"  - Available samples: {len(loader)}")
    print(f"  - Samples to process: {n_samples}")
    print(f"  - Severities: {severities}")
    print(f"  - Degradation types: {degradation_types}")
    
    # Build list of all conditions
    conditions = ['clean']
    for deg_type in degradation_types:
        for sev in severities:
            conditions.append(f"{deg_type}_{sev}")
    
    total_images = n_samples * len(conditions)
    print(f"  - Total conditions: {len(conditions)}")
    print(f"  - Total images to generate: {total_images}")
    
    # Estimate disk space (~150KB per image average)
    estimated_gb = (total_images * 150 * 1024) / (1024**3)
    print(f"  - Estimated disk space: ~{estimated_gb:.1f} GB")
    
    # Create directories
    print(f"\nCreating directory structure...")
    for condition in conditions:
        img_dir = os.path.join(output_path, 'images', condition)
        os.makedirs(img_dir, exist_ok=True)
    
    os.makedirs(os.path.join(output_path, 'annotations'), exist_ok=True)
    
    # Storage for annotations
    annotations = []
    failed_samples = []
    
    # Process each sample
    print(f"\nGenerating images...")
    
    for idx in tqdm(range(n_samples), desc="Processing"):
        try:
            sample = loader[idx]
            image = sample['image']
            
            if image is None:
                failed_samples.append(idx)
                continue
            
            filename = f"{idx:06d}.jpg"
            
            # Save clean image
            clean_path = os.path.join(output_path, 'images', 'clean', filename)
            cv2.imwrite(clean_path, image)
            
            # Save degraded versions for each severity
            for sev in severities:
                pipeline = DegradationPipeline(severity=sev)
                
                for deg_type in degradation_types:
                    # Apply degradation
                    if deg_type == 'fog':
                        degraded = pipeline.add_fog(image)
                    elif deg_type == 'smoke':
                        degraded = pipeline.add_smoke(image)
                    elif deg_type == 'thermal':
                        degraded = pipeline.add_thermal(image)
                    else:
                        continue
                    
                    # Save
                    save_path = os.path.join(output_path, 'images', f"{deg_type}_{sev}", filename)
                    cv2.imwrite(save_path, degraded)
            
            # Store annotation
            bbox = sample['bbox']
            annotations.append({
                'id': idx,
                'filename': filename,
                'expression': sample['expression'],
                'all_expressions': sample['all_expressions'],
                'bbox': bbox,
                'bbox_xyxy': [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]],
                'image_width': image.shape[1],
                'image_height': image.shape[0],
                'ref_id': sample['ref_id'],
                'original_coco_id': sample['image_id']
            })
            
        except Exception as e:
            failed_samples.append(idx)
            print(f"\nError at sample {idx}: {e}")
            continue
    
    # Create dataset info
    dataset_info = {
        'info': {
            'name': 'RefCOCO-Degraded',
            'description': 'Visual Grounding Benchmark Under Adverse Imaging Conditions',
            'version': '1.0',
            'date_created': datetime.now().isoformat(),
            'source_dataset': 'RefCOCO',
            'source_images': 'MS COCO',
            'split': split,
            'num_samples': len(annotations),
            'num_failed': len(failed_samples),
            'severities': severities,
            'degradation_types': degradation_types,
            'conditions': conditions
        },
        'statistics': {
            'total_images': len(annotations) * len(conditions),
            'images_per_condition': len(annotations),
            'num_conditions': len(conditions)
        },
        'annotations': annotations
    }
    
    # Save main annotations file
    ann_path = os.path.join(output_path, 'annotations', 'annotations.json')
    with open(ann_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"\nSaved annotations to {ann_path}")
    
    # Create evaluation prompts for VLMs
    evaluation_prompts = []
    for ann in annotations:
        evaluation_prompts.append({
            'id': ann['id'],
            'filename': ann['filename'],
            'expression': ann['expression'],
            'ground_truth_bbox': ann['bbox'],
            'image_size': [ann['image_width'], ann['image_height']],
            'prompts': {
                'grounding': f"Locate '{ann['expression']}' in this image. Return the bounding box as [x, y, width, height] in pixels.",
                'detection': f"Is there '{ann['expression']}' in this image? Answer yes/no with confidence (0-100%).",
                'description': f"Describe what is at region [{int(ann['bbox'][0])}, {int(ann['bbox'][1])}, {int(ann['bbox'][0]+ann['bbox'][2])}, {int(ann['bbox'][1]+ann['bbox'][3])}].",
                'verification': f"Does the region [{int(ann['bbox'][0])}, {int(ann['bbox'][1])}, {int(ann['bbox'][0]+ann['bbox'][2])}, {int(ann['bbox'][1]+ann['bbox'][3])}] contain '{ann['expression']}'?"
            }
        })
    
    prompts_path = os.path.join(output_path, 'annotations', 'evaluation_prompts.json')
    with open(prompts_path, 'w') as f:
        json.dump(evaluation_prompts, f, indent=2)
    print(f"Saved evaluation prompts to {prompts_path}")
    
    # Create splits file
    splits_info = {
        'split': split,
        'num_samples': len(annotations),
        'sample_ids': [ann['id'] for ann in annotations],
        'conditions': conditions
    }
    splits_path = os.path.join(output_path, 'annotations', f'{split}_split.json')
    with open(splits_path, 'w') as f:
        json.dump(splits_info, f, indent=2)
    
    # Create README
    readme_content = f"""# RefCOCO-Degraded Benchmark Dataset

## Overview
A benchmark for evaluating visual grounding robustness under adverse imaging conditions.

## Dataset Statistics
| Property | Value |
|----------|-------|
| Source | RefCOCO ({split} split) |
| Samples | {len(annotations)} |
| Conditions | {len(conditions)} |
| Total Images | {len(annotations) * len(conditions)} |
| Severities | {severities} |
| Degradations | {degradation_types} |

## Directory Structure
```
{os.path.basename(output_path)}/
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
│   └── {split}_split.json     # Split information
└── README.md
```

## Annotation Format
```json
{{
  "id": 0,
  "filename": "000000.jpg",
  "expression": "the lady with the blue shirt",
  "bbox": [x, y, width, height],
  "bbox_xyxy": [x1, y1, x2, y2],
  "image_width": 640,
  "image_height": 480
}}
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
image_path = f'images/{{condition}}/{{sample["filename"]}}'
image = Image.open(image_path)
```

### Evaluating a Model
```python
for sample in annotations:
    for condition in conditions:
        image_path = f'images/{{condition}}/{{sample["filename"]}}'
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
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    readme_path = os.path.join(output_path, 'README.md')
    with open(readme_path, 'w') as f:
        f.write(readme_content)
    
    # Calculate actual size
    total_size = 0
    for root, dirs, files in os.walk(output_path):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    
    # Final summary
    print("\n" + "=" * 70)
    print("DATASET CREATION COMPLETE")
    print("=" * 70)
    print(f"\nLocation: {output_path}")
    print(f"\nContents:")
    print(f"  - Samples: {len(annotations)}")
    print(f"  - Conditions: {len(conditions)}")
    print(f"  - Total images: {len(annotations) * len(conditions)}")
    print(f"  - Failed samples: {len(failed_samples)}")
    print(f"  - Total size: {total_size / (1024**3):.2f} GB")
    print(f"\nFiles created:")
    print(f"  - annotations/annotations.json")
    print(f"  - annotations/evaluation_prompts.json")
    print(f"  - annotations/{split}_split.json")
    print(f"  - README.md")
    print(f"  - images/ ({len(conditions)} folders)")
    print("=" * 70)
    
    return dataset_info


# ==================== MAIN ====================
if __name__ == "__main__":
    
    # ===== CONFIGURATION =====
    REFCOCO_PATH = 'rq1_datasets/refcoco/refer/data/refcoco'
    COCO_IMAGES_PATH = 'rq1_datasets/coco/images/train2014/'
    OUTPUT_PATH = 'datasets/refcoco_degraded_benchmark/'
    
    # Dataset options
    SPLIT = 'val'                           # Use validation split
    SEVERITIES = [0.25, 0.5, 0.75, 1.0]          # Four severity levels
    DEGRADATION_TYPES = ['fog', 'smoke', 'thermal']
    MAX_SAMPLES = None                       # None = all samples, or set number like 500
    
    # Create dataset
    create_full_dataset(
        refcoco_path=REFCOCO_PATH,
        coco_images_path=COCO_IMAGES_PATH,
        output_path=OUTPUT_PATH,
        split=SPLIT,
        severities=SEVERITIES,
        degradation_types=DEGRADATION_TYPES,
        max_samples=MAX_SAMPLES
    )