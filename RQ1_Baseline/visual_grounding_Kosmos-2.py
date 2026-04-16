# Setup and install packages
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import re
import time
from datetime import datetime
import matplotlib.pyplot as plt
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
RQ1: Visual Grounding with Kosmos-2
Microsoft's Kosmos-2 model for visual grounding evaluation.
"""

# Configuration

DATASET_PATH = '/content/drive/MyDrive/TheDay/Projet_The_Day/HorusEye/horuseye_VLM/refcoco_degraded_benchmark'
CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
NUM_SAMPLES = None  # None = all
CHECKPOINT_EVERY = 50
MODEL_NAME = "kosmos2"

print(f"✓ Configuration set")
print(f"  Dataset: {DATASET_PATH}")
print(f"  Samples: {NUM_SAMPLES if NUM_SAMPLES else 'All'}")


# Load model

print("Loading Kosmos-2...")

processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
model = AutoModelForImageTextToText.from_pretrained(
    "microsoft/kosmos-2-patch14-224",
    torch_dtype=torch.float16
).to("cuda")

model.eval()
print(f"✓ Kosmos-2 loaded on {model.device}!")


# Define functions

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def predict_bbox_kosmos2(image_path, expression, model, processor):
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    prompt = f"<image> User: <grounding><phrase> {expression} </phrase> Assistant:"
    inputs = processor(
        text=prompt,
        images=image,
        return_tensors="pt"
    )

    inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    if 'pixel_values' in inputs:
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.float16)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            use_cache=True,
            max_new_tokens=64,
            repetition_penalty=1.1,
            suppress_tokens=[processor.tokenizer.pad_token_id]
        )

    # Decode the generated text
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Post-process to extract entities and bboxes
    try:
        processed_text, entities = processor.post_process_generation(generated_text)
    except Exception as e:
        entities = None
    bbox = None

    # Method 1: Use processor's entities
    if entities:
        for entity in entities:
            if len(entity) >= 3:
                entity_name, span, boxes = entity
                if boxes and len(boxes) > 0:
                    box = boxes[0]
                    if len(box) == 4:
                        x1, y1, x2, y2 = box
                        x1 = x1 * img_width
                        y1 = y1 * img_height
                        x2 = x2 * img_width
                        y2 = y2 * img_height
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        break
            elif len(entity) >= 2:
                entity_name, boxes = entity[0], entity[1]
                if boxes:
                    if isinstance(boxes, tuple) and len(boxes) == 4:
                        x1, y1, x2, y2 = boxes
                        x1 = x1 * img_width
                        y1 = y1 * img_height
                        x2 = x2 * img_width
                        y2 = y2 * img_height
                        bbox = [x1, y1, x2 - x1, y2 - y1]
                        break
                    elif isinstance(boxes, list) and len(boxes) > 0:
                        box = boxes[0]
                        if len(box) == 4:
                            x1, y1, x2, y2 = box
                            x1 = x1 * img_width
                            y1 = y1 * img_height
                            x2 = x2 * img_width
                            y2 = y2 * img_height
                            bbox = [x1, y1, x2 - x1, y2 - y1]
                            break

    # Method 2: Manual parsing if entities failed
    if bbox is None:
        patch_pattern = r'<patch_index_(\d+)><patch_index_(\d+)>'
        matches = re.findall(patch_pattern, generated_text)

        if matches:
            try:
                idx1, idx2 = int(matches[0][0]), int(matches[0][1])
                row1, col1 = idx1 // 32, idx1 % 32
                row2, col2 = idx2 // 32, idx2 % 32

                x1 = col1 / 32 * img_width
                y1 = row1 / 32 * img_height
                x2 = (col2 + 1) / 32 * img_width
                y2 = (row2 + 1) / 32 * img_height

                bbox = [x1, y1, x2 - x1, y2 - y1]
            except:
                pass

        # Method 3: Look for <box> tags
        box_pattern = r'<box>(\d+),\s*(\d+),\s*(\d+),\s*(\d+)</box>'
        match = re.search(box_pattern, generated_text)
        if match and bbox is None:
            x1, y1, x2, y2 = map(int, match.groups())
            if max(x1, y1, x2, y2) <= 1000:
                x1 = x1 / 1000 * img_width
                y1 = y1 / 1000 * img_height
                x2 = x2 / 1000 * img_width
                y2 = y2 / 1000 * img_height
            bbox = [x1, y1, x2 - x1, y2 - y1]

        # Method 4: Look for coordinates in brackets
        bracket_pattern = r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]'
        match = re.search(bracket_pattern, generated_text)
        if match and bbox is None:
            x1, y1, x2, y2 = map(float, match.groups())
            if max(x1, y1, x2, y2) <= 1.0:
                x1 *= img_width
                y1 *= img_height
                x2 *= img_width
                y2 *= img_height
            elif max(x1, y1, x2, y2) <= 1000:
                x1 = x1 / 1000 * img_width
                y1 = y1 / 1000 * img_height
                x2 = x2 / 1000 * img_width
                y2 = y2 / 1000 * img_height
            bbox = [x1, y1, x2 - x1, y2 - y1]

    return bbox, generated_text


def load_dataset(dataset_path, num_samples=None):
    """Load dataset from RQ1 results file."""
    rq1_path = os.path.join(dataset_path, 'results', 'rq1_with_bboxes.json')

    with open(rq1_path) as f:
        rq1_data = json.load(f)

    samples = rq1_data['samples']

    for s in samples:
        if 'gt_bbox' in s and 'bbox' not in s:
            s['bbox'] = s['gt_bbox']

    if num_samples:
        samples = samples[:num_samples]

    print(f"✓ Loaded {len(samples)} samples")
    return samples


def run_evaluation(dataset_path, conditions, num_samples=None, checkpoint_every=50):
    """Run RQ1 evaluation with Kosmos-2."""

    results_dir = os.path.join(dataset_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(results_dir, f'rq1_{MODEL_NAME}_checkpoint.json')

    start_idx = 0
    condition_ious = {cond: [] for cond in conditions}
    processed_samples = []

    if os.path.exists(checkpoint_path):
        print("\n✓ Found checkpoint, resuming...")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        start_idx = checkpoint.get('last_index', 0) + 1
        condition_ious = checkpoint.get('condition_ious', condition_ious)
        processed_samples = checkpoint.get('samples', [])
        print(f"  Resuming from sample {start_idx}")
        print(f"  Already processed: {len(processed_samples)} samples")

        # Print current stats
        for cond in conditions:
            if condition_ious[cond]:
                print(f"  {cond}: Mean IoU = {np.mean(condition_ious[cond]):.4f}")

    samples = load_dataset(dataset_path, num_samples)

    print(f"\n  Total samples: {len(samples)}")
    print(f"  Remaining: {len(samples) - start_idx}")

    total = len(samples) * len(conditions)
    completed = start_idx * len(conditions)
    pbar = tqdm(total=total, initial=completed, desc=f"RQ1 {MODEL_NAME}")

    errors = 0
    debug_count = 0

    for idx, sample in enumerate(samples):
        if idx < start_idx:
            continue

        sample_id = sample['sample_id']
        filename = sample['filename']
        expression = sample['expression']
        gt_bbox = sample.get('bbox') or sample.get('gt_bbox')

        sample_result = {
            'sample_id': sample_id,
            'filename': filename,
            'expression': expression,
            'gt_bbox': gt_bbox,
            'conditions': {}
        }

        for condition in conditions:
            image_path = os.path.join(dataset_path, 'images', condition, filename)

            if not os.path.exists(image_path):
                pbar.update(1)
                continue

            try:
                pred_bbox, response = predict_bbox_kosmos2(
                    image_path, expression, model, processor
                )

                # Debug: Print first few results
                if debug_count < 5 and condition == 'clean':
                    print(f"\n[DEBUG] Sample {idx}, {condition}")
                    print(f"  Expression: {expression}")
                    print(f"  Response: {response[:200]}...")
                    print(f"  Pred bbox: {pred_bbox}")
                    print(f"  GT bbox: {gt_bbox}")
                    debug_count += 1

                if pred_bbox:
                    iou = calculate_iou(pred_bbox, gt_bbox)
                else:
                    iou = 0.0
                    pred_bbox = [0, 0, 0, 0]

                sample_result['conditions'][condition] = {
                    'predicted_bbox': pred_bbox,
                    'iou': iou,
                    'response': response[:500]  # Store truncated response for debugging
                }
                condition_ious[condition].append(iou)

            except Exception as e:
                errors += 1
                sample_result['conditions'][condition] = {
                    'predicted_bbox': None,
                    'iou': 0.0,
                    'error': str(e)
                }
                condition_ious[condition].append(0.0)

                if errors <= 3:
                    print(f"\n[ERROR] {filename} ({condition}): {e}")

            pbar.update(1)

            if len(condition_ious[condition]) > 0:
                avg_iou = np.mean(condition_ious[condition])
                pbar.set_postfix({'cond': condition[:6], 'IoU': f'{avg_iou:.3f}', 'err': errors})

        processed_samples.append(sample_result)

        # Save checkpoint
        if (idx + 1) % checkpoint_every == 0:
            checkpoint = {
                'last_index': idx,
                'condition_ious': {k: [float(v) for v in vals] for k, vals in condition_ious.items()},
                'samples': processed_samples,
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
                f.flush()
                os.fsync(f.fileno())
            tqdm.write(f"  ✓ Checkpoint saved at sample {idx + 1}")

    pbar.close()

    # Print results
    summary = {}
    print(f"\n{'Condition':<15} {'Mean IoU':<12} {'Std':<10} {'N':<8}")
    
    for condition in conditions:
        ious = condition_ious[condition]
        if len(ious) > 0:
            mean_iou = np.mean(ious)
            std_iou = np.std(ious)
            summary[condition] = {
                'mean_iou': float(mean_iou),
                'std_iou': float(std_iou),
                'n': len(ious)
            }
            print(f"{condition:<15} {mean_iou:<12.4f} {std_iou:<10.4f} {len(ious):<8}")

    # Save results
    results = {
        'metadata': {
            'model': 'Kosmos-2-patch14-224',
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(processed_samples),
            'conditions': conditions,
            'errors': errors
        },
        'summary': summary,
        'samples': processed_samples
    }

    save_path = os.path.join(results_dir, f'rq1_{MODEL_NAME}_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("✓ Checkpoint deleted")

    return results, summary

# Test on 5 samples to verify bbox parsing works
print("Testing Kosmos-2 on 5 samples...")

test_samples = load_dataset(DATASET_PATH, num_samples=5)

for i, sample in enumerate(test_samples):
    filename = sample['filename']
    expression = sample['expression']
    gt_bbox = sample.get('bbox') or sample.get('gt_bbox')

    image_path = os.path.join(DATASET_PATH, 'images', 'clean', filename)

    if os.path.exists(image_path):
        bbox, response = predict_bbox_kosmos2(image_path, expression, model, processor)

        if bbox:
            iou = calculate_iou(bbox, gt_bbox)
        else:
            iou = 0.0

        print(f"\n[Test {i+1}]")
        print(f"  Expression: {expression}")
        print(f"  GT bbox: {gt_bbox}")
        print(f"  Pred bbox: {bbox}")
        print(f"  IoU: {iou:.4f}")
        print(f"  Response: {response[:150]}...")


print("If IoU > 0 for most samples, proceed to full evaluation.")
print("If IoU = 0 for all samples, there's a parsing issue.")


# Run full evaluation
results, summary = run_evaluation(
    dataset_path=DATASET_PATH,
    conditions=CONDITIONS,
    num_samples=NUM_SAMPLES,
    checkpoint_every=CHECKPOINT_EVERY
)

print("\n✓ Evaluation complete!")