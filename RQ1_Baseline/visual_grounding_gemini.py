"""
Evaluate RefCOCO-Degraded with Gemini 2.0 Flash
"""

# SETUP

# !pip install google-genai pillow tqdm matplotlib -q

import os
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Import Gemini
try:
    from google import genai
    USE_NEW_API = True
    print("✓ Using google-genai (new API)")
except ImportError:
    import google.generativeai as genai
    USE_NEW_API = False
    print("✓ Using google-generativeai")


# Dataset path 
DATASET_PATH = '/content/refcoco_degraded_benchmark'

# API Key - Set your Google API key
os.environ['GOOGLE_API_KEY'] = '[GCP_API_KEY]'

# Evaluation settings
NUM_SAMPLES = None  # None = all (3,811), or set number like 100 for testing
CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
REQUESTS_PER_MINUTE = 15  # Conservative rate limit
MAX_WORKERS = 4  # Parallel threads per sample


# HELPER FUNCTIONS

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x, y, w, h]"""
    if box1 is None or box2 is None:
        return 0.0
    
    x1_1, y1_1 = box1[0], box1[1]
    x2_1, y2_1 = box1[0] + box1[2], box1[1] + box1[3]
    x1_2, y1_2 = box2[0], box2[1]
    x2_2, y2_2 = box2[0] + box2[2], box2[1] + box2[3]
    
    x1_i, y1_i = max(x1_1, x1_2), max(y1_1, y1_2)
    x2_i, y2_i = min(x2_1, x2_2), min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1, area2 = box1[2] * box1[3], box2[2] * box2[3]
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def extract_bbox_normalized(text, img_w, img_h):
    """
    Converts Gemini's [ymin, xmin, ymax, xmax] (0-1000) 
    back to RefCOCO's [x, y, w, h] (pixels).
    """
    if not text:
        return None
    
    # Look for Gemini's typical bracketed output: [ymin, xmin, ymax, xmax]
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(pattern, text)
    
    if match:
        try:
            ymin, xmin, ymax, xmax = [int(v) for v in match.groups()]
            
            # Scale normalized 1000-scale back to pixels
            px_xmin = (xmin / 1000.0) * img_w
            px_ymin = (ymin / 1000.0) * img_h
            px_xmax = (xmax / 1000.0) * img_w
            px_ymax = (ymax / 1000.0) * img_h
            
            # Convert to RefCOCO format [x, y, width, height]
            return [px_xmin, px_ymin, px_xmax - px_xmin, px_ymax - px_ymin]
        except:
            return None
    return None


# EVALUATOR

class GeminiEvaluator:
    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = 'gemini-2.0-flash'
        self.request_delay = 4.0
        self.last_request = 0

    def _wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request = time.time()

    def predict(self, image_path, expression):
        self._wait()

        try:
            img = Image.open(image_path)
            img_w, img_h = img.size

            prompt = (
                f"Identify the object described as '{expression}'. "
                f"Return its bounding box in [ymin, xmin, ymax, xmax] format "
                f"using a normalized scale of 0-1000. Output ONLY the list."
            )

            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, img]
            )

            bbox = extract_bbox(response.text, img_w, img_h)

            return {
                'bbox': bbox,
                'response': response.text[:100],
                'success': True
            }
        except Exception as e:
            return {'bbox': None, 'response': str(e), 'success': False}


# CHECKPOINT FUNCTIONS

def load_checkpoint(results_dir):
    path = os.path.join(results_dir, 'rq1_checkpoint.json')
    if os.path.exists(path):
        with open(path) as f:
            ckpt = json.load(f)
        print(f"✓ Resuming from sample {ckpt['n_done']}")
        return ckpt['results'], ckpt['n_done']
    return None, 0


def save_checkpoint(results, n_done, results_dir):
    path = os.path.join(results_dir, 'rq1_checkpoint.json')
    with open(path, 'w') as f:
        json.dump({'n_done': n_done, 'results': results}, f)

# MAIN EVALUATION

def run_rq1_with_bboxes(dataset_path, num_samples=None, conditions=None, checkpoint_every=50):
    """
    Run RQ1 and save BOTH IoU AND predicted bboxes for RQ2.
    """

    # Load annotations
    print("Loading dataset...")
    ann_path = os.path.join(dataset_path, 'annotations', 'annotations.json')
    with open(ann_path) as f:
        data = json.load(f)

    annotations = data['annotations']
    if num_samples:
        annotations = annotations[:num_samples]

    if conditions is None:
        conditions = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']

    # Results directory
    results_dir = os.path.join(dataset_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load checkpoint
    checkpoint_results, start_idx = load_checkpoint(results_dir)

    # Initialize results
    if checkpoint_results:
        results = checkpoint_results
    else:
        results = {
            'samples': [],  # Will store per-sample data
            'summary': {}
        }

    # Evaluator
    evaluator = GeminiEvaluator()

    # Progress
    total = (len(annotations) - start_idx) * len(conditions)
    pbar = tqdm(total=total, desc="RQ1 Evaluation")

    
    print("RQ1: Evaluating Visual Grounding Under Degradation")
    print(f"  Samples: {len(annotations)}")
    print(f"  Conditions: {conditions}")
    print(f"  Saving: IoU + Predicted Bboxes (for RQ2)")
    

    # Evaluate
    for sample_idx, ann in enumerate(annotations[start_idx:], start=start_idx):

        sample_results = {
            'sample_id': sample_idx,
            'filename': ann['filename'],
            'expression': ann['expression'],
            'gt_bbox': ann['bbox'],
            'conditions': {}
        }

        for condition in conditions:
            image_path = os.path.join(dataset_path, 'images', condition, ann['filename'])

            if not os.path.exists(image_path):
                pbar.update(1)
                continue

            # Get prediction
            result = evaluator.predict(image_path, ann['expression'])

            # Calculate IoU
            iou = calculate_iou(result['bbox'], ann['bbox'])

            # Store BOTH IoU AND predicted bbox
            sample_results['conditions'][condition] = {
                'predicted_bbox': result['bbox'],  # ← SAVE THIS!
                'iou': iou,                        # ← AND THIS!
                'success': result['success']
            }

            pbar.update(1)
            pbar.set_postfix({'cond': condition[:6], 'iou': f'{iou:.2f}'})

        results['samples'].append(sample_results)

        # Checkpoint
        if (sample_idx + 1) % checkpoint_every == 0:
            save_checkpoint(results, sample_idx + 1, results_dir)
            pbar.write(f"  ✓ Checkpoint at {sample_idx + 1}")

    pbar.close()

    # Calculate summary
    print("\nCalculating summary...")
    summary = {}
    for condition in conditions:
        ious = []
        for sample in results['samples']:
            if condition in sample['conditions']:
                ious.append(sample['conditions'][condition]['iou'])

        if ious:
            summary[condition] = {
                'mean_iou': float(np.mean(ious)),
                'std_iou': float(np.std(ious)),
                'acc_0.5': float(np.mean([i >= 0.5 for i in ious])),
                'acc_0.7': float(np.mean([i >= 0.7 for i in ious])),
                'total': len(ious),
                'wrong_count': sum(1 for i in ious if i < 0.5)  # For RQ2
            }

    results['summary'] = summary

    # Print results
    print("\n" + "="*70)
    print("RQ1 RESULTS")
    print("="*70)
    print(f"\n{'Condition':<15} {'Mean IoU':<10} {'Acc@0.5':<10} {'Wrong (IoU<0.5)':<15}")
    print("-"*60)

    for cond in conditions:
        if cond in summary:
            s = summary[cond]
            print(f"{cond:<15} {s['mean_iou']:<10.3f} {s['acc_0.5']:<10.3f} {s['wrong_count']:<15}")

    print("-"*60)

    # Save final results
    save_path = os.path.join(results_dir, 'rq1_with_bboxes.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {save_path}")
    print(f"  → Contains IoU + Predicted Bboxes for RQ2")

    # Download
    try:
        from google.colab import files
        files.download(save_path)
    except:
        pass

    return results

# RUN
results = run_rq1_with_bboxes(
    dataset_path=DATASET_PATH,
    num_samples=NUM_SAMPLES,
    conditions=CONDITIONS,
    checkpoint_every=CHECKPOINT_EVERY
)

print("\n✓ RQ1 complete! Ready for RQ2.")
