"""
RQ1: Visual Grounding Degradation with Qwen2-VL-2B

Compare Qwen2-VL-2B vs Gemini on RefCOCO-Degraded benchmark.
"""

# SETUP & INSTALL
#!pip install transformers accelerate bitsandbytes pillow tqdm -q
#!pip install qwen-vl-utils -q

import os
import json
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import re

#from google.colab import drive
#drive.mount('/content/drive')

print("✓ Packages installed")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# CONFIGURATION
# === CHANGE THIS PATH ===
DATASET_PATH = '/HorusEye/horuseye_VLM/refcoco_degraded_benchmark'

CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
NUM_SAMPLES = None  # None = all samples, or set a number like 100
CHECKPOINT_EVERY = 50  # Save checkpoint every N samples

print("✓ Configuration set")
print(f"  DATASET_PATH: {DATASET_PATH}")
print(f"  CONDITIONS: {CONDITIONS}")
print(f"  NUM_SAMPLES: {NUM_SAMPLES if NUM_SAMPLES else 'All'}")
print(f"  CHECKPOINT_EVERY: {CHECKPOINT_EVERY}")


# LOAD MODEL
print("Loading Qwen2-VL-2B model...")
print("  This may take 2-3 minutes...")

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    trust_remote_code=True
)

print("✓ Model loaded!")
print(f"  Model dtype: {model.dtype}")
print(f"  Device: {model.device}")


# DEFINE FUNCTIONS

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Box format: [x, y, width, height]
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to [x1, y1, x2, y2]
    box1_x2, box1_y2 = x1 + w1, y1 + h1
    box2_x2, box2_y2 = x2 + w2, y2 + h2

    # Intersection
    inter_x1 = max(x1, x2)
    inter_y1 = max(y1, y2)
    inter_x2 = min(box1_x2, box2_x2)
    inter_y2 = min(box1_y2, box2_y2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    # Union
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def parse_bbox_from_response(response_text, image_width, image_height):
    """
    Parse bounding box from Qwen2-VL response.
    Handles multiple output formats.
    """

    # Pattern 1: <box>(x1, y1), (x2, y2)</box>
    box_pattern = r'<box>\s*\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\s*</box>'
    match = re.search(box_pattern, response_text)

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        # Qwen uses 0-1000 normalized coordinates
        x1 = x1 / 1000 * image_width
        y1 = y1 / 1000 * image_height
        x2 = x2 / 1000 * image_width
        y2 = y2 / 1000 * image_height
        return [x1, y1, x2 - x1, y2 - y1]

    # Pattern 2: <ref>object</ref><box>(x1, y1), (x2, y2)</box>
    ref_box_pattern = r'<ref>.*?</ref>\s*<box>\s*\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\s*</box>'
    match = re.search(ref_box_pattern, response_text)

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        x1 = x1 / 1000 * image_width
        y1 = y1 / 1000 * image_height
        x2 = x2 / 1000 * image_width
        y2 = y2 / 1000 * image_height
        return [x1, y1, x2 - x1, y2 - y1]

    # Pattern 3: [x1, y1, x2, y2]
    bracket_pattern = r'\[(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*),\s*(\d+\.?\d*)\]'
    match = re.search(bracket_pattern, response_text)

    if match:
        x1, y1, x2, y2 = map(float, match.groups())
        if max(x1, y1, x2, y2) <= 1.0:
            x1 *= image_width
            y1 *= image_height
            x2 *= image_width
            y2 *= image_height
        elif max(x1, y1, x2, y2) <= 1000:
            x1 = x1 / 1000 * image_width
            y1 = y1 / 1000 * image_height
            x2 = x2 / 1000 * image_width
            y2 = y2 / 1000 * image_height
        return [x1, y1, x2 - x1, y2 - y1]

    # Pattern 4: Plain numbers (x1, y1, x2, y2)
    num_pattern = r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)'
    match = re.search(num_pattern, response_text)

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        if max(x1, y1, x2, y2) <= 1000:
            x1 = x1 / 1000 * image_width
            y1 = y1 / 1000 * image_height
            x2 = x2 / 1000 * image_width
            y2 = y2 / 1000 * image_height
        return [x1, y1, x2 - x1, y2 - y1]

    return None


def predict_bbox_qwen(image_path, expression, model, processor):
    """
    Use Qwen2-VL to predict bounding box for expression.
    """

    # Load image
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    # Create grounding prompt
    prompt = f"""Locate the object: "{expression}"

Output the bounding box in this format:
<box>(x1, y1), (x2, y2)</box>

Use coordinates from 0 to 1000."""

    # Prepare input
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ]
        }
    ]

    # Process
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False
        )

    # Decode
    response = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # Extract assistant response
    if "assistant" in response.lower():
        response = response.split("assistant")[-1].strip()

    # Parse bbox
    bbox = parse_bbox_from_response(response, img_width, img_height)

    return bbox, response


def load_dataset(dataset_path, num_samples=None):
    """Load dataset from RQ1 results file."""

    rq1_path = os.path.join(dataset_path, 'results', 'rq1_with_bboxes.json')

    with open(rq1_path) as f:
        rq1_data = json.load(f)

    samples = rq1_data['samples']

    # Ensure 'bbox' key exists
    for s in samples:
        if 'gt_bbox' in s and 'bbox' not in s:
            s['bbox'] = s['gt_bbox']

    if num_samples:
        samples = samples[:num_samples]

    print(f"✓ Loaded {len(samples)} samples from rq1_with_bboxes.json")

    return samples


def run_rq1_evaluation(dataset_path, conditions, num_samples=None, checkpoint_every=50):
    """
    Run RQ1 evaluation with checkpoints.
    Saves progress every checkpoint_every samples.
    Auto-resumes from checkpoint if exists.
    """
    print("RQ1: Visual Grounding with Qwen2-VL-2B")
 

    results_dir = os.path.join(dataset_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(results_dir, 'rq1_qwen2vl_checkpoint.json')

    # Load checkpoint if exists
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

    # Load dataset
    samples = load_dataset(dataset_path, num_samples)

    print(f"\n  Total samples: {len(samples)}")
    print(f"  Remaining: {len(samples) - start_idx}")
    print(f"  Conditions: {conditions}")
    print(f"  Checkpoint every: {checkpoint_every} samples")

    # Progress bar
    total = len(samples) * len(conditions)
    completed = start_idx * len(conditions)
    pbar = tqdm(total=total, initial=completed, desc="RQ1 Qwen2-VL")

    errors = 0

    for idx, sample in enumerate(samples):

        # Skip already processed
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
                # Predict
                pred_bbox, response = predict_bbox_qwen(image_path, expression, model, processor)

                if pred_bbox:
                    iou = calculate_iou(pred_bbox, gt_bbox)
                else:
                    iou = 0.0
                    pred_bbox = [0, 0, 0, 0]

                sample_result['conditions'][condition] = {
                    'predicted_bbox': pred_bbox,
                    'iou': iou
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

            pbar.update(1)

            # Update progress display
            if len(condition_ious[condition]) > 0:
                avg_iou = np.mean(condition_ious[condition])
                pbar.set_postfix({
                    'cond': condition[:6],
                    'IoU': f'{avg_iou:.3f}',
                    'err': errors
                })

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
            tqdm.write(f"  ✓ Checkpoint saved at sample {idx + 1}")

    pbar.close()

    # Calculate summary statistics
    print("RQ1 RESULTS: Qwen2-VL-2B")

    summary = {}

    print(f"\n{'Condition':<15} {'Mean IoU':<12} {'Std':<10} {'N':<8}")
    print("-"*50)

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


    # Save final results
    results = {
        'metadata': {
            'model': 'Qwen2-VL-2B-Instruct',
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(processed_samples),
            'conditions': conditions
        },
        'summary': summary,
        'samples': processed_samples
    }

    save_path = os.path.join(results_dir, 'rq1_qwen2vl_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {save_path}")

    # Delete checkpoint after successful completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("✓ Checkpoint deleted (evaluation complete)")

    return results, summary


def compare_models(dataset_path):
    """
    Compare Qwen2-VL-2B results with Gemini results.
    """

    results_dir = os.path.join(dataset_path, 'results')

    # Load Gemini results
    gemini_path = os.path.join(results_dir, 'rq1_with_bboxes.json')
    qwen_path = os.path.join(results_dir, 'rq1_qwen2vl_results.json')

    if not os.path.exists(gemini_path):
        print("❌ Gemini results not found")
        return None

    if not os.path.exists(qwen_path):
        print("❌ Qwen2-VL results not found. Run evaluation first.")
        return None

    with open(gemini_path) as f:
        gemini_data = json.load(f)

    with open(qwen_path) as f:
        qwen_data = json.load(f)

    print("MODEL COMPARISON: Gemini vs Qwen2-VL-2B")

    conditions = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']

    # Calculate Gemini IoUs from samples
    gemini_ious = {cond: [] for cond in conditions}
    for sample in gemini_data.get('samples', []):
        for cond in conditions:
            if cond in sample:
                gemini_ious[cond].append(sample[cond].get('iou', 0))

    print(f"\n{'Condition':<15} {'Gemini IoU':<15} {'Qwen2-VL IoU':<15} {'Difference':<12}")
    print("-"*60)

    comparison = {}

    for cond in conditions:
        # Gemini mean
        gemini_mean = np.mean(gemini_ious[cond]) if gemini_ious[cond] else 0

        # Qwen mean
        qwen_mean = qwen_data.get('summary', {}).get(cond, {}).get('mean_iou', 0)

        diff = qwen_mean - gemini_mean

        comparison[cond] = {
            'gemini': float(gemini_mean),
            'qwen': float(qwen_mean),
            'diff': float(diff)
        }

        diff_str = f"{diff:+.4f}"
        print(f"{cond:<15} {gemini_mean:<15.4f} {qwen_mean:<15.4f} {diff_str:<12}")

    print("-"*60)

    # Calculate averages
    avg_gemini = np.mean([comparison[c]['gemini'] for c in conditions])
    avg_qwen = np.mean([comparison[c]['qwen'] for c in conditions])
    avg_diff = avg_qwen - avg_gemini

    print(f"{'AVERAGE':<15} {avg_gemini:<15.4f} {avg_qwen:<15.4f} {avg_diff:+.4f}")

    # Plot comparison
    plot_comparison(comparison, conditions, results_dir)

    # Save comparison
    comparison_path = os.path.join(results_dir, 'rq1_model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    print(f"\n✓ Comparison saved to {comparison_path}")

    return comparison


def plot_comparison(comparison, conditions, results_dir):
    """Plot model comparison."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RQ1: Visual Grounding - Model Comparison', fontsize=14, fontweight='bold')

    # Plot 1: Side by side bars
    ax1 = axes[0]
    x = np.arange(len(conditions))
    width = 0.35

    gemini_ious = [comparison[c]['gemini'] for c in conditions]
    qwen_ious = [comparison[c]['qwen'] for c in conditions]

    bars1 = ax1.bar(x - width/2, gemini_ious, width, label='Gemini 2.0 Flash', color='#4285F4')
    bars2 = ax1.bar(x + width/2, qwen_ious, width, label='Qwen2-VL-2B', color='#9C27B0')

    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Mean IoU')
    ax1.set_title('Mean IoU by Condition')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_0.5', '') for c in conditions])
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    for bar, val in zip(bars1, gemini_ious):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)
    for bar, val in zip(bars2, qwen_ious):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', fontsize=9)

    # Plot 2: Degradation comparison (relative to clean)
    ax2 = axes[1]

    gemini_clean = comparison['clean']['gemini']
    qwen_clean = comparison['clean']['qwen']

    gemini_degradation = [(comparison[c]['gemini'] - gemini_clean) / gemini_clean * 100
                          if gemini_clean > 0 else 0 for c in conditions]
    qwen_degradation = [(comparison[c]['qwen'] - qwen_clean) / qwen_clean * 100
                        if qwen_clean > 0 else 0 for c in conditions]

    x = np.arange(len(conditions))
    bars1 = ax2.bar(x - width/2, gemini_degradation, width, label='Gemini 2.0 Flash', color='#4285F4')
    bars2 = ax2.bar(x + width/2, qwen_degradation, width, label='Qwen2-VL-2B', color='#9C27B0')

    ax2.set_xlabel('Condition')
    ax2.set_ylabel('% Change from Clean')
    ax2.set_title('Degradation Impact (% change from clean)')
    ax2.set_xticks(x)
    ax2.set_xticklabels([c.replace('_0.5', '') for c in conditions])
    ax2.legend()
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    for bar, val in zip(bars1, gemini_degradation):
        color = 'green' if val >= 0 else 'red'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:+.1f}%', ha='center', fontsize=8, color=color)
    for bar, val in zip(bars2, qwen_degradation):
        color = 'green' if val >= 0 else 'red'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:+.1f}%', ha='center', fontsize=8, color=color)

    plt.tight_layout()

    save_path = os.path.join(results_dir, 'rq1_model_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Comparison plot saved to {save_path}")
    plt.show()


print("✓ Functions defined")


# RUN EVALUATION

print("Starting RQ1 evaluation with Qwen2-VL-2B...")
print("This will auto-resume from checkpoint if runtime disconnected.\n")

results, summary = run_rq1_evaluation(
    dataset_path=DATASET_PATH,
    conditions=CONDITIONS,
    num_samples=NUM_SAMPLES,
    checkpoint_every=CHECKPOINT_EVERY
)

print("\n✓ Evaluation complete!")


# COMPARE WITH GEMINI

print("Comparing Qwen2-VL-2B with Gemini results...\n")

comparison = compare_models(DATASET_PATH)

if comparison:
    print("\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    # Which model is better overall?
    avg_gemini = np.mean([comparison[c]['gemini'] for c in CONDITIONS])
    avg_qwen = np.mean([comparison[c]['qwen'] for c in CONDITIONS])

    if avg_gemini > avg_qwen:
        print(f"\n  Gemini outperforms Qwen2-VL by {(avg_gemini - avg_qwen):.4f} IoU on average")
    else:
        print(f"\n  Qwen2-VL outperforms Gemini by {(avg_qwen - avg_gemini):.4f} IoU on average")

    # Which is more robust to degradation?
    gemini_drop = comparison['clean']['gemini'] - comparison['thermal_0.5']['gemini']
    qwen_drop = comparison['clean']['qwen'] - comparison['thermal_0.5']['qwen']

    print(f"\n  Degradation impact (clean → thermal):")
    print(f"    Gemini: -{gemini_drop:.4f} IoU")
    print(f"    Qwen2-VL: -{qwen_drop:.4f} IoU")

    if gemini_drop < qwen_drop:
        print(f"\n  → Gemini is more ROBUST to degradation")
    else:
        print(f"\n  → Qwen2-VL is more ROBUST to degradation")

print("\n✓ Analysis complete!")