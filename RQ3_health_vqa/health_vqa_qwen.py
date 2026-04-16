# SETUP
import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

#!pip install transformers accelerate pillow tqdm -q

# Check GPU
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB" if torch.cuda.is_available() else "")

# CONFIGURATION

DATASET_PATH = '/content/refcoco_degraded_benchmark'
ANNOTATIONS_PATH = '/content/refcoco_degraded_benchmark/rq3_health_annotations.json'
RESULTS_PATH = '/content/refcoco_degraded_benchmark/results'

CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']

# Model
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

print(f"Dataset: {DATASET_PATH}")
print(f"Annotations: {ANNOTATIONS_PATH}")

# LOAD MODEL

print("Loading Qwen2-VL-2B model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

model.eval()
print("✓ Model loaded!")

# HELPER FUNCTIONS

def crop_image(image, bbox):
    """
    Crop image to bounding box.
    bbox format: [x, y, width, height]
    """
    if bbox is None:
        return image

    x, y, w, h = [int(v) for v in bbox]
    img_w, img_h = image.size

    # Clamp to image bounds
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return image

    return image.crop((x, y, x + w, y + h))


def ask_posture_qwen(image, method="full"):
    """
    Ask Qwen2-VL about person's posture.
    Uses VQA-style prompt for best results.

    Args:
        image: PIL Image
        method: "full" or "cropped"

    Returns:
        str: STANDING, SITTING, LYING, or UNCLEAR
    """

    # VQA-style prompt (critical for good results)
    if method == "full":
        prompt = "Question: What is the posture of the person in this image? Answer with one word: STANDING, SITTING, or LYING. Answer:"
    else:
        prompt = "Question: What is this person's posture? Answer with one word: STANDING, SITTING, or LYING. Answer:"

    # Prepare inputs for Qwen2-VL
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

    # Generate - decode ONLY new tokens (critical fix!)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id
        )

    # Decode only the NEW tokens (not the input)
    input_len = inputs['input_ids'].shape[1]
    new_tokens = outputs[0][input_len:]
    response = processor.decode(new_tokens, skip_special_tokens=True).strip()

    # Parse response
    return parse_posture(response)


def parse_posture(text):
    """Parse posture from model response."""
    text = text.upper().strip()

    if "LYING" in text or "LIE" in text or "DOWN" in text or "HORIZONTAL" in text:
        return "LYING"
    elif "SITTING" in text or "SIT" in text or "SEATED" in text:
        return "SITTING"
    elif "STANDING" in text or "STAND" in text or "UPRIGHT" in text:
        return "STANDING"
    else:
        return "UNCLEAR"


print("✓ Helper functions defined")

# LOAD ANNOTATIONS
print("Loading health annotations...")

with open(ANNOTATIONS_PATH) as f:
    annotations = json.load(f)

samples = annotations['samples']
print(f"  → {len(samples)} total samples")

# Filter samples with posture annotations
valid_samples = [s for s in samples if s.get('posture') and s['posture'].strip() != ""]
print(f"  → {len(valid_samples)} samples with posture annotations")

if len(valid_samples) == 0:
    print("\n❌ ERROR: No annotated samples found!")
    print("   Please fill in the posture field in your annotation file first.")
else:
    print("✓ Annotations loaded!")

# RUN RQ3 EVALUATION

def run_rq3_qwen(dataset_path, valid_samples, conditions):
    """
    Run RQ3 evaluation for Qwen2-VL.
    Compares Full Image vs Cropped accuracy.
    """

    # Results storage
    results = {
        cond: {
            'posture': {'full': [], 'cropped': [], 'gt': []},
            'raw_responses': {'full': [], 'cropped': []}
        }
        for cond in conditions
    }

    total = len(valid_samples) * len(conditions)
    pbar = tqdm(total=total, desc="Qwen2-VL RQ3")


    print("RQ3: Health VQA - Qwen2-VL-2B")
    print(f"  Samples: {len(valid_samples)}")
    print(f"  Conditions: {conditions}")
    print(f"  Task: Posture Classification (STANDING/SITTING/LYING)")


    for sample in valid_samples:
        filename = sample['filename']
        gt_posture = sample['posture'].upper().strip()

        # Get bbox for cropping (use ground truth bbox)
        bbox = sample.get('gt_bbox') or sample.get('rq2_final_bbox')

        for condition in conditions:
            image_path = os.path.join(dataset_path, 'images', condition, filename)

            if not os.path.exists(image_path):
                pbar.update(1)
                continue

            try:
                # Load image
                img = Image.open(image_path).convert('RGB')

                # Method A: Full image
                full_response = ask_posture_qwen(img, method="full")

                # Method B: Cropped image
                cropped_img = crop_image(img, bbox)
                crop_response = ask_posture_qwen(cropped_img, method="cropped")

                # Store results
                results[condition]['posture']['full'].append(full_response)
                results[condition]['posture']['cropped'].append(crop_response)
                results[condition]['posture']['gt'].append(gt_posture)

                pbar.update(1)
                pbar.set_postfix({
                    'cond': condition[:6],
                    'full': full_response[:3],
                    'crop': crop_response[:3]
                })

            except Exception as e:
                print(f"\nError on {filename}: {e}")
                pbar.update(1)
                continue

    pbar.close()

    return results


# Run evaluation
print("Starting Qwen2-VL RQ3 evaluation...")
results = run_rq3_qwen(DATASET_PATH, valid_samples, CONDITIONS)

# RUN RQ3 EVALUATION
def run_rq3_qwen(dataset_path, valid_samples, conditions):
    """
    Run RQ3 evaluation for Qwen2-VL.
    Compares Full Image vs Cropped accuracy.
    """

    # Results storage
    results = {
        cond: {
            'posture': {'full': [], 'cropped': [], 'gt': []},
            'raw_responses': {'full': [], 'cropped': []}
        }
        for cond in conditions
    }

    total = len(valid_samples) * len(conditions)
    pbar = tqdm(total=total, desc="Qwen2-VL RQ3")


    print("RQ3: Health VQA - Qwen2-VL-2B")
    print(f"  Samples: {len(valid_samples)}")
    print(f"  Conditions: {conditions}")
    print(f"  Task: Posture Classification (STANDING/SITTING/LYING)")


    for sample in valid_samples:
        filename = sample['filename']
        gt_posture = sample['posture'].upper().strip()

        # Get bbox for cropping (use ground truth bbox)
        bbox = sample.get('gt_bbox') or sample.get('rq2_final_bbox')

        for condition in conditions:
            image_path = os.path.join(dataset_path, 'images', condition, filename)

            if not os.path.exists(image_path):
                pbar.update(1)
                continue

            try:
                # Load image
                img = Image.open(image_path).convert('RGB')

                # Method A: Full image
                full_response = ask_posture_qwen(img, method="full")

                # Method B: Cropped image
                cropped_img = crop_image(img, bbox)
                crop_response = ask_posture_qwen(cropped_img, method="cropped")

                # Store results
                results[condition]['posture']['full'].append(full_response)
                results[condition]['posture']['cropped'].append(crop_response)
                results[condition]['posture']['gt'].append(gt_posture)

                pbar.update(1)
                pbar.set_postfix({
                    'cond': condition[:6],
                    'full': full_response[:3],
                    'crop': crop_response[:3]
                })

            except Exception as e:
                print(f"\nError on {filename}: {e}")
                pbar.update(1)
                continue

    pbar.close()

    return results


# Run evaluation
print("Starting Qwen2-VL RQ3 evaluation...")
results = run_rq3_qwen(DATASET_PATH, valid_samples, CONDITIONS)

# CALCULATE ACCURACY
def calculate_accuracy(results, conditions):
    """Calculate accuracy for each condition and method."""
    summary = {}

    print("RQ3 RESULTS: Qwen2-VL Posture Classification Accuracy")
    print(f"\n{'Condition':<15} {'Full Image':<15} {'Cropped':<15} {'Change':<10}")


    for cond in conditions:
        full_preds = results[cond]['posture']['full']
        crop_preds = results[cond]['posture']['cropped']
        gt_labels = results[cond]['posture']['gt']

        if len(gt_labels) == 0:
            continue

        # Calculate accuracy
        full_correct = sum(1 for p, g in zip(full_preds, gt_labels) if p == g)
        crop_correct = sum(1 for p, g in zip(crop_preds, gt_labels) if p == g)

        full_acc = full_correct / len(gt_labels) * 100
        crop_acc = crop_correct / len(gt_labels) * 100
        change = crop_acc - full_acc

        summary[cond] = {
            'full_accuracy': full_acc,
            'cropped_accuracy': crop_acc,
            'change': change,
            'n': len(gt_labels)
        }

        marker = "✓" if change > 0 else ("✗" if change < -5 else "~")
        print(f"{cond:<15} {full_acc:<15.1f} {crop_acc:<15.1f} {change:+.1f}% {marker}")

    # Average
    if summary:
        avg_full = np.mean([summary[c]['full_accuracy'] for c in conditions if c in summary])
        avg_crop = np.mean([summary[c]['cropped_accuracy'] for c in conditions if c in summary])
        avg_change = avg_crop - avg_full

        print(f"{'AVERAGE':<15} {avg_full:<15.1f} {avg_crop:<15.1f} {avg_change:+.1f}%")

        summary['average'] = {
            'full_accuracy': avg_full,
            'cropped_accuracy': avg_crop,
            'change': avg_change
        }

    return summary

summary = calculate_accuracy(results, CONDITIONS)

#  SAVE RESULTS
os.makedirs(RESULTS_PATH, exist_ok=True)

output = {
    'metadata': {
        'model': 'Qwen2-VL-2B-Instruct',
        'experiment': 'RQ3 - Health VQA Posture Classification',
        'timestamp': datetime.now().isoformat(),
        'num_samples': len(valid_samples),
        'conditions': CONDITIONS
    },
    'summary': summary,
    'raw': {
        cond: {
            'full': results[cond]['posture']['full'],
            'cropped': results[cond]['posture']['cropped'],
            'gt': results[cond]['posture']['gt']
        }
        for cond in CONDITIONS
    }
}

save_path = os.path.join(RESULTS_PATH, 'rq3_qwen2vl_results.json')
with open(save_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\n✓ Results saved to {save_path}")

# VISUALIZATION

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('RQ3: Qwen2-VL Posture Classification - Full vs Cropped',
             fontsize=14, fontweight='bold')

valid_conds = [c for c in CONDITIONS if c in summary and c != 'average']

# Plot 1: Accuracy comparison
ax1 = axes[0]
x = np.arange(len(valid_conds))
width = 0.35

full_accs = [summary[c]['full_accuracy'] for c in valid_conds]
crop_accs = [summary[c]['cropped_accuracy'] for c in valid_conds]

bars1 = ax1.bar(x - width/2, full_accs, width, label='Full Image', color='#95a5a6', alpha=0.85)
bars2 = ax1.bar(x + width/2, crop_accs, width, label='Cropped', color='#9b59b6', alpha=0.85)

ax1.set_xlabel('Condition')
ax1.set_ylabel('Posture Accuracy (%)')
ax1.set_title('Accuracy: Full Image vs Cropped')
ax1.set_xticks(x)
ax1.set_xticklabels([c.replace('_0.5', '') for c in valid_conds])
ax1.legend()
ax1.set_ylim(0, 100)

for bar, acc in zip(bars1, full_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.0f}%', ha='center', fontsize=9)
for bar, acc in zip(bars2, crop_accs):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.0f}%', ha='center', fontsize=9, fontweight='bold')

# Plot 2: Change
ax2 = axes[1]
changes = [summary[c]['change'] for c in valid_conds]
colors = ['#27ae60' if c > 0 else '#e74c3c' for c in changes]

bars = ax2.bar(range(len(valid_conds)), changes, color=colors, alpha=0.85, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xticks(range(len(valid_conds)))
ax2.set_xticklabels([c.replace('_0.5', '') for c in valid_conds])
ax2.set_ylabel('Accuracy Change (%)')
ax2.set_title('Effect of Cropping')

for bar, change in zip(bars, changes):
    color = '#27ae60' if change > 0 else '#c0392b'
    offset = 1 if change > 0 else -2
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + offset,
             f'{change:+.1f}%', ha='center', fontsize=11, fontweight='bold', color=color)

plt.tight_layout()

plot_path = os.path.join(RESULTS_PATH, 'rq3_qwen2vl_results.png')
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to {plot_path}")
plt.show()

# FINAL SUMMARY
print("RQ3 QWEN2-VL EVALUATION COMPLETE")

print(f"""
MODEL: Qwen2-VL-2B-Instruct
TASK: Posture Classification (STANDING/SITTING/LYING)
SAMPLES: {len(valid_samples)}

RESULTS:
""")

for cond in CONDITIONS:
    if cond in summary and cond != 'average':
        s = summary[cond]
        marker = "✓" if s['change'] > 0 else ("⚠️" if s['change'] < -10 else "~")
        print(f"  {cond:<15}: Full {s['full_accuracy']:.1f}% → Crop {s['cropped_accuracy']:.1f}% ({s['change']:+.1f}%) {marker}")

if 'average' in summary:
    avg = summary['average']
    print(f"\n  {'AVERAGE':<15}: Full {avg['full_accuracy']:.1f}% → Crop {avg['cropped_accuracy']:.1f}% ({avg['change']:+.1f}%)")

print(f"""
FILES SAVED:
  - {save_path}
  - {plot_path}
""")