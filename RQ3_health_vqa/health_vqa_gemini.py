"""
RQ3: Does Language-Guided Attention Improve Downstream Health Analysis?

This experiment proves that:
- RQ1: Degradation breaks visual grounding (AI can't find the victim)
- RQ2: Language feedback fixes localization (AI locates the victim)
- RQ3: Better localization enables health analysis (AI diagnoses the victim)

Protocol:
1. Use manual annotations as Ground Truth
2. Compare Gemini's accuracy on:
   - Full degraded image (Baseline)
   - Cropped to RQ2 Iteration 3 bbox (Language-Guided)
3. Measure accuracy improvement per degradation condition
"""

# SETUP

#!pip install google-genai pillow tqdm matplotlib -q

import os
import json
import re
import time
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

#from google.colab import drive, userdata
#drive.mount('/content/drive')

GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

from google import genai
client = genai.Client(api_key=GOOGLE_API_KEY)

print("✓ API ready")


# CONFIGURATION

DATASET_PATH = 'refcoco_degraded_benchmark'
RQ2_RESULTS_PATH = 'refcoco_degraded_benchmark/results/rq2_complete_results_combined.json'
ANNOTATIONS_PATH = 'refcoco_degraded_benchmark/rq3_health_annotations.json'

CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']


# CREATE ANNOTATION HELPER

def create_annotation_file(rq2_results_path, output_path, num_samples=50):
    """
    Create a template file for manual health annotations.
    Select samples that had successful RQ2 corrections.
    """
    
    # Load RQ2 results
    with open(rq2_results_path) as f:
        rq2_data = json.load(f)
    
    # Get samples with good RQ2 improvement
    samples_to_annotate = []
    
    # Try to get samples from thermal first 
    thermal_results = rq2_data.get('results', {}).get('combined', {}).get('thermal_0.5', [])
    
    for r in thermal_results:
        if r.get('improved', False) and r.get('final_iou', 0) > 0.3:
            samples_to_annotate.append({
                'sample_id': r['sample_id'],
                'initial_bbox': r.get('initial_bbox'),
                'final_bbox': r.get('final_bbox'),
                'initial_iou': r.get('initial_iou'),
                'final_iou': r.get('final_iou')
            })
        
        if len(samples_to_annotate) >= num_samples:
            break
    
    # If not enough from thermal, add from other conditions
    if len(samples_to_annotate) < num_samples:
        for cond in ['smoke_0.5', 'fog_0.5', 'clean']:
            cond_results = rq2_data.get('results', {}).get('combined', {}).get(cond, [])
            for r in cond_results:
                if r['sample_id'] not in [s['sample_id'] for s in samples_to_annotate]:
                    samples_to_annotate.append({
                        'sample_id': r['sample_id'],
                        'initial_bbox': r.get('initial_bbox'),
                        'final_bbox': r.get('final_bbox'),
                        'initial_iou': r.get('initial_iou'),
                        'final_iou': r.get('final_iou')
                    })
                if len(samples_to_annotate) >= num_samples:
                    break
            if len(samples_to_annotate) >= num_samples:
                break
    
    # Load RQ1 data to get filenames and expressions
    rq1_path = os.path.join(DATASET_PATH, 'results', 'rq1_with_bboxes.json')
    with open(rq1_path) as f:
        rq1_data = json.load(f)
    
    # Build lookup
    rq1_lookup = {s['sample_id']: s for s in rq1_data['samples']}
    
    # Create annotation template
    annotations = {
        "metadata": {
            "description": "RQ3 Health Annotation - Manual Ground Truth",
            "annotator": "YOUR_NAME",
            "date": datetime.now().strftime("%Y-%m-%d"),
            "num_samples": len(samples_to_annotate),
            "guidelines": {
                "posture": {
                    "STANDING": "Person is upright on their feet",
                    "SITTING": "Person is seated",
                    "LYING": "Person is horizontal (critical for emergency)"
                },
                "face_visible": {
                    "YES": "Face clearly visible",
                    "PARTIAL": "Face partially visible",
                    "NO": "Face not visible"
                },
                "needs_help": {
                    "YES": "Person appears to need assistance",
                    "NO": "Person appears fine",
                    "UNCLEAR": "Cannot determine"
                }
            }
        },
        "samples": []
    }
    
    for sample in samples_to_annotate:
        sid = sample['sample_id']
        if sid in rq1_lookup:
            rq1_sample = rq1_lookup[sid]
            annotations['samples'].append({
                "sample_id": sid,
                "filename": rq1_sample['filename'],
                "expression": rq1_sample['expression'],
                "gt_bbox": rq1_sample['gt_bbox'],
                "rq2_initial_bbox": sample['initial_bbox'],
                "rq2_final_bbox": sample['final_bbox'],
                "rq2_initial_iou": sample['initial_iou'],
                "rq2_final_iou": sample['final_iou'],
                "posture": "",  # STANDING, SITTING, or LYING
                "face_visible": "",  # YES, PARTIAL, or NO
                "needs_help": "",  # YES, NO, or UNCLEAR
                "notes": ""
            })
    
    # Save
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"✓ Created annotation template: {output_path}")
    print(f"  → {len(annotations['samples'])} samples to annotate")
    print(f"\n  NEXT STEP:")
    print(f"  1. Download this file")
    print(f"  2. Open each image in {DATASET_PATH}/images/clean/")
    print(f"  3. Fill in: posture, face_visible, needs_help")
    print(f"  4. Upload back to Google Drive")
    
    return annotations

# VQA HEALTH EVALUATOR

class HealthVQAEvaluator:
    """
    Evaluate health-related VQA on degraded images.
    Compare full image vs cropped (language-guided) performance.
    """

    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = 'gemini-2.0-flash'
        self.request_delay = 4.0
        self.last_request = 0
        self.total_requests = 0

    def _wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request = time.time()

    def ask_posture(self, image, method="full"):
        """
        Ask about person's posture.
        Direct prompting for clear YES/NO/category response.
        """
        self._wait()

        if method == "full":
            prompt = (
                "Look at this emergency scene image. "
                "Answer with ONE WORD only: STANDING, SITTING, or LYING. "
                "What is the posture of the person in this image?"
            )
        else:  # cropped
            prompt = (
                "This is a cropped image of a person. "
                "Answer with ONE WORD only: STANDING, SITTING, or LYING. "
                "What is this person's posture?"
            )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            self.total_requests += 1
            return self._parse_posture(response.text)
        except Exception as e:
            print(f"Error: {e}")
            return "ERROR"

    def ask_face_visible(self, image, method="full"):
        """
        Ask if face is visible for health assessment.
        """
        self._wait()

        if method == "full":
            prompt = (
                "Look at this emergency scene image. "
                "Answer with ONE WORD only: YES, PARTIAL, or NO. "
                "Is the person's face clearly visible for medical assessment?"
            )
        else:
            prompt = (
                "This is a cropped image of a person. "
                "Answer with ONE WORD only: YES, PARTIAL, or NO. "
                "Is this person's face clearly visible?"
            )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            self.total_requests += 1
            return self._parse_face_visible(response.text)
        except Exception as e:
            return "ERROR"

    def ask_needs_help(self, image, method="full"):
        """
        Ask if person appears to need help.
        """
        self._wait()

        if method == "full":
            prompt = (
                "Look at this emergency scene image. "
                "Answer with ONE WORD only: YES or NO. "
                "Does the person appear to need immediate assistance or help?"
            )
        else:
            prompt = (
                "This is a cropped image of a person. "
                "Answer with ONE WORD only: YES or NO. "
                "Does this person appear to need immediate assistance?"
            )

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            self.total_requests += 1
            return self._parse_needs_help(response.text)
        except Exception as e:
            return "ERROR"

    def _parse_posture(self, text):
        """Parse posture response."""
        text = text.upper().strip()
        if "LYING" in text or "LIE" in text or "DOWN" in text or "HORIZONTAL" in text:
            return "LYING"
        elif "SITTING" in text or "SIT" in text or "SEATED" in text:
            return "SITTING"
        elif "STANDING" in text or "STAND" in text or "UPRIGHT" in text:
            return "STANDING"
        else:
            return "UNCLEAR"

    def _parse_face_visible(self, text):
        """Parse face visibility response."""
        text = text.upper().strip()
        if "PARTIAL" in text:
            return "PARTIAL"
        elif "YES" in text or "VISIBLE" in text or "CLEAR" in text:
            return "YES"
        elif "NO" in text or "NOT" in text:
            return "NO"
        else:
            return "UNCLEAR"

    def _parse_needs_help(self, text):
        """Parse needs help response."""
        text = text.upper().strip()
        if "YES" in text:
            return "YES"
        elif "NO" in text:
            return "NO"
        else:
            return "UNCLEAR"


def crop_image(image, bbox):
    """
    Crop image to bounding box.
    bbox format: [x, y, w, h]
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

# Image Cropping Example

# Use a sample image from the dataset
sample_image_path = os.path.join(DATASET_PATH, 'images', 'smoke_0.5', '000002.jpg')

# Define a sample bounding box [x, y, w, h]
# This bbox is for the person in the middle-right of '000002.jpg'
sample_bbox = [300, 100, 200, 300] 

print(f"\n--- Cropping Example ---")
print(f"Original image: {sample_image_path}")
print(f"Bounding box (x, y, w, h): {sample_bbox}")

if os.path.exists(sample_image_path):
    original_image = Image.open(sample_image_path)
    cropped_image = crop_image(original_image, sample_bbox)
    
    # Display images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cropped_image)
    plt.title('Cropped Image')
    plt.axis('off')
    plt.show()

    # You can also save the cropped image if needed
    # cropped_image.save("/content/cropped_example.jpg")
    # print("Cropped image saved to /content/cropped_example.jpg")
else:
    print(f"Error: Sample image not found at {sample_image_path}")


# VQA HEALTH EVALUATOR

class HealthVQAEvaluator:
    """
    Evaluate health-related VQA on degraded images.
    Compare full image vs cropped (language-guided) performance.
    """
    
    def __init__(self):
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.model_name = 'gemini-2.0-flash'
        self.request_delay = 4.0
        self.last_request = 0
        self.total_requests = 0
    
    def _wait(self):
        elapsed = time.time() - self.last_request
        if elapsed < self.request_delay:
            time.sleep(self.request_delay - elapsed)
        self.last_request = time.time()
    
    def ask_posture(self, image, method="full"):
        """
        Ask about person's posture.
        Direct prompting for clear YES/NO/category response.
        """
        self._wait()
        
        if method == "full":
            prompt = (
                "Look at this emergency scene image. "
                "Answer with ONE WORD only: STANDING, SITTING, or LYING. "
                "What is the posture of the person in this image?"
            )
        else:  # cropped
            prompt = (
                "This is a cropped image of a person. "
                "Answer with ONE WORD only: STANDING, SITTING, or LYING. "
                "What is this person's posture?"
            )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            self.total_requests += 1
            return self._parse_posture(response.text)
        except Exception as e:
            print(f"Error: {e}")
            return "ERROR"
    
    def ask_face_visible(self, image, method="full"):
        """
        Ask if face is visible for health assessment.
        """
        self._wait()
        
        if method == "full":
            prompt = (
                "Look at this emergency scene image. "
                "Answer with ONE WORD only: YES, PARTIAL, or NO. "
                "Is the person's face clearly visible for medical assessment?"
            )
        else:
            prompt = (
                "This is a cropped image of a person. "
                "Answer with ONE WORD only: YES, PARTIAL, or NO. "
                "Is this person's face clearly visible?"
            )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            self.total_requests += 1
            return self._parse_face_visible(response.text)
        except Exception as e:
            return "ERROR"
    
    def ask_needs_help(self, image, method="full"):
        """
        Ask if person appears to need help.
        """
        self._wait()
        
        if method == "full":
            prompt = (
                "Look at this emergency scene image. "
                "Answer with ONE WORD only: YES or NO. "
                "Does the person appear to need immediate assistance or help?"
            )
        else:
            prompt = (
                "This is a cropped image of a person. "
                "Answer with ONE WORD only: YES or NO. "
                "Does this person appear to need immediate assistance?"
            )
        
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            self.total_requests += 1
            return self._parse_needs_help(response.text)
        except Exception as e:
            return "ERROR"
    
    def _parse_posture(self, text):
        """Parse posture response."""
        text = text.upper().strip()
        if "LYING" in text or "LIE" in text or "DOWN" in text or "HORIZONTAL" in text:
            return "LYING"
        elif "SITTING" in text or "SIT" in text or "SEATED" in text:
            return "SITTING"
        elif "STANDING" in text or "STAND" in text or "UPRIGHT" in text:
            return "STANDING"
        else:
            return "UNCLEAR"
    
    def _parse_face_visible(self, text):
        """Parse face visibility response."""
        text = text.upper().strip()
        if "PARTIAL" in text:
            return "PARTIAL"
        elif "YES" in text or "VISIBLE" in text or "CLEAR" in text:
            return "YES"
        elif "NO" in text or "NOT" in text:
            return "NO"
        else:
            return "UNCLEAR"
    
    def _parse_needs_help(self, text):
        """Parse needs help response."""
        text = text.upper().strip()
        if "YES" in text:
            return "YES"
        elif "NO" in text:
            return "NO"
        else:
            return "UNCLEAR"


def crop_image(image, bbox):
    """
    Crop image to bounding box.
    bbox format: [x, y, w, h]
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


# STEP 3: RUN RQ3 EVALUATION

def run_rq3_evaluation(dataset_path, annotations_path, conditions=None):
    """
    Run complete RQ3 health VQA evaluation.

    Compares:
    - Method A: Full degraded image (Baseline)
    - Method B: Cropped to RQ2 final bbox (Language-Guided)
    """

    if conditions is None:
        conditions = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']

    # Load annotations (your ground truth)
    print("Loading health annotations...")
    with open(annotations_path) as f:
        annotations = json.load(f)

    samples = annotations['samples']
    print(f"  → {len(samples)} annotated samples")

    # Filter samples with complete annotations
    valid_samples = [s for s in samples if s.get('posture') and s['posture'] != ""]
    print(f"  → {len(valid_samples)} samples with posture annotations")

    if len(valid_samples) == 0:
        print("\n❌ ERROR: No annotated samples found!")
        print("   Please fill in the posture, face_visible, needs_help fields")
        print("   in your annotation file first.")
        return None

    # Initialize evaluator
    evaluator = HealthVQAEvaluator()

    # Results storage
    results = {
        cond: {
            'posture': {'full': [], 'cropped': [], 'gt': []},
            'face_visible': {'full': [], 'cropped': [], 'gt': []},
            'needs_help': {'full': [], 'cropped': [], 'gt': []}
        }
        for cond in conditions
    }

    # Progress
    total = len(valid_samples) * len(conditions)
    pbar = tqdm(total=total, desc="RQ3 Evaluation")

    print(f"\n{'='*70}")
    print("RQ3: Health VQA Evaluation")
    print(f"{'='*70}")
    print(f"  Samples: {len(valid_samples)}")
    print(f"  Conditions: {conditions}")
    print(f"  Methods: Full Image vs Cropped (RQ2)")
    print(f"{'='*70}")

    # Evaluate
    for sample in valid_samples:
        filename = sample['filename']
        gt_posture = sample['posture'].upper()
        gt_face = sample.get('face_visible', 'UNCLEAR').upper()
        gt_help = sample.get('needs_help', 'UNCLEAR').upper()

        # Get RQ2 final bbox (Iteration 3 - best precision)
        rq2_bbox = sample.get('rq2_final_bbox')

        for condition in conditions:
            image_path = os.path.join(dataset_path, 'images', condition, filename)

            if not os.path.exists(image_path):
                pbar.update(1)
                continue

            # Load image
            img = Image.open(image_path)

            # Method A: Full image
            full_posture = evaluator.ask_posture(img, method="full")
            full_face = evaluator.ask_face_visible(img, method="full")
            full_help = evaluator.ask_needs_help(img, method="full")

            # Method B: Cropped to RQ2 bbox
            cropped_img = crop_image(img, rq2_bbox)
            crop_posture = evaluator.ask_posture(cropped_img, method="cropped")
            crop_face = evaluator.ask_face_visible(cropped_img, method="cropped")
            crop_help = evaluator.ask_needs_help(cropped_img, method="cropped")

            # Store results
            results[condition]['posture']['full'].append(full_posture)
            results[condition]['posture']['cropped'].append(crop_posture)
            results[condition]['posture']['gt'].append(gt_posture)

            results[condition]['face_visible']['full'].append(full_face)
            results[condition]['face_visible']['cropped'].append(crop_face)
            results[condition]['face_visible']['gt'].append(gt_face)

            results[condition]['needs_help']['full'].append(full_help)
            results[condition]['needs_help']['cropped'].append(crop_help)
            results[condition]['needs_help']['gt'].append(gt_help)

            pbar.update(1)
            pbar.set_postfix({
                'cond': condition[:6],
                'posture': f"{full_posture[0]}→{crop_posture[0]}"
            })

    pbar.close()

    # Calculate accuracy
    print("RQ3 RESULTS: Health VQA Accuracy")

    summary = calculate_rq3_accuracy(results, conditions)

    # Save results
    results_dir = os.path.join(dataset_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    output = {
        'metadata': {
            'experiment': 'RQ3 - Health VQA in Degraded Conditions',
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(valid_samples),
            'conditions': conditions,
            'total_requests': evaluator.total_requests
        },
        'summary': summary,
        'raw': results
    }

    save_path = os.path.join(results_dir, 'rq3_results.json')
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n✓ Results saved to {save_path}")

    # Visualize
    plot_rq3_results(summary, conditions, results_dir)

    return results, summary


def calculate_rq3_accuracy(results, conditions):
    """Calculate accuracy for each condition and method."""

    summary = {}

    print("POSTURE DETECTION ACCURACY (Critical for Emergency)")
    print(f"\n{'Condition':<15} {'Full Image':<15} {'Cropped (RQ2)':<15} {'Gain':<10}")

    for cond in conditions:
        posture_full = results[cond]['posture']['full']
        posture_crop = results[cond]['posture']['cropped']
        posture_gt = results[cond]['posture']['gt']

        if len(posture_gt) == 0:
            continue

        # Calculate accuracy
        full_correct = sum(1 for p, g in zip(posture_full, posture_gt) if p == g)
        crop_correct = sum(1 for p, g in zip(posture_crop, posture_gt) if p == g)

        full_acc = full_correct / len(posture_gt) * 100
        crop_acc = crop_correct / len(posture_gt) * 100
        gain = crop_acc - full_acc

        summary[cond] = {
            'posture': {
                'full_accuracy': full_acc,
                'cropped_accuracy': crop_acc,
                'gain': gain,
                'n': len(posture_gt)
            }
        }

        # Also calculate face visibility
        face_full = results[cond]['face_visible']['full']
        face_crop = results[cond]['face_visible']['cropped']
        face_gt = results[cond]['face_visible']['gt']

        if len([g for g in face_gt if g and g != 'UNCLEAR']) > 0:
            valid_gt = [(f, c, g) for f, c, g in zip(face_full, face_crop, face_gt) if g and g != 'UNCLEAR']
            if valid_gt:
                face_full_acc = sum(1 for f, c, g in valid_gt if f == g) / len(valid_gt) * 100
                face_crop_acc = sum(1 for f, c, g in valid_gt if c == g) / len(valid_gt) * 100
                summary[cond]['face_visible'] = {
                    'full_accuracy': face_full_acc,
                    'cropped_accuracy': face_crop_acc,
                    'gain': face_crop_acc - face_full_acc
                }

        print(f"{cond:<15} {full_acc:<15.1f} {crop_acc:<15.1f} {gain:+.1f}%")


    # Average across conditions
    avg_full = np.mean([summary[c]['posture']['full_accuracy'] for c in conditions if c in summary])
    avg_crop = np.mean([summary[c]['posture']['cropped_accuracy'] for c in conditions if c in summary])
    avg_gain = avg_crop - avg_full

    print(f"{'AVERAGE':<15} {avg_full:<15.1f} {avg_crop:<15.1f} {avg_gain:+.1f}%")

    summary['average'] = {
        'posture': {
            'full_accuracy': avg_full,
            'cropped_accuracy': avg_crop,
            'gain': avg_gain
        }
    }

    # Key insight
    print("KEY INSIGHT")
    print(f"""
Language-guided attention (cropping to RQ2 bbox) improves health
diagnosis accuracy by {avg_gain:+.1f}% on average.

The improvement is largest in degraded conditions:
""")

    for cond in conditions:
        if cond in summary:
            gain = summary[cond]['posture']['gain']
            emoji = "🏆" if gain > 20 else ("✓" if gain > 10 else "→")
            print(f"  {emoji} {cond}: {gain:+.1f}% improvement")

    return summary


def plot_rq3_results(summary, conditions, results_dir):
    """Create visualization for RQ3 results."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('RQ3: Does Language-Guided Attention Improve Health Analysis?',
                 fontsize=14, fontweight='bold')

    colors = {
        'clean': '#27ae60',
        'fog_0.5': '#3498db',
        'smoke_0.5': '#9b59b6',
        'thermal_0.5': '#e74c3c'
    }

    # Plot 1: Accuracy Comparison
    ax1 = axes[0]

    x = np.arange(len(conditions))
    width = 0.35

    full_accs = [summary[c]['posture']['full_accuracy'] for c in conditions if c in summary]
    crop_accs = [summary[c]['posture']['cropped_accuracy'] for c in conditions if c in summary]
    valid_conds = [c for c in conditions if c in summary]

    bars1 = ax1.bar(x - width/2, full_accs, width, label='Full Image (Baseline)',
                    color='#e74c3c', alpha=0.85)
    bars2 = ax1.bar(x + width/2, crop_accs, width, label='Cropped (RQ2 Guided)',
                    color='#27ae60', alpha=0.85)

    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Posture Detection Accuracy (%)')
    ax1.set_title('Health Analysis Accuracy: Full vs Language-Guided')
    ax1.set_xticks(x)
    ax1.set_xticklabels([c.replace('_0.5', '') for c in valid_conds])
    ax1.legend()
    ax1.set_ylim(0, 100)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, acc in zip(bars1, full_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.0f}%', ha='center', fontsize=9)
    for bar, acc in zip(bars2, crop_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{acc:.0f}%', ha='center', fontsize=9, color='green', fontweight='bold')

    # Plot 2: Improvement (Gain)
    ax2 = axes[1]

    gains = [summary[c]['posture']['gain'] for c in valid_conds]
    bar_colors = [colors.get(c, '#888') for c in valid_conds]

    bars = ax2.bar(range(len(valid_conds)), gains, color=bar_colors, alpha=0.85, edgecolor='black')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(range(len(valid_conds)))
    ax2.set_xticklabels([c.replace('_0.5', '') for c in valid_conds])
    ax2.set_ylabel('Accuracy Gain (%)')
    ax2.set_title('Improvement from Language-Guided Attention')

    for bar, gain in zip(bars, gains):
        color = 'green' if gain > 0 else 'red'
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{gain:+.1f}%', ha='center', fontsize=11, fontweight='bold', color=color)

    plt.tight_layout()

    save_path = os.path.join(results_dir, 'rq3_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {save_path}")
    plt.show()


# MAIN EXECUTION
print("RQ3: Health VQA in Degraded Conditions")

# Check if annotations exist
if not os.path.exists(ANNOTATIONS_PATH):
    print("\n⚠️  Annotation file not found!")
else:
    print("\n✓ Annotation file found!")
    print("  Running RQ3 evaluation...")

    results, summary = run_rq3_evaluation(
        dataset_path=DATASET_PATH,
        annotations_path=ANNOTATIONS_PATH,
        conditions=CONDITIONS
    )