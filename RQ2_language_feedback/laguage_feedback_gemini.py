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
#from google.colab import drive
#drive.mount('/content/drive')

# Import Gemini
try:
    from google import genai
    USE_NEW_API = True
    print("✓ Using google-genai (new API)")
except ImportError:
    import google.generativeai as genai
    USE_NEW_API = False
    print("✓ Using google-generativeai")

# CONFIGURATION
DATASET_PATH = 'HorusEye/horuseye_VLM/refcoco_degraded_benchmark'
RQ1_RESULTS_PATH = 'HorusEye/horuseye_VLM/refcoco_degraded_benchmark/results/rq1_with_bboxes.json'

NUM_SAMPLES = None          # Samples PER CONDITION (set None for all)
MAX_ITERATIONS = 3         # Baseline + 2 feedback rounds
FEEDBACK_TYPES = ['combined']  # Test all types
IOU_THRESHOLD = 0.5        # Only correct predictions below this
CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
CHECKPOINT_EVERY = 25


# FUNCTIONS

def calculate_iou(box1, box2):
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
    union = box1[2] * box1[3] + box2[2] * box2[3] - intersection
    return intersection / union if union > 0 else 0.0


def extract_bbox(text, img_w, img_h):
    if not text:
        return None
    pattern = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'
    match = re.search(pattern, text)
    if match:
        try:
            ymin, xmin, ymax, xmax = [int(v) for v in match.groups()]
            px_xmin = (xmin / 1000.0) * img_w
            px_ymin = (ymin / 1000.0) * img_h
            px_xmax = (xmax / 1000.0) * img_w
            px_ymax = (ymax / 1000.0) * img_h
            return [px_xmin, px_ymin, px_xmax - px_xmin, px_ymax - px_ymin]
        except:
            return None
    return None


def bbox_to_normalized(bbox, img_w, img_h):
    """Convert [x, y, w, h] pixels to [ymin, xmin, ymax, xmax] (0-1000)"""
    if bbox is None:
        return None
    x, y, w, h = bbox
    xmin = int((x / img_w) * 1000)
    ymin = int((y / img_h) * 1000)
    xmax = int(((x + w) / img_w) * 1000)
    ymax = int(((y + h) / img_h) * 1000)
    return [ymin, xmin, ymax, xmax]


def get_position_description(bbox, img_w, img_h):
    """Describe position in human terms."""
    cx = bbox[0] + bbox[2] / 2
    cy = bbox[1] + bbox[3] / 2

    if cx < img_w * 0.33:
        h_pos = "left"
    elif cx > img_w * 0.67:
        h_pos = "right"
    else:
        h_pos = "center"

    if cy < img_h * 0.33:
        v_pos = "top"
    elif cy > img_h * 0.67:
        v_pos = "bottom"
    else:
        v_pos = "middle"

    return f"{v_pos}-{h_pos}"

# FEEDBACK GENERATORS (RQ2.3: Test Different Types)

def generate_feedback_spatial(pred_bbox, gt_bbox, expression, img_w, img_h):
    """
    SPATIAL FEEDBACK: Direction-based corrections
    Example: "Move RIGHT and LOWER"
    """
    pred_norm = bbox_to_normalized(pred_bbox, img_w, img_h)

    if pred_bbox is None:
        return f"You did not detect '{expression}'. Please try again."

    # Calculate direction differences
    pred_cx = pred_bbox[0] + pred_bbox[2] / 2
    pred_cy = pred_bbox[1] + pred_bbox[3] / 2
    gt_cx = gt_bbox[0] + gt_bbox[2] / 2
    gt_cy = gt_bbox[1] + gt_bbox[3] / 2

    dx = (gt_cx - pred_cx) / img_w
    dy = (gt_cy - pred_cy) / img_h

    parts = []
    if dx > 0.15:
        parts.append("move significantly to the RIGHT")
    elif dx > 0.05:
        parts.append("move slightly to the RIGHT")
    elif dx < -0.15:
        parts.append("move significantly to the LEFT")
    elif dx < -0.05:
        parts.append("move slightly to the LEFT")

    if dy > 0.15:
        parts.append("move significantly LOWER")
    elif dy > 0.05:
        parts.append("move slightly LOWER")
    elif dy < -0.15:
        parts.append("move significantly HIGHER")
    elif dy < -0.05:
        parts.append("move slightly HIGHER")

    if not parts:
        parts.append("adjust the position slightly")

    direction = ", ".join(parts)

    return (
        f"Your previous prediction was [{pred_norm[0]}, {pred_norm[1]}, {pred_norm[2]}, {pred_norm[3]}]. "
        f"This is incorrect for '{expression}'. "
        f"You need to {direction}. "
        f"Provide a corrected bounding box in [ymin, xmin, ymax, xmax] format (0-1000 scale). "
        f"Output ONLY the coordinates."
    )


def generate_feedback_size(pred_bbox, gt_bbox, expression, img_w, img_h):
    """
    SIZE FEEDBACK: Size-based corrections
    Example: "Make the box LARGER/SMALLER"
    """
    pred_norm = bbox_to_normalized(pred_bbox, img_w, img_h)

    if pred_bbox is None:
        return f"You did not detect '{expression}'. Please try again."

    # Calculate size differences
    pred_area = pred_bbox[2] * pred_bbox[3]
    gt_area = gt_bbox[2] * gt_bbox[3]
    area_ratio = gt_area / max(pred_area, 1)

    width_ratio = gt_bbox[2] / max(pred_bbox[2], 1)
    height_ratio = gt_bbox[3] / max(pred_bbox[3], 1)

    parts = []
    if width_ratio > 1.4:
        parts.append("make the box much WIDER")
    elif width_ratio > 1.15:
        parts.append("make the box slightly WIDER")
    elif width_ratio < 0.7:
        parts.append("make the box much NARROWER")
    elif width_ratio < 0.85:
        parts.append("make the box slightly NARROWER")

    if height_ratio > 1.4:
        parts.append("make the box much TALLER")
    elif height_ratio > 1.15:
        parts.append("make the box slightly TALLER")
    elif height_ratio < 0.7:
        parts.append("make the box much SHORTER")
    elif height_ratio < 0.85:
        parts.append("make the box slightly SHORTER")

    if not parts:
        parts.append("adjust the size slightly")

    size_feedback = ", ".join(parts)

    return (
        f"Your previous prediction was [{pred_norm[0]}, {pred_norm[1]}, {pred_norm[2]}, {pred_norm[3]}]. "
        f"The size is incorrect for '{expression}'. "
        f"You need to {size_feedback}. "
        f"Provide a corrected bounding box in [ymin, xmin, ymax, xmax] format (0-1000 scale). "
        f"Output ONLY the coordinates."
    )


def generate_feedback_descriptive(pred_bbox, gt_bbox, expression, img_w, img_h):
    """
    DESCRIPTIVE FEEDBACK: Location description
    Example: "The target is in the top-left area of the image"
    """
    pred_norm = bbox_to_normalized(pred_bbox, img_w, img_h)

    if pred_bbox is None:
        return f"You did not detect '{expression}'. Please try again."

    # Describe where the target actually is
    gt_position = get_position_description(gt_bbox, img_w, img_h)
    pred_position = get_position_description(pred_bbox, img_w, img_h)

    return (
        f"Your previous prediction was [{pred_norm[0]}, {pred_norm[1]}, {pred_norm[2]}, {pred_norm[3]}]. "
        f"You predicted the {pred_position} area, but '{expression}' is actually in the {gt_position} area. "
        f"Look for the object in the {gt_position} of the image. "
        f"Provide a corrected bounding box in [ymin, xmin, ymax, xmax] format (0-1000 scale). "
        f"Output ONLY the coordinates."
    )


def generate_feedback_combined(pred_bbox, gt_bbox, expression, img_w, img_h):
    """
    COMBINED FEEDBACK: Spatial + Size + Descriptive
    """
    pred_norm = bbox_to_normalized(pred_bbox, img_w, img_h)

    if pred_bbox is None:
        return f"You did not detect '{expression}'. Please try again."

    # Spatial
    pred_cx = pred_bbox[0] + pred_bbox[2] / 2
    pred_cy = pred_bbox[1] + pred_bbox[3] / 2
    gt_cx = gt_bbox[0] + gt_bbox[2] / 2
    gt_cy = gt_bbox[1] + gt_bbox[3] / 2

    dx = (gt_cx - pred_cx) / img_w
    dy = (gt_cy - pred_cy) / img_h

    spatial_parts = []
    if dx > 0.1:
        spatial_parts.append("RIGHT")
    elif dx < -0.1:
        spatial_parts.append("LEFT")
    if dy > 0.1:
        spatial_parts.append("LOWER")
    elif dy < -0.1:
        spatial_parts.append("HIGHER")

    # Size
    width_ratio = gt_bbox[2] / max(pred_bbox[2], 1)
    height_ratio = gt_bbox[3] / max(pred_bbox[3], 1)

    size_parts = []
    if width_ratio > 1.3:
        size_parts.append("WIDER")
    elif width_ratio < 0.7:
        size_parts.append("NARROWER")
    if height_ratio > 1.3:
        size_parts.append("TALLER")
    elif height_ratio < 0.7:
        size_parts.append("SHORTER")

    # Position description
    gt_position = get_position_description(gt_bbox, img_w, img_h)

    # Build combined feedback
    feedback_parts = []
    if spatial_parts:
        feedback_parts.append(f"move {' and '.join(spatial_parts)}")
    if size_parts:
        feedback_parts.append(f"make it {' and '.join(size_parts)}")

    if feedback_parts:
        correction = ", ".join(feedback_parts)
    else:
        correction = "make small adjustments"

    return (
        f"Your previous prediction was [{pred_norm[0]}, {pred_norm[1]}, {pred_norm[2]}, {pred_norm[3]}]. "
        f"This is incorrect. The object '{expression}' is in the {gt_position} area. "
        f"You need to {correction}. "
        f"Provide a corrected bounding box in [ymin, xmin, ymax, xmax] format (0-1000 scale). "
        f"Output ONLY the coordinates."
    )


# Feedback generator mapping
FEEDBACK_GENERATORS = {
    'spatial': generate_feedback_spatial,
    'size': generate_feedback_size,
    'descriptive': generate_feedback_descriptive,
    'combined': generate_feedback_combined
}
# EVALUATOR

class RQ2Evaluator:
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

    def predict(self, image, prompt):
        self._wait()
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt, image]
            )
            self.total_requests += 1
            return response.text
        except Exception as e:
            if 'quota' in str(e).lower() or '429' in str(e):
                time.sleep(60)
                return self.predict(image, prompt)
            return None

    def correct_with_feedback(self, image_path, expression, gt_bbox,
                               initial_bbox, initial_iou,
                               max_iterations=3, feedback_type='spatial'):
        """
        Apply iterative feedback to correct a wrong prediction.

        Args:
            image_path: Path to image
            expression: Referring expression
            gt_bbox: Ground truth bounding box
            initial_bbox: Wrong prediction from RQ1
            initial_iou: IoU from RQ1
            max_iterations: Number of correction attempts
            feedback_type: Type of feedback to use

        Returns:
            Dictionary with iteration results
        """
        img = Image.open(image_path)
        img_w, img_h = img.size

        # Get feedback generator
        generate_feedback = FEEDBACK_GENERATORS[feedback_type]

        results = {
            'initial_iou': initial_iou,
            'initial_bbox': initial_bbox,
            'iterations': [],
            'final_iou': initial_iou,
            'final_bbox': initial_bbox,
            'improved': False,
            'improvement': 0.0
        }

        current_bbox = initial_bbox
        current_iou = initial_iou

        # Iteration 1 is the RQ1 baseline (already have it)
        results['iterations'].append({
            'iteration': 1,
            'bbox': initial_bbox,
            'iou': initial_iou,
            'feedback': None
        })

        # Apply feedback for remaining iterations
        for iteration in range(2, max_iterations + 1):

            # Generate feedback based on current prediction
            feedback = generate_feedback(current_bbox, gt_bbox, expression, img_w, img_h)

            # Get corrected prediction
            response = self.predict(img, feedback)

            if response is None:
                results['iterations'].append({
                    'iteration': iteration,
                    'bbox': None,
                    'iou': 0.0,
                    'feedback': feedback[:100],
                    'error': True
                })
                continue

            # Extract new bbox
            new_bbox = extract_bbox(response, img_w, img_h)
            new_iou = calculate_iou(new_bbox, gt_bbox)

            results['iterations'].append({
                'iteration': iteration,
                'bbox': new_bbox,
                'iou': new_iou,
                'feedback': feedback[:100]
            })

            # Update current prediction for next iteration
            current_bbox = new_bbox if new_bbox else current_bbox
            current_iou = new_iou

        # Calculate final results
        results['final_iou'] = current_iou
        results['final_bbox'] = current_bbox
        results['improvement'] = current_iou - initial_iou
        results['improved'] = current_iou > initial_iou + 0.01

        return results

# LOAD RQ1 RESULTS

def load_wrong_predictions(rq1_path, conditions, iou_threshold=0.5, max_per_condition=None):
    """
    Load wrong predictions from RQ1 results.

    Returns:
        Dictionary: {condition: [list of wrong samples]}
    """
    print(f"Loading RQ1 results from {rq1_path}...")

    with open(rq1_path) as f:
        rq1_data = json.load(f)

    wrong_predictions = {cond: [] for cond in conditions}

    for sample in rq1_data['samples']:
        for cond in conditions:
            if cond not in sample['conditions']:
                continue

            cond_data = sample['conditions'][cond]

            # Filter: only wrong predictions
            if cond_data['iou'] < iou_threshold and cond_data['predicted_bbox'] is not None:
                wrong_predictions[cond].append({
                    'sample_id': sample['sample_id'],
                    'filename': sample['filename'],
                    'expression': sample['expression'],
                    'gt_bbox': sample['gt_bbox'],
                    'predicted_bbox': cond_data['predicted_bbox'],
                    'initial_iou': cond_data['iou']
                })

    # Limit samples per condition if specified
    if max_per_condition:
        for cond in conditions:
            wrong_predictions[cond] = wrong_predictions[cond][:max_per_condition]

    # Print summary
    print(f"\nWrong predictions loaded (IoU < {iou_threshold}):")
    for cond in conditions:
        print(f"  {cond}: {len(wrong_predictions[cond])} samples")

    return wrong_predictions
    
# CHECKPOINT FUNCTIONS

def load_checkpoint(results_dir, feedback_type):
    path = os.path.join(results_dir, f'rq2_{feedback_type}_checkpoint.json')
    if os.path.exists(path):
        with open(path) as f:
            ckpt = json.load(f)
        print(f"✓ Resuming {feedback_type} from checkpoint")
        return ckpt['results'], ckpt['progress']
    return None, {}


def save_checkpoint(results, progress, results_dir, feedback_type):
    path = os.path.join(results_dir, f'rq2_{feedback_type}_checkpoint.json')
    with open(path, 'w') as f:
        json.dump({'results': results, 'progress': progress}, f)

# MAIN EVALUATION

def run_rq2_evaluation(dataset_path, rq1_path,
                       num_samples=200,
                       max_iterations=3,
                       feedback_types=['spatial', 'size', 'descriptive', 'combined'],
                       conditions=['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5'],
                       iou_threshold=0.5,
                       checkpoint_every=25):
    """
    Run complete RQ2 evaluation.

    Tests all feedback types on wrong predictions from RQ1.
    """

    # Results directory
    results_dir = os.path.join(dataset_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    # Load wrong predictions from RQ1
    wrong_predictions = load_wrong_predictions(
        rq1_path, conditions, iou_threshold, max_per_condition=num_samples
    )

    # Initialize evaluator
    evaluator = RQ2Evaluator()

    # Store all results
    all_results = {}

    # Test each feedback type
    for feedback_type in feedback_types:
        print(f"\n{'='*70}")
        print(f"Testing Feedback Type: {feedback_type.upper()}")
        print(f"{'='*70}")

        # Load checkpoint if exists
        checkpoint_results, progress = load_checkpoint(results_dir, feedback_type)

        if checkpoint_results:
            results = checkpoint_results
        else:
            results = {cond: [] for cond in conditions}
            progress = {cond: 0 for cond in conditions}

        # Evaluate each condition
        for cond in conditions:
            samples = wrong_predictions[cond]
            start_idx = progress.get(cond, 0)

            if start_idx >= len(samples):
                print(f"  {cond}: Already complete")
                continue

            print(f"\n  {cond}: {len(samples) - start_idx} samples remaining")

            pbar = tqdm(samples[start_idx:], desc=f"  {cond}")

            for i, sample in enumerate(pbar, start=start_idx):
                image_path = os.path.join(dataset_path, 'images', cond, sample['filename'])

                if not os.path.exists(image_path):
                    continue

                # Apply feedback correction
                result = evaluator.correct_with_feedback(
                    image_path=image_path,
                    expression=sample['expression'],
                    gt_bbox=sample['gt_bbox'],
                    initial_bbox=sample['predicted_bbox'],
                    initial_iou=sample['initial_iou'],
                    max_iterations=max_iterations,
                    feedback_type=feedback_type
                )

                result['sample_id'] = sample['sample_id']
                results[cond].append(result)

                pbar.set_postfix({
                    'init': f"{sample['initial_iou']:.2f}",
                    'final': f"{result['final_iou']:.2f}",
                    'Δ': f"{result['improvement']:+.2f}"
                })

                # Checkpoint
                if (i + 1) % checkpoint_every == 0:
                    progress[cond] = i + 1
                    save_checkpoint(results, progress, results_dir, feedback_type)

            progress[cond] = len(samples)
            save_checkpoint(results, progress, results_dir, feedback_type)

        all_results[feedback_type] = results

        # Save results for this feedback type
        save_path = os.path.join(results_dir, f'rq2_{feedback_type}_results.json')
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Saved: {save_path}")

    # Calculate and print summary
    print_rq2_summary(all_results, conditions, feedback_types, max_iterations, evaluator.total_requests)

    # Create visualization
    plot_rq2_results(all_results, conditions, feedback_types, max_iterations, results_dir)

    # Save complete results
    final_path = os.path.join(results_dir, 'rq2_complete_results.json')
    with open(final_path, 'w') as f:
        json.dump({
            'metadata': {
                'experiment': 'RQ2 - Iterative Language Feedback',
                'feedback_types': feedback_types,
                'max_iterations': max_iterations,
                'iou_threshold': iou_threshold,
                'timestamp': datetime.now().isoformat()
            },
            'results': all_results
        }, f, indent=2)

    print(f"\n✓ Complete results saved to {final_path}")

    return all_results


def print_rq2_summary(all_results, conditions, feedback_types, max_iterations, total_requests):
    """Print comprehensive RQ2 summary answering all sub-questions."""

    print("RQ2 COMPLETE RESULTS")
        

    # Calculate summary statistics
    summary = {}

    for feedback_type in feedback_types:
        summary[feedback_type] = {}

        for cond in conditions:
            results = all_results[feedback_type][cond]

            if not results:
                continue

            initial_ious = [r['initial_iou'] for r in results]
            final_ious = [r['final_iou'] for r in results]
            improvements = [r['improvement'] for r in results]
            improved_count = sum(1 for r in results if r['improved'])

            # IoU per iteration
            iter_ious = {i: [] for i in range(1, max_iterations + 1)}
            for r in results:
                for iter_data in r['iterations']:
                    iter_num = iter_data['iteration']
                    iter_ious[iter_num].append(iter_data['iou'])

            summary[feedback_type][cond] = {
                'n': len(results),
                'initial_iou': np.mean(initial_ious),
                'final_iou': np.mean(final_ious),
                'improvement': np.mean(improvements),
                'improved_pct': improved_count / len(results) * 100,
                'iter_ious': {k: np.mean(v) if v else 0 for k, v in iter_ious.items()}
            }

    # RQ2.1: Does feedback improve accuracy?
    print("RQ2.1: Does feedback improve accuracy?")
    print(f"\n{'Feedback Type':<15} {'Initial IoU':<12} {'Final IoU':<12} {'Δ IoU':<10} {'Improved%':<10}")

    for feedback_type in feedback_types:
        all_initial = []
        all_final = []
        all_improved = []

        for cond in conditions:
            if cond in summary[feedback_type]:
                s = summary[feedback_type][cond]
                all_initial.extend([s['initial_iou']] * s['n'])
                all_final.extend([s['final_iou']] * s['n'])
                all_improved.append(s['improved_pct'])

        if all_initial:
            init_avg = np.mean(all_initial)
            final_avg = np.mean(all_final)
            delta = final_avg - init_avg
            improved_avg = np.mean(all_improved)

            emoji = "✓" if delta > 0 else "✗"
            print(f"{feedback_type:<15} {init_avg:<12.3f} {final_avg:<12.3f} {delta:+.3f}     {improved_avg:.1f}%  {emoji}")

    # RQ2.2: How many rounds needed?
    print("RQ2.2: How many iterations are needed?")

    # Use combined feedback as example
    best_type = 'combined' if 'combined' in feedback_types else feedback_types[0]

    print(f"\nUsing '{best_type}' feedback:")
    print(f"\n{'Condition':<15}", end="")
    for i in range(1, max_iterations + 1):
        print(f" {'Iter'+str(i):<10}", end="")
    print(f" {'Best Iter':<10}")
    print("-"*60)

    for cond in conditions:
        if cond in summary[best_type]:
            s = summary[best_type][cond]
            print(f"{cond:<15}", end="")

            best_iter = 1
            best_iou = 0
            for i in range(1, max_iterations + 1):
                iou = s['iter_ious'].get(i, 0)
                print(f" {iou:<10.3f}", end="")
                if iou > best_iou:
                    best_iou = iou
                    best_iter = i
            print(f" {best_iter}")

    # RQ2.3: What feedback types work best?
    print("RQ2.3: What feedback types work best?")

    print(f"\n{'Condition':<15}", end="")
    for ft in feedback_types:
        print(f" {ft:<12}", end="")
    print(f" {'Best Type':<12}")
    print("-"*70)

    for cond in conditions:
        print(f"{cond:<15}", end="")
        best_type_for_cond = None
        best_improvement = -999

        for ft in feedback_types:
            if cond in summary[ft]:
                imp = summary[ft][cond]['improvement']
                print(f" {imp:+.3f}      ", end="")
                if imp > best_improvement:
                    best_improvement = imp
                    best_type_for_cond = ft
            else:
                print(f" {'—':<12}", end="")

        print(f" {best_type_for_cond}")

    # RQ2.4: Does feedback help more on degraded images?
    print("RQ2.4: Does feedback help more on degraded images?")

    print(f"\n{'Condition':<15} {'Improvement':<12} {'Success Rate':<12} {'Verdict':<20}")

    # Average across feedback types
    for cond in conditions:
        improvements = []
        success_rates = []

        for ft in feedback_types:
            if cond in summary[ft]:
                improvements.append(summary[ft][cond]['improvement'])
                success_rates.append(summary[ft][cond]['improved_pct'])

        if improvements:
            avg_imp = np.mean(improvements)
            avg_success = np.mean(success_rates)

            if avg_imp > 0.05:
                verdict = "✓ Feedback helps!"
            elif avg_imp > 0:
                verdict = "~ Marginal help"
            else:
                verdict = "✗ Feedback hurts"

            print(f"{cond:<15} {avg_imp:+.3f}       {avg_success:.1f}%        {verdict}")

    print(f"Total API requests: {total_requests}")

    return summary


def plot_rq2_results(all_results, conditions, feedback_types, max_iterations, results_dir):
    """Create comprehensive RQ2 visualization."""

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle('RQ2: Iterative Language Feedback for Grounding Correction',
                 fontsize=14, fontweight='bold', y=0.98)

    colors = {
        'clean': '#27ae60',
        'fog_0.5': '#3498db',
        'smoke_0.5': '#9b59b6',
        'thermal_0.5': '#e74c3c'
    }

    feedback_colors = {
        'spatial': '#3498db',
        'size': '#e74c3c',
        'descriptive': '#9b59b6',
        'combined': '#27ae60'
    }

    # Calculate summary for plotting
    summary = {}
    for ft in feedback_types:
        summary[ft] = {}
        for cond in conditions:
            results = all_results[ft][cond]
            if results:
                summary[ft][cond] = {
                    'initial': np.mean([r['initial_iou'] for r in results]),
                    'final': np.mean([r['final_iou'] for r in results]),
                    'improvement': np.mean([r['improvement'] for r in results]),
                    'improved_pct': sum(1 for r in results if r['improved']) / len(results) * 100
                }

    # Plot 1: Improvement by Feedback Type
    ax1 = fig.add_subplot(2, 2, 1)
    x = np.arange(len(conditions))
    width = 0.2

    for i, ft in enumerate(feedback_types):
        improvements = [summary[ft].get(c, {}).get('improvement', 0) for c in conditions]
        ax1.bar(x + i * width, improvements, width, label=ft, color=feedback_colors[ft], alpha=0.85)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Condition')
    ax1.set_ylabel('Δ IoU')
    ax1.set_title('RQ2.3: Improvement by Feedback Type')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels([c.replace('_0.5', '') for c in conditions])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Success Rate by Condition
    ax2 = fig.add_subplot(2, 2, 2)

    for i, ft in enumerate(feedback_types):
        success_rates = [summary[ft].get(c, {}).get('improved_pct', 0) for c in conditions]
        ax2.bar(x + i * width, success_rates, width, label=ft, color=feedback_colors[ft], alpha=0.85)

    ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Condition')
    ax2.set_ylabel('% Samples Improved')
    ax2.set_title('RQ2.4: Success Rate by Condition')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels([c.replace('_0.5', '') for c in conditions])
    ax2.legend()
    ax2.set_ylim(0, 100)
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Before vs After (Combined Feedback)
    ax3 = fig.add_subplot(2, 2, 3)

    best_ft = 'combined' if 'combined' in feedback_types else feedback_types[0]

    initial = [summary[best_ft].get(c, {}).get('initial', 0) for c in conditions]
    final = [summary[best_ft].get(c, {}).get('final', 0) for c in conditions]

    x = np.arange(len(conditions))
    width = 0.35

    ax3.bar(x - width/2, initial, width, label='Initial (RQ1)', color='#e74c3c', alpha=0.85)
    ax3.bar(x + width/2, final, width, label='After Feedback', color='#27ae60', alpha=0.85)

    ax3.set_xlabel('Condition')
    ax3.set_ylabel('Mean IoU')
    ax3.set_title(f'RQ2.1: Before vs After ({best_ft} feedback)')
    ax3.set_xticks(x)
    ax3.set_xticklabels([c.replace('_0.5', '') for c in conditions])
    ax3.legend()
    ax3.set_ylim(0, 0.7)
    ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Summary Table
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')

    # Create summary text
    summary_text = "RQ2 ANSWERS SUMMARY\n" + "="*50 + "\n\n"

    summary_text += "RQ2.1: Does feedback improve accuracy?\n"
    for ft in feedback_types:
        total_imp = np.mean([summary[ft].get(c, {}).get('improvement', 0) for c in conditions])
        emoji = "YES ✓" if total_imp > 0 else "NO ✗"
        summary_text += f"  {ft}: {total_imp:+.3f} → {emoji}\n"

    summary_text += "\nRQ2.4: Best conditions for feedback?\n"
    for cond in conditions:
        avg_imp = np.mean([summary[ft].get(cond, {}).get('improvement', 0) for ft in feedback_types])
        avg_success = np.mean([summary[ft].get(cond, {}).get('improved_pct', 0) for ft in feedback_types])
        emoji = "✓" if avg_imp > 0 else "✗"
        summary_text += f"  {cond}: {avg_imp:+.3f}, {avg_success:.0f}% success {emoji}\n"

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='gray'))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    save_path = os.path.join(results_dir, 'rq2_complete_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Plot saved to {save_path}")
    plt.show()

# RUN

print("RQ2: Iterative Language Feedback Evaluation")
print(f"\nThis evaluation will:")
print(f"  1. Load wrong predictions from RQ1 (IoU < 0.5)")
print(f"  2. Apply {MAX_ITERATIONS} iterations of feedback")
print(f"  3. Test {len(FEEDBACK_TYPES)} feedback types: {FEEDBACK_TYPES}")
print(f"  4. Answer all RQ2 sub-questions")

all_results = run_rq2_evaluation(
    dataset_path=DATASET_PATH,
    rq1_path=RQ1_RESULTS_PATH,
    num_samples=NUM_SAMPLES,
    max_iterations=MAX_ITERATIONS,
    feedback_types=FEEDBACK_TYPES,
    conditions=CONDITIONS,
    iou_threshold=IOU_THRESHOLD,
    checkpoint_every=CHECKPOINT_EVERY
)

print("\n✓ RQ2 Evaluation Complete!")