#  SETUP & INSTALL
import os
import shutil
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

#!pip install transformers accelerate pillow tqdm bitsandbytes qwen-vl-utils -q

print("✓ Packages installed")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# CONFIGURATION

DATASET_PATH = '/content/refcoco_degraded_benchmark'
CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
MODEL_NAME = "qwen2vl"
NUM_SAMPLES = None  # ALL samples (3,811 images × 4 conditions = 15,244 total)
NUM_ROUNDS = 3

# Output paths
OUTPUT_DIR = '/content/refcoco_degraded_benchmark/results'
CHECKPOINT_PATH = f'{OUTPUT_DIR}/rq2_qwen2vl_checkpoint.json'
RESULTS_PATH = f'{OUTPUT_DIR}/rq2_qwen2vl_fixed_results.json'
CHECKPOINT_EVERY = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"✓ Configuration set")
print(f"  Dataset: {DATASET_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Samples: ALL, Rounds: {NUM_ROUNDS}")
print(f"  Checkpoint every: {CHECKPOINT_EVERY} samples")

# LOAD MODEL
# 4-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=quantization_config
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

model.eval()

#  HELPER FUNCTIONS
import re

def calculate_iou(bbox1, bbox2):
    """Calculate IoU between two [x, y, w, h] bboxes."""
    if not bbox1 or not bbox2:
        return 0.0

    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    # Convert to corners
    x1_min, y1_min = x1, y1
    x1_max, y1_max = x1 + w1, y1 + h1
    x2_min, y2_min = x2, y2
    x2_max, y2_max = x2 + w2, y2 + h2

    # Intersection
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0

    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def parse_bbox_from_response(response, img_width, img_height):
    """
    Parse bounding box from Qwen2-VL response.
    Handles formats: <|box_start|>(x1,y1),(x2,y2)<|box_end|> or just (x1,y1),(x2,y2)
    Returns [x, y, w, h] in pixel coordinates.
    """
    response = response.strip()

    # Pattern 1: With box tokens - <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    pattern1 = r'\((\d+),(\d+)\),\((\d+),(\d+)\)'

    # Pattern 2: Plain coordinates
    pattern2 = r'\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]'

    # Try pattern 1 first
    match = re.search(pattern1, response)
    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        # Convert from 0-1000 normalized to pixels
        x1 = x1 * img_width / 1000
        y1 = y1 * img_height / 1000
        x2 = x2 * img_width / 1000
        y2 = y2 * img_height / 1000
        return [x1, y1, x2 - x1, y2 - y1]

    # Try pattern 2
    match = re.search(pattern2, response)
    if match:
        coords = list(map(int, match.groups()))
        return coords

    return None

def generate_feedback(pred_bbox, gt_bbox, img_width, img_height):
    """Generate language feedback comparing predicted bbox to ground truth."""
    if not pred_bbox:
        return "I couldn't detect the bounding box. Please look more carefully at the image and provide coordinates in the format (x1,y1),(x2,y2)."

    px, py, pw, ph = pred_bbox
    gx, gy, gw, gh = gt_bbox

    # Calculate centers
    pred_cx, pred_cy = px + pw/2, py + ph/2
    gt_cx, gt_cy = gx + gw/2, gy + gh/2

    feedback_parts = []

    # Horizontal feedback
    x_diff = gt_cx - pred_cx
    threshold = img_width * 0.05  # 5% threshold

    if x_diff > threshold:
        feedback_parts.append("The box should be more to the RIGHT")
    elif x_diff < -threshold:
        feedback_parts.append("The box should be more to the LEFT")

    # Vertical feedback
    y_diff = gt_cy - pred_cy
    threshold = img_height * 0.05

    if y_diff > threshold:
        feedback_parts.append("The box should be more DOWN")
    elif y_diff < -threshold:
        feedback_parts.append("The box should be more UP")

    # Size feedback
    pred_area = pw * ph
    gt_area = gw * gh
    area_ratio = pred_area / gt_area if gt_area > 0 else 1

    if area_ratio < 0.7:
        feedback_parts.append("The box should be LARGER")
    elif area_ratio > 1.4:
        feedback_parts.append("The box should be SMALLER")

    if not feedback_parts:
        feedback_parts.append("The box is close but can be more precise")

    feedback = ". ".join(feedback_parts) + ". Please provide a corrected bounding box."
    return feedback


print("✓ Helper functions defined")

#  PREDIC WITH FEEDBACK FUNCTION

def predict_with_feedback_fixed(image_path, expression, gt_bbox, model, processor, num_rounds=3, iou_threshold=0.5):
    """
    Run multi-round prediction with language feedback.

    1. Image passed in ALL rounds
    2. Proper conversation history maintained
    3. Decode only NEW tokens
    4. last_response properly initialized
    5. Filter: Only run feedback if R1 IoU < threshold

    Args:
        iou_threshold: Skip feedback rounds if R1 IoU >= this value (default 0.5)

    Returns:
        dict with IoU for each round, final bbox, and 'skipped' flag
    """
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    results = {
        'rounds': [],
        'final_bbox': None,
        'final_iou': 0.0,
        'skipped': False  # True if R1 IoU >= threshold
    }

    current_bbox = None
    last_response = ""

    # Maintain full conversation history
    full_conversation = []  # Proper history tracking

    for round_num in range(num_rounds):
        if round_num == 0:
            # Round 1: Initial prediction
            prompt = f"Locate '{expression}' in the image and provide the bounding box coordinates in format (x1,y1),(x2,y2)."

            # First message includes image
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        else:
            # Round 2+: Feedback-based correction
            feedback = generate_feedback(current_bbox, gt_bbox, img_width, img_height)
            prompt = feedback

            # Build PROPER multi-turn conversation
            # Include image in first user message, then continue conversation
            messages = [
                # Original user message with image
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Locate '{expression}' in the image and provide the bounding box coordinates in format (x1,y1),(x2,y2)."}
                    ]
                },
                # Assistant's previous response
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": last_response}]
                },
                # User's feedback
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # ALWAYS pass the image
        inputs = processor(
            text=[text],
            images=[image],  # Always include image!
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Store input length
        input_length = inputs['input_ids'].shape[1]

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        # Decode only new tokens (not the prompt)
        generated_ids = outputs[0][input_length:]
        response = processor.decode(generated_ids, skip_special_tokens=True).strip()

        # Store for next round
        last_response = response

        # Parse bbox
        current_bbox = parse_bbox_from_response(response, img_width, img_height)

        # Calculate IoU
        if current_bbox:
            iou = calculate_iou(current_bbox, gt_bbox)
        else:
            iou = 0.0

        results['rounds'].append({
            'round': round_num + 1,
            'prompt': prompt[:200],
            'response': response[:500],
            'bbox': current_bbox,
            'iou': float(iou)
        })

        # Debug output for first few samples
        if round_num == 0:
            print(f"R1: IoU={iou:.3f}, bbox={current_bbox}")

            # FILTER: Skip feedback rounds if R1 IoU >= threshold
            if iou >= iou_threshold:
                print(f"SKIPPED: R1 IoU ({iou:.3f}) >= {iou_threshold} threshold")
                results['skipped'] = True
                results['final_bbox'] = current_bbox
                results['final_iou'] = iou
                return results
        else:
            print(f"    R{round_num+1}: IoU={iou:.3f}, feedback='{prompt[:50]}...'")

    # Set final results
    if current_bbox:
        results['final_bbox'] = current_bbox
        results['final_iou'] = calculate_iou(current_bbox, gt_bbox)

    return results


print("✓ predict_with_feedback function defined")

# LOAD DATASET
def load_dataset(dataset_path, num_samples=None):
    """Load dataset from RQ1 results file."""
    rq1_path = os.path.join(dataset_path, 'results', 'rq1_with_bboxes.json')

    with open(rq1_path) as f:
        rq1_data = json.load(f)

    samples = rq1_data['samples']

    # Ensure bbox field exists
    for s in samples:
        if 'gt_bbox' in s and 'bbox' not in s:
            s['bbox'] = s['gt_bbox']

    if num_samples:
        samples = samples[:num_samples]

    print(f"✓ Loaded {len(samples)} samples")
    return samples

# TEST BEFORE FULL RUN
print("Filter: R1 IoU < 0.5 will get feedback, R1 IoU >= 0.5 will be skipped")

samples = load_dataset(DATASET_PATH, num_samples=5)

# Test on first 2 samples
for i, sample in enumerate(samples[:2]):
    filename = sample['filename']
    expression = sample['expression']
    gt_bbox = sample['bbox']

    # Test on clean condition
    image_path = f"{DATASET_PATH}/images/clean/{filename}"

    if os.path.exists(image_path):
        print(f"\n[Test {i+1}] {filename}")
        print(f"  Expression: '{expression}'")
        print(f"  GT bbox: {gt_bbox}")

        result = predict_with_feedback_fixed(
            image_path, expression, gt_bbox,
            model, processor, num_rounds=NUM_ROUNDS,
            iou_threshold=0.5
        )

        if result.get('skipped', False):
            print(f"\n SKIPPED (R1 IoU >= 0.5)")
            print(f" R1 IoU: {result['rounds'][0]['iou']:.3f}")
        else:
            print(f"\n  Round IoUs: {[r['iou'] for r in result['rounds']]}")
            print(f"  Improvement: R1={result['rounds'][0]['iou']:.3f} → R3={result['rounds'][-1]['iou']:.3f}")

            # Check if rounds are different
            ious = [r['iou'] for r in result['rounds']]
            if len(ious) == 3 and ious[0] == ious[1] == ious[2]:
                print(f" WARNING: All rounds identical! Feedback may not be working.")
            else:
                print(f"  ✓ Rounds show variation - feedback is working!")

# RUN FULL EVALUATION
def save_checkpoint(checkpoint_data):
    """Save checkpoint to Google Drive for persistence."""
    # Save checkpoint
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(checkpoint_data, f, indent=2, default=float)
    # Also save intermediate results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(checkpoint_data['all_results'], f, indent=2, default=float)
    print(f" Saved: {checkpoint_data['current_condition']} - {checkpoint_data['current_idx']}/{checkpoint_data['total_samples']}")

def load_checkpoint():
    """Load checkpoint if exists. Falls back to results file."""
    if os.path.exists(CHECKPOINT_PATH):
        print(f" Found checkpoint: {CHECKPOINT_PATH}")
        try:
            with open(CHECKPOINT_PATH) as f:
                checkpoint_data = json.load(f)
            # Validate essential keys
            if 'current_condition' in checkpoint_data and \
               'current_idx' in checkpoint_data and \
               'total_samples' in checkpoint_data and \
               'all_results' in checkpoint_data:
                return checkpoint_data
            else:
                print(f" Warning: Checkpoint file at {CHECKPOINT_PATH} is incomplete or corrupted. Starting fresh.")
                return None
        except json.JSONDecodeError:
            print(f" Warning: Checkpoint file at {CHECKPOINT_PATH} is malformed. Starting fresh.")
            return None
    elif os.path.exists(RESULTS_PATH):
        print(f" No checkpoint, but found results file: {RESULTS_PATH}")
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
        if all_results.get('samples'):
            last_sample = all_results['samples'][-1]
            last_condition = last_sample.get('condition', 'clean')
            condition_samples = [s for s in all_results['samples'] if s.get('condition') == last_condition]
            completed_conditions = list(all_results.get('summary', {}).keys())
            if last_condition in completed_conditions:
                completed_conditions.remove(last_condition)
            return {
                'current_condition': last_condition,
                'current_idx': len(condition_samples),
                'total_samples': all_results['metadata'].get('num_samples', 3811),
                'completed_conditions': completed_conditions,
                'all_results': all_results
            }
    return None

def run_rq2_evaluation(dataset_path, conditions, model, processor, num_samples, num_rounds):
    """Run RQ2 feedback evaluation across all conditions with checkpointing."""

    samples = load_dataset(dataset_path, num_samples)

    # Check for existing checkpoint
    checkpoint = load_checkpoint()
    if checkpoint:
        print(f"\n RESUMING from checkpoint:")
        print(f"   Condition: {checkpoint['current_condition']}")
        print(f"   Sample: {checkpoint['current_idx']}/{checkpoint['total_samples']}")
        all_results = checkpoint['all_results']
        start_condition_idx = conditions.index(checkpoint['current_condition'])
        start_sample_idx = checkpoint['current_idx']
        completed_conditions = checkpoint.get('completed_conditions', [])
    else:
        print("\n Starting fresh evaluation...")
        all_results = {
            'metadata': {
                'model': 'Qwen2-VL-2B-Instruct',
                'num_samples': len(samples),
                'num_rounds': num_rounds,
                'timestamp': datetime.now().isoformat(),
                'iou_filter_threshold': 0.5,
                'filter_description': 'Feedback rounds (R2, R3) only run on samples with R1 IoU < 0.5',
                'fixes_applied': [
                    'Image passed in ALL rounds',
                    'Proper conversation history',
                    'Decode only NEW tokens',
                    'last_response initialized',
                    'Filter: Skip feedback if R1 IoU >= 0.5'
                ]
            },
            'summary': {},
            'samples': []
        }
        start_condition_idx = 0
        start_sample_idx = 0
        completed_conditions = []

    for cond_idx, condition in enumerate(conditions):
        # Skip already completed conditions
        if condition in completed_conditions:
            print(f"\n⏭ Skipping {condition} (already completed)")
            continue

        # Skip conditions before resume point
        if cond_idx < start_condition_idx:
            continue

        print(f"Processing condition: {condition}")

        # Initialize or load from checkpoint
        if cond_idx == start_condition_idx and start_sample_idx > 0:
            # Resuming mid-condition: load existing results
            condition_results = [s for s in all_results['samples'] if s.get('condition') == condition]
            round_ious = {r: [] for r in range(1, num_rounds + 1)}
            skipped_count = 0
            processed_count = 0
            # Rebuild stats from existing results
            for r in condition_results:
                if r.get('skipped', False):
                    skipped_count += 1
                    round_ious[1].append(r['rounds'][0]['iou'])
                else:
                    processed_count += 1
                    for rnd in r['rounds']:
                        round_ious[rnd['round']].append(rnd['iou'])
            print(f"  📂 Loaded {len(condition_results)} existing results, resuming from sample {start_sample_idx}")
        else:
            condition_results = []
            round_ious = {r: [] for r in range(1, num_rounds + 1)}
            skipped_count = 0
            processed_count = 0

        for idx, sample in enumerate(tqdm(samples, desc=condition)):
            # Skip samples before resume point
            if cond_idx == start_condition_idx and idx < start_sample_idx:
                continue

            filename = sample['filename']
            expression = sample['expression']
            gt_bbox = sample['bbox']

            image_path = os.path.join(dataset_path, 'images', condition, filename)

            if not os.path.exists(image_path):
                continue

            try:
                result = predict_with_feedback_fixed(
                    image_path, expression, gt_bbox,
                    model, processor, num_rounds,
                    iou_threshold=0.5  # Filter: only run feedback if R1 IoU < 0.5
                )

                # Track skipped vs processed
                if result.get('skipped', False):
                    skipped_count += 1
                    # Only R1 exists for skipped samples
                    round_ious[1].append(result['rounds'][0]['iou'])
                else:
                    processed_count += 1
                    # Track all round IoUs for processed samples
                    for r in result['rounds']:
                        round_ious[r['round']].append(r['iou'])

                sample_result = {
                    'sample_id': sample.get('sample_id', idx),
                    'filename': filename,
                    'expression': expression,
                    'gt_bbox': gt_bbox,
                    'rounds': result['rounds'],
                    'final_iou': result['final_iou'],
                    'skipped': result.get('skipped', False),
                    'condition': condition  # Add condition early for checkpoint
                }
                condition_results.append(sample_result)
                all_results['samples'].append(sample_result)

                # Save checkpoint every N samples
                if (idx + 1) % CHECKPOINT_EVERY == 0:
                    save_checkpoint({
                        'current_condition': condition,
                        'current_idx': idx + 1,
                        'total_samples': len(samples),
                        'completed_conditions': completed_conditions,
                        'all_results': all_results
                    })

            except Exception as e:
                print(f"Error on {filename}: {e}")
                continue

        # Calculate summary for this condition
        if condition_results:
            # R1 includes ALL samples, R2/R3 only processed (non-skipped)
            r1_mean = np.mean(round_ious[1]) if round_ious[1] else 0
            r2_mean = np.mean(round_ious[2]) if round_ious[2] else 0
            r3_mean = np.mean(round_ious[3]) if round_ious[3] else 0

            # For improvement calculation, use only processed samples
            processed_r1_ious = [r['rounds'][0]['iou'] for r in condition_results if not r.get('skipped', False)]
            processed_r1_mean = np.mean(processed_r1_ious) if processed_r1_ious else 0

            all_results['summary'][condition] = {
                'round_1': {'mean_iou': float(r1_mean), 'count': len(round_ious[1])},
                'round_2': {'mean_iou': float(r2_mean), 'count': len(round_ious[2])},
                'round_3': {'mean_iou': float(r3_mean), 'count': len(round_ious[3])},
                'processed_r1_mean': float(processed_r1_mean),  # R1 mean for samples that got feedback
                'improvement_r1_to_r3': float(r3_mean - processed_r1_mean) if processed_r1_mean > 0 else 0,
                'improvement_pct': float((r3_mean - processed_r1_mean) / processed_r1_mean * 100) if processed_r1_mean > 0 else 0,
                'total_samples': len(condition_results),
                'processed_samples': processed_count,
                'skipped_samples': skipped_count,
                'skip_rate': float(skipped_count / len(condition_results) * 100) if condition_results else 0
            }

            print(f"\n{condition} Summary:")
            print(f"  Total: {len(condition_results)} | Processed: {processed_count} | Skipped (IoU≥0.5): {skipped_count}")
            print(f"  R1 (all): {r1_mean:.4f}")
            print(f"  R1 (processed only): {processed_r1_mean:.4f}")
            print(f"  R2 (processed): {r2_mean:.4f}")
            print(f"  R3 (processed): {r3_mean:.4f}")
            if processed_r1_mean > 0:
                print(f"  Improvement: {r3_mean - processed_r1_mean:+.4f} ({(r3_mean - processed_r1_mean) / processed_r1_mean * 100:+.1f}%) Kishan)")

        # Mark condition as completed
        completed_conditions.append(condition)

        # Save checkpoint after completing condition
        save_checkpoint({
            'current_condition': condition,
            'current_idx': len(samples),  # Completed
            'total_samples': len(samples),
            'completed_conditions': completed_conditions,
            'all_results': all_results
        })
        print(f"  {condition} completed!")

        # Reset start_sample_idx for next conditions
        start_sample_idx = 0

    # All done - remove checkpoint file
    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        print("\n Checkpoint file removed (evaluation complete)")

    # Print final summary
    print("FINAL SUMMARY: RQ2 Qwen2-VL")


    print(f"\n{'Condition':<15} {'R1 IoU':<10} {'R2 IoU':<10} {'R3 IoU':<10} {'Δ (R1→R3)':<12}")

    for condition in conditions:
        if condition in all_results['summary']:
            s = all_results['summary'][condition]
            delta = s['improvement_r1_to_r3']
            print(f"{condition:<15} {s['round_1']['mean_iou']:<10.4f} {s['round_2']['mean_iou']:<10.4f} {s['round_3']['mean_iou']:<10.4f} {delta:+.4f} ({s['improvement_pct']:+.1f}%)")

    # Save results
    with open(RESULTS_PATH, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    print(f"\n✓ Results saved to {RESULTS_PATH}")

    return all_results


# Run evaluation
print("Starting RQ2 evaluation with Qwen2-VL...")
results = run_rq2_evaluation(
    DATASET_PATH, CONDITIONS, model, processor,
    NUM_SAMPLES, NUM_ROUNDS
)

