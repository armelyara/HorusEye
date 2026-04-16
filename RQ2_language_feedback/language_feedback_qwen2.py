# SETUP & INSTALL

#!pip install transformers accelerate pillow tqdm bitsandbytes qwen-vl-utils -q

import os
import json
import re
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

print("✓ Packages installed")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# CONFIGURATION

DATASET_PATH = 'HorusEye/horuseye_VLM/refcoco_degraded_benchmark'
CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
NUM_SAMPLES = 100  # Use subset for RQ2 (feedback is slow)
CHECKPOINT_EVERY = 20
MODEL_NAME = "qwen2vl"
NUM_ROUNDS = 3  # Number of feedback rounds

print(f"✓ Configuration set")
print(f"  Dataset: {DATASET_PATH}")
print(f"  Samples: {NUM_SAMPLES}")
print(f"  Feedback rounds: {NUM_ROUNDS}")


# LOAD MODEL

print("Loading Qwen2-VL model...")
print("  This may take 3-5 minutes...")

# 4-bit quantization for T4 GPU
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16
)

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct",
    trust_remote_code=True
)

model.eval()
print("✓ Qwen2-VL loaded!")


# DEFINE FUNCTIONS

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


def parse_bbox_from_response(response, img_width, img_height):
    """
    Parse bounding box from Qwen2-VL response.
    Qwen2-VL returns bbox in format: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    Coordinates are normalized 0-1000.
    """
    # Pattern 1: <|box_start|>(x1,y1),(x2,y2)<|box_end|>
    pattern1 = r'<\|box_start\|>\((\d+),(\d+)\),\((\d+),(\d+)\)<\|box_end\|>'
    match = re.search(pattern1, response)

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        x1 = x1 / 1000 * img_width
        y1 = y1 / 1000 * img_height
        x2 = x2 / 1000 * img_width
        y2 = y2 / 1000 * img_height
        return [x1, y1, x2 - x1, y2 - y1]

    # Pattern 2: [x1, y1, x2, y2] or (x1, y1, x2, y2)
    pattern2 = r'[\[\(](\d+)[,\s]+(\d+)[,\s]+(\d+)[,\s]+(\d+)[\]\)]'
    match = re.search(pattern2, response)

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        if max(x1, y1, x2, y2) <= 1000:
            x1 = x1 / 1000 * img_width
            y1 = y1 / 1000 * img_height
            x2 = x2 / 1000 * img_width
            y2 = y2 / 1000 * img_height
        return [x1, y1, x2 - x1, y2 - y1]

    return None


def generate_feedback(pred_bbox, gt_bbox, img_width, img_height):
    """
    Generate natural language feedback about the prediction error.
    """
    if pred_bbox is None:
        return "I couldn't find a bounding box in your response. Please provide coordinates in the format (x1,y1),(x2,y2)."

    px, py, pw, ph = pred_bbox
    gx, gy, gw, gh = gt_bbox

    # Calculate center points
    pred_cx, pred_cy = px + pw/2, py + ph/2
    gt_cx, gt_cy = gx + gw/2, gy + gh/2

    feedback_parts = []

    # Position feedback
    dx = gt_cx - pred_cx
    dy = gt_cy - pred_cy

    if abs(dx) > img_width * 0.1:
        direction = "right" if dx > 0 else "left"
        feedback_parts.append(f"The box should be more to the {direction}")

    if abs(dy) > img_height * 0.1:
        direction = "down" if dy > 0 else "up"
        feedback_parts.append(f"The box should be more {direction}")

    # Size feedback
    size_ratio_w = gw / (pw + 1e-6)
    size_ratio_h = gh / (ph + 1e-6)

    if size_ratio_w > 1.2:
        feedback_parts.append("The box should be wider")
    elif size_ratio_w < 0.8:
        feedback_parts.append("The box should be narrower")

    if size_ratio_h > 1.2:
        feedback_parts.append("The box should be taller")
    elif size_ratio_h < 0.8:
        feedback_parts.append("The box should be shorter")

    if not feedback_parts:
        return "The box is close but can be more precise. Please refine the bounding box."

    return ". ".join(feedback_parts) + ". Please provide a corrected bounding box."


def predict_with_feedback(image_path, expression, gt_bbox, model, processor, num_rounds=3):
    """
    Run multi-round prediction with language feedback.

    Returns:
        dict with IoU for each round and final bbox
    """
    image = Image.open(image_path).convert('RGB')
    img_width, img_height = image.size

    results = {
        'rounds': [],
        'final_bbox': None,
        'final_iou': 0.0
    }

    conversation_history = []
    current_bbox = None

    for round_num in range(num_rounds):
        if round_num == 0:
            # Round 1: Initial prediction
            prompt = f"Locate '{expression}' in the image and provide the bounding box coordinates."
        else:
            # Round 2+: Feedback-based correction
            feedback = generate_feedback(current_bbox, gt_bbox, img_width, img_height)
            prompt = f"{feedback}"

        # Build conversation
        conversation_history.append({
            "role": "user",
            "content": [
                {"type": "image", "image": image} if round_num == 0 else {"type": "text", "text": ""},
                {"type": "text", "text": prompt}
            ]
        })

        # For Qwen2-VL, we need to handle the conversation format
        if round_num == 0:
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
            # Continue conversation with feedback
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": f"Locate '{expression}' in the image and provide the bounding box coordinates."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": last_response}]
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process inputs
        inputs = processor(
            text=[text],
            images=[image] if round_num == 0 else None,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False
            )

        # Decode response
        response = processor.decode(outputs[0], skip_special_tokens=True)
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

    # Set final results
    if current_bbox:
        results['final_bbox'] = current_bbox
        results['final_iou'] = calculate_iou(current_bbox, gt_bbox)

    return results


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


print("✓ Functions defined")


# TEST BEFORE FULL RUN

print("="*70)
print("Testing Qwen2-VL RQ2 Feedback Loop on 2 samples...")
print("="*70)

samples = load_dataset(DATASET_PATH, num_samples=2)

for i, sample in enumerate(samples[:2]):
    filename = sample['filename']
    expression = sample['expression']
    gt_bbox = sample.get('bbox') or sample.get('gt_bbox')

    # Test on clean condition
    image_path = f"{DATASET_PATH}/images/clean/{filename}"

    if os.path.exists(image_path):
        print(f"\n[Test {i+1}] {filename}")
        print(f"  Expression: {expression}")
        print(f"  GT bbox: {gt_bbox}")

        result = predict_with_feedback(
            image_path, expression, gt_bbox,
            model, processor, num_rounds=NUM_ROUNDS
        )

        for r in result['rounds']:
            print(f"  Round {r['round']}: IoU = {r['iou']:.3f}")

        print(f"  Final IoU: {result['final_iou']:.3f}")

print("\n" + "="*70)
print("If feedback loop is working, proceed to Cell 6 for full evaluation.")
print("="*70)


# RUN EVALUATION

def run_rq2_evaluation(dataset_path, conditions, num_samples=None, num_rounds=3, checkpoint_every=20):
    """Run RQ2 feedback evaluation with Qwen2-VL."""

    print(f"RQ2: Language Feedback with Qwen2-VL ({num_rounds} rounds)")

    results_dir = os.path.join(dataset_path, 'results')
    os.makedirs(results_dir, exist_ok=True)

    checkpoint_path = os.path.join(results_dir, f'rq2_{MODEL_NAME}_checkpoint.json')

    # Load checkpoint if exists
    start_idx = 0
    round_ious = {cond: {f'round_{r+1}': [] for r in range(num_rounds)} for cond in conditions}
    processed_samples = []

    if os.path.exists(checkpoint_path):
        print("\n✓ Found checkpoint, resuming...")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        start_idx = checkpoint.get('last_index', 0) + 1
        round_ious = checkpoint.get('round_ious', round_ious)
        processed_samples = checkpoint.get('samples', [])
        print(f"  Resuming from sample {start_idx}")

    samples = load_dataset(dataset_path, num_samples)

    print(f"\n  Total samples: {len(samples)}")
    print(f"  Remaining: {len(samples) - start_idx}")

    total = len(samples) * len(conditions)
    completed = start_idx * len(conditions)
    pbar = tqdm(total=total, initial=completed, desc=f"RQ2 {MODEL_NAME}")

    errors = 0

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
                result = predict_with_feedback(
                    image_path, expression, gt_bbox,
                    model, processor, num_rounds=num_rounds
                )

                sample_result['conditions'][condition] = result

                # Track IoU by round
                for r in result['rounds']:
                    round_key = f"round_{r['round']}"
                    round_ious[condition][round_key].append(r['iou'])

            except Exception as e:
                errors += 1
                sample_result['conditions'][condition] = {
                    'error': str(e),
                    'rounds': [],
                    'final_iou': 0.0
                }

            pbar.update(1)

            # Show progress
            if len(round_ious[condition]['round_1']) > 0:
                r1_avg = np.mean(round_ious[condition]['round_1'])
                r3_avg = np.mean(round_ious[condition][f'round_{num_rounds}'])
                pbar.set_postfix({'cond': condition[:6], 'R1': f'{r1_avg:.3f}', f'R{num_rounds}': f'{r3_avg:.3f}'})

        processed_samples.append(sample_result)

        # Save checkpoint
        if (idx + 1) % checkpoint_every == 0:
            checkpoint = {
                'last_index': idx,
                'round_ious': {
                    cond: {k: [float(v) for v in vals] for k, vals in cond_rounds.items()}
                    for cond, cond_rounds in round_ious.items()
                },
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
    print(f"RQ2 RESULTS: Qwen2-VL")

    summary = {}

    # Header
    header = f"{'Condition':<15}"
    for r in range(num_rounds):
        header += f"{'Round '+str(r+1):<12}"
    header += f"{'Δ(R1→R3)':<12}"
    print(f"\n{header}")
    print("-"*len(header))

    for condition in conditions:
        summary[condition] = {}
        row = f"{condition:<15}"

        round_avgs = []
        for r in range(num_rounds):
            round_key = f"round_{r+1}"
            ious = round_ious[condition][round_key]
            if len(ious) > 0:
                avg = np.mean(ious)
                round_avgs.append(avg)
                summary[condition][round_key] = {
                    'mean_iou': float(avg),
                    'std': float(np.std(ious)),
                    'n': len(ious)
                }
                row += f"{avg:<12.4f}"
            else:
                round_avgs.append(0)
                row += f"{'N/A':<12}"

        # Calculate improvement
        if len(round_avgs) >= 2 and round_avgs[0] > 0:
            delta = round_avgs[-1] - round_avgs[0]
            delta_pct = delta / round_avgs[0] * 100
            summary[condition]['improvement'] = {
                'absolute': float(delta),
                'relative_pct': float(delta_pct)
            }
            row += f"{delta:+.4f} ({delta_pct:+.1f}%)"
        else:
            row += "N/A"

        print(row)

    print("-"*len(header))

    # Key findings
    print("KEY FINDINGS")

    for condition in conditions:
        if 'improvement' in summary.get(condition, {}):
            imp = summary[condition]['improvement']
            print(f"  {condition}: {imp['absolute']:+.4f} IoU ({imp['relative_pct']:+.1f}%)")

    # Save results
    output = {
        'metadata': {
            'model': 'Qwen2-VL-2B-Instruct',
            'experiment': 'RQ2 - Language Feedback Loop',
            'num_rounds': num_rounds,
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(processed_samples)
        },
        'summary': summary,
        'samples': processed_samples
    }

    save_path = os.path.join(results_dir, f'rq2_{MODEL_NAME}_results.json')
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {save_path}")

    # Delete checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print("✓ Checkpoint deleted")

    return output, summary


# Run evaluation
print("Starting RQ2 evaluation with Qwen2-VL...")
results, summary = run_rq2_evaluation(
    dataset_path=DATASET_PATH,
    conditions=CONDITIONS,
    num_samples=NUM_SAMPLES,
    num_rounds=NUM_ROUNDS,
    checkpoint_every=CHECKPOINT_EVERY
)


# COMPARE WITH GEMINI RQ2

def compare_rq2_models(dataset_path):
    """Compare RQ2 results across models."""

    print("RQ2: MODEL COMPARISON")

    results_dir = os.path.join(dataset_path, 'results')

    models = {
        'Gemini': 'rq2_complete_results_combined.json',
        'Qwen2-VL': 'rq2_qwen2vl_results.json'
    }

    conditions = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']

    model_results = {}

    for model_name, filename in models.items():
        filepath = os.path.join(results_dir, filename)
        if os.path.exists(filepath):
            with open(filepath) as f:
                data = json.load(f)

            if 'summary' in data:
                model_results[model_name] = data['summary']
                print(f"✓ Loaded {model_name}")
            else:
                print(f"⚠ {model_name} has no summary")
        else:
            print(f"✗ {model_name} not found")

    if not model_results:
        print("No model results found!")
        return

    # Print Round 1 IoU comparison
    print("ROUND 1 IoU (Initial Prediction)")

    header = f"{'Condition':<15}" + "".join(f"{m:<15}" for m in model_results.keys())
    print(f"\n{header}")
    print("-"*len(header))

    for cond in conditions:
        row = f"{cond:<15}"
        for model_name in model_results.keys():
            if cond in model_results[model_name]:
                r1 = model_results[model_name][cond].get('round_1', {}).get('mean_iou', 0)
                row += f"{r1:<15.4f}"
            else:
                row += f"{'N/A':<15}"
        print(row)

    # Print Improvement comparison
    print("IMPROVEMENT (Round 1 → Round 3)")

    print(f"\n{header}")
    print("-"*len(header))

    for cond in conditions:
        row = f"{cond:<15}"
        for model_name in model_results.keys():
            if cond in model_results[model_name]:
                imp = model_results[model_name][cond].get('improvement', {})
                if 'relative_pct' in imp:
                    row += f"{imp['relative_pct']:+.1f}%".ljust(15)
                else:
                    row += f"{'N/A':<15}"
            else:
                row += f"{'N/A':<15}"
        print(row)

    # Save comparison
    comparison_path = os.path.join(results_dir, 'rq2_all_models_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(model_results, f, indent=2, default=float)

    print(f"\n✓ Comparison saved to {comparison_path}")


# Run comparison
compare_rq2_models(DATASET_PATH)


print("RQ2 QWEN2-VL COMPLETE!")
