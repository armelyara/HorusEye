"""
RQ3: Health VQA with BLIP-2
Posture classification: STANDING, SITTING, or LYING.

Proper VQA-style prompting and response decoding.
"""

# SETUP & INSTALL

#!pip install transformers accelerate pillow tqdm bitsandbytes -q

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from transformers import Blip2Processor, Blip2ForConditionalGeneration

#from google.colab import drive
#drive.mount('/content/drive')

print("✓ Packages installed")
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")


# CONFIGURATION

DATASET_PATH = 'HorusEye/horuseye_VLM/refcoco_degraded_benchmark'
CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
MODEL_NAME = "blip2"

print(f"✓ Configuration set")
print(f"  Dataset: {DATASET_PATH}")


# LOAD MODEL
print("Loading BLIP-2 model...")
print("  This may take 3-5 minutes...")

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    torch_dtype=torch.float16,
    device_map="auto"
)

model.eval()
print("✓ BLIP-2 loaded!")


# DEFINE FUNCTIONS

def load_annotations(dataset_path):
    """Load RQ3 human annotations."""
    anno_path = os.path.join(dataset_path, 'rq3_health_annotations.json')
    
    if not os.path.exists(anno_path):
        print(f"❌ Annotations not found at {anno_path}")
        return None
    
    with open(anno_path) as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        annotations = data
    elif 'samples' in data:
        annotations = data['samples']
    elif 'annotations' in data:
        annotations = data['annotations']
    else:
        annotations = data
    
    print(f"✓ Loaded {len(annotations)} annotations")
    return annotations


def predict_posture_blip2(image_path, model, processor):
    """
    Use BLIP-2 to predict posture.
    Returns: (posture, raw_response)
    """
    image = Image.open(image_path).convert('RGB')
    
    # VQA prompt for posture classification
    prompt = "Question: What is this person's posture? Is the person standing, sitting, or lying down? Answer:"
    
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    
    # Store input length for later
    input_length = inputs['input_ids'].shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=20,
            do_sample=False
        )
    
    # Decode only the new tokens, not the input prompt
    generated_ids = outputs[0][input_length:]
    response = processor.decode(generated_ids, skip_special_tokens=True).strip()
    
    # Extract posture from response
    response_upper = response.upper()
    
    if "STANDING" in response_upper or "STAND" in response_upper:
        return "STANDING", response
    elif "SITTING" in response_upper or "SIT" in response_upper:
        return "SITTING", response
    elif "LYING" in response_upper or "LY" in response_upper or "LAY" in response_upper:
        return "LYING", response
    else:
        # If unclear, return the raw response for debugging
        return "UNKNOWN", response


def crop_image(image_path, bbox):
    """Crop image to bounding box and save to temp file."""
    image = Image.open(image_path).convert('RGB')
    x, y, w, h = bbox
    
    # Ensure valid crop region
    img_w, img_h = image.size
    x = max(0, min(int(x), img_w))
    y = max(0, min(int(y), img_h))
    w = min(int(w), img_w - x)
    h = min(int(h), img_h - y)
    
    if w <= 10 or h <= 10:
        return None
    
    cropped = image.crop((x, y, x + w, y + h))
    
    # Save to temp file
    temp_path = "/tmp/cropped_image.jpg"
    cropped.save(temp_path)
    return temp_path


print("✓ Functions defined")


# TEST BEFORE FULL RUN
print("Testing BLIP-2 posture prediction on 3 samples...")


annotations = load_annotations(DATASET_PATH)

if annotations:
    test_samples = annotations[:3]
    
    for i, anno in enumerate(test_samples):
        filename = anno.get('filename')
        gt_posture = anno.get('posture', '').upper()
        
        # Test on clean condition
        image_path = f"{DATASET_PATH}/images/clean/{filename}"
        
        if os.path.exists(image_path):
            pred, response = predict_posture_blip2(image_path, model, processor)
            correct = "✓" if pred == gt_posture else "✗"
            
            print(f"\n[Test {i+1}] {filename}")
            print(f"  GT: {gt_posture}")
            print(f"  Pred: {pred} {correct}")
            print(f"  Raw response: '{response}'")


print("If predictions look reasonable, proceed to Cell 6 for full evaluation.")


# RUN EVALUATION

def run_rq3_evaluation(dataset_path, conditions, model, processor):
    """
    Run RQ3 Health VQA evaluation.
    Compares full image vs cropped (RQ2 bbox) accuracy.
    """
    
    print("RQ3: Health VQA with BLIP-2")
    
    # Load annotations
    annotations = load_annotations(dataset_path)
    if not annotations:
        return None
    
    # Load RQ2 bboxes (for cropping)
    rq2_path = os.path.join(dataset_path, 'results', 'rq2_complete_results_combined.json')
    rq2_samples = {}
    
    if os.path.exists(rq2_path):
        with open(rq2_path) as f:
            rq2_data = json.load(f)
        rq2_samples = {s['sample_id']: s for s in rq2_data.get('samples', [])}
        print(f"✓ Loaded {len(rq2_samples)} RQ2 samples for cropping")
    else:
        print("⚠ No RQ2 results found, using ground truth bbox for cropping")
    
    results = {cond: {
        'full_correct': 0, 
        'crop_correct': 0, 
        'total': 0,
        'full_predictions': [],
        'crop_predictions': []
    } for cond in conditions}
    
    processed_samples = []
    
    for anno in tqdm(annotations, desc="RQ3 BLIP-2"):
        sample_id = anno.get('sample_id')
        filename = anno.get('filename')
        gt_posture = anno.get('posture', '').upper()
        gt_bbox = anno.get('bbox') or anno.get('gt_bbox')
        
        if not gt_posture or not filename:
            continue
        
        # Get RQ2 corrected bbox if available
        rq2_bbox = None
        if sample_id in rq2_samples:
            rq2_sample = rq2_samples[sample_id]
            if 'final_bbox' in rq2_sample:
                rq2_bbox = rq2_sample['final_bbox']
        
        crop_bbox = rq2_bbox if rq2_bbox else gt_bbox
        
        sample_result = {
            'sample_id': sample_id,
            'filename': filename,
            'gt_posture': gt_posture,
            'conditions': {}
        }
        
        for condition in conditions:
            image_path = os.path.join(dataset_path, 'images', condition, filename)
            
            if not os.path.exists(image_path):
                continue
            
            try:
                # Method A: Full image
                pred_full, resp_full = predict_posture_blip2(image_path, model, processor)
                full_correct = pred_full == gt_posture
                
                # Method B: Cropped image
                if crop_bbox:
                    cropped_path = crop_image(image_path, crop_bbox)
                    if cropped_path:
                        pred_crop, resp_crop = predict_posture_blip2(cropped_path, model, processor)
                        crop_correct = pred_crop == gt_posture
                    else:
                        pred_crop, resp_crop = pred_full, resp_full
                        crop_correct = full_correct
                else:
                    pred_crop, resp_crop = pred_full, resp_full
                    crop_correct = full_correct
                
                # Track results
                results[condition]['total'] += 1
                if full_correct:
                    results[condition]['full_correct'] += 1
                if crop_correct:
                    results[condition]['crop_correct'] += 1
                
                results[condition]['full_predictions'].append({
                    'gt': gt_posture, 'pred': pred_full, 'correct': full_correct
                })
                results[condition]['crop_predictions'].append({
                    'gt': gt_posture, 'pred': pred_crop, 'correct': crop_correct
                })
                
                sample_result['conditions'][condition] = {
                    'pred_full': pred_full,
                    'resp_full': resp_full[:100],
                    'pred_crop': pred_crop,
                    'resp_crop': resp_crop[:100],
                    'full_correct': full_correct,
                    'crop_correct': crop_correct
                }
                
            except Exception as e:
                print(f"Error on {filename} ({condition}): {e}")
                continue
        
        processed_samples.append(sample_result)
    
    # Print results
    print("RQ3 RESULTS: BLIP-2")
    print(f"\n{'Condition':<15} {'Full Image':<15} {'Cropped':<15} {'Gain':<10}")
    
    summary = {}
    for condition in conditions:
        r = results[condition]
        if r['total'] > 0:
            full_acc = r['full_correct'] / r['total'] * 100
            crop_acc = r['crop_correct'] / r['total'] * 100
            gain = crop_acc - full_acc
            
            summary[condition] = {
                'full_accuracy': float(full_acc),
                'crop_accuracy': float(crop_acc),
                'gain': float(gain),
                'total': r['total']
            }
            
            gain_str = f"{gain:+.1f}%"
            print(f"{condition:<15} {full_acc:<15.1f} {crop_acc:<15.1f} {gain_str:<10}")
    
    # Key finding
    print("KEY FINDINGS")
    
    if summary:
        clean_full = summary.get('clean', {}).get('full_accuracy', 0)
        clean_crop = summary.get('clean', {}).get('crop_accuracy', 0)
        thermal_full = summary.get('thermal_0.5', {}).get('full_accuracy', 0)
        thermal_crop = summary.get('thermal_0.5', {}).get('crop_accuracy', 0)
        
        print(f"\n1. Clean: Full={clean_full:.1f}% → Cropped={clean_crop:.1f}% ({clean_crop-clean_full:+.1f}%)")
        print(f"2. Thermal: Full={thermal_full:.1f}% → Cropped={thermal_crop:.1f}% ({thermal_crop-thermal_full:+.1f}%)")
        
        if thermal_crop > thermal_full:
            print(f"\n   ✓ Cropping helps in thermal conditions!")
        elif thermal_full > thermal_crop:
            print(f"\n   ⚠ Full image performs better in thermal (crop may cut off body parts)")
    
    # Save results
    output = {
        'metadata': {
            'model': 'BLIP-2-OPT-2.7B',
            'experiment': 'RQ3 - Health VQA Posture Classification',
            'timestamp': datetime.now().isoformat()
        },
        'summary': summary,
        'samples': processed_samples
    }
    
    save_path = os.path.join(dataset_path, 'results', f'rq3_{MODEL_NAME}_results.json')
    with open(save_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to {save_path}")
    
    return output


# Run the evaluation
print("Starting RQ3 evaluation with BLIP-2...")
results = run_rq3_evaluation(DATASET_PATH, CONDITIONS, model, processor)


# DEBUG - CHECK PREDICTIONS

def debug_check_predictions(dataset_path, num_to_check=5):
    """Check BLIP-2 predictions in detail."""
    print("DEBUG: Checking BLIP-2 Posture Predictions")
    
    results_path = os.path.join(dataset_path, 'results', f'rq3_{MODEL_NAME}_results.json')
    
    if not os.path.exists(results_path):
        print("No results file found!")
        return
    
    with open(results_path) as f:
        data = json.load(f)
    
    samples = data['samples'][:num_to_check]
    
    for sample in samples:
        print(f"\n[{sample['filename']}]")
        print(f"  GT Posture: {sample['gt_posture']}")
        
        for cond in CONDITIONS:
            if cond in sample.get('conditions', {}):
                c = sample['conditions'][cond]
                full_mark = "✓" if c['full_correct'] else "✗"
                crop_mark = "✓" if c['crop_correct'] else "✗"
                
                print(f"\n  [{cond}]")
                print(f"    Full: {c['pred_full']} {full_mark} ('{c['resp_full'][:50]}...')")
                print(f"    Crop: {c['pred_crop']} {crop_mark} ('{c['resp_crop'][:50]}...')")

# Run debug
debug_check_predictions(DATASET_PATH, num_to_check=3)

print("RQ3 BLIP-2 COMPLETE!")
