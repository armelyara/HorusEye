"""
RQ4: Hallucination Risk with BLIP-2
Research Question: Do VLMs hallucinate more when vision is degraded?
"""

# SETUP & INSTALL

#!pip install transformers accelerate pillow tqdm bitsandbytes matplotlib -q

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
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
RQ1_RESULTS_PATH = os.path.join(DATASET_PATH, 'results', 'rq1_with_bboxes.json')

CONDITIONS = ['clean', 'fog_0.5', 'smoke_0.5', 'thermal_0.5']
NUM_SAMPLES = 100
CHECKPOINT_EVERY = 25
MODEL_NAME = "blip2"

print(f"✓ Configuration set")
print(f"  Dataset: {DATASET_PATH}")
print(f"  Samples: {NUM_SAMPLES}")


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


# DEFINE HALLUCINATION DETECTION FUNCTIONS

# Objects that are commonly hallucinated
COMMON_HALLUCINATED_OBJECTS = [
    'phone', 'bag', 'hat', 'umbrella', 'book', 'cup', 'bottle', 
    'glasses', 'watch', 'backpack', 'purse', 'wallet', 'keys',
    'dog', 'cat', 'bird', 'car', 'bicycle', 'tree', 'flower'
]

# Hallucination indicator phrases
UNCERTAINTY_PHRASES = [
    'might be', 'could be', 'possibly', 'appears to', 'seems like', 
    'i think', 'maybe', 'probably', 'unclear', 'hard to tell', 
    'cannot determine', 'not sure', 'difficult to see'
]

OVERCONFIDENCE_PHRASES = [
    'clearly', 'definitely', 'certainly', 'obviously', 
    'without doubt', 'absolutely', 'undoubtedly'
]

# Colors (hallucination indicator in thermal images)
COLORS = [
    'red', 'blue', 'green', 'yellow', 'white', 'black', 
    'orange', 'purple', 'pink', 'brown'
]


def load_dataset(num_samples=100):
    """Load dataset from RQ1 results, filtering for person-related expressions."""
    with open(RQ1_RESULTS_PATH) as f:
        rq1_data = json.load(f)
    
    samples = rq1_data['samples']
    
    # Filter for person-related expressions
    person_keywords = ['person', 'man', 'woman', 'boy', 'girl', 'guy', 'lady', 
                       'people', 'kid', 'child', 'player', 'worker']
    person_samples = [s for s in samples if any(kw in s['expression'].lower() for kw in person_keywords)]
    
    if len(person_samples) < num_samples:
        remaining = num_samples - len(person_samples)
        other_samples = [s for s in samples if s not in person_samples][:remaining]
        person_samples.extend(other_samples)
    
    samples = person_samples[:num_samples]
    
    print(f"✓ Loaded {len(samples)} samples (person-related filter applied)")
    return samples


def analyze_hallucination_text(response_text, expression):
    """
    Analyze text response for hallucination indicators.
    Matches the exact methodology used for Gemini and Qwen.
    
    Returns:
        dict with hallucination metrics
    """
    response_lower = response_text.lower()
    expression_lower = expression.lower()
    
    # Count uncertainty phrases
    uncertainty_count = sum(1 for phrase in UNCERTAINTY_PHRASES if phrase in response_lower)
    
    # Count fabricated objects
    fabrication_count = 0
    fabricated_items = []
    for obj in COMMON_HALLUCINATED_OBJECTS:
        if obj in response_lower and obj not in expression_lower:
            fabrication_count += 1
            fabricated_items.append(obj)
    
    # Count overconfidence phrases
    overconfidence_count = sum(1 for phrase in OVERCONFIDENCE_PHRASES if phrase in response_lower)
    
    # Response length
    word_count = len(response_text.split())
    
    # Color mentions (hallucination in thermal images)
    color_mentions = sum(1 for color in COLORS if color in response_lower)
    
    # Hallucination score: higher = more hallucination risk
    # Formula from original: fabrication + (overconfidence * 0.5) - (uncertainty * 0.3)
    hallucination_score = fabrication_count + (overconfidence_count * 0.5) - (uncertainty_count * 0.3)
    
    return {
        'uncertainty_count': uncertainty_count,
        'fabrication_count': fabrication_count,
        'fabricated_items': fabricated_items,
        'overconfidence_count': overconfidence_count,
        'word_count': word_count,
        'color_mentions': color_mentions,
        'hallucination_score': hallucination_score
    }


def get_detailed_description_blip2(image_path, expression, model, processor):
    """
    Ask BLIP-2 for detailed description.
    Uses the EXACT same prompt as Gemini and Qwen.
    """
    image = Image.open(image_path).convert('RGB')
    
    # Exact prompt from original RQ4 methodology
    prompt = f"""Look at this image and describe what you see in detail.

Focus on the object described as: "{expression}"

Describe:
1. The object's appearance (color, size, shape)
2. The object's position in the scene
3. What the object is doing
4. Any other objects or people nearby

Be specific and detailed."""
    
    inputs = processor(
        images=image,
        text=prompt,
        return_tensors="pt"
    ).to(model.device, torch.float16)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,  # Same as Qwen
            do_sample=False
        )
    
    response = processor.decode(outputs[0], skip_special_tokens=True).strip()
    return response


print("✓ Functions defined")


# RUN HALLUCINATION TEST

def run_blip2_hallucination_test(dataset_path, num_samples=100, checkpoint_every=25):
    """
    Test BLIP-2 for hallucinations across degraded conditions.
    Matches the exact methodology used for Gemini and Qwen.
    """
    print("RQ4: BLIP-2 Hallucination Test")
    
    results_dir = os.path.join(dataset_path, 'results')
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_path = os.path.join(results_dir, f'rq4_{MODEL_NAME}_checkpoint.json')
    
    # Load checkpoint if exists
    start_idx = 0
    processed_samples = []
    
    if os.path.exists(checkpoint_path):
        print("✓ Found checkpoint, resuming...")
        with open(checkpoint_path) as f:
            checkpoint = json.load(f)
        start_idx = checkpoint.get('last_index', 0) + 1
        processed_samples = checkpoint.get('samples', [])
        print(f"  Resuming from sample {start_idx}")
    
    # Load samples
    samples = load_dataset(num_samples)
    
    # Progress
    total = len(samples) * len(CONDITIONS)
    completed = start_idx * len(CONDITIONS)
    pbar = tqdm(total=total, initial=completed, desc="BLIP-2 Hallucination")
    
    for idx, sample in enumerate(samples):
        if idx < start_idx:
            continue
        
        filename = sample['filename']
        expression = sample['expression']
        
        sample_result = {
            'sample_id': sample['sample_id'],
            'filename': filename,
            'expression': expression,
            'conditions': {}
        }
        
        for condition in CONDITIONS:
            image_path = os.path.join(dataset_path, 'images', condition, filename)
            
            if not os.path.exists(image_path):
                pbar.update(1)
                continue
            
            try:
                # Get detailed description
                description = get_detailed_description_blip2(
                    image_path, expression, model, processor
                )
                
                # Analyze for hallucinations
                analysis = analyze_hallucination_text(description, expression)
                
                sample_result['conditions'][condition] = {
                    'description': description[:500],  # Truncate for storage
                    'hallucination_analysis': analysis
                }
                
            except Exception as e:
                sample_result['conditions'][condition] = {'error': str(e)}
            
            pbar.update(1)
        
        processed_samples.append(sample_result)
        
        # Save checkpoint
        if (idx + 1) % checkpoint_every == 0:
            checkpoint = {
                'last_index': idx,
                'samples': processed_samples,
                'timestamp': datetime.now().isoformat()
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f)
                f.flush()
                os.fsync(f.fileno())
            tqdm.write(f"  ✓ Checkpoint saved at sample {idx + 1}")
    
    pbar.close()
    
    # Save final results
    results = {
        'metadata': {
            'experiment': 'RQ4 - BLIP-2 Hallucination Test',
            'model': 'BLIP-2-OPT-2.7B',
            'timestamp': datetime.now().isoformat(),
            'num_samples': len(processed_samples)
        },
        'samples': processed_samples
    }
    
    save_path = os.path.join(results_dir, f'rq4_{MODEL_NAME}_results.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {save_path}")
    
    # Delete checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
    
    return results


# Run the test
blip2_results = run_blip2_hallucination_test(DATASET_PATH, NUM_SAMPLES, CHECKPOINT_EVERY)


# ANALYZE RESULTS

def analyze_blip2_results(dataset_path):
    """Analyze BLIP-2 hallucination results."""
    
    print("RQ4: BLIP-2 Hallucination Analysis")
    
    results_dir = os.path.join(dataset_path, 'results')
    blip2_path = os.path.join(results_dir, f'rq4_{MODEL_NAME}_results.json')
    
    with open(blip2_path) as f:
        blip2_data = json.load(f)
    
    # Initialize analysis storage
    analysis = {cond: {
        'fabrication': [], 
        'uncertainty': [], 
        'overconfidence': [], 
        'word_count': [], 
        'color_mentions': [], 
        'hallucination_score': []
    } for cond in CONDITIONS}
    
    # Process results
    for sample in blip2_data['samples']:
        for cond in CONDITIONS:
            if cond in sample.get('conditions', {}):
                cond_data = sample['conditions'][cond]
                if 'hallucination_analysis' in cond_data:
                    ha = cond_data['hallucination_analysis']
                    analysis[cond]['fabrication'].append(ha['fabrication_count'])
                    analysis[cond]['uncertainty'].append(ha['uncertainty_count'])
                    analysis[cond]['overconfidence'].append(ha['overconfidence_count'])
                    analysis[cond]['word_count'].append(ha['word_count'])
                    analysis[cond]['color_mentions'].append(ha['color_mentions'])
                    analysis[cond]['hallucination_score'].append(ha['hallucination_score'])
    
    # Print summary
    print("BLIP-2 Hallucination Metrics")
    print(f"\n{'Condition':<15} {'Fabrication':<12} {'Uncertainty':<12} {'Overconf':<12} {'Colors':<10} {'H-Score':<10}")
    
    summary = {}
    for cond in CONDITIONS:
        fab = np.mean(analysis[cond]['fabrication']) if analysis[cond]['fabrication'] else 0
        unc = np.mean(analysis[cond]['uncertainty']) if analysis[cond]['uncertainty'] else 0
        ovc = np.mean(analysis[cond]['overconfidence']) if analysis[cond]['overconfidence'] else 0
        col = np.mean(analysis[cond]['color_mentions']) if analysis[cond]['color_mentions'] else 0
        hsc = np.mean(analysis[cond]['hallucination_score']) if analysis[cond]['hallucination_score'] else 0
        wc = np.mean(analysis[cond]['word_count']) if analysis[cond]['word_count'] else 0
        
        summary[cond] = {
            'fabrication': float(fab),
            'uncertainty': float(unc),
            'overconfidence': float(ovc),
            'color_mentions': float(col),
            'h_score': float(hsc),
            'avg_word_count': float(wc)
        }
        print(f"{cond:<15} {fab:<12.2f} {unc:<12.2f} {ovc:<12.2f} {col:<10.2f} {hsc:<10.2f}")
    
    # Key findings
    print("KEY FINDINGS")
    
    clean_h = summary['clean']['h_score']
    thermal_h = summary['thermal_0.5']['h_score']
    clean_col = summary['clean']['color_mentions']
    thermal_col = summary['thermal_0.5']['color_mentions']
    avg_wc = np.mean([summary[c]['avg_word_count'] for c in CONDITIONS])
    
    print(f"\n1. Average response length: {avg_wc:.1f} words")
    print(f"\n2. H-Score change (clean → thermal): {clean_h:.2f} → {thermal_h:.2f} ({thermal_h - clean_h:+.2f})")
    print(f"\n3. Color mentions (clean → thermal): {clean_col:.2f} → {thermal_col:.2f}")
    
    if thermal_col > clean_col:
        print(f"\n   ⚠️ BLIP-2 hallucinates colors in thermal images!")
    
    if thermal_h > clean_h:
        print(f"\n   ✗ BLIP-2 hallucinates MORE in degraded conditions")
    else:
        print(f"\n   ✓ BLIP-2 does NOT show increased hallucination in degraded conditions")
    
    # Save summary
    summary_data = {
        'model': 'BLIP-2-OPT-2.7B',
        'timestamp': datetime.now().isoformat(),
        'summary': summary,
        'findings': {
            'clean_to_thermal_h_change': float(thermal_h - clean_h),
            'clean_to_thermal_color_change': float(thermal_col - clean_col),
            'avg_word_count': float(avg_wc)
        }
    }
    
    summary_path = os.path.join(results_dir, f'rq4_{MODEL_NAME}_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_path}")
    
    return summary


# Run analysis
summary = analyze_blip2_results(DATASET_PATH)
